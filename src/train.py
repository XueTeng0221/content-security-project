import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.datautils import MELDDataset, FakeAVCelebDataset, FFPlusPlusDataset
from src.datautils.meld import EMOTION2ID
from src.models import AVDeepfakeDetector
from src.utils import (
    evaluate_metrics, evaluate_metrics_multiclass,
    plot_roc, plot_pr, plot_confusion, plot_loss_curve,
)
from src.utils.config import load_config

_DEFAULTS = dict(
    dataset="meld",
    data_root="data/MELD",
    csv_path="",
    out_dir="outputs",
    pretrain_ckpt="",       # path to pretrained encoder weights (for finetune stage)
    pretrain_out_dir="",    # where to save pretrained weights (for meld stage)
    epochs=20, batch_size=8, lr=1e-4, embed_dim=256,
    freeze_visual=True, num_workers=4, num_classes=7,
    align_weight=0.1, seed=42,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    frames, mels, labels = zip(*batch)
    max_t = max(m.shape[-1] for m in mels)
    mels_padded = torch.stack([nn.functional.pad(m, (0, max_t - m.shape[-1])) for m in mels])
    return torch.stack(frames), mels_padded, torch.stack(labels)


def _balanced_sampler(dataset):
    labels = torch.tensor([lbl for _, lbl in dataset.samples])
    counts = torch.bincount(labels)
    weights = (1.0 / counts)[labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def build_loaders(cfg):
    kw = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, collate_fn=collate_fn)
    if cfg.dataset == "fakeavceleb":
        train_set = FakeAVCelebDataset(cfg.data_root, cfg.csv_path.replace("{split}", "train"))
        val_set   = FakeAVCelebDataset(cfg.data_root, cfg.csv_path.replace("{split}", "val"))
        sampler = _balanced_sampler(train_set)
        return DataLoader(train_set, sampler=sampler, **kw), DataLoader(val_set, shuffle=False, **kw)
    elif cfg.dataset == "ffplusplus":
        train_set = FFPlusPlusDataset(cfg.data_root, cfg.csv_path.replace("{split}", "train"))
        val_set   = FFPlusPlusDataset(cfg.data_root, cfg.csv_path.replace("{split}", "val"))
        return DataLoader(train_set, shuffle=True, **kw), DataLoader(val_set, shuffle=False, **kw)
    else:  # meld
        train_set = MELDDataset(cfg.data_root, split="train")
        val_set   = MELDDataset(cfg.data_root, split="dev")
        return DataLoader(train_set, shuffle=True, **kw), DataLoader(val_set, shuffle=False, **kw)


def _load_pretrained_encoders(model, ckpt_path):
    """Load visual_enc and audio_enc weights from a pretrained checkpoint."""
    state = torch.load(ckpt_path, map_location="cpu")
    enc_state = {k: v for k, v in state.items()
                 if k.startswith("visual_enc.") or k.startswith("audio_enc.")}
    missing, unexpected = model.load_state_dict(enc_state, strict=False)
    print(f"Loaded pretrained encoders from {ckpt_path} "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")


def _run_epoch(model, loader, optimizer, criterion, scaler, device, cfg, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for frames, mel, labels in loader:
            frames, mel, labels = frames.to(device), mel.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                logits, align_loss = model(frames, mel)
                loss = criterion(logits, labels) + cfg.align_weight * align_loss
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device, out_dir, num_classes, align_weight=0.1):
    model.eval()
    total_loss, all_labels, all_preds, all_scores = 0.0, [], [], []
    with torch.no_grad():
        for frames, mel, labels in loader:
            frames, mel, labels = frames.to(device), mel.to(device), labels.to(device)
            with torch.amp.autocast("cuda"):
                logits, align_loss = model(frames, mel)
            total_loss += (criterion(logits, labels) + align_weight * align_loss).item()
            probs = torch.softmax(logits, dim=1)
            all_preds.extend(probs.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            if num_classes == 2:
                all_scores.extend(probs[:, 1].cpu().tolist())

    if num_classes == 2:
        metrics = evaluate_metrics(all_labels, all_scores)
        plot_roc(all_labels, all_scores, save_path=os.path.join(out_dir, "roc.png"))
        plot_pr(all_labels, all_scores,  save_path=os.path.join(out_dir, "pr.png"))
        plot_confusion(all_labels, all_preds, save_path=os.path.join(out_dir, "confusion.png"))
    else:
        metrics = evaluate_metrics_multiclass(all_labels, all_preds)
        class_names = [name for name, _ in sorted(EMOTION2ID.items(), key=lambda kv: kv[1])]
        plot_confusion(all_labels, all_preds,
                       save_path=os.path.join(out_dir, "confusion.png"),
                       class_names=class_names)
    return total_loss / len(loader), metrics


def _train_loop(model, train_loader, val_loader, criterion, device, cfg):
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = GradScaler()
    os.makedirs(cfg.out_dir, exist_ok=True)
    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    for epoch in range(cfg.epochs):
        train_loss = _run_epoch(model, train_loader, optimizer, criterion, scaler, device, cfg, train=True)
        scheduler.step()
        val_loss, metrics = evaluate(model, val_loader, criterion, device, cfg.out_dir, cfg.num_classes, cfg.align_weight)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        metrics_str = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"Epoch {epoch+1}/{cfg.epochs} | train={train_loss:.4f} val={val_loss:.4f} {metrics_str}")
        torch.save(model.state_dict(), os.path.join(cfg.out_dir, "ckpt_last.pt"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(cfg.out_dir, "ckpt_best.pt"))

    plot_loss_curve(train_losses, val_losses, save_path=os.path.join(cfg.out_dir, "loss.png"))


def pretrain(cfg):
    """Stage 1: train AV emotion-consistency representation on MELD (7-class + InfoNCE)."""
    assert cfg.dataset == "meld", "pretrain stage requires dataset=meld"
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_loaders(cfg)

    model = AVDeepfakeDetector(
        embed_dim=cfg.embed_dim,
        freeze_visual=cfg.freeze_visual,
        num_classes=cfg.num_classes,   # 7 for MELD
    ).to(device)
    print(f"[pretrain] device={device}  params={sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    criterion = nn.CrossEntropyLoss()
    _train_loop(model, train_loader, val_loader, criterion, device, cfg)

    # Save encoder-only weights for downstream finetune
    save_dir = cfg.pretrain_out_dir or cfg.out_dir
    os.makedirs(save_dir, exist_ok=True)
    enc_state = {k: v for k, v in model.state_dict().items()
                 if k.startswith("visual_enc.") or k.startswith("audio_enc.")}
    torch.save(enc_state, os.path.join(save_dir, "pretrained_encoders.pt"))
    print(f"[pretrain] encoder weights saved to {save_dir}/pretrained_encoders.pt")


def finetune(cfg):
    """Stage 2: binary Real/Fake classification on FakeAVCeleb or FF++, optionally loading pretrained encoders."""
    assert cfg.dataset in ("fakeavceleb", "ffplusplus"), \
        "finetune stage requires dataset=fakeavceleb or ffplusplus"
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_loaders(cfg)

    model = AVDeepfakeDetector(
        embed_dim=cfg.embed_dim,
        freeze_visual=cfg.freeze_visual,
        num_classes=2,
    ).to(device)

    if cfg.pretrain_ckpt:
        _load_pretrained_encoders(model, cfg.pretrain_ckpt)

    print(f"[finetune/{cfg.dataset}] device={device}  params={sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    criterion = nn.CrossEntropyLoss()
    _train_loop(model, train_loader, val_loader, criterion, device, cfg)


def train(cfg):
    """Legacy single-stage entry point (used when dataset != meld and no pretrain_ckpt)."""
    if cfg.dataset == "meld":
        pretrain(cfg)
    else:
        finetune(cfg)


if __name__ == "__main__":
    train(load_config(_DEFAULTS))
