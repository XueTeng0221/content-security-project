import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from src.data import MELDDataset, FakeAVCelebDataset, FFPlusPlusDataset
from src.data.meld import EMOTION2ID
from src.models import AVDeepfakeDetector
from src.utils import (
    evaluate_metrics, evaluate_metrics_multiclass,
    plot_roc, plot_pr, plot_confusion, plot_loss_curve,
)
from src.utils.config import load_config

_DEFAULTS = dict(
    dataset="meld",       # meld | fakeavceleb | ffplusplus
    data_root="data/MELD",
    csv_path="",          # for fakeavceleb/ffplusplus: path with {split} placeholder
    out_dir="outputs",
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
    mels_padded = torch.stack([
        torch.nn.functional.pad(m, (0, max_t - m.shape[-1])) for m in mels
    ])
    return torch.stack(frames), mels_padded, torch.stack(labels)


def build_loaders(cfg):
    kw = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, collate_fn=collate_fn)
    if cfg.dataset == "fakeavceleb":
        train_set = FakeAVCelebDataset(cfg.data_root, cfg.csv_path.replace("{split}", "train"))
        val_set   = FakeAVCelebDataset(cfg.data_root, cfg.csv_path.replace("{split}", "val"))
    elif cfg.dataset == "ffplusplus":
        train_set = FFPlusPlusDataset(cfg.data_root, cfg.csv_path.replace("{split}", "train"))
        val_set   = FFPlusPlusDataset(cfg.data_root, cfg.csv_path.replace("{split}", "val"))
    else:
        train_set = MELDDataset(cfg.data_root, split="train")
        val_set   = MELDDataset(cfg.data_root, split="dev")
    return DataLoader(train_set, shuffle=True, **kw), DataLoader(val_set, shuffle=False, **kw)


def train(cfg):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_loaders(cfg)

    model = AVDeepfakeDetector(
        embed_dim=cfg.embed_dim,
        freeze_visual=cfg.freeze_visual,
        num_classes=cfg.num_classes,
    ).to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = GradScaler()

    os.makedirs(cfg.out_dir, exist_ok=True)
    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        for frames, mel, labels in train_loader:
            frames, mel, labels = frames.to(device), mel.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                logits, align_loss = model(frames, mel)
                loss = criterion(logits, labels) + cfg.align_weight * align_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        scheduler.step()

        train_loss = total_loss / len(train_loader)
        val_loss, metrics = evaluate(model, val_loader, criterion, device, cfg.out_dir, cfg.num_classes, cfg.align_weight)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        metrics_str = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"Epoch {epoch+1}/{cfg.epochs} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} {metrics_str}")

        torch.save(model.state_dict(), os.path.join(cfg.out_dir, "ckpt_last.pt"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(cfg.out_dir, "ckpt_best.pt"))

    plot_loss_curve(train_losses, val_losses, save_path=os.path.join(cfg.out_dir, "loss.png"))


def evaluate(model, loader, criterion, device, out_dir, num_classes, align_weight=0.1):
    model.eval()
    total_loss, all_labels, all_preds, all_scores = 0.0, [], [], []
    with torch.no_grad():
        for frames, mel, labels in loader:
            frames, mel, labels = frames.to(device), mel.to(device), labels.to(device)
            with autocast():
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


if __name__ == "__main__":
    train(load_config(_DEFAULTS))
