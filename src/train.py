import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data import MELDDataset
from src.data.meld import EMOTION2ID
from src.models import AVDeepfakeDetector
from src.utils import (
    evaluate_metrics, evaluate_metrics_multiclass,
    plot_roc, plot_pr, plot_confusion, plot_loss_curve,
)
from src.utils.config import load_config

_DEFAULTS = dict(data_root="data/MELD", out_dir="outputs",
                 epochs=20, batch_size=8, lr=1e-4, embed_dim=256,
                 freeze_visual=True, num_workers=4, num_classes=7)


def collate_fn(batch):
    frames, mels, labels = zip(*batch)
    # Pad mel to same length
    max_t = max(m.shape[-1] for m in mels)
    mels_padded = torch.stack([
        torch.nn.functional.pad(m, (0, max_t - m.shape[-1])) for m in mels
    ])
    return torch.stack(frames), mels_padded, torch.stack(labels)


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = MELDDataset(cfg.data_root, split="train")
    val_set   = MELDDataset(cfg.data_root, split="dev")
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size,
                              shuffle=True,  num_workers=cfg.num_workers, collate_fn=collate_fn)
    val_loader   = DataLoader(val_set,   batch_size=cfg.batch_size,
                              shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_fn)

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

    os.makedirs(cfg.out_dir, exist_ok=True)
    train_losses, val_losses = [], []

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        for frames, mel, labels in train_loader:
            frames, mel, labels = frames.to(device), mel.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(frames, mel)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        train_loss = total_loss / len(train_loader)
        val_loss, metrics = evaluate(model, val_loader, criterion, device, cfg.out_dir, cfg.num_classes)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        metrics_str = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"Epoch {epoch+1}/{cfg.epochs} | "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} {metrics_str}")

        torch.save(model.state_dict(), os.path.join(cfg.out_dir, f"ckpt_epoch{epoch+1}.pt"))

    plot_loss_curve(train_losses, val_losses,
                    save_path=os.path.join(cfg.out_dir, "loss.png"))


def evaluate(model, loader, criterion, device, out_dir, num_classes):
    model.eval()
    total_loss, all_labels, all_preds, all_scores = 0.0, [], [], []
    with torch.no_grad():
        for frames, mel, labels in loader:
            frames, mel, labels = frames.to(device), mel.to(device), labels.to(device)
            logits = model(frames, mel)
            total_loss += criterion(logits, labels).item()
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
        if len(class_names) != num_classes:
            class_names = [str(i) for i in range(num_classes)]
        plot_confusion(all_labels, all_preds,
                       save_path=os.path.join(out_dir, "confusion.png"),
                       class_names=class_names)
    return total_loss / len(loader), metrics


if __name__ == "__main__":
    train(load_config(_DEFAULTS))
