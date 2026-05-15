import os
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.models import AVDeepfakeDetector
from src.models.ablation_models import VisualOnlyDetector, AudioOnlyDetector, ConcatFusionDetector
from src.utils import evaluate_metrics, evaluate_metrics_multiclass
from src.utils.config import load_config
from src.train import set_seed, build_loaders

_DEFAULTS = dict(
    dataset="meld", data_root="data/MELD", csv_path="",
    out_dir="outputs/ablation",
    epochs=10, batch_size=8, lr=1e-4, embed_dim=256,
    freeze_visual=True, num_workers=4, num_classes=7,
    align_weight=0.1, seed=42,
)

VARIANTS = {
    "visual_only":    VisualOnlyDetector,
    "audio_only":     AudioOnlyDetector,
    "concat_fusion":  ConcatFusionDetector,
    "cross_attention": AVDeepfakeDetector,
}


def _forward(model, frames, mel):
    out = model(frames, mel)
    if isinstance(out, tuple):
        return out[0], out[1]
    return out, torch.tensor(0.0, device=frames.device)


def run_variant(name, model_cls, loaders, device, cfg):
    set_seed(cfg.seed)
    kwargs = dict(embed_dim=cfg.embed_dim, num_classes=cfg.num_classes)
    if model_cls is AVDeepfakeDetector:
        kwargs["freeze_visual"] = cfg.freeze_visual
        
    model = model_cls(**kwargs).to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    for _ in range(cfg.epochs):
        model.train()
        for frames, mel, labels in loaders["train"]:
            frames, mel, labels = frames.to(device), mel.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                logits, align_loss = _forward(model, frames, mel)
                loss = criterion(logits, labels) + cfg.align_weight * align_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

    model.eval()
    all_labels, all_preds, all_scores = [], [], []
    with torch.no_grad():
        for frames, mel, labels in loaders["val"]:
            frames, mel, labels = frames.to(device), mel.to(device), labels.to(device)
            with torch.amp.autocast("cuda"):
                logits, _ = _forward(model, frames, mel)
                
            probs = torch.softmax(logits, dim=1)
            all_preds.extend(probs.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            if cfg.num_classes == 2:
                all_scores.extend(probs[:, 1].cpu().tolist())

    metrics = (evaluate_metrics(all_labels, all_scores) if cfg.num_classes == 2
               else evaluate_metrics_multiclass(all_labels, all_preds))
    print(f"[{name}] " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    return metrics


def plot_ablation(results: dict, out_dir: str):
    rows = [{"variant": k, "metric": m, "value": v} for k, metrics in results.items() for m, v in metrics.items()]
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=df, x="variant", y="value", hue="metric", ax=ax)
    ax.set(title="Ablation Study", ylabel="Score", xlabel="")
    ax.legend(loc="lower right")
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ablation.png"), dpi=150)
    plt.close(fig)
    df.pivot(index="variant", columns="metric", values="value").to_csv(os.path.join(out_dir, "ablation.csv"))


if __name__ == "__main__":
    cfg = load_config(_DEFAULTS)
    cfg.out_dir = os.path.join(cfg.out_dir, "ablation")
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_loaders(cfg)
    loaders = {"train": train_loader, "val": val_loader}
    results = {name: run_variant(name, cls, loaders, device, cfg) for name, cls in VARIANTS.items()}
    plot_ablation(results, cfg.out_dir)
    print(f"\nSaved ablation.png and ablation.csv to {cfg.out_dir}")
