import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.data import MELDDataset
from src.models import AVDeepfakeDetector
from src.models.ablation_models import VisualOnlyDetector, AudioOnlyDetector, ConcatFusionDetector
from src.utils import evaluate_metrics, evaluate_metrics_multiclass
from src.utils.config import load_config
from src.train import collate_fn

_DEFAULTS = dict(data_root="data/MELD", out_dir="outputs/ablation",
                 epochs=10, batch_size=8, lr=1e-4, embed_dim=256,
                 num_workers=4, num_classes=7)


VARIANTS = {
    "visual_only":    VisualOnlyDetector,
    "audio_only":     AudioOnlyDetector,
    "concat_fusion":  ConcatFusionDetector,
    "cross_attention": AVDeepfakeDetector,
}


def run_variant(name, model_cls, loader, device, cfg):
    model = model_cls(embed_dim=cfg.embed_dim, num_classes=cfg.num_classes).to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr
    )
    criterion = nn.CrossEntropyLoss()

    for _ in range(cfg.epochs):
        model.train()
        for frames, mel, labels in loader["train"]:
            frames, mel, labels = frames.to(device), mel.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(frames, mel), labels)
            loss.backward()
            optimizer.step()

    model.eval()
    all_labels, all_preds, all_scores = [], [], []
    with torch.no_grad():
        for frames, mel, labels in loader["val"]:
            frames, mel, labels = frames.to(device), mel.to(device), labels.to(device)
            probs = torch.softmax(model(frames, mel), dim=1)
            all_preds.extend(probs.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            if cfg.num_classes == 2:
                all_scores.extend(probs[:, 1].cpu().tolist())

    if cfg.num_classes == 2:
        metrics = evaluate_metrics(all_labels, all_scores)
    else:
        metrics = evaluate_metrics_multiclass(all_labels, all_preds)
    metrics_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    print(f"[{name}] {metrics_str}")
    return metrics


def plot_ablation(results: dict, out_dir: str):
    rows = [{"variant": k, "metric": m, "value": v}
            for k, metrics in results.items()
            for m, v in metrics.items()]
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=df, x="variant", y="value", hue="metric", ax=ax)
    ax.set(title="Ablation Study", ylabel="Score", xlabel="")
    ax.legend(loc="lower right")
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ablation.png"), dpi=150)
    plt.close(fig)

    df.pivot(index="variant", columns="metric", values="value").to_csv(
        os.path.join(out_dir, "ablation.csv")
    )


if __name__ == "__main__":
    cfg = load_config(_DEFAULTS)
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = {
        split: DataLoader(
            MELDDataset(cfg.data_root, split=split),
            batch_size=cfg.batch_size, shuffle=(split == "train"),
            num_workers=cfg.num_workers, collate_fn=collate_fn,
        )
        for split in ("train", "dev")
    }

    results = {}
    for name, cls in VARIANTS.items():
        results[name] = run_variant(name, cls, loaders, device, cfg)  # noqa: E501

    plot_ablation(results, cfg.out_dir)
    print(f"\nSaved ablation.png and ablation.csv to {cfg.out_dir}")
