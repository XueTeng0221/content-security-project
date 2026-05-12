import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    precision_recall_curve, confusion_matrix,
    accuracy_score, f1_score
)


def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return (fpr[idx] + fnr[idx]) / 2


def evaluate_metrics(labels, scores):
    preds = (np.array(scores) >= 0.5).astype(int)
    return {
        "auc":  roc_auc_score(labels, scores),
        "ap":   average_precision_score(labels, scores),
        "eer":  compute_eer(labels, scores),
    }


def evaluate_metrics_multiclass(labels, preds):
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def plot_roc(labels, scores, save_path="roc.png"):
    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    fig, ax = plt.subplots()
    sns.lineplot(x=fpr, y=tpr, ax=ax, label=f"AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set(xlabel="FPR", ylabel="TPR", title="ROC Curve")
    ax.legend()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pr(labels, scores, save_path="pr.png"):
    prec, rec, _ = precision_recall_curve(labels, scores)
    ap = average_precision_score(labels, scores)
    fig, ax = plt.subplots()
    sns.lineplot(x=rec, y=prec, ax=ax, label=f"AP={ap:.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="PR Curve")
    ax.legend()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(labels, preds, save_path="confusion.png", class_names=("Real", "Fake")):
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(max(4, len(class_names)), max(4, len(class_names))))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set(xlabel="Predicted", ylabel="True", title="Confusion Matrix")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curve(train_losses, val_losses, save_path="loss.png"):
    fig, ax = plt.subplots()
    sns.lineplot(x=range(len(train_losses)), y=train_losses, ax=ax, label="train")
    sns.lineplot(x=range(len(val_losses)),   y=val_losses,   ax=ax, label="val")
    ax.set(xlabel="Epoch", ylabel="Loss", title="Training Loss")
    ax.legend()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
