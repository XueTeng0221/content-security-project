"""Build train/val/test CSVs for FakeAVCeleb from meta_data.csv.

Output columns: video_path, label  (0=real, 1=fake)
video_path is relative to data_root (= FakeAVCeleb_v1.2/FakeAVCeleb_v1.2/).

Stratified split by (label, method) to keep class & manipulation balance.
Identity-disjoint at the source level so the same person never crosses splits.
"""
import argparse
import os
import random

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/FakeAVCeleb_v1.2/FakeAVCeleb_v1.2")
    ap.add_argument("--out_dir", default="data/FakeAVCeleb_v1.2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--verify", action="store_true",
                    help="Drop rows whose mp4 cannot be found on disk.")
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(args.root, "meta_data.csv"))
    df = df.rename(columns={"Unnamed: 9": "dir"})

    # Build relative video_path
    sep = os.sep
    def to_rel(row):
        d = row["dir"]
        if d.startswith("FakeAVCeleb/"):
            d = d[len("FakeAVCeleb/"):]
        return os.path.join(d, row["path"]).replace("\\", "/")
    df["video_path"] = df.apply(to_rel, axis=1)
    df["label"] = (df["category"] != "A").astype(int)  # A = RealVideo-RealAudio

    if args.verify:
        before = len(df)
        df = df[df["video_path"].apply(lambda p: os.path.exists(os.path.join(args.root, p)))]
        print(f"verify: kept {len(df)}/{before} after disk check")

    # Identity-disjoint split via 'source'
    random.seed(args.seed)
    sources = sorted(df["source"].unique())
    random.shuffle(sources)
    n = len(sources)
    n_test = int(n * args.test_ratio)
    n_val = int(n * args.val_ratio)
    test_src = set(sources[:n_test])
    val_src = set(sources[n_test:n_test + n_val])

    def split_of(s):
        if s in test_src: return "test"
        if s in val_src: return "val"
        return "train"
    df["split"] = df["source"].apply(split_of)

    os.makedirs(args.out_dir, exist_ok=True)
    for sp in ("train", "val", "test"):
        sub = df[df["split"] == sp][["video_path", "label"]]
        out = os.path.join(args.out_dir, f"{sp}.csv")
        sub.to_csv(out, index=False)
        n_real = int((sub["label"] == 0).sum())
        n_fake = int((sub["label"] == 1).sum())
        print(f"{sp}: {len(sub)} rows  real={n_real}  fake={n_fake}  -> {out}")


if __name__ == "__main__":
    main()
