import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
import pandas as pd
from .transforms import video_transform, extract_mel

NUM_FRAMES = 8


def _sample_frames(cap, n):
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = torch.linspace(0, max(total - 1, 0), n).long().tolist()
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            frame = __import__('numpy').zeros((224, 224, 3), dtype=__import__('numpy').uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(video_transform(Image.fromarray(frame)))
    return torch.stack(frames)


class FFPlusPlusDataset(Dataset):
    """
    FaceForensics++ visual-only deepfake dataset.

    Expected CSV columns: video_path, label  (0=real, 1=fake)
    Audio branch receives a zero mel (silent placeholder) since FF++ has no audio forgery.
    """

    def __init__(self, root: str, csv_path: str):
        self.root = root
        df = pd.read_csv(csv_path)
        self.samples = list(zip(df["video_path"], df["label"].astype(int)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        path = os.path.join(self.root, rel_path)

        cap = cv2.VideoCapture(path)
        frames = _sample_frames(cap, NUM_FRAMES)
        cap.release()

        # FF++ has no audio forgery — zero mel as placeholder
        mel = torch.zeros(1, 80, 100)
        return frames, mel, torch.tensor(label, dtype=torch.long)
