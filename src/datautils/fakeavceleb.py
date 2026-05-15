import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
import pandas as pd
import torchaudio
from .transforms import video_transform, extract_mel

NUM_FRAMES = 8
TARGET_FPS = 25  # enforce A-V sync at 25fps


def _sample_frames_at_fps(cap, n, fps):
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    # sample n evenly-spaced frames from the first (n/fps) seconds
    duration_frames = min(total, int(n / fps * src_fps) + 1)
    indices = torch.linspace(0, max(duration_frames - 1, 0), n).long().tolist()
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            frame = __import__('numpy').zeros((224, 224, 3), dtype=__import__('numpy').uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(video_transform(Image.fromarray(frame)))
    return torch.stack(frames)  # (N, 3, 224, 224)


class FakeAVCelebDataset(Dataset):
    """
    FakeAVCeleb binary deepfake dataset.

    Expected CSV columns: video_path, label  (0=real, 1=fake)
    video_path is relative to root.
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
        frames = _sample_frames_at_fps(cap, NUM_FRAMES, TARGET_FPS)
        cap.release()

        try:
            waveform, sr = torchaudio.load(path)
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
        except Exception:
            waveform = torch.zeros(1, 16000)

        mel = extract_mel(waveform)
        return frames, mel, torch.tensor(label, dtype=torch.long)
