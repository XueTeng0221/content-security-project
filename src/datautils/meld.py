import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
import pandas as pd
import numpy as np
import subprocess
import tempfile
import soundfile as sf
from .transforms import video_transform, extract_mel

# MELD structure:
# root/
#   train_splits/  dev_splits/  test_splits/
#     dia{d}_utt{u}.mp4
#   train_sent_emo.csv  (Utterance, Emotion, Sentiment, ...)
#
# Emotion labels: neutral, surprise, fear, sadness, joy, disgust, anger

EMOTION2ID = {e: i for i, e in enumerate(
    ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]
)}

NUM_FRAMES = 8


def _sample_frames(cap, n):
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = torch.linspace(0, max(total - 1, 0), n).long().tolist()
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            import numpy as np
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(video_transform(Image.fromarray(frame)))
    return torch.stack(frames)  # (N, 3, 224, 224)


class MELDDataset(Dataset):
    """MELD dataset for emotion-consistency pretraining."""

    def __init__(self, root: str, split: str = "train"):
        split_map = {
            "train": ("train/train_splits", "train/train_sent_emo.csv"),
            "dev":   ("dev/dev_splits_complete",     "dev_sent_emo.csv"),
            "test":  ("test/output_repeated_splits_test",   "test_sent_emo.csv"),
        }
        video_dir, csv_name = split_map[split]
        self.video_dir = os.path.join(root, video_dir)
        df = pd.read_csv(os.path.join(root, csv_name))
        self.samples = [
            (row["Dialogue_ID"], row["Utterance_ID"],
             EMOTION2ID.get(row["Emotion"].lower(), 0))
            for _, row in df.iterrows()
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dia_id, utt_id, emotion = self.samples[idx]
        path = os.path.join(self.video_dir, f"dia{dia_id}_utt{utt_id}.mp4")

        cap = cv2.VideoCapture(path)
        frames = _sample_frames(cap, NUM_FRAMES)
        cap.release()

        # Extract audio using ffmpeg
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            subprocess.run(
                ["ffmpeg", "-i", path, "-ar", "16000", "-ac", "1", "-y", tmp_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
            )
            waveform, sr = sf.read(tmp_path)
            waveform = torch.from_numpy(waveform).float().unsqueeze(0)  # (1, T)
        except Exception:
            # Fallback: silent audio
            waveform = torch.zeros(1, 16000)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        mel = extract_mel(waveform)

        return frames, mel, torch.tensor(emotion, dtype=torch.long)
