import torch
import torchaudio
import torchvision.transforms as T
import numpy as np

SAMPLE_RATE = 16000
N_MELS = 80
HOP_LENGTH = 160  # 10ms at 16kHz
N_FFT = 400

video_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
)

def extract_mel(waveform: torch.Tensor) -> torch.Tensor:
    """waveform: (1, T) -> mel: (1, N_MELS, time_frames)"""
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    mel = mel_transform(waveform)
    mel = torch.log(mel + 1e-6)
    return mel
