"""High-level inference helper for SONAR.

Wraps SONAR-Full into a single function: `score_audio(path) -> dict`.
Designed for the Gradio demo and end-user CLI consumers.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F


SAMPLE_RATE = 16_000
TARGET_SAMPLES = 64_600  # ~4 s @ 16 kHz, paper convention


@dataclass
class Detection:
    bonafide_prob: float       # softmax probability of class 1 (real)
    spoof_prob: float          # softmax probability of class 0 (fake)
    label: str                 # "real" or "fake" (argmax)
    lf_hf_alignment: float     # cosine similarity between content and noise embeddings (high = real, low = fake)
    sample_rate: int = SAMPLE_RATE
    n_samples: int = TARGET_SAMPLES


def _pad_or_trim(x: np.ndarray, n: int = TARGET_SAMPLES) -> np.ndarray:
    if x.shape[0] >= n:
        return x[:n]
    out = np.zeros(n, dtype=np.float32)
    out[: x.shape[0]] = x
    return out


def _load_audio(path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        except ImportError as e:
            raise RuntimeError(f"audio at {sr} Hz; install librosa to resample to {target_sr} Hz") from e
    return _pad_or_trim(audio, TARGET_SAMPLES)


class SONARDetector:
    """Lazy-loaded SONAR-Full deepfake detector.

    Usage:
        det = SONARDetector(ckpt="checkpoints/sonar_full_xlsr_aasist_eer6.pth")
        result = det.score_audio("clip.wav")
        print(result.bonafide_prob, result.label)
    """

    def __init__(self, ckpt: str, device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 xlsr_ckpt: Optional[str] = None):
        from argparse import Namespace
        from sonar.guided_model import GuidedModel
        if xlsr_ckpt:
            os.environ["SONAR_XLSR_CKPT"] = xlsr_ckpt
        self.device = torch.device(device)
        args = Namespace(algo=4, batch_size=1, device=str(self.device))
        self.model = GuidedModel(args, self.device).to(self.device)
        sd = torch.load(ckpt, map_location=self.device)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        if isinstance(sd, dict) and any(k.startswith("module.") for k in sd):
            sd = {k[7:]: v for k, v in sd.items() if k.startswith("module.")}
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[SONARDetector] missing={len(missing)} unexpected={len(unexpected)}")
        self.model.eval()

    @torch.no_grad()
    def score_audio(self, path_or_array: Union[str, np.ndarray]) -> Detection:
        if isinstance(path_or_array, str):
            audio = _load_audio(path_or_array)
        else:
            audio = _pad_or_trim(np.asarray(path_or_array, dtype=np.float32))
        x = torch.from_numpy(audio).to(self.device).unsqueeze(0).unsqueeze(0)  # (1, 1, T)
        out = self.model(x)
        # GuidedModel.forward returns (logits, x_low, x_high)
        if isinstance(out, (tuple, list)) and len(out) >= 3:
            logits, lf, hf = out[0], out[1], out[2]
        else:
            logits, lf, hf = out, None, None
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        bonafide_prob = float(probs[1]) if probs.shape[0] >= 2 else 1.0 - float(probs[0])
        spoof_prob = 1.0 - bonafide_prob
        label = "real" if bonafide_prob >= 0.5 else "fake"
        if lf is not None and hf is not None:
            cos = F.cosine_similarity(lf.mean(dim=1), hf.mean(dim=1), dim=-1).item()
        else:
            cos = float("nan")
        return Detection(
            bonafide_prob=bonafide_prob,
            spoof_prob=spoof_prob,
            label=label,
            lf_hf_alignment=cos,
        )
