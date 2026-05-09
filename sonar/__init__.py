"""SONAR — Spectral-Contrastive Audio Residuals for Generalizable Deepfake Detection.

Public API:
    from sonar.inference import SONARDetector
    detector = SONARDetector(ckpt="checkpoints/sonar_full_xlsr_aasist_eer6.pth")
    result = detector.score_audio("clip.wav")
"""

__version__ = "1.0.0"
