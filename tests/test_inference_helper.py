"""Lightweight tests for sonar.inference helpers (no fairseq / checkpoint required)."""
import numpy as np

from sonar.inference import _pad_or_trim, TARGET_SAMPLES


def test_pad_short_signal():
    x = np.zeros(1000, dtype=np.float32)
    out = _pad_or_trim(x)
    assert out.shape == (TARGET_SAMPLES,)
    assert out.dtype == np.float32
    assert (out[:1000] == 0).all()
    assert (out[1000:] == 0).all()


def test_trim_long_signal():
    x = np.arange(TARGET_SAMPLES + 5000, dtype=np.float32)
    out = _pad_or_trim(x)
    assert out.shape == (TARGET_SAMPLES,)
    assert out[0] == 0.0
    assert out[-1] == TARGET_SAMPLES - 1
