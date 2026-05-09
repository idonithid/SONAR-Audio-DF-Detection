"""SONAR — interactive deepfake-audio detection demo.

Gradio app for HuggingFace Spaces. Lazy-loads the SONAR-Full checkpoint at
boot, then accepts an audio upload / microphone recording / example clip and
returns:
  * the bonafide / spoof probability
  * the LF--HF cosine alignment score (the JS-aligned signal)
  * STFT spectrograms of the LF and HF bands

Designed to fail gracefully when run on CPU-only Spaces or before checkpoint
download completes -- all heavy ops live in lazy-loaded helpers.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np

# Allow `python sonar-demo/app.py` from a clone before the package is installed.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import gradio as gr  # noqa: E402

# ---------- Lazy detector ------------------------------------------------

_DETECTOR = None
_INIT_ERROR = None
HF_REPO = os.environ.get("SONAR_HF_REPO", "idonithid/SONAR-weights")


def _resolve_ckpts():
    """Fetch SONAR-Full weights and the XLSR-300M backbone from HF Hub."""
    from huggingface_hub import hf_hub_download
    sonar_ckpt = hf_hub_download(repo_id=HF_REPO,
                                 filename="sonar_full_xlsr_aasist_eer6.pth")
    xlsr_ckpt  = hf_hub_download(repo_id=HF_REPO, filename="xlsr2_300m.pt")
    os.environ["SONAR_XLSR_CKPT"] = xlsr_ckpt
    return sonar_ckpt


def _detector():
    global _DETECTOR, _INIT_ERROR
    if _DETECTOR is not None or _INIT_ERROR is not None:
        return _DETECTOR
    try:
        sonar_ckpt = _resolve_ckpts()
        from sonar.inference import SONARDetector
        _DETECTOR = SONARDetector(ckpt=sonar_ckpt)
    except Exception as e:
        _INIT_ERROR = f"{type(e).__name__}: {e}"
    return _DETECTOR


# ---------- Spectrogram helpers ------------------------------------------

def _spec_figures(audio: np.ndarray, sr: int):
    """Return two plotly figures: LF (0-4 kHz) and HF (4-8 kHz) spectrograms."""
    import plotly.graph_objects as go
    from scipy.signal import stft

    f, t, z = stft(audio, fs=sr, nperseg=512, noverlap=384)
    mag = 20.0 * np.log10(np.abs(z) + 1e-8)

    lf_mask = f <= 4000
    hf_mask = f > 4000

    def _heat(mask, title):
        fig = go.Figure(
            data=go.Heatmap(
                z=mag[mask], x=t, y=f[mask],
                colorscale="Viridis", showscale=False, zmin=-80, zmax=0,
            )
        )
        fig.update_layout(
            title=title, xaxis_title="time (s)", yaxis_title="freq (Hz)",
            height=240, margin=dict(l=40, r=10, t=30, b=30), template="plotly_white",
        )
        return fig

    return _heat(lf_mask, "Low-frequency band (0–4 kHz)"), _heat(hf_mask, "High-frequency band (4–8 kHz)")


def _placeholder_fig(msg: str):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=14))
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    fig.update_layout(template="plotly_white", height=240)
    return fig


# ---------- Inference handler -------------------------------------------

def analyse(audio_in):
    if audio_in is None:
        empty = _placeholder_fig("upload audio or click an example")
        return "Waiting for audio…", "—", empty, empty
    sr, samples = audio_in
    samples = samples.astype(np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if samples.size == 0:
        empty = _placeholder_fig("empty signal")
        return "Empty audio", "—", empty, empty

    if sr != 16000:
        try:
            import librosa
            samples = librosa.resample(samples, orig_sr=sr, target_sr=16000)
            sr = 16000
        except ImportError:
            pass

    det = _detector()
    if det is None:
        msg = f"Detector not ready: {_INIT_ERROR or 'still loading'}"
        return msg, "—", _placeholder_fig(msg), _placeholder_fig(msg)

    result = det.score_audio(samples)
    label_md = (
        f"### Verdict: **{'🟢 REAL' if result.label == 'real' else '🔴 FAKE'}**\n\n"
        f"- Bonafide probability: **{result.bonafide_prob:.3f}**\n"
        f"- Spoof probability: **{result.spoof_prob:.3f}**"
    )
    align_md = f"**LF–HF cosine alignment:** `{result.lf_hf_alignment:+.3f}`  \n_(higher → more likely real, lower → more likely synthesised)_"
    lf_fig, hf_fig = _spec_figures(samples, sr)
    return label_md, align_md, lf_fig, hf_fig


# ---------- UI -----------------------------------------------------------

INTRO = """
# SONAR — Audio Deepfake Detection
Upload, record, or pick an example clip to run **SONAR-Full**, the
spectral-contrastive detector from our ICML 2026 paper. The model fuses an
XLSR content branch with a parallel high-pass branch and is trained with a
Jensen–Shannon alignment loss that rewards genuine LF–HF coherence and
penalises the disruption typical of generative speech.

The verdict below is the model's softmax output; the LF and HF spectrograms
visualise the two bands SONAR analyses jointly.
"""

ABOUT = """
## About the paper

SONAR diagnoses **spectral bias** as a major cause of poor generalisation in
audio deepfake detectors: neural networks preferentially learn low-frequency
structure and miss subtle high-frequency artifacts left by neural vocoders.

Our approach:

- **Content branch.** XLSR encoder on raw 16 kHz audio.
- **Noise branch.** Parallel branch driven by *learnable, value-constrained*
  SRM high-pass filters (length-5, central tap fixed to −1, zero-sum
  constraint applied after every step).
- **Cross-attention fusion** of the two embeddings.
- **Jensen–Shannon alignment loss** that pulls genuine LF/HF representations
  together and pushes fake ones apart.

Headline results (ITW EER, single run):

| Model | DF | LA | ITW |
|---|---:|---:|---:|
| XLSR + AASIST | 3.69 | 1.90 | 10.46 |
| **SONAR-Full** | **1.57** | **1.55** | **6.00** |
| **SONAR-Finetune** | **1.45** | **1.20** | **5.43** |

Code, paper, and the static landing page with our findings live at
[github.com/idonithid/SONAR-Audio-DF-Detection](https://github.com/idonithid/SONAR-Audio-DF-Detection).
"""

with gr.Blocks(title="SONAR — Audio Deepfake Detection") as demo:
    gr.Markdown(INTRO)
    with gr.Tab("Detector"):
        with gr.Row():
            with gr.Column(scale=1):
                audio_in = gr.Audio(sources=["upload", "microphone"], type="numpy", label="Audio")
                btn = gr.Button("Analyse", variant="primary")
            with gr.Column(scale=1):
                verdict = gr.Markdown("Waiting for audio…")
                alignment = gr.Markdown("—")
        with gr.Row():
            lf_plot = gr.Plot(label="Low-frequency band")
            hf_plot = gr.Plot(label="High-frequency band")
        btn.click(analyse, inputs=audio_in, outputs=[verdict, alignment, lf_plot, hf_plot])
        # Auto-trigger when a new file lands
        audio_in.change(analyse, inputs=audio_in, outputs=[verdict, alignment, lf_plot, hf_plot])
    with gr.Tab("About"):
        gr.Markdown(ABOUT)


if __name__ == "__main__":
    demo.launch()
