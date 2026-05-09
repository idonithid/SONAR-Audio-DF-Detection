---
title: SONAR — Audio Deepfake Detection
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.36.0"
app_file: app.py
pinned: false
license: mit
short_description: Detect AI-generated audio with the ICML 2026 SONAR model
---

# SONAR — Audio Deepfake Detection

This Space runs the **SONAR-Full** detector from our ICML 2026 paper.

- 🎙️ Upload or record an audio clip → bonafide / spoof verdict
- 📈 Visualise the low- and high-frequency bands SONAR analyses jointly
- 🔬 LF–HF cosine alignment score (high = real, low = synthesised)

The detector lazy-loads on first request; the first analysis after a cold
start can take 30–60 seconds while the XLSR-300M backbone is moved to GPU.
Once warm, inference is ~57 ms / clip.

If the Space is asleep or broken, the [GitHub Pages landing page](https://idonithid.github.io/SONAR-Audio-DF-Detection/)
hosts the paper findings and figures and works regardless.

📄 [Paper / GitHub](https://github.com/idonithid/SONAR-Audio-DF-Detection)
