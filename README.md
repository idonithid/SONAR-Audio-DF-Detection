# SONAR — Spectral-Contrastive Audio Residuals for Generalizable Deepfake Detection

Code for the paper *SONAR: Spectral-Contrastive Audio Residuals for Generalizable Deepfake Detection* (ICML 2026, regular track).

> **TL;DR.** Modern speech-synthesis systems leave subtle high-frequency (HF) artifacts that frequency-agnostic detectors ignore — a manifestation of *spectral bias*. SONAR is a dual-path detector that fuses an XLSR content branch with a parallel branch driven by learnable, value-constrained SRM high-pass filters, and trains them with a Jensen–Shannon alignment loss that pulls genuine LF/HF representations together while pushing fake ones apart. SONAR achieves single-run state-of-the-art on ASVspoof 2021 and In-the-Wild, converges 4× faster than strong baselines, and degrades gracefully under codecs and bandwidth shifts.

## Headline results (ITW EER, single run)

| Model                    | DF EER ↓ | LA EER ↓ | ITW EER ↓ |
|--------------------------|---------:|---------:|----------:|
| XLSR + AASIST (baseline) |     3.69 |     1.90 |     10.46 |
| XLSR-Mamba               |     1.88 |     0.93 |      6.71 |
| **SONAR-Full**           | **1.57** | **1.55** |  **6.00** |
| **SONAR-Finetune**       | **1.45** | **1.20** |  **5.43** |

## Quickstart

```bash
git clone https://github.com/idonithid/SONAR-Audio-DF-Detection.git
cd SONAR
pip install -r requirements.txt

# Download pretrained checkpoints
python scripts/download_checkpoints.py --out_dir checkpoints
export SONAR_XLSR_CKPT="$(pwd)/checkpoints/xlsr2_300m.pt"

# Score a single clip
python -c "
from sonar.inference import SONARDetector
det = SONARDetector(ckpt='checkpoints/sonar_full_xlsr_aasist_eer6.pth')
print(det.score_audio('your_clip.wav'))
"
```

## Repo layout

```
SONAR/
├── sonar/                         core package
│   ├── model.py                   single-encoder XLSR + AASIST baseline
│   ├── guided_model.py            SONAR-Full (paper main model)
│   ├── small_guided_model.py      SONAR-Lite (Sec. 4.3)
│   ├── srm_filters.py             constrained-SRM Conv1d
│   ├── HFFM.py                    high-freq focus modules
│   ├── eval_metric_LA.py          ASVspoof 2021 LA EER + min-tDCF
│   ├── eval_metric_DF.py          ASVspoof 2021 DF EER
│   ├── inference.py               SONARDetector — single-clip helper
│   └── data/                      dataset loaders (asvspoof, in-the-wild, RawBoost)
├── scripts/
│   └── download_checkpoints.py    fetch weights from HuggingFace Hub
├── bench/
│   ├── flops_compare.py           params/FLOPs/latency table
│   └── dac_plot.py                neural-codec robustness figure
├── checkpoints/                   (gitignored) downloaded weights live here
├── docs/
└── tests/
```

## Datasets

Set the dataset roots at run time (no hard-coded `/mnt/storage/...`):

```bash
export ASVSPOOF2019_ROOT=/path/to/ASVspoof2019/LA
export ASVSPOOF2021_ROOT=/path/to/ASVspoof2021
export ITW_ROOT=/path/to/release_in_the_wild
```

| Dataset             | Where to get it                                                                                        |
|---------------------|---------------------------------------------------------------------------------------------------------|
| ASVspoof 2019 (LA)  | <https://datashare.ed.ac.uk/handle/10283/3336>                                                          |
| ASVspoof 2021 (LA / DF) | <https://doi.org/10.5281/zenodo.4837263>                                                            |
| In-the-Wild         | <https://deepfake-total.com/in_the_wild>                                                                |

## Training

```bash
python scripts/train.py \
  --train_data_path  $ASVSPOOF2019_ROOT/ASVspoof2019_LA_train \
  --train_protocols_path $ASVSPOOF2019_ROOT/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt \
  --batch_size 28 --num_epochs 20 --lr 1e-5 --algo 4
```

(See `scripts/train.py --help` for the full flag list.)

## Evaluation

```bash
python scripts/eval_LA.py SCORE_FILE KEYS_DIR PHASE   # ASVspoof 2021 LA
python scripts/eval_DF.py SCORE_FILE KEYS_DIR PHASE   # ASVspoof 2021 DF
```

## Demo

A live, browser-based demo is available at the [SONAR HuggingFace Space](https://huggingface.co/spaces/idonithid/SONAR-demo); the source for it lives in [`sonar-demo/`](sonar-demo/). A static landing page with the paper's findings is hosted on [GitHub Pages](https://idonithid.github.io/SONAR-Audio-DF-Detection/).

## Citation

```bibtex
@inproceedings{hidekel2026sonar,
  title     = {{SONAR}: Spectral-Contrastive Audio Residuals for Generalizable Deepfake Detection},
  author    = {Hidekel, Ido Nitzan and Lifshitz, Gal and Cohen, Khen and Raviv, Dan},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

## License

Code: MIT (see `LICENSE`).
Pretrained weights: derivative of [XLSR](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec/xlsr) (CC-BY-NC-4.0); see `docs/licenses.md`.

## Acknowledgements

We build on [fairseq XLSR](https://github.com/facebookresearch/fairseq), [AASIST](https://github.com/clovaai/aasist), [XLSR-Mamba](https://github.com/swagshaw/XLSR-Mamba), and [RawBoost](https://github.com/TakHemlata/RawBoost-antispoofing). Thanks to the ICML 2026 reviewers and area chair for feedback that shaped this version.
