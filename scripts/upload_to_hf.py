"""Upload SONAR pretrained checkpoints to a HuggingFace model repo.

Usage:
    pip install --upgrade huggingface_hub
    huggingface-cli login                 # paste a write-scoped token
    python scripts/upload_to_hf.py \\
        --repo_id idonithid/SONAR-weights \\
        --create

This will (1) create the repo if missing, (2) upload the four .pth files
from CHECKPOINT_LOCAL_PATHS, (3) upload a model card describing them.
"""
from __future__ import annotations
import argparse
import os
import sys
import textwrap
from pathlib import Path

# (HF Hub filename) -> (local path on this machine)
CHECKPOINT_LOCAL_PATHS = {
    "xlsr2_300m.pt":                       "/mnt/storage/datasets/ido_audio_df/models/ssl/trained_models/xlsr2_300m.pt",
    "baseline_xlsr_aasist.pth":            "/home/initzan/Frequency_df/Best_LA_model_for_DF.pth",
    "sonar_full_xlsr_aasist_eer6.pth":     "/mnt/storage/datasets/ido_audio_df/xlsr_aassist_guided_eer_6.pth",
    "sonar_finetune_xlsr_mamba_eer5p5.pth":"/mnt/storage/datasets/ido_audio_df/models/ssl/trained_models/checkpoints/ido_checkpoint/mamba_finetune_srm/second_stage/mamba_ep7_loss0.006_eer0.055.pth",
}

MODEL_CARD = textwrap.dedent("""\
    ---
    license: cc-by-nc-4.0
    library_name: pytorch
    tags:
      - audio
      - deepfake-detection
      - icml-2026
    ---

    # SONAR weights

    Pretrained checkpoints for *SONAR: Spectral-Contrastive Audio Residuals
    for Generalizable Deepfake Detection* (ICML 2026).

    | File | ITW EER | Architecture | License |
    |---|---:|---|---|
    | `xlsr2_300m.pt` | — | XLSR-300M backbone (fairseq, derivative of [facebookresearch/fairseq](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec/xlsr)). | CC-BY-NC-4.0 (upstream) |
    | `baseline_xlsr_aasist.pth` | ~10.5% | Single XLSR + AASIST baseline (paper Table 1 row "XLSR+AASIST"). | CC-BY-NC-4.0 |
    | `sonar_full_xlsr_aasist_eer6.pth` | **6.0%** | SONAR-Full: dual XLSR + RFE + cross-attention + AASIST + JS-alignment loss. Matches `guided_model.GuidedModel`. | CC-BY-NC-4.0 |
    | `sonar_finetune_xlsr_mamba_eer5p5.pth` | **5.5%** | SONAR-Finetune: frozen XLSR-Mamba content branch + RFE/NFE + cross-attention + Conformer head + JS-alignment loss. | CC-BY-NC-4.0 |

    Code: <https://github.com/idonithid/SONAR-Audio-DF-Detection>
    Project page: <https://idonithid.github.io/SONAR-Audio-DF-Detection/>

    ## Loading

    ```python
    from huggingface_hub import hf_hub_download
    import torch
    from argparse import Namespace
    from sonar.guided_model import GuidedModel

    ckpt = hf_hub_download(repo_id="idonithid/SONAR-weights",
                           filename="sonar_full_xlsr_aasist_eer6.pth")
    xlsr = hf_hub_download(repo_id="idonithid/SONAR-weights",
                           filename="xlsr2_300m.pt")
    import os; os.environ["SONAR_XLSR_CKPT"] = xlsr

    model = GuidedModel(Namespace(algo=4, batch_size=1, device="cuda"), "cuda").cuda()
    model.load_state_dict(torch.load(ckpt, map_location="cuda"), strict=False)
    model.eval()
    ```

    ## Citation

    ```bibtex
    @inproceedings{hidekel2026sonar,
      title     = {{SONAR}: Spectral-Contrastive Audio Residuals for Generalizable Deepfake Detection},
      author    = {Hidekel, Ido Nitzan and Lifshitz, Gal and Cohen, Khen and Raviv, Dan},
      booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
      year      = {2026}
    }
    ```
""")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id", default="idonithid/SONAR-weights")
    p.add_argument("--create", action="store_true",
                   help="Create the repo if it doesn't exist (idempotent).")
    p.add_argument("--only", help="Comma-separated subset of filenames; default uploads all.")
    args = p.parse_args()

    try:
        from huggingface_hub import HfApi, login
    except ImportError:
        sys.exit("missing dep: pip install --upgrade huggingface_hub")

    api = HfApi()
    try:
        api.whoami()
    except Exception as e:
        sys.exit(f"not logged in to HF Hub: {e}\nrun: huggingface-cli login")

    if args.create:
        api.create_repo(repo_id=args.repo_id, repo_type="model",
                        exist_ok=True, private=False)
        print(f"  ensured model repo exists: {args.repo_id}")

    # Model card
    card_path = Path("/tmp/SONAR_README.md")
    card_path.write_text(MODEL_CARD)
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=args.repo_id, repo_type="model",
    )
    print("  uploaded README.md")

    keys = (args.only.split(",") if args.only else CHECKPOINT_LOCAL_PATHS.keys())
    for filename in keys:
        local = CHECKPOINT_LOCAL_PATHS.get(filename.strip())
        if local is None:
            print(f"  unknown filename: {filename!r}, skipping"); continue
        if not os.path.exists(local):
            print(f"  missing local file: {local}, skipping"); continue
        size_gb = os.path.getsize(local) / (1024 ** 3)
        print(f"  uploading {filename}  ({size_gb:.2f} GB) ...")
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=filename,
            repo_id=args.repo_id, repo_type="model",
        )
        print(f"     -> {args.repo_id}/{filename}")
    print("\nall done.")


if __name__ == "__main__":
    main()
