"""Fetch SONAR pretrained checkpoints from HuggingFace Hub.

Usage:
    python scripts/download_checkpoints.py [--out_dir checkpoints] [--which all]

Available checkpoints (when uploaded by the authors to HuggingFace Hub):
  - xlsr2_300m.pt              -- XLSR-300M backbone (CC-BY-NC-4.0, derivative of fairseq XLSR)
  - baseline_xlsr_aasist.pth   -- single-encoder XLSR + AASIST baseline (~10.5% ITW EER)
  - sonar_full_xlsr_aasist_eer6.pth -- SONAR-Full (6.0% ITW EER, paper Table 1)
  - sonar_finetune_xlsr_mamba_eer5p5.pth -- SONAR-Finetune on XLSR-Mamba (5.5% ITW EER)

NOTE: until the authors push to the Hub, set SONAR_XLSR_CKPT and the SONAR
checkpoint paths via environment variables to point at local files.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

REPO_ID = "idonithid/SONAR-weights"
FILES = {
    "xlsr2_300m.pt":                          "xlsr2_300m.pt",
    "baseline_xlsr_aasist.pth":               "baseline_xlsr_aasist.pth",
    "sonar_full_xlsr_aasist_eer6.pth":        "sonar_full_xlsr_aasist_eer6.pth",
    "sonar_finetune_xlsr_mamba_eer5p5.pth":   "sonar_finetune_xlsr_mamba_eer5p5.pth",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="checkpoints")
    p.add_argument("--which", default="all", help="comma-separated subset, or 'all'")
    p.add_argument("--repo_id", default=REPO_ID)
    args = p.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit("Missing dep: `pip install huggingface_hub`")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.which == "all":
        keys = list(FILES.keys())
    else:
        keys = [k.strip() for k in args.which.split(",")]

    for k in keys:
        if k not in FILES:
            print(f"unknown: {k} (skipping)"); continue
        print(f"fetching {k} from {args.repo_id} ...")
        path = hf_hub_download(repo_id=args.repo_id, filename=FILES[k], local_dir=str(out))
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
