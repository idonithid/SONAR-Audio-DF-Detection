"""FLOPs and parameter comparison for SONAR vs single-encoder baseline.

Compares:
  - model.Model              (single-encoder XLSR + AASIST)
  - guided_model.GuidedModel (dual-encoder SONAR-Full)
  - small_guided_model.GuidedModel (SONAR-Lite)

Run:  python bench/flops_compare.py [--device cuda:0]
"""
from __future__ import annotations
import argparse, csv, os, sys, gc
from argparse import Namespace
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def fmt_si(n: float, unit: str = "") -> str:
    for prefix, scale in [("T", 1e12), ("G", 1e9), ("M", 1e6), ("K", 1e3)]:
        if n >= scale:
            return f"{n / scale:.2f}{prefix}{unit}"
    return f"{n:.0f}{unit}"


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_flops(model: torch.nn.Module, input_shape: tuple[int, ...], device: str) -> int | None:
    from calflops import calculate_flops
    model.eval()
    try:
        flops, _macs, _params = calculate_flops(
            model=model,
            input_shape=input_shape,
            output_as_string=False,
            output_precision=4,
            print_results=False,
            print_detailed=False,
        )
        return int(flops)
    except Exception as e:
        print(f"  calflops failed: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def measure_with_dummy_forward(model: torch.nn.Module, input_shape: tuple[int, ...], device: str) -> dict:
    """Manual forward to confirm the model runs and to time inference."""
    model.eval()
    x = torch.randn(*input_shape, device=device)
    torch.cuda.synchronize() if device.startswith("cuda") else None
    import time
    with torch.no_grad():
        for _ in range(3):  # warmup
            _ = model(x)
        torch.cuda.synchronize() if device.startswith("cuda") else None
        t0 = time.perf_counter()
        n = 5
        for _ in range(n):
            _ = model(x)
        torch.cuda.synchronize() if device.startswith("cuda") else None
        dt = (time.perf_counter() - t0) / n
    return {"latency_s": dt}


def build_args() -> Namespace:
    return Namespace(algo=4, batch_size=1, device="cuda:0")


def free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_csv", default=str(REPO / "bench" / "flops_results.csv"))
    parser.add_argument("--out_tex", default=str(REPO / "papr_files" / "figures" / "flops_table.tex"))
    parser.add_argument("--input_samples", type=int, default=64600)
    parser.add_argument("--skip_flops", action="store_true", help="Skip calflops (params + latency only)")
    args_cli = parser.parse_args()

    device = args_cli.device
    args = build_args()
    input_shape = (1, 1, args_cli.input_samples)

    rows = []

    targets = [
        ("model.Model", "SONAR-baseline (single XLSR + AASIST)", "model", "Model"),
        ("guided_model.GuidedModel", "SONAR-Full (dual XLSR + RFE + cross-attn + AASIST)", "guided_model", "GuidedModel"),
    ]

    # NOTE: model.py:46 has a shape-indexing bug (`input_data[:, :, 0]` takes a
    # single time step instead of squeezing the channel axis). Patch it here so
    # the single-encoder baseline can run for fair comparison. The fix mirrors
    # guided_model.py:63 which uses `[:, 0, :]`.
    import model as _model_mod
    _orig_extract = _model_mod.SSLModel.extract_feat
    def _patched_extract_feat(self, input_data):
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
            self.model.train()
        if input_data.ndim == 3:
            input_tmp = input_data[:, 0, :]
        else:
            input_tmp = input_data
        return self.model(input_tmp, mask=False, features_only=True)['x']
    _model_mod.SSLModel.extract_feat = _patched_extract_feat

    # model.Model.forward also calls `x.squeeze(-1)` (vs guided_model's
    # `x.squeeze(1)`); for a (B,1,T) input squeeze(-1) is a no-op (T != 1)
    # so extract_feat sees (B,1,T) — the patched extract_feat handles it.

    for label, desc, mod_name, cls_name in targets:
        print(f"\n=== {label} — {desc} ===")
        free()
        try:
            mod = __import__(mod_name)
            cls = getattr(mod, cls_name)
            net = cls(args, device).to(device)
        except Exception as e:
            print(f"  build failed: {type(e).__name__}: {e}")
            rows.append({"label": label, "desc": desc, "params_total": None, "params_trainable": None,
                         "flops": None, "latency_s": None, "error": str(e)})
            continue

        total, trainable = count_params(net)
        print(f"  params total:     {total:,}  ({fmt_si(total)})")
        print(f"  params trainable: {trainable:,}  ({fmt_si(trainable)})")

        flops = None
        if not args_cli.skip_flops:
            flops = measure_flops(net, input_shape, device)
            if flops is not None:
                print(f"  FLOPs (1 fwd):    {flops:,}  ({fmt_si(flops)})")

        latency = None
        try:
            stats = measure_with_dummy_forward(net, input_shape, device)
            latency = stats["latency_s"]
            print(f"  latency:          {latency * 1000:.1f} ms / forward")
        except Exception as e:
            print(f"  latency measurement failed: {type(e).__name__}: {e}")

        rows.append({"label": label, "desc": desc, "params_total": total,
                     "params_trainable": trainable, "flops": flops, "latency_s": latency, "error": ""})

        del net
        free()

    Path(args_cli.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args_cli.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV → {args_cli.out_csv}")

    Path(args_cli.out_tex).parent.mkdir(parents=True, exist_ok=True)
    with open(args_cli.out_tex, "w") as f:
        f.write("% Auto-generated by bench/flops_compare.py — do not edit by hand.\n")
        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        f.write("\\caption{Parameter count, FLOPs (one 4-second @ 16\\,kHz forward), and inference latency.}\n")
        f.write("\\label{tab:flops}\n")
        f.write("\\begin{tabular}{lrrr}\n\\toprule\n")
        f.write("Model & Params & FLOPs & Latency (ms) \\\\\n\\midrule\n")
        for r in rows:
            name = {"model.Model": "Single-encoder baseline",
                    "guided_model.GuidedModel": "SONAR-Full (ours)",
                    "small_guided_model.GuidedModel": "SONAR-Lite (ours)"}.get(r["label"], r["label"])
            p = fmt_si(r["params_total"]) if r["params_total"] else "--"
            fl = fmt_si(r["flops"]) if r["flops"] else "--"
            lat = f"{r['latency_s'] * 1000:.1f}" if r["latency_s"] else "--"
            f.write(f"{name} & {p} & {fl} & {lat} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"TeX → {args_cli.out_tex}")


if __name__ == "__main__":
    main()
