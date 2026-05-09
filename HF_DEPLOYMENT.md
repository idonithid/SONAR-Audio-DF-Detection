# Deploying SONAR to HuggingFace

Two HF repos are needed:

| Repo                          | What lives there                                                       |
|-------------------------------|------------------------------------------------------------------------|
| `idonithid/SONAR-weights`     | Pretrained checkpoints (`*.pth` and the XLSR-300M backbone). ~10 GB.   |
| `idonithid/SONAR-demo`        | The Gradio Space that serves the live detector.                         |

The Space lazy-loads weights from the model repo at first request, so you upload them once and the Space costs nothing extra to deploy.

---

## 0. One-time: get a write token

1. https://huggingface.co/settings/tokens → **Create new token**
2. Type: **Write**, name e.g. `sonar-deploy`. Copy the value.
3. On this machine:
```bash
hf auth login
# paste the token when prompted
```

This caches the token at `~/.cache/huggingface/token`. All scripts below pick it up automatically.

> **Note** — the legacy `huggingface-cli` is deprecated; use `hf` (already installed via the `huggingface_hub` package). The Python library API used by `scripts/upload_to_hf.py` is unaffected.

---

## 1. Upload the weights to `idonithid/SONAR-weights`

```bash
cd /home/initzan/SONAR
python scripts/upload_to_hf.py --repo_id idonithid/SONAR-weights --create
```

What it does:
- creates the model repo (if missing)
- uploads `xlsr2_300m.pt`, `baseline_xlsr_aasist.pth`, `sonar_full_xlsr_aasist_eer6.pth`, `sonar_finetune_xlsr_mamba_eer5p5.pth` from their canonical local paths (the ones already symlinked at `/home/initzan/Frequency_df/checkpoints/`)
- writes a model card (`README.md`) describing the four files

Total upload is ~10 GB. On a residential link expect 30–90 minutes; on a campus link, much less. The script handles LFS automatically.

To re-upload only one file later:
```bash
python scripts/upload_to_hf.py --only sonar_full_xlsr_aasist_eer6.pth
```

---

## 2. Create the Space `idonithid/SONAR-demo`

The Space is just a small clone-and-push from `sonar-demo/`.

```bash
# Create the Space (one-time)
hf repos create idonithid/SONAR-demo --type space --space-sdk gradio
```

Then push the contents:

```bash
cd /tmp
git clone https://huggingface.co/spaces/idonithid/SONAR-demo
cp /home/initzan/SONAR/sonar-demo/app.py          SONAR-demo/
cp /home/initzan/SONAR/sonar-demo/requirements.txt SONAR-demo/
cp /home/initzan/SONAR/sonar-demo/README.md       SONAR-demo/
cd SONAR-demo
git add . && git commit -m "Initial deploy"
git push
```

The Space rebuilds automatically; takes ~3–5 minutes.

`requirements.txt` pulls the SONAR package straight from GitHub, so any future code change you push to `idonithid/SONAR-Audio-DF-Detection` will be picked up by the Space the next time the Space rebuilds (forced by a small commit, e.g. bumping a version comment in `app.py`).

---

## 3. (Optional) Add example clips

Drop a few short `.wav` clips into `SONAR-demo/examples/` (e.g. one bonafide, one TTS, one neural-codec) and `git push`. They show up as one-click examples in the Gradio UI.

---

## Failure modes

| Symptom                                              | Fix                                                                              |
|------------------------------------------------------|----------------------------------------------------------------------------------|
| `hf: not logged in`                                  | Re-run `hf auth login` with a Write-scope token.                                  |
| Upload hangs at one file                             | The script is resumable — re-run with `--only <filename>` to retry that file.    |
| Space build error: "fairseq install failed"          | Pin a different fairseq version in `sonar-demo/requirements.txt` (e.g. `fairseq==0.12.2` is most stable).                                                  |
| Space cold start times out (>5 min)                  | Pre-load weights at build time: add `python -c "from huggingface_hub import hf_hub_download; hf_hub_download('idonithid/SONAR-weights', 'xlsr2_300m.pt')"` to a `prerun.sh`. |
| Detector returns `Detector not ready: …`            | First request triggers weight download — wait ~60 s and retry. The status is shown in the verdict box.                                                          |

Once the Space is live, the project page's "Live demo (coming soon)" button automatically flips to "Try the live demo →" via `status.js`.
