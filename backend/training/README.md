# Stage 2b — LoRA fine-tuning for Depth Anything

This is DepthWizard's primary differentiator: teaching the depth model
to emit plausible **absolute** heights directly from image content, so
it works even on images with zero location metadata (PNG/JPG), beyond
the official PS's minimum requirement (relative-only for that case).

**Everything in `train_lora.py` was checked against the real
`LiheYoung/depth-anything-small-hf` model in this repo's own backend
venv before being written** — the LoRA target module names, the
decoder-fine-tuning mechanism, the save/load round trip, and the
image-preprocessing approach are all verified working (see
`backend/tests/test_lora_calibration.py`, which exercises the same
round trip as an automated test). What is **not** verified is actual
training convergence or accuracy — this dev environment has no GPU and
no training data. That's what this README gets you set up for.

## 1. Get the datasets

You need registered access to at least the urban pairs before this is
worth running at all. See the root `README.md` > Datasets section for
links; summary:

| Dataset | Where | Account needed? | Speed |
|---|---|---|---|
| ISPRS Vaihingen + Potsdam | isprs.org benchmark page | Free request form, any email | Hours–days approval |
| IEEE GRSS DFC2018/2019 | dase.ticinumaerospace.com | Free registration + contest terms | Usually fast |
| GEDI / ICESat-2 (forest coverage) | NASA Earthdata + `earthaccess` | Free Earthdata Login | Instant |
| NASADEM (for self-built hilly/sparse pairs) | OpenTopography API | Free API key | Instant |

For self-built hilly/sparse pairs: download Sentinel-2 or Landsat RGB
tiles for your chosen region, and matching NASADEM/Copernicus DEM
elevation for the same bounds (`backend/app/services/calibration/dem_calibration.py`
already has working Copernicus DEM fetch code you can reuse/adapt for
bulk tile downloading instead of per-request).

## 2. Build a manifest.csv

`dataset.py` reads a flat CSV — see its docstring for the exact
columns. `build_manifest.py` is a starting-point script for the common
case (RGB and height files sharing a basename in two directories):

```bash
python build_manifest.py \
    --rgb-dir /path/to/vaihingen/rgb --height-dir /path/to/vaihingen/dsm \
    --category urban --out manifest_vaihingen.csv
```

Run once per dataset/terrain category, then concatenate the CSVs
(keep one header row) into `manifest_train.csv` / `manifest_val.csv`
(hold out ~10-20% of tiles per category for validation — the project's
evaluation criteria wants accuracy reported per terrain type, so make
sure your val split actually covers all your categories, not just the
biggest one).

**If your downloaded folder structure doesn't pair by matching
basenames** (likely, since ISPRS/DFC ship in their own layouts),
`build_manifest.py`'s pairing logic is meant to be edited, not treated
as guaranteed-correct — we don't have the actual files to test it
against in this repo.

## 3. Run on Colab or Kaggle (free GPU)

Paste into a Colab/Kaggle GPU-runtime notebook:

```python
!git clone <your-repo-url> depthwizard
%cd depthwizard/backend/training
!pip install -q -r requirements-train.txt   # does NOT touch torch — see the file's comment

# Upload or mount your manifest CSVs and the RGB/height files they reference first.

!python train_lora.py \
    --manifest manifest_train.csv \
    --val-manifest manifest_val.csv \
    --output-dir lora_adapter_out \
    --epochs 5 \
    --batch-size 16
```

Expected runtime on a free-tier T4: rough order of magnitude minutes-
per-epoch for a few thousand tiles at 518×518/batch 16 — actual time
depends entirely on dataset size; the script prints per-epoch timing so
you'll know quickly whether 5 epochs fits your session limit. If it
doesn't, reduce `--batch-size` or run fewer epochs and check the
`training_report.json` trend before deciding whether more epochs would
help.

For the highest-accuracy report run (not the live-demo default), pass
`--model-id LiheYoung/depth-anything-large-hf`.

## 4. Get the trained adapter back into the app

Download `lora_adapter_out/` (contains `adapter_config.json`,
`adapter_model.safetensors`, `training_report.json`) and place it at:

```
backend/data/lora_adapter/
```

(or point `DEPTHWIZARD_LORA_ADAPTER_DIR` at wherever you put it). No
code changes needed —
`app/services/calibration/lora_calibration.py::is_lora_available()`
picks it up automatically the next time the backend starts.

## 5. Report the real numbers

`training_report.json` has per-epoch train loss and, if you passed
`--val-manifest`, per-category RMSE/MAE/correlation. Paste the final
epoch's validation numbers into the root README's Calibration table —
**do not** claim accuracy that wasn't actually measured. If the numbers
are bad, that's useful information too (maybe more epochs, more data,
or a different `--decoder-lr` is needed) — report it rather than
hiding it.
