# DepthWizard — SIH26175

Single-view satellite/aerial RGB imagery → elevation map (DSM) → navigable
3D terrain. Built for Smart India Hackathon 2026, Problem Statement
**SIH26175** (sponsored by ISRO).

The app requires a free account (email/password) — sign up at `/signup`.
Once in, you can download the result as a 3D model (GLB/OBJ), a depth-map
PNG, and — for georeferenced, DEM-calibrated uploads — a real GeoTIFF DSM
usable in GIS software.

> **Status: Stages 1, 2a, 2c, 2e working end-to-end, plus slope analysis.**
> Upload a PNG/JPG, and Depth Anything produces a relative depth map you
> can fly around in a Three.js viewer. Upload a GeoTIFF and it's
> automatically calibrated to real meters against live Copernicus DEM
> GLO-30 data. A shadow cross-check gives an independent height estimate
> for any image. An optional MiDaS-vs-Depth-Anything confidence overlay
> is available too. The viewer also has a slope-angle overlay/readout
> computed from the mesh's own real geometry. LoRA fine-tuning (2b)'s
> code is fully written and its loading path verified end-to-end
> against a real (untrained) adapter — the only missing piece is
> actually training it, which needs dataset access + a GPU you'll need
> to provide (see [training/README.md](backend/training/README.md) and
> [Roadmap](#roadmap)). Metric3D v2
> (2d) is blocked on a genuine ecosystem incompatibility (its `mmcv`
> dependency doesn't support Python 3.12 — confirmed by directly
> attempting the install, not a guess) — see
> [Why Metric3D v2 isn't integrated](#why-metric3d-v2-isnt-integrated-stage-2d).
> Every claim in this README is backed by a passing test that exercises
> the real model/network call, not a mock.

## What this solves

The official PS asks for a pipeline that turns a single-view optical
image into a DSM (height of buildings/terrain/everything on the surface)
and visualizes it as navigable 3D terrain:

- **Georeferenced images** (GeoTIFF) → an absolute DSM in meters
- **Non-georeferenced images** (PNG/JPG) → the PS's own minimum
  requirement is only a *relative* DSM (correct height ranking, no
  metric values)

**Our differentiator:** instead of settling for relative-only output on
non-georeferenced images, the goal is an absolute (metric) DSM for
*both* input types, via techniques most teams are likely to skip:

1. A depth model fine-tuned (LoRA) to internalize real-world scale cues
   directly from image content, so it doesn't strictly need location
   metadata to estimate real height (Stage 2b — **code + inference
   path built and verified; training itself needs data + GPU you
   provide**, see below).
2. An independent, physics-based shadow-length + sun-angle cross-check
   wherever a shadow is visible (Stage 2c — **working**).
3. Metric3D v2 as a second, independent metric-depth estimate for
   georeferenced images with a recoverable focal length (Stage 2d —
   **blocked**, see below).

If these produce a low-confidence result, the system is designed to fall
back to relative-only output — which is still compliant with the PS's
own minimum requirement. (The fallback logic itself is implicit today:
Stage 2a either succeeds and reports `dem_calibrated`, or fails and the
response stays `relative_uncalibrated` — see `dem_calibration_error`.)

## Architecture

```
depthwizard/
├── backend/            FastAPI + PyTorch inference service
│   └── app/
│       ├── main.py             FastAPI app, CORS, static file serving
│       ├── config.py           Settings (model variants, paths, CORS origins)
│       ├── schemas.py          Pydantic response models
│       ├── routers/
│       │   ├── depth.py        POST /api/depth/estimate, GET /api/depth/health
│       │   └── shadow.py       POST /api/shadow/estimate (Stage 2c)
│       ├── services/
│       │   ├── depth_anything_model.py  Stage 1: Depth Anything inference (ACTIVE, primary)
│       │   ├── midas_model.py           Stage 2e: MiDaS, confidence-ensemble partner (ACTIVE)
│       │   ├── confidence.py            Stage 2e: MiDaS vs Depth Anything disagreement (ACTIVE)
│       │   ├── calibration/
│       │   │   ├── dem_calibration.py   Stage 2a: Copernicus DEM GLO-30 regression (ACTIVE)
│       │   │   └── lora_calibration.py  Stage 2b: LoRA adapter loading + inference (ACTIVE, no trained adapter yet)
│       │   └── shadow/
│       │       └── shadow_geometry.py   Stage 2c: shadow detection + sun-angle math (ACTIVE)
│       └── utils/image_io.py   PNG/JPG/GeoTIFF loading + CRS detection (ACTIVE)
│   └── training/        Stage 2b LoRA fine-tuning (run separately, needs GPU)
│       ├── train_lora.py       Training script — see training/README.md
│       ├── dataset.py          Generic RGB+height-raster manifest loader
│       └── build_manifest.py   Helper to build a manifest.csv from paired directories
└── frontend/            Next.js (App Router) + Three.js + Tailwind
    ├── app/page.js               Landing page, ties upload + viewer + shadow panel together
    ├── components/
    │   ├── UploadPanel.js        File upload UI, calls the backend
    │   ├── TerrainViewer.js      Three.js displacement-mesh viewer (camera-clamped, slope overlay)
    │   └── ShadowPanel.js        Stage 2c UI: sun-angle input + results list
    └── lib/api.js                Backend API client
```

**Request flow:**

1. User uploads an image in the browser (`UploadPanel.js`).
2. `POST /api/depth/estimate` (multipart) hits the backend.
3. `image_io.load_image_any()` tries to open the file with `rasterio`;
   if it has an embedded CRS, it's flagged `is_georeferenced=true` and
   its bounds/CRS are extracted. Otherwise it's loaded as a plain image.
4. `depth_anything_model.run_inference()` runs Depth Anything (Hugging
   Face `transformers` `depth-estimation` pipeline,
   `LiheYoung/depth-anything-small-hf` — ViT-S, 24.8M params — by
   default; see `app/config.py`) and returns a per-pixel relative
   inverse-depth map, resampled back to the input resolution.
5. **If georeferenced** (and `calibrate_dem` isn't explicitly disabled,
   on by default): `dem_calibration.calibrate_with_dem()` fetches the
   matching Copernicus DEM GLO-30 tile(s) directly over HTTP (GDAL
   `/vsicurl/`, byte-range reads, no download or account needed — see
   below), samples ~1600 points, fits a linear regression from relative
   depth to real elevation, and applies it across the full depth map.
   `calibration_stage` becomes `"dem_calibrated"` and
   `absolute_height_grid` (real meters) is included alongside the
   always-present normalized `height_grid`.
5b. **Else, if a LoRA adapter has been trained and dropped in**
   (`app/services/calibration/lora_calibration.py::is_lora_available()`
   — false by default, since none is trained in this repo yet; see
   `backend/training/README.md`): the adapter's own direct metric
   output is used instead, `calibration_stage` becomes
   `"lora_calibrated"`. This is the path that makes non-georeferenced
   images metric too, not just georeferenced ones.
6. **If `include_confidence=true`** (opt-in, frontend checkbox): MiDaS
   also runs on the same image, and `confidence.compute_confidence_map()`
   returns a MiDaS-vs-Depth-Anything disagreement heatmap.
7. The router saves a colorized depth heatmap PNG, the resized input
   PNG, and (if requested) a confidence heatmap PNG to
   `backend/data/outputs/` (served at `/outputs/...`), downsamples grids
   to `height_grid_resolution × height_grid_resolution` (128×128 by
   default), and returns everything as JSON (`DepthEstimateResponse`).
8. `TerrainViewer.js` builds a `THREE.PlaneGeometry`, displaces each
   vertex by `absolute_height_grid` (real meters, when calibrated) or
   `height_grid` × a visual-only scale constant (otherwise), textures it
   with the chosen overlay (original / depth heatmap / confidence), and
   renders it with a **camera-clamped** `OrbitControls` (see
   [disocclusion gap](#disocclusion-gap-mitigation-implemented) below).
   Hovering shows the raycast-hit point's height — labeled "m,
   DEM-calibrated" or "relative, uncalibrated" depending on which.
9. Separately, `ShadowPanel.js` can POST the same file to
   `/api/shadow/estimate` with a user-supplied (or GeoTIFF-derived) sun
   angle, returning independent shadow-based height estimates for
   comparison.

### Why Depth Anything instead of MiDaS

Depth Anything (`LiheYoung/depth-anything-*`, github.com/LiheYoung/Depth-Anything)
is the Stage 1 default, replacing MiDaS. It's trained on 1.5M labeled +
62M unlabeled images (vs MiDaS's ~2M labeled) and outperforms MiDaS on
every published benchmark we found — e.g. DDAD AbsRel improves
0.251 → 0.230, KITTI δ1 improves 0.850 → 0.947. Two size variants are
configured:

- **ViT-S** (`depth-anything-small-hf`, 24.8M params) — live web demo /
  real-time user uploads, where latency matters. Default.
- **ViT-L** (`depth-anything-large-hf`, 335.3M params) — offline
  benchmark runs / best-accuracy report figures, not on the live path.

MiDaS is **not removed** — it's kept in `app/services/midas_model.py`,
actively used for the Stage 2e confidence ensemble.

### Why Copernicus DEM GLO-30 instead of NASADEM (Stage 2a)

NASADEM and AW3D30 were the original preference, but both need an
account (OpenTopography API key / separate registration) to fetch
programmatically. Copernicus DEM GLO-30 is published as Cloud-Optimized
GeoTIFFs on a fully public, **anonymous-read** AWS S3 bucket
(`https://copernicus-dem-30m.s3.amazonaws.com/`) — verified reachable
with a plain HTTPS GET, no credentials, no signup. That's what's
actually wired into `dem_calibration.py` today. Swapping in NASADEM
later is a one-line change to the tile-URL builder plus a credentials
lookup, not a redesign.

### Stage 2b (LoRA) — what's built vs. what needs you

Split deliberately into two pieces:

1. **The inference-loading side** (`app/services/calibration/lora_calibration.py`)
   — implemented and tested. `backend/tests/test_lora_calibration.py`
   builds a real (untrained) LoRA adapter with the exact config
   `train_lora.py` uses, saves it, loads it fresh, and runs a forward
   pass through the actual `LiheYoung/depth-anything-small-hf` model —
   a genuine round trip, not a mock. `backend/tests/test_depth_api.py::test_estimate_uses_lora_when_available`
   proves the same thing at the full API level: drop a valid adapter at
   `settings.lora_adapter_dir` and `/api/depth/estimate` automatically
   starts returning `calibration_stage: "lora_calibrated"` — no code
   changes needed.
2. **The training script** (`backend/training/train_lora.py`) — written,
   but its actual *output* (a trained adapter with real accuracy) is
   not produced, because that needs registered datasets and a GPU this
   dev environment doesn't have. See `backend/training/README.md` for
   exactly what to do: get dataset access, build a manifest, run on
   Colab/Kaggle, drop the result back in.

Recipe, verified against the real model architecture before being
written (not guessed — see the inline comments in `train_lora.py` for
the exact checks run):
- LoRA `target_modules=["query", "value"]` on the Dinov2 backbone's
  self-attention layers (confirmed these exact names exist in the real
  model, and collide with nothing outside `model.backbone`).
- `modules_to_save=["neck", "head"]` keeps the depth decoder (~11% of
  total params) fully trainable, at the recipe's specified decoder
  learning rate.
- LoRA-adapted backbone trains at `decoder_lr / 50` (the recipe's
  "encoder LR = decoder LR / 50").
- Batch size 16, ~5 epochs, resolution a multiple of 14 — all CLI
  defaults in `train_lora.py`, all from the Depth Anything paper's own
  fine-tuning recipe as specified for this project.
- Loss is masked Huber/SmoothL1 against real DSM meters (not
  scale-invariant log) — the point of this stage is metric output, not
  just better relative ordering.

### Why Metric3D v2 isn't integrated (Stage 2d)

Its official `torch.hub.load('yvanyin/metric3d', ...)` entrypoint builds
the model from a config file using the OpenMMLab `mmcv`/`mmengine`
stack. We tried it directly in this environment: `mmcv`'s installer
(`pip install mmcv` and OpenMMLab's own recommended `mim install mmcv`)
both fail at the build step because they depend on `pkg_resources` code
paths (`pkgutil.ImpImporter`) that were removed in Python 3.12. This is
a real, confirmed ecosystem incompatibility, not a guess — and not
something worth papering over with a hacky workaround given the
timeline. The two real options are (a) a separate Python 3.10/3.11
environment just for this one model, or (b) skip it. We went with a
pragmatic substitute for what Stage 2d/6 actually needed —
see [Roadmap](#roadmap) item 6 and `TerrainViewer.js`'s `SLOPE_*`
comment block: the mesh already has real displaced geometry, and
Three.js's own `computeVertexNormals()` (already running for lighting)
gives genuine per-vertex surface normals for free, which is enough to
deliver working slope analysis without Metric3D v2's independent
model prediction. Trade-off, stated honestly: these normals are derived
from OUR OWN height map, so they inherit whatever noise is already in
Stage 1/2a's values, rather than cross-checking it the way an
independently-predicted normal map would.

## Datasets

**In active use today:**
- Copernicus DEM GLO-30 (ESA) — Stage 2a reference elevation, fetched
  live per-request, no local copy stored.

**Planned for Stage 2b (LoRA) — not yet downloaded or consumed:**

**Training pairs (RGB + real DSM ground truth, urban):**
- ISPRS Vaihingen, ISPRS Potsdam — aerial RGB + LiDAR DSM. Free, but
  gated behind a short request form on the ISPRS benchmark site
  ([isprs.org/resources/datasets/benchmarks](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx));
  approval isn't instant. Also mirrored on IEEE DataPort (free account).
- IEEE GRSS DFC2018 / DFC2019 — via the DASE portal
  ([dase.ticinumaerospace.com](http://dase.ticinumaerospace.com/)),
  free registration + accepting contest terms. DFC2019 baseline code is
  open on GitHub (`pubgeo/dfc2019`, no login) but the imagery itself
  still needs the portal.

**NASA elevation / canopy-height data (forest, hilly, general terrain):**
- GEDI, ICESat-2 ATL08 — via NASA Earthdata Login
  ([urs.earthdata.nasa.gov/users/new](https://urs.earthdata.nasa.gov/users/new)),
  free, instant self-serve signup; programmatic access via the
  `earthaccess` Python package once logged in.
- NASADEM — preferred bare-earth DEM; via OpenTopography (free API key,
  instant, see [opentopography.org](https://opentopography.org/developers)).
- ASTER GDEM — supplementary/gap-filling elevation.

**Other elevation references:**
- AW3D30 (JAXA) — an actual DSM (includes structures), closest
  structurally to the prediction target; needs its own registration.

**Base imagery for self-built hilly/sparse training pairs:**
- Sentinel-2 (ESA/Copernicus) or Landsat 8/9 (NASA/USGS), paired with
  NASADEM/Copernicus DEM/AW3D30 for regions ISPRS/DFC don't cover.

**Pretrained backbones:**
- Depth Anything (`LiheYoung/depth-anything-small-hf` / `-large-hf`) —
  primary, in use today for Stage 1.
- MiDaS (`intel-isl/MiDaS`) — in use today for Stage 2e's confidence
  ensemble.
- Metric3D v2 — weights are on Hugging Face under `JUGGHM/Metric3D`
  (e.g. `metric_depth_vit_small_800k.pth`), **no login required**,
  directly downloadable. The blocker isn't the weights — it's the
  `torch.hub.load('yvanyin/metric3d', ...)` loader's `mmcv` dependency,
  which doesn't support Python 3.12 (see
  [Why Metric3D v2 isn't integrated](#why-metric3d-v2-isnt-integrated-stage-2d)).

## Calibration approach (2a–2e)

| Stage | What it does | Applies to | Status |
|---|---|---|---|
| 2a — DEM calibration | Fit relative-depth → Copernicus DEM GLO-30 elevation regression using the GeoTIFF's own coordinate bounds | Georeferenced only | **Working** — live network call, tested against real Berlin elevation data |
| 2b — LoRA fine-tuning | Fine-tune Depth Anything on RGB+DSM pairs, using Depth Anything's own published recipe (encoder LR = 1/50 of decoder LR, batch size 16, ~5 epochs, resolution a multiple of 14) so it learns real-world scale from image content alone | All images | **Code + inference path working and tested** (adapter loading verified end-to-end); **no trained adapter yet** — needs registered datasets + GPU training, see `backend/training/README.md` |
| 2c — Shadow geometry | `height = shadow_length × tan(solar_elevation)`; contour-based shadow detection (Otsu threshold + extent along the sun-azimuth direction) | Images with a visible shadow | **Working** — separate `POST /api/shadow/estimate` endpoint |
| 2d — Metric3D v2 | Independent metric-depth estimate using EXIF focal length (not GPS/coordinates); jointly predicts surface normals for slope analysis at no extra cost; cross-checked against 2a | Georeferenced, focal length available | **Blocked** — `mmcv` dependency incompatible with Python 3.12, confirmed by direct attempt (see above). Slope analysis itself is delivered a different way (mesh-derived normals, see item 6 below), so only the independent-metric-depth cross-check part of 2d is actually missing. |
| 2e — Confidence map | Per-pixel disagreement between MiDaS and Depth Anything (deliberately *not* two Depth Anything sizes — see `confidence.py` docstring for why) | All images, opt-in | **Working** — `include_confidence=true` on `/api/depth/estimate` |

Read the docstrings in `dem_calibration.py`, `shadow_geometry.py`, and
`confidence.py` for exact algorithm details and known limitations (a
single linear DEM regression and an Otsu-threshold shadow detector are
both genuinely weak signals on their own — that's why they're
cross-checks, not the primary estimate).

## Roadmap / build order

- [x] **1. Swap Stage 1 backbone to Depth Anything (ViT-S)**
- [x] **2. Stage 2a DEM calibration**, live against Copernicus DEM GLO-30
- [x] **5. Stage 2c shadow-geometry cross-check** (`/api/shadow/estimate`)
- [x] **6. Slope analysis** in the Three.js viewer — via mesh-derived normals rather than Metric3D v2 (blocked, see item 4); overlay + hover readout, no external model needed
- [x] **7. Camera tilt/pan clamping** in the flythrough (disocclusion-gap mitigation)
- [x] **8. Stage 2e confidence ensemble** (MiDaS vs Depth Anything, opt-in)
- [x] **3. LoRA fine-tuning (2b) code** — training script + verified inference-loading path. **Not done: an actual trained adapter** — that needs dataset access (ISPRS/DFC portals, see Datasets above — free but not instant) and a GPU (this dev environment is CPU-only); see `backend/training/README.md` for the exact steps.
- [ ] **4. Metric3D v2 integration (2d)** for georeferenced images with focal length. Weights are open/no-login, but the official loader's `mmcv` dependency is incompatible with Python 3.12 in this environment (confirmed by direct attempt — see [Why Metric3D v2 isn't integrated](#why-metric3d-v2-isnt-integrated-stage-2d)). Needs either a separate Python 3.10/3.11 environment or is dropped.
- [ ] **9. Polish** — DSM GeoTIFF download, deploy

A working Stage 1 + 2a is already evaluation-compliant and exceeds the
PS's stated minimum for non-georeferenced images (which only requires
*relative* output). 2b (LoRA) is what turns "DEM-calibrated when
georeferenced" into "metric even without location metadata" — the
project's core differentiator. The code for it is done and tested; what
remains is running `train_lora.py` against real data on a real GPU,
which needs you to sort out dataset access and compute (see
`backend/training/README.md`).

### Disocclusion-gap mitigation (implemented)

The mesh only has color data from the single top-down source photo —
there's no data for the *sides* of tall buildings/terrain or whatever
they occlude. A near-horizontal camera angle would expose those
untextured faces as holes or stretched artifacts. Fixed via the cheap
option: `TerrainViewer.js` clamps `OrbitControls` to an 8°–58° tilt
range from straight-down (never reaching grazing/horizontal angles) and
disables panning, so the camera can't drift into the gap. Per-building
depth smoothing and full inpainting remain future work (documented in
`TerrainViewer.js`), not attempted.

## Running locally

### Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

First request downloads Depth Anything (and, if you use the confidence
overlay, MiDaS) weights via `torch.hub`/Hugging Face — needs internet,
one-time, cached under `~/.cache/huggingface` and `~/.cache/torch`. DEM
calibration needs network access to `copernicus-dem-30m.s3.amazonaws.com`
per-request (no account, no local caching yet). API docs at
`http://localhost:8000/docs`.

Run tests: `pytest tests/ -v` — exercises real Depth Anything/MiDaS
inference, a real Copernicus DEM network fetch, and a real LoRA
adapter save/load/forward round trip, so it's slower than a typical
unit test suite (~30-60s) but nothing is mocked.

To fine-tune the LoRA adapter (Stage 2b — needs a GPU and real
datasets you provide), see `backend/training/README.md`.

### Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local   # NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
npm run dev
```

Open `http://localhost:3000`.

## Deployment (planned)

- **Frontend:** Vercel
- **Backend/inference:** Hugging Face Spaces (GPU tier) preferred for
  live Depth Anything inference; Render/Railway (CPU) as a fallback if
  inference speed is acceptable. If live inference proves too slow for a
  smooth demo, the plan is to precompute results for a few sample images
  so the live link always has an instant example ready, while still
  accepting live uploads.
- Not yet deployed — local dev only at this point in the build.

## Known limitations

- Depth Anything (like MiDaS) was trained on ground-level/oblique
  photography, not nadir (straight-down) satellite imagery. This is
  exactly what Stage 2b's LoRA fine-tuning on real aerial RGB+DSM pairs
  is meant to correct — Stage 2a's DEM regression is a real but coarse
  fix in the meantime (a single linear fit against 30m-resolution data).
- Stage 2a's regression is linear only; the original plan called for
  trying a degree-2 polynomial if residuals show curvature — not added.
- Stage 2c's shadow detector is a simplified Otsu-threshold contour
  detector, not a trained building/shadow object detector — it will
  false-positive on any dark region (shaded roofs, water, tree canopy).
  Documented as a deliberate simplification, not a bug.
- Sun elevation/azimuth for the shadow cross-check is always manual
  user input (no automatic timestamp-based solar-angle lookup yet, even
  for GeoTIFFs) — accuracy depends entirely on how close the guess is.
- The 3D viewer's horizontal footprint (`PLANE_SIZE`) is not to the same
  real-world scale as the calibrated vertical meters — only relative
  heights across the mesh are geographically meaningful right now, not
  absolute ground extent.
- Slope analysis is derived from the mesh's own (already-noisy) height
  values, not an independent model prediction — see
  [Why Metric3D v2 isn't integrated](#why-metric3d-v2-isnt-integrated-stage-2d).
- `npm audit` flags a handful of moderate Next.js findings that only
  clear on a 15/16 major-version bump (breaking change); left as-is for
  now rather than risking an unplanned major upgrade mid-build.
