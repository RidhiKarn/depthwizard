"""
Helper to build a manifest.csv (see dataset.py) from two directories of
matching RGB / height-raster files, paired by SAME BASENAME (e.g.
rgb_dir/tile_042.tif <-> height_dir/tile_042.tif). This covers the
common case for ISPRS Vaihingen/Potsdam and DFC2018/2019 once you've
downloaded and (if needed) renamed files to match — their actual
zip/folder layouts differ per dataset and per download, so this script
intentionally does NOT try to parse them directly. Adjust the pairing
logic below once you have the real files in front of you; this is a
starting point, not a guarantee-it-works-blind script (we don't have
the datasets to test it against in this repo).

Usage:
    python build_manifest.py \\
        --rgb-dir /path/to/rgb --height-dir /path/to/dsm \\
        --category urban --out manifest_urban.csv

Run once per dataset/category, then concatenate the resulting CSVs (or
pass multiple --manifest args isn't supported by train_lora.py — just
cat them together, keeping one header row) into your final train/val
manifests.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

HEIGHT_EXTENSIONS = {".tif", ".tiff", ".npy"}
RGB_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rgb-dir", required=True, type=Path)
    p.add_argument("--height-dir", required=True, type=Path)
    p.add_argument("--category", default="unknown")
    p.add_argument("--height-scale", type=float, default=1.0)
    p.add_argument("--height-offset", type=float, default=0.0)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    rgb_by_stem = {
        f.stem: f for f in args.rgb_dir.iterdir() if f.suffix.lower() in RGB_EXTENSIONS
    }
    height_by_stem = {
        f.stem: f for f in args.height_dir.iterdir() if f.suffix.lower() in HEIGHT_EXTENSIONS
    }

    shared_stems = sorted(set(rgb_by_stem) & set(height_by_stem))
    missing_rgb = sorted(set(height_by_stem) - set(rgb_by_stem))
    missing_height = sorted(set(rgb_by_stem) - set(height_by_stem))

    if missing_rgb:
        print(f"WARNING: {len(missing_rgb)} height file(s) with no matching RGB by basename, skipped.")
    if missing_height:
        print(f"WARNING: {len(missing_height)} RGB file(s) with no matching height file by basename, skipped.")
    if not shared_stems:
        raise SystemExit(
            "No matching (rgb, height) pairs found by basename. This script assumes "
            "identical filenames across the two directories — check your actual "
            "downloaded folder structure and adjust the pairing logic if it differs."
        )

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rgb_path", "height_path", "height_scale", "height_offset", "category"])
        for stem in shared_stems:
            writer.writerow(
                [
                    str(rgb_by_stem[stem].resolve()),
                    str(height_by_stem[stem].resolve()),
                    args.height_scale,
                    args.height_offset,
                    args.category,
                ]
            )

    print(f"Wrote {len(shared_stems)} pairs to {args.out}")


if __name__ == "__main__":
    main()
