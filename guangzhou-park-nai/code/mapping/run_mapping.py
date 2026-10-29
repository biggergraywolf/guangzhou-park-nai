"""Run the study's seasonal prediction, masking, clipping, aggregation and trend stages."""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True

from package_paths import (
    CODE_DIR, PACKAGE_ROOT, FROZEN, DEFAULT_MODEL, SEASONS,
    input_path, output_path, reproduction_root, select_seasons,
)

FEATURES = ("doy_sin", "doy_cos", "hls_model_evi2", "hls_model_ndmi",
            "gpm_precip_uncal_d0_mm", "gpm_precip_uncal_prev7d_mm")


def complete_features(path, seasons):
    return all((path / tag / (feature + ".tif")).is_file()
               for tag in seasons for feature in FEATURES)


def raster_source(stage, seasons, fresh, override=None):
    candidates = [input_path(override)] if override else [fresh / stage, FROZEN / stage]
    for candidate in candidates:
        if all((candidate / f"GZ_NAI_{tag}.tif").is_file() for tag in seasons):
            return candidate
    raise FileNotFoundError("No complete seasonal input in: " + ", ".join(map(str, candidates)))


def run(script, *arguments):
    command = [sys.executable, "-B", str(CODE_DIR / script), *map(str, arguments)]
    print("Running:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True, cwd=PACKAGE_ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("prepare", "predict", "mask", "clip", "aggregate", "trend"))
    parser.add_argument("--seasons", help="Comma-separated subset of 2016Q1–2024Q4; default all 36 seasons.")
    parser.add_argument("--inputs", help="External aligned feature directory for predict.")
    parser.add_argument("--gee-dir", help="External historical GEE raster exports for prepare.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--source-dir", help="Explicit source rasters for mask/clip/aggregate, or annual inputs for trend.")
    parser.add_argument("--output-dir", help="External reproduction root; stage files go into its mapping subdirectory.")
    args = parser.parse_args()
    seasons = select_seasons(args.seasons)
    if args.step in ("aggregate", "trend") and set(seasons) != set(SEASONS):
        parser.error("aggregate/trend require all 36 seasons in the fixed nine-year analysis")
    external = reproduction_root(args.output_dir)
    fresh = external / "mapping"
    os.environ["NAI_REPRODUCED_DIR"] = str(external)
    fresh.mkdir(parents=True, exist_ok=True)
    if args.step == "prepare":
        if not args.gee_dir:
            parser.error("prepare requires --gee-dir; historical GEE input rasters are not bundled")
        run("prepare_model_inputs.py", "--gee-dir", input_path(args.gee_dir),
            "--out-dir", fresh / "model_inputs", "--only", ",".join(seasons),
            "--doy-mode", "daily_mean", "--precip-mode", "daily_equiv", "--resampling", "nearest")
    elif args.step == "predict":
        inputs = input_path(args.inputs) if args.inputs else fresh / "model_inputs"
        if not complete_features(inputs, seasons):
            raise FileNotFoundError("Missing aligned inputs; run prepare or supply --inputs explicitly: " + str(inputs))
        manifest = fresh / "mapping_seasons.csv"
        with manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("season", "feat_dir", "out_name"))
            for tag in seasons:
                writer.writerow((tag, os.path.relpath(inputs / tag, manifest.parent).replace("\\", "/"),
                                 f"GZ_NAI_{tag}.tif"))
        run("apply_m4_urban_raster_v2.py", "--model", input_path(args.model), "--manifest", manifest,
            "--out-dir", fresh / "NAI_output", "--season-mode", "daily_mean")
    elif args.step == "mask":
        source = raster_source("NAI_output", seasons, fresh, args.source_dir)
        run("mask_nai_output.py", "--in-dir", source, "--out-dir", fresh / "NAI_masked",
            "--keep", "1", "--only", ",".join(seasons))
    elif args.step == "clip":
        source = raster_source("NAI_masked", seasons, fresh, args.source_dir)
        run("clip_urban_parks.py", "--in-dir", source, "--out-dir", fresh / "NAI_urban_parks_clip",
            "--only", ",".join(seasons))
    elif args.step == "aggregate":
        source = raster_source("NAI_urban_parks_clip", seasons, fresh, args.source_dir)
        run("aggregate_seasonal_annual.py", "--in-dir", source,
            "--out-dir", fresh / "NAI_urban_parks_clip")
    elif args.step == "trend":
        arguments = ["--out-dir", fresh / "NAI_urban_parks_clip" / "trend_theilsen_mk_2016_2024"]
        if args.source_dir:
            arguments += ["--annual-dir", input_path(args.source_dir)]
        run("trend_raster_theilsen_mk.py", *arguments)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc))
