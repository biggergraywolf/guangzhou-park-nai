# Core research code

These scripts reproduce the study's preprocessing, model comparisons, urban M4 export and raster analyses using the current `data/`, `models/` and `results/` directories. Figure layout authoring is outside this core code set. Old diagnostic reports, duplicate dependency files and optional candidate-feature scripts have been removed.

## Model workflow

Run from the project directory, or use the absolute path to `code/run.py` from another working directory:

```text
python code/run.py check
python code/run.py preprocess
python code/run.py validate
python code/run.py export
```

- `check` verifies the 5,998-row master, fixed station groups, formal M4 LOSO results, and daily/monthly/contiguous-season metrics and same-fold station-mean baselines. It does not train models.
- `preprocess` merges the six station-level GEE exports in `data/gee_station_exports/`. Their names use calendar quarters. The generated master matches `data/master_features.csv`.
- `validate` fits the four models under the original evaluation design and assembles the current comparison: A/B estimates, M1-M3 grouped LOSO, and formal M4 time-blocked OOF/grouped LOSO with training-fold smearing where specified. The individual stages are `ab`, `comparators`, `m4`, and `assemble`.
- `export` fits a new urban M4 artifact for mapping. The existing `models/production/m4_urban_model.joblib` remains the frozen production artifact. The separate reference bundle is archival, not the mapping entry point.

The implementation files are `evaluate_models_ab.py`, `evaluate_loso_comparators.py`, `evaluate_m4.py`, `assemble_validation.py`, `export_urban_model.py`, `check.py`, `grouping.py`, `package_paths.py`, and `preprocessing/merge_09_master_core_for_ml_v2.py`. `run.py` is the common entry point.

The seven M1-M3 predictors and six dynamic M4 residual predictors, hyperparameters, data splits and mathematical calculations are retained. The fixed station names in the source monitoring/modeling tables are join keys and are not translated.

## Raster workflow

Raster workflows need the archived GeoTIFF layers, which are not stored in this repository:

```text
export NAI_RASTER_DIR=/path/to/raster_archive      # Windows: set NAI_RASTER_DIR=...
python code/mapping/run_mapping.py clip
python code/mapping/run_mapping.py aggregate
python code/mapping/run_mapping.py trend
```

The inputs are `NAI_RASTER_DIR/NAI_masked/` (by default `results/raster/NAI_masked/`) and `data/spatial/urban_parks.shp` with its companion files. Aggregation and trend analysis can also start from complete frozen preceding stages in the same archive.

`aggregate` also regenerates Q2-minus-Q4 amplitude rasters and their summary using the existing fixed 2,400 ions cm^-3 percentage reference. Six archived park-amplitude rasters have an older grid origin inconsistent with their seasonal sources. Their numerical arrays are retained, while new outputs use the corresponding seasonal grid; see the English trend guide before spatial reuse. The current annual-mean and trend grids are unaffected.

To prepare and predict new seasonal maps, provide the external historical inputs explicitly:

```text
python code/mapping/run_mapping.py prepare --gee-dir /path/to/gee_exports
python code/mapping/run_mapping.py predict --inputs /path/to/aligned_model_inputs
python code/mapping/run_mapping.py mask
```

The large raw GEE/aligned historical inputs are not bundled in this reduced project. The three retained Earth Engine scripts under `mapping/RS_GEE_scripts/` cover station core exports, historical seasonal features and sensor cross-calibration. They require the researcher's own Earth Engine assets and export configuration.

Historical mapping uses 36 three-month seasons grouped into nine March-to-February seasonal years, 2016-2024, covering March 2016-February 2025. Park-specific filenames and result-table labels use the English names used in the associated manuscript. The trend folder includes an English guide describing the balanced annual panel, the separate 90%-coverage core panel, and BH adjustment.

## Output location and environment

All new outputs go **outside this submission directory**, by default to the sibling `NAI_reproduced/`. Use `--output-dir ../NAI_reproduced/run_01` or `NAI_REPRODUCED_DIR` to select another external root. Mapping outputs are placed in its `mapping/` subdirectory. Attempts to write into this submission project are rejected.

`requirements.txt` lists the core libraries. Raster clipping, aggregation and trends require a GDAL Python binding with matching GDAL/OGR and PROJ native libraries; use an environment that supplies these together. Pure model checks use NumPy/pandas; fitting or loading artifacts also requires the relevant model libraries. Numerical platforms and dependency versions can affect new training results.

Packaging checks did not refit models. They reproduced the master byte-for-byte, checked the current 80-row comparison assembly, loaded the production model for sample predictions, and checked raster processing against frozen numerical outputs. New runs never replace the supplied source data or the frozen models and results.
