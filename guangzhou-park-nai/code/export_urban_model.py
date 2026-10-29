"""Fit and export the six-predictor urban M4 model; published models remain unchanged."""
import os
import numpy as np
import pandas as pd
from package_paths import get_paths
from grouping import assign_fixed_site_groups
from pathlib import Path
import joblib
import lightgbm as lgb

PATHS = get_paths()
CSV = str(PATHS.master_table)
OUTDIR = str(PATHS.output("model_export"))
os.makedirs(OUTDIR, exist_ok=True)



ANOM_FEATS = ["doy_sin", "doy_cos", "hls_model_evi2",
              "hls_model_ndmi", "gpm_precip_uncal_d0_mm", "gpm_precip_uncal_prev7d_mm"]

df = pd.read_csv(CSV)
df["date"] = pd.to_datetime(df["date"])
y = df["average_value"].to_numpy(float)
y_log = np.log1p(y)
df["group"] = assign_fixed_site_groups(df)

def fit_lgbm(X, yv, seed=0, n=1000):
    m = lgb.LGBMRegressor(n_estimators=n, learning_rate=0.03, num_leaves=31,
                          min_child_samples=20, subsample=0.9, colsample_bytree=0.85,
                          reg_lambda=10, n_jobs=-1, random_state=seed, verbose=-1)
    m.fit(X, yv)
    return m

# ---------------- urban subset ----------------
sub = df[df["group"] == "A_urban"].reset_index(drop=True)
yv = sub["average_value"].to_numpy(float)
yv_log = np.log1p(yv)
sites = sub["area"].unique()

site_mean_log = sub.groupby("area")["average_value"].apply(lambda v: np.log1p(v.mean()))
grp_mean_log = float(np.log1p(sub["average_value"].mean()))
base_tr = sub["area"].map(site_mean_log).to_numpy()
resid = yv_log - base_tr
ma_full = fit_lgbm(sub[ANOM_FEATS], resid, 0)
smear_full = float(np.mean(np.exp(yv_log - base_tr - ma_full.predict(sub[ANOM_FEATS]))))

artifact = {
    "model_name": "M4_urban_group_two_part_anomaly_lgbm",
    "version": "v9_paper_final",
    "created_for": "monthly-seasonal NAI raster mapping over unmonitored urban parks (Guangzhou-like)",
    "anomaly_model": ma_full,
    "anom_feats": ANOM_FEATS,
    "group_mean_log1p_urban": grp_mean_log,
    "group_mean_linear_urban": float(np.expm1(grp_mean_log)),
    "site_mean_log1p_urban": {k: float(v) for k, v in site_mean_log.to_dict().items()},
    "smear": smear_full,
    "target": "log1p(NAI ions/cm3 daily mean)",
    "prediction_formula": "pred = clip(expm1(grp_mean_log1p_urban + anomaly_model.predict(X_anom)) * smear, 0, None)",
    "feature_ranges_train": {c: [float(sub[c].min()), float(sub[c].max())] for c in ANOM_FEATS},
    "train_n": int(len(sub)),
    "train_sites": sorted(sub["area"].unique().tolist()),
    "train_period": [str(sub["date"].min().date()), str(sub["date"].max().date())],
}
joblib.dump(artifact, str(Path(OUTDIR) / 'm4_urban_model.joblib'))
print("saved ->", str(Path(OUTDIR) / 'm4_urban_model.joblib'))
