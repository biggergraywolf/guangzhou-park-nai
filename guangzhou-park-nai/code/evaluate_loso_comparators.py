"""Grouped leave-one-station-out evaluation of M1, M2 and their equal-weight ensemble."""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from package_paths import get_paths
from grouping import assign_fixed_site_groups
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

PATHS = get_paths()
CSV = str(PATHS.master_table)
OUTDIR = str(PATHS.output("comparison"))
ABDIR = OUTDIR

CORE8 = ["doy_sin", "doy_cos", "dem_elevation_m", "dw_baseline2023_trees_prob",
         "hls_model_evi2", "hls_model_ndmi", "gpm_precip_uncal_d0_mm", "days_since_start"]
CORE7 = [c for c in CORE8 if c != "days_since_start"]



def fit_lgbm(X, y_log, seed=0, n=1000):
    m = lgb.LGBMRegressor(n_estimators=n, learning_rate=0.03, num_leaves=31,
                          min_child_samples=20, subsample=0.9, colsample_bytree=0.85,
                          reg_lambda=10, n_jobs=-1, random_state=seed, verbose=-1)
    m.fit(X, y_log)
    return m


def fit_xgb(X, y_log, seed=0):
    m = xgb.XGBRegressor(n_estimators=1200, learning_rate=0.03, max_depth=5,
                         min_child_weight=5, subsample=0.9, colsample_bytree=0.85,
                         reg_lambda=10, objective="reg:squarederror",
                         n_jobs=-1, random_state=seed)
    m.fit(X, y_log)
    return m


def smear_of(model, X, y_log):
    return float(np.mean(np.exp(y_log - model.predict(X))))


def back(pred_log, smear):
    return np.clip(np.expm1(pred_log) * smear, 0, None)


def predict_candidates(df, tr, te, seed=0):
    X = df[CORE7]
    y = df["average_value"].to_numpy(float)
    y_log = np.log1p(y)
    out = {}
    m1 = fit_lgbm(X[tr], y_log[tr], seed)
    p1 = back(m1.predict(X[te]), smear_of(m1, X[tr], y_log[tr]))
    m2 = fit_xgb(X[tr], y_log[tr], seed)
    p2 = back(m2.predict(X[te]), smear_of(m2, X[tr], y_log[tr]))
    out["M1_LGBM"] = p1
    out["M2_XGB"] = p2
    out["M3_Ensemble"] = (p1 + p2) / 2
    return out


df = pd.read_csv(CSV)
df["date"] = pd.to_datetime(df["date"])
y = df["average_value"].to_numpy(float)
df["group"] = assign_fixed_site_groups(df)

rows = []
for grp in ["A_urban", "B_mountain"]:
    for site in df.loc[df["group"] == grp, "area"].unique():
        te = (df["area"] == site).to_numpy()
        tr = ((df["group"] == grp) & (df["area"] != site)).to_numpy()
        for name, p in predict_candidates(df, tr, te, 0).items():
            rows.append({"protocol": f"C_LOSO_{grp}", "site": site, "model": name,
                         "R2": r2_score(y[te], p),
                         "RMSE": float(np.sqrt(mean_squared_error(y[te], p))),
                         "MAE": float(mean_absolute_error(y[te], p))})
abC = pd.DataFrame(rows)
abC.to_csv(str(Path(ABDIR) / 'loso_comparators.csv'), index=False, encoding="utf-8-sig")
print("M1-M3 grouped LOSO saved,", len(abC), "rows")
