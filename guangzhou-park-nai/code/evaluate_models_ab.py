"""Compare the four final models under repeated random and fixed cross-year validation."""
import os
import json
import warnings
import numpy as np
import pandas as pd
from package_paths import get_paths
from grouping import assign_fixed_site_groups
from pathlib import Path
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

PATHS = get_paths()
CSV = str(PATHS.master_table)
OUTDIR = str(PATHS.output("comparison"))
ABDIR = OUTDIR
os.makedirs(ABDIR, exist_ok=True)



CORE8 = ["doy_sin", "doy_cos", "dem_elevation_m", "dw_baseline2023_trees_prob",
         "hls_model_evi2", "hls_model_ndmi", "gpm_precip_uncal_d0_mm", "days_since_start"]
CORE7 = [c for c in CORE8 if c != "days_since_start"]
ANOM_FEATS_BASE = ["doy_sin", "doy_cos", "days_since_start", "hls_model_evi2",
                   "hls_model_ndmi", "gpm_precip_uncal_d0_mm", "gpm_precip_uncal_prev7d_mm"]
ANOM_FEATS = [c for c in ANOM_FEATS_BASE if c != "days_since_start"]


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


def metrics(y_true, y_pred):
    return {"R2": r2_score(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "Bias": float(np.mean(y_pred - y_true))}


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
    tr_df = df[tr]
    site_mean_log = tr_df.groupby("area")["average_value"].apply(lambda s: np.log1p(s.mean()))
    grp_mean_log = tr_df.groupby("group")["average_value"].apply(lambda s: np.log1p(s.mean()))
    base_map = df["area"].map(site_mean_log).to_numpy()
    base_log_te = base_map[te].astype(float).copy()
    miss = np.isnan(base_log_te)
    if miss.any():
        base_log_te[miss] = df["group"].map(grp_mean_log).to_numpy()[te][miss]
    resid = y_log[tr] - base_map[tr]
    ma = fit_lgbm(df[ANOM_FEATS][tr], resid, seed)
    out["M4_TwoPart"] = back(base_log_te + ma.predict(df[ANOM_FEATS][te]), 1.0)
    return out


def stratified_split(df, seed, test_size=0.3):
    rng = np.random.RandomState(seed)
    d = df.copy()
    d["q"] = d.groupby("area")["average_value"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
    test_idx = []
    for _, g in d.groupby(["area", "q"]):
        idx = g.index.to_numpy()
        test_idx.extend(rng.choice(idx, size=max(1, int(round(len(idx) * test_size))), replace=False))
    mask = df.index.isin(test_idx)
    return ~mask, mask


def main():
    df = pd.read_csv(CSV)
    df["date"] = pd.to_datetime(df["date"])
    y = df["average_value"].to_numpy(float)
    df["group"] = assign_fixed_site_groups(df)

    comp = []

    # ---- Protocol A ----
    for seed in [42, 2026, 7, 123, 555]:
        tr, te = stratified_split(df, seed)
        for name, p in predict_candidates(df, tr, te, seed).items():
            m = metrics(y[te], p)
            comp.append({"protocol": "A_random_stratified", "seed": seed, "model": name,
                         "n_test": int(te.sum()), **m})
            print(f"A s{seed} {name:12s} R2={m['R2']:.4f} RMSE={m['RMSE']:.0f}")

    # ---- Protocol B ----
    tr = (df["date"] <= "2024-12-31").to_numpy()
    te = ~tr
    for name, p in predict_candidates(df, tr, te, 0).items():
        m = metrics(y[te], p)
        comp.append({"protocol": "B_time_forward", "seed": 0, "model": name,
                     "n_test": int(te.sum()), **m})
        print(f"B {name:12s} R2={m['R2']:.4f} RMSE={m['RMSE']:.0f}")

    ab = pd.DataFrame(comp)
    ab.to_csv(str(Path(ABDIR) / 'model_comparison_ab.csv'), index=False, encoding="utf-8-sig")

    print("Saved four-model Protocol A/B comparison; run the separate grouped LOSO stages for Protocol C.")


if __name__ == "__main__":
    main()
