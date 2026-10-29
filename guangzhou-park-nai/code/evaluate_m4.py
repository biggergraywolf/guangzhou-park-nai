"""Evaluate M4 with six dynamic residual predictors: blocked OOF and grouped LOSO."""
import json
import os
import warnings
import numpy as np
import pandas as pd
from package_paths import get_paths
from grouping import assign_fixed_site_groups
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

PATHS = get_paths()
CSV = str(PATHS.master_table)
OUTDIR = str(PATHS.output("m4_final"))



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


# ============ 1) 时间阻塞5折 OOF（已监测公园模式） ============
dates = np.sort(df["date"].unique())
blocks = np.array_split(dates, 5)
oof = np.full(len(df), np.nan)
for blk in blocks:
    te = df["date"].isin(blk).to_numpy()
    tr = ~te
    tr_df = df[tr]
    site_mean_log = tr_df.groupby("area")["average_value"].apply(lambda v: np.log1p(v.mean()))
    grp_mean_log = tr_df.groupby("group")["average_value"].apply(lambda v: np.log1p(v.mean()))
    base_tr = df["area"].map(site_mean_log).to_numpy()
    base_te = df["area"].map(site_mean_log).to_numpy()
    # 站点在某训练块缺失时回退组均值（如大夫山仅190天）
    miss_tr = np.isnan(base_tr)
    if miss_tr.any():
        base_tr[miss_tr] = df["group"].map(grp_mean_log).to_numpy()[miss_tr]
    miss_te = np.isnan(base_te)
    if miss_te.any():
        base_te[miss_te] = df["group"].map(grp_mean_log).to_numpy()[miss_te]
    resid = y_log - base_tr
    ma = fit_lgbm(df[ANOM_FEATS][tr], resid[tr], 0)
    pred_tr_log = base_tr + ma.predict(df[ANOM_FEATS])
    smear = float(np.mean(np.exp(y_log[tr] - pred_tr_log[tr])))
    pred_te = np.clip(np.expm1(base_te[te] + ma.predict(df[ANOM_FEATS][te])) * smear, 0, None)
    oof[te] = pred_te

df["oof_pred_m4"] = oof
df[["row_id", "area", "group", "date", "average_value", "oof_pred_m4"]].to_csv(
    str(Path(OUTDIR) / 'oof_m4_timeblocked_predictions.csv'), index=False, encoding="utf-8-sig")

m_all = {"R2": float(r2_score(y, oof)),
         "RMSE": float(np.sqrt(mean_squared_error(y, oof))),
         "MAE": float(mean_absolute_error(y, oof))}
pr = []
for site, g in df.groupby("area"):
    pr.append({"area": site, "group": g["group"].iloc[0], "n": len(g),
               "R2_temporal": float(r2_score(g["average_value"], g["oof_pred_m4"])),
               "RMSE": float(np.sqrt(mean_squared_error(g["average_value"], g["oof_pred_m4"]))),
               "MAE": float(mean_absolute_error(g["average_value"], g["oof_pred_m4"]))})
pr_df = pd.DataFrame(pr)
pr_df.to_csv(str(Path(OUTDIR) / 'per_site_temporal_r2_M4.csv'), index=False, encoding="utf-8-sig")
med = float(pr_df["R2_temporal"].median())
n_pos = int((pr_df["R2_temporal"] > 0).sum())
with open(str(Path(OUTDIR) / 'oof_m4_summary.json'), "w", encoding="utf-8") as fp:
    json.dump({"oof_M4": m_all, "per_site_temporal_r2_median": med,
               "n_sites_positive": n_pos}, fp, ensure_ascii=False, indent=2)
print(f"[OOF-M4] R2={m_all['R2']:.4f} RMSE={m_all['RMSE']:.0f} MAE={m_all['MAE']:.0f}")
print(f"[OOF-M4] per-site temporal R2 median={med:.3f}, positive sites={n_pos}/14")
print(pr_df.sort_values("R2_temporal")[["area", "R2_temporal"]].to_string(index=False))

# ============ 2) 分组LOSO（未监测公园模式，含合并行） ============
rows = []
for grp in ["A_urban", "B_mountain"]:
    sub = df[df["group"] == grp].reset_index(drop=True)
    sites = sub["area"].unique()
    preds = np.full(len(sub), np.nan)
    yv = sub["average_value"].to_numpy(float)
    yv_log = np.log1p(yv)
    for site in sites:
        te = (sub["area"] == site).to_numpy()
        tr = ~te
        tr_sub = sub[tr]
        site_mean_log = tr_sub.groupby("area")["average_value"].apply(lambda v: np.log1p(v.mean()))
        grp_mean_log = float(np.log1p(tr_sub["average_value"].mean()))
        base_tr = sub["area"].map(site_mean_log).to_numpy()
        resid = yv_log - base_tr
        ma = fit_lgbm(sub[ANOM_FEATS][tr], resid[tr], 0)
        smear = float(np.mean(np.exp(yv_log[tr] - base_tr[tr] - ma.predict(sub[ANOM_FEATS][tr]))))
        base_te = np.full(te.sum(), grp_mean_log)
        preds[te] = np.clip(np.expm1(base_te + ma.predict(sub[ANOM_FEATS][te])) * smear, 0, None)
        obs_s, pred_s = yv[te], preds[te]
        rows.append({"group": grp, "test_site": site, "n": int(te.sum()),
                     "mean_obs": obs_s.mean(), "mean_pred": pred_s.mean(),
                     "RMSE": float(np.sqrt(mean_squared_error(obs_s, pred_s))),
                     "MAE": float(mean_absolute_error(obs_s, pred_s)),
                     "Bias": float((pred_s - obs_s).mean()),
                     "Bias_pct": float(100 * (pred_s - obs_s).mean() / obs_s.mean())})
    r2 = r2_score(yv, preds)
    rmse = float(np.sqrt(mean_squared_error(yv, preds)))
    mae = float(mean_absolute_error(yv, preds))
    # 组均值常数基线
    base_pred = np.zeros_like(yv)
    for site in sites:
        te = (sub["area"] == site).to_numpy()
        base_pred[te] = yv[~te].mean()
    rows.append({"group": grp, "test_site": "__POOLED_MODEL__", "n": len(sub),
                 "mean_obs": yv.mean(), "mean_pred": preds.mean(),
                 "RMSE": rmse, "MAE": mae,
                 "Bias": float((preds - yv).mean()),
                 "Bias_pct": float(100 * (preds - yv).mean() / yv.mean())})
    rows.append({"group": grp, "test_site": "__POOLED_BASELINE_mean__", "n": len(sub),
                 "mean_obs": yv.mean(), "mean_pred": base_pred.mean(),
                 "RMSE": float(np.sqrt(mean_squared_error(yv, base_pred))),
                 "MAE": float(mean_absolute_error(yv, base_pred)),
                 "Bias": float((base_pred - yv).mean()), "Bias_pct": 0.0})
    print(f"[LOSO-M4] {grp}: pooled R2={r2:.3f} RMSE={rmse:.0f} MAE={mae:.0f} | "
          f"baseline MAE={mean_absolute_error(yv, base_pred):.0f}")

out = pd.DataFrame(rows)
out.to_csv(str(Path(OUTDIR) / 'grouped_loso_M4_results.csv'), index=False, encoding="utf-8-sig")
print("grouped_loso_M4_results.csv saved (with pooled rows)")
print("ALL DONE")
