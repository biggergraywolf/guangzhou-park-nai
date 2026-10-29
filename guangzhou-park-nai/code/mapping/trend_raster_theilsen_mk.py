# -*- coding: utf-8 -*-
"""
7 个公园 9 年 (2016-2024) 逐像元 Theil-Sen + Mann-Kendall 趋势分析
====================================================================
输入 : annual_2016_2024/all_parks/GZ_NAI_YYYY_mean.tif  (9 个年均栅格)
输出 : trend_theilsen_mk_2016_2024/

方法要点
  - 逐像元趋势, 只用"9 年均有效"的平衡面板像元 -> 每个像元 n=9, 检验严格可比
  - Theil-Sen 斜率 = 36 个两两斜率的中位数 (ions/cm3/年)
  - Sen 95% CI 基于 S 的方差
  - MK 检验含并列(ties)校正, 正态近似 (n=9 适用), 双侧
  - 同时输出未校正显著性与 Benjamini-Hochberg FDR 校正后显著性
    (数千个像元同时检验, 未校正必然产生大量假阳性)
"""
import os, csv, math, warnings
import numpy as np
from package_paths import PACKAGE_ROOT, PARKS, FROZEN, REPRODUCED, input_path, output_path, preferred_stage, configure_gdal_data, park_label, park_slug
configure_gdal_data()
from osgeo import gdal, ogr, osr

warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
import argparse


def main():
    fresh = REPRODUCED / 'NAI_urban_parks_clip' / 'annual_2016_2024' / 'all_parks'
    frozen = FROZEN / 'NAI_urban_parks_clip' / 'annual_2016_2024' / 'all_parks'
    ap = argparse.ArgumentParser(description='Nine-seasonal-year pixel Sen/MK/BH trend analysis.')
    ap.add_argument('--annual-dir', default=str(fresh if fresh.is_dir() else frozen))
    ap.add_argument('--out-dir', default=str(REPRODUCED / 'NAI_urban_parks_clip' / 'trend_theilsen_mk_2016_2024'))
    ap.add_argument('--parks', default=str(PARKS))
    args = ap.parse_args()

    gdal.UseExceptions(); ogr.UseExceptions()

    BASE = str(PACKAGE_ROOT)
    CLIP = str(REPRODUCED / "NAI_urban_parks_clip")
    ADIR = str(input_path(args.annual_dir))
    SHP = str(input_path(args.parks))
    OUT = str(output_path(args.out_dir))
    YEARS = list(range(2016, 2025)); N = len(YEARS)
    Z95 = 1.959964
    os.makedirs(os.path.join(OUT, "all_parks"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "by_park"), exist_ok=True)

    # ------------------------------------------------------------------ 数据
    r = gdal.Open(os.path.join(ADIR, "GZ_NAI_%d_mean.tif" % YEARS[0]))
    W, H, GT, PRJ = r.RasterXSize, r.RasterYSize, r.GetGeoTransform(), r.GetProjection()
    r = None
    A = np.full((N, H, W), np.nan, np.float32)
    for i, y in enumerate(YEARS):
        d = gdal.Open(os.path.join(ADIR, "GZ_NAI_%d_mean.tif" % y))
        A[i] = d.GetRasterBand(1).ReadAsArray(); d = None
    VAL = ~np.isnan(A)
    print("输入: %d 个年均栅格, %d x %d" % (N, W, H))

    ds = ogr.Open(SHP); lyr = ds.GetLayer(0)
    feats = [(park_label(f.GetField("name")), f.GetGeometryRef().Clone()) for f in lyr]
    lab = np.zeros((H, W), np.int32)
    dm = gdal.GetDriverByName("MEM"); dsh = ogr.GetDriverByName("Memory")
    srs = osr.SpatialReference(); srs.ImportFromEPSG(4326)
    for i, (nm, g) in enumerate(feats, 1):
        vds = dsh.CreateDataSource("t"); vl = vds.CreateLayer("l", srs, ogr.wkbPolygon)
        ft = ogr.Feature(vl.GetLayerDefn()); ft.SetGeometry(g); vl.CreateFeature(ft)
        m = dm.Create("", W, H, 1, gdal.GDT_Byte); m.SetGeoTransform(GT); m.SetProjection(PRJ)
        gdal.RasterizeLayer(m, [1], vl, burn_values=[1])
        lab[m.GetRasterBand(1).ReadAsArray().astype(bool)] = i
        m = None; vds = None
    PN = [f[0] for f in feats]
    PARK = (lab > 0)
    BAL = PARK & VAL.all(axis=0)          # 平衡面板: 9 年均有效
    print("公园像元 %d, 其中 9 年全有效 %d (%.1f%%)"
          % (PARK.sum(), BAL.sum(), 100.0 * BAL.sum() / PARK.sum()))

    # ------------------------------------------------------------------ 逐像元检验
    rows, cols = np.where(BAL)
    X = A[:, rows, cols].astype(np.float64)        # (9, M)
    M = X.shape[1]
    iu = np.triu_indices(N, 1)
    gap = (iu[1] - iu[0]).astype(np.float64)
    DIFF = X[iu[1], :] - X[iu[0], :]               # (36, M)
    SLOPES = DIFF / gap[:, None]                   # (36, M)
    SEN = np.median(SLOPES, axis=0)

    S = np.sign(DIFF).sum(axis=0).astype(np.float64)
    # 并列(ties)校正
    tie = np.zeros(M)
    for j in range(M):
        _, c = np.unique(X[:, j], return_counts=True)
        if c.max() > 1:
            tie[j] = sum(k * (k - 1) * (2 * k + 5) for k in c if k > 1)
    VAR = (N * (N - 1) * (2 * N + 5) - tie) / 18.0
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = np.where(S > 0, (S - 1) / np.sqrt(VAR),
                     np.where(S < 0, (S + 1) / np.sqrt(VAR), 0.0))
        PV = 2 * (1 - 0.5 * (1 + np.vectorize(math.erf)(np.abs(Z) / math.sqrt(2))))
    Z = np.nan_to_num(Z); PV = np.clip(np.nan_to_num(PV, nan=1.0), 0, 1)

    # Sen 95% CI
    SS = np.sort(SLOPES, axis=0); NN = SS.shape[0]
    C = Z95 * np.sqrt(VAR)
    lo_i = np.clip(np.floor((NN - C) / 2.0).astype(int), 0, NN - 1)
    hi_i = np.clip(np.ceil((NN + C) / 2.0).astype(int), 0, NN - 1)
    SEN_LO = SS[lo_i, np.arange(M)]; SEN_HI = SS[hi_i, np.arange(M)]

    # 显著性
    SIG = PV < 0.05
    CLS = np.where(SIG, np.where(SEN > 0, 1.0, -1.0), 0.0)

    # BH-FDR 校正 (在所有参与检验的像元上)
    def bh(p, q=0.05):
        m = len(p); order = np.argsort(p); ps = p[order]
        thr = q * (np.arange(1, m + 1)) / m
        ok = ps <= thr; out = np.zeros(m, bool)
        if ok.any():
            k = int(np.max(np.where(ok)[0])) + 1
            out[order[:k]] = True
        return out

    FDR = bh(PV, 0.05)
    CLS_FDR = np.where(FDR, np.where(SEN > 0, 1.0, -1.0), 0.0)

    print("逐像元检验完成: M = %d" % M)
    print("  未校正 p<0.05 : %d (%.2f%%), 期望假阳性 %.0f"
          % (SIG.sum(), 100.0 * SIG.sum() / M, 0.05 * M))
    print("  BH-FDR q<0.05 : %d (%.2f%%)" % (FDR.sum(), 100.0 * FDR.sum() / M))

    # ------------------------------------------------------------------ 写栅格
    def put(vals, fill=np.nan, dtype=np.float32):
        a = np.full((H, W), fill, dtype)
        a[rows, cols] = vals
        return a

    def write(arr, path, nodata=None):
        if arr.dtype == np.int16:
            opts = ["COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES",
                    "BLOCKXSIZE=256", "BLOCKYSIZE=256", "BIGTIFF=IF_SAFER"]
            dt = gdal.GDT_Int16
        else:
            opts = ["COMPRESS=DEFLATE", "PREDICTOR=3", "TILED=YES",
                    "BLOCKXSIZE=256", "BLOCKYSIZE=256", "BIGTIFF=IF_SAFER"]
            dt = gdal.GDT_Float32
        o = gdal.GetDriverByName("GTiff").Create(path, W, H, 1, dt, options=opts)
        o.SetGeoTransform(GT); o.SetProjection(PRJ)
        b = o.GetRasterBand(1)
        b.SetNoDataValue(float("nan") if nodata is None else nodata)
        b.WriteArray(arr); o.FlushCache(); o = None

    LAYERS = [
        ("sen_slope", put(SEN.astype(np.float32)), None, "Theil-Sen 斜率 (ions/cm3/年)"),
        ("sen_lo95", put(SEN_LO.astype(np.float32)), None, "Sen 斜率 95% CI 下界"),
        ("sen_hi95", put(SEN_HI.astype(np.float32)), None, "Sen 斜率 95% CI 上界"),
        ("mk_S", put(S.astype(np.float32)), None, "Mann-Kendall S 统计量"),
        ("mk_Z", put(Z.astype(np.float32)), None, "Mann-Kendall Z 统计量"),
        ("mk_p", put(PV.astype(np.float32)), None, "Mann-Kendall 双侧 p 值"),
        ("trend_class", put(CLS.astype(np.int16), fill=np.int16(-99), dtype=np.int16),
         -99, "显著性分类 -1=显著下降 0=不显著 1=显著上升 (未校正)"),
        ("trend_class_fdr", put(CLS_FDR.astype(np.int16), fill=np.int16(-99), dtype=np.int16),
         -99, "BH-FDR(q=0.05) 校正后显著性分类"),
    ]
    for name, arr, nd, desc in LAYERS:
        write(arr, os.path.join(OUT, "all_parks", "GZ_NAI_%s_2016_2024.tif" % name), nd)
    print("all_parks 栅格: %d 个" % len(LAYERS))

    # 分公园
    nb = 0
    for i, (nm, g) in enumerate(feats):
        pm = (lab == i + 1) & BAL
        if not pm.any():
            continue
        rr, cc = np.where(pm)
        r0, r1, c0, c1 = rr.min(), rr.max() + 1, cc.min(), cc.max() + 1
        # Bounding boxes can overlap; restrict output to this park label.
        sel = ((rows >= r0) & (rows < r1) & (cols >= c0) & (cols < c1)
               & (lab[rows, cols] == i + 1))
        for name, arr, nd, desc in LAYERS:
            sub = np.full((r1 - r0, c1 - c0),
                          np.nan if arr.dtype != np.int16 else np.int16(-99),
                          np.float32 if arr.dtype != np.int16 else np.int16)
            sub[rows[sel] - r0, cols[sel] - c0] = arr[rows[sel], cols[sel]]
            p = os.path.join(OUT, "by_park", "%s_%s_2016_2024.tif" % (park_slug(nm), name))
            if sub.dtype == np.int16:
                o = gdal.GetDriverByName("GTiff").Create(
                    p, sub.shape[1], sub.shape[0], 1, gdal.GDT_Int16,
                    options=["COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES"])
                o.SetGeoTransform((GT[0] + c0 * GT[1], GT[1], 0,
                                   GT[3] + r0 * GT[5], 0, GT[5]))
                o.SetProjection(PRJ)
                b = o.GetRasterBand(1); b.SetNoDataValue(-99); b.WriteArray(sub)
                o.FlushCache(); o = None
            else:
                o = gdal.GetDriverByName("GTiff").Create(
                    p, sub.shape[1], sub.shape[0], 1, gdal.GDT_Float32,
                    options=["COMPRESS=DEFLATE", "PREDICTOR=3", "TILED=YES"])
                o.SetGeoTransform((GT[0] + c0 * GT[1], GT[1], 0,
                                   GT[3] + r0 * GT[5], 0, GT[5]))
                o.SetProjection(PRJ)
                b = o.GetRasterBand(1); b.SetNoDataValue(float("nan")); b.WriteArray(sub)
                o.FlushCache(); o = None
            nb += 1
    print("by_park 栅格: %d 个" % nb)

    # ------------------------------------------------------------------ CSV
    def pmask(i):
        return (lab == i + 1)

    # 公园均值序列(9年)的趋势, 作为对照
    park_sen, park_p, park_S = [], [], []
    for i, (nm, g) in enumerate(feats):
        m = pmask(i)
        ser = np.array([A[t][m & ~np.isnan(A[t])].mean() for t in range(N)])
        Xs = ser.astype(np.float64)
        sl = (Xs[iu[1]] - Xs[iu[0]]) / gap
        park_sen.append(float(np.median(sl)))
        Ss = float(np.sign(Xs[iu[1]] - Xs[iu[0]]).sum())
        _, c = np.unique(Xs, return_counts=True)
        tv = sum(k * (k - 1) * (2 * k + 5) for k in c if k > 1)
        Vv = (N * (N - 1) * (2 * N + 5) - tv) / 18.0
        Zv = (Ss - 1) / math.sqrt(Vv) if Ss > 0 else ((Ss + 1) / math.sqrt(Vv) if Ss < 0 else 0)
        park_p.append(float(2 * (1 - 0.5 * (1 + math.erf(abs(Zv) / math.sqrt(2))))))
        park_S.append(int(Ss))

    rows3 = []
    for i, (nm, g) in enumerate(feats):
        sel = (lab[rows, cols] == i + 1)
        m = int(sel.sum())
        if m == 0:
            continue
        s_, p_, f_ = SEN[sel], PV[sel], FDR[sel]
        c_ = CLS[sel]
        rows3.append([
            nm, int(pmask(i).sum()), m,
            round(park_sen[i], 2), round(park_p[i], 4),
            round(float(np.median(s_)), 2), round(float(s_.mean()), 2),
            round(float(np.percentile(s_, 25)), 2), round(float(np.percentile(s_, 75)), 2),
            round(100.0 * float((s_ > 0).mean()), 1),
            int((p_ < 0.05).sum()), round(100.0 * float((p_ < 0.05).mean()), 2),
            round(0.05 * m, 1),
            int(f_.sum()), round(100.0 * float(f_.mean()), 2),
            int((c_ > 0).sum()), int((c_ < 0).sum()),
        ])

    hdr = ["park", "pixel_count", "tested_pixel_count", "park_mean_series_sen_slope", "park_mean_series_mk_p",
           "pixel_sen_median", "pixel_sen_mean", "pixel_sen_P25", "pixel_sen_P75", "positive_slope_fraction_pct",
           "unadjusted_p_lt_0_05_pixel_count", "unadjusted_p_lt_0_05_fraction_pct", "expected_false_positive_count",
           "fdr_significant_pixel_count", "fdr_significant_fraction_pct", "unadjusted_significant_increase_pixel_count", "unadjusted_significant_decrease_pixel_count"]
    with open(os.path.join(OUT, "park_trend_summary.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        c = csv.writer(f); c.writerow(hdr); c.writerows(rows3)

    rows3b = []
    for i, (nm, g) in enumerate(feats):
        sel = (lab[rows, cols] == i + 1)
        if not sel.any():
            continue
        s_ = SEN[sel]
        q = np.percentile(s_, [0, 5, 25, 50, 75, 95, 100])
        rows3b.append([nm, int(sel.sum())] + [round(float(v), 2) for v in q] +
                      [round(float(s_.mean()), 2), round(float(s_.std(ddof=1)), 2)])
    with open(os.path.join(OUT, "park_pixel_slope_distribution.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        c = csv.writer(f)
        c.writerow(["park", "tested_pixel_count", "Min", "P5", "P25", "median", "P75", "P95",
                    "Max", "mean", "standard_deviation"])
        c.writerows(rows3b)

    # 全公园合并汇总
    selall = np.ones(M, bool)
    with open(os.path.join(OUT, "all_parks_trend_summary.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        c = csv.writer(f)
        c.writerow(["metric", "value"])
        c.writerow(["tested_pixel_count_total", M])
        c.writerow(["pixel_sen_slope_median", round(float(np.median(SEN)), 2)])
        c.writerow(["pixel_sen_slope_mean", round(float(SEN.mean()), 2)])
        c.writerow(["pixel_sen_slope_P25", round(float(np.percentile(SEN, 25)), 2)])
        c.writerow(["pixel_sen_slope_P75", round(float(np.percentile(SEN, 75)), 2)])
        c.writerow(["positive_slope_pixel_count", int((SEN > 0).sum())])
        c.writerow(["negative_slope_pixel_count", int((SEN < 0).sum())])
        c.writerow(["unadjusted_p_lt_0_05_pixel_count", int(SIG.sum())])
        c.writerow(["unadjusted_p_lt_0_05_fraction_pct", round(100.0 * SIG.sum() / M, 2)])
        c.writerow(["expected_false_positive_count_at_0_05", round(0.05 * M, 1)])
        c.writerow(["bh_fdr_significant_pixel_count", int(FDR.sum())])
        c.writerow(["unadjusted_significant_increase_pixel_count", int((CLS > 0).sum())])
        c.writerow(["unadjusted_significant_decrease_pixel_count", int((CLS < 0).sum())])
        c.writerow(["sen_ci_excludes_zero_pixel_count", int(((SEN_LO > 0) | (SEN_HI < 0)).sum())])

    print()
    print("=" * 100); print("逐公园趋势结果"); print("=" * 100)
    print("%-20s %8s %12s %10s %12s %10s %10s" %
          ("park", "检验像元", "均值Sen斜率", "均值MK_p", "像元Sen中位", "p<.05像元", "FDR显著"))
    print("-" * 90)
    for r_ in rows3:
        print("%-20s %8d %12.2f %10.4f %12.2f %10d %10d"
              % (r_[0], r_[2], r_[3], r_[4], r_[5], r_[10], r_[13]))
    print("-" * 90)
    print("单位 ions/cm3/年")
    print()
    print("全公园合并: 像元 Sen 斜率中位数 %.2f, 正斜率 %d / 负斜率 %d"
          % (np.median(SEN), (SEN > 0).sum(), (SEN < 0).sum()))
    print("            p<0.05 像元 %d (%.2f%%), 期望假阳性 %.0f, FDR 后 %d"
          % (SIG.sum(), 100.0 * SIG.sum() / M, 0.05 * M, FDR.sum()))
    print("输出目录: %s" % OUT)


if __name__ == "__main__":
    main()
