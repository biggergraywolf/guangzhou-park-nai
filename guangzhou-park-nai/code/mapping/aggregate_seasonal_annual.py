# -*- coding: utf-8 -*-
"""
需求1(季节) + 需求2(年度) 聚合
================================
统一时段: 2016-2024 (9 个完整年, 36 个季度)

需求1  多年不同季节均值对比(CSV) + 多年四季均值栅格
需求2  逐公园逐年均值对比(CSV) + 逐年年均栅格 + 9年总体均值栅格

栅格输出两套:
  all_parks/  7 个公园合并在一个文件(沿用裁剪结果约定)
  by_park/    每个公园单独一个文件, 已裁到该公园包围盒

统计口径:
  栅格 = 逐像元 nanmean (气候态/均值场)
  公园均值 = 该公园掩膜内有效像元均值
  同时给出"核心面板"(在 2016-2024 内 >=90% 期有效的像元)的稳健均值
"""
import os, glob, csv, math, warnings
import numpy as np
from package_paths import PACKAGE_ROOT, PARKS, FROZEN, REPRODUCED, input_path, output_path, preferred_stage, configure_gdal_data, park_label, park_slug, PARK_NAMES
configure_gdal_data()
from osgeo import gdal, ogr, osr

# 公园外像元在全部期均为 NaN, nanmean 会给出 "Mean of empty slice" 警告, 结果仍为 NaN, 属预期
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

import argparse


def write_amplitude_products(season_r, park_names, masks, seasonal_dir, write_tif, bbox):
    """Reproduce the archived Q2-Q4 products from assess_mapping_suitability.py.

    The relative-amplitude column retains the original fixed 2400 ions/cm3
    reference. It is not divided by each park's own annual mean.
    """
    out = os.path.join(seasonal_dir, "derived")
    for folder in ("all_parks", "by_park"):
        os.makedirs(os.path.join(out, folder), exist_ok=True)
    amplitude = (season_r["Q2"] - season_r["Q4"]).astype(np.float32)
    write_tif(amplitude, os.path.join(out, "all_parks", "GZ_NAI_amp_Q2_minus_Q4_2016_2024.tif"))
    rows = []
    # Preserve the original source-filename order while translating its labels.
    for source_name in sorted(PARK_NAMES):
        name = park_label(source_name)
        mask = masks[park_names.index(name)]
        r0, r1, c0, c1 = bbox(mask)
        tile = np.full((r1 - r0, c1 - c0), np.nan, np.float32)
        inside = mask[r0:r1, c0:c1]
        source = amplitude[r0:r1, c0:c1]
        tile[inside] = source[inside]
        write_tif(tile, os.path.join(out, "by_park", park_slug(name) + "_amp_Q2_minus_Q4.tif"),
                  row_offset=r0, col_offset=c0)
        values = tile[~np.isnan(tile)]
        rows.append([name, round(float(values.mean()), 1),
                     round(float(np.percentile(values, 5)), 1),
                     round(float(np.percentile(values, 95)), 1),
                     round(100.0 * float(values.mean()) / 2400, 1)])
    with open(os.path.join(out, "park_seasonal_amplitude.csv"), "w", newline="",
              encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["park", "mean_amplitude", "P5", "P95", "amplitude_relative_to_2400_pct"])
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description='Aggregate the 36 paper seasons into seasonal and annual outputs.')
    ap.add_argument('--in-dir', default=str(preferred_stage('NAI_urban_parks_clip')))
    ap.add_argument('--out-dir', default=str(REPRODUCED / 'NAI_urban_parks_clip'))
    ap.add_argument('--parks', default=str(PARKS))
    args = ap.parse_args()

    gdal.UseExceptions(); ogr.UseExceptions()

    BASE = str(PACKAGE_ROOT)
    CLIP = str(input_path(args.in_dir))
    OUT_ROOT = str(output_path(args.out_dir))
    SHP = str(input_path(args.parks))
    OUT1 = os.path.join(OUT_ROOT, "seasonal_2016_2024")
    OUT2 = os.path.join(OUT_ROOT, "annual_2016_2024")
    YEARS = list(range(2016, 2025))
    SEASONS = ["Q1", "Q2", "Q3", "Q4"]
    SNAME = {"Q1": "Spring (March-May)", "Q2": "Summer (June-August)", "Q3": "Autumn (September-November)", "Q4": "Winter (December-February)"}

    for d in (OUT1, OUT2,
              os.path.join(OUT1, "all_parks"), os.path.join(OUT1, "by_park"),
              os.path.join(OUT2, "all_parks"), os.path.join(OUT2, "by_park")):
        os.makedirs(d, exist_ok=True)

    # ---------------------------------------------------------------- 读取
    periods = ["%d%s" % (y, q) for y in YEARS for q in SEASONS]
    files = {p: os.path.join(CLIP, "GZ_NAI_%s.tif" % p) for p in periods}
    missing = [p for p, f in files.items() if not os.path.exists(f)]
    if missing:
        raise SystemExit("缺少栅格: %s" % missing)

    ref = gdal.Open(files[periods[0]])
    W, H, GT, PRJ = ref.RasterXSize, ref.RasterYSize, ref.GetGeoTransform(), ref.GetProjection()
    ref = None
    print("时段: %s ~ %s  (%d 期)" % (periods[0], periods[-1], len(periods)))
    print("栅格: %d x %d, 100 m, EPSG:32649" % (W, H))

    stack = np.full((len(periods), H, W), np.nan, np.float32)
    for i, p in enumerate(periods):
        d = gdal.Open(files[p]); stack[i] = d.GetRasterBand(1).ReadAsArray(); d = None
    valid = ~np.isnan(stack)
    print("载入完成, 总体有效像元率 %.1f%%" % (100.0 * valid.mean()))

    # ---------------------------------------------------------------- 公园掩膜
    ds = ogr.Open(SHP); lyr = ds.GetLayer(0)
    feats = [(park_label(f.GetField("name")), f.GetGeometryRef().Clone()) for f in lyr]
    lab = np.zeros((H, W), np.int32)
    dm = gdal.GetDriverByName("MEM"); dsh = ogr.GetDriverByName("Memory")
    srs = osr.SpatialReference(); srs.ImportFromEPSG(4326)
    for i, (nm, geom) in enumerate(feats, 1):
        vds = dsh.CreateDataSource("t"); vl = vds.CreateLayer("l", srs, ogr.wkbPolygon)
        ft = ogr.Feature(vl.GetLayerDefn()); ft.SetGeometry(geom); vl.CreateFeature(ft)
        m = dm.Create("", W, H, 1, gdal.GDT_Byte); m.SetGeoTransform(GT); m.SetProjection(PRJ)
        gdal.RasterizeLayer(m, [1], vl, burn_values=[1])
        lab[m.GetRasterBand(1).ReadAsArray().astype(bool)] = i
        m = None; vds = None
    PN = [f[0] for f in feats]
    masks = [(lab == i + 1) for i in range(len(feats))]
    print("公园: %d 个" % len(PN))

    # 核心面板掩膜 (在 36 期内 >=90% 有效)
    cov = valid.mean(axis=0)
    core = cov >= 0.90

    def bbox(m):
        r = np.where(m.any(axis=1))[0]; c = np.where(m.any(axis=0))[0]
        return r.min(), r.max() + 1, c.min(), c.max() + 1

    def write_tif(arr, path, row_offset=0, col_offset=0):
        drv = gdal.GetDriverByName("GTiff")
        o = drv.Create(path, arr.shape[1], arr.shape[0], 1, gdal.GDT_Float32,
                       options=["COMPRESS=DEFLATE", "PREDICTOR=3", "TILED=YES",
                                "BLOCKXSIZE=256", "BLOCKYSIZE=256", "BIGTIFF=IF_SAFER"])
        # A cropped park tile must inherit its own window origin.
        o.SetGeoTransform((GT[0] + col_offset * GT[1] + row_offset * GT[2],
                           GT[1], GT[2],
                           GT[3] + col_offset * GT[4] + row_offset * GT[5],
                           GT[4], GT[5])); o.SetProjection(PRJ)
        b = o.GetRasterBand(1); b.SetNoDataValue(float("nan")); b.WriteArray(arr)
        o.FlushCache(); o = None

    with np.errstate(invalid="ignore"):
        # ============================================================ 需求1
        print()
        print("=" * 80); print("需求1: 季节均值 (2016-2024, 每季 9 年)"); print("=" * 80)
        season_r = {}
        for q in SEASONS:
            idx = [i for i, p in enumerate(periods) if p.endswith(q)]
            r = np.nanmean(stack[idx], axis=0).astype(np.float32)
            season_r[q] = r
            np_ = int(np.count_nonzero(~np.isnan(r) & (lab > 0)))
            write_tif(r, os.path.join(OUT1, "all_parks", "GZ_NAI_%s_mean_2016_2024.tif" % q))
            print("  %s  %s  -> all_parks/GZ_NAI_%s_mean_2016_2024.tif  (公园内像元 %d)"
                  % (q, SNAME[q], q, np_))

        for i, nm in enumerate(PN):
            m = masks[i]; r0, r1, c0, c1 = bbox(m)
            for q in SEASONS:
                a = np.full((r1 - r0, c1 - c0), np.nan, np.float32)
                sub = season_r[q][r0:r1, c0:c1]; sm = m[r0:r1, c0:c1]
                a[sm] = sub[sm]
                write_tif(a, os.path.join(OUT1, "by_park",
                                          "%s_%s_mean_2016_2024.tif" % (park_slug(nm), q)), row_offset=r0, col_offset=c0)
        print("  分公园栅格: %d 个 (7 公园 x 4 季)" % (len(PN) * 4))

        # ============================================================ 需求2
        print()
        print("=" * 80); print("需求2: 年均值 (2016-2024)"); print("=" * 80)
        year_r = {}
        for y in YEARS:
            idx = [i for i, p in enumerate(periods) if p.startswith(str(y))]
            r = np.nanmean(stack[idx], axis=0).astype(np.float32)
            year_r[y] = r
            write_tif(r, os.path.join(OUT2, "all_parks", "GZ_NAI_%d_mean.tif" % y))
        allr = np.nanmean(stack, axis=0).astype(np.float32)
        write_tif(allr, os.path.join(OUT2, "all_parks", "GZ_NAI_2016_2024_mean.tif"))
        print("  逐年栅格: %d 个 (2016-2024)" % len(YEARS))
        print("  总体均值栅格: all_parks/GZ_NAI_2016_2024_mean.tif")

        for i, nm in enumerate(PN):
            m = masks[i]; r0, r1, c0, c1 = bbox(m)
            sm = m[r0:r1, c0:c1]
            for y in YEARS:
                a = np.full((r1 - r0, c1 - c0), np.nan, np.float32)
                sub = year_r[y][r0:r1, c0:c1]; a[sm] = sub[sm]
                write_tif(a, os.path.join(OUT2, "by_park", "%s_%d_mean.tif" % (park_slug(nm), y)), row_offset=r0, col_offset=c0)
            a = np.full((r1 - r0, c1 - c0), np.nan, np.float32)
            sub = allr[r0:r1, c0:c1]; a[sm] = sub[sm]
            write_tif(a, os.path.join(OUT2, "by_park", "%s_2016_2024_mean.tif" % park_slug(nm)), row_offset=r0, col_offset=c0)
        print("  分公园栅格: %d 个 (7 公园 x (9 年 + 总体))" % (len(PN) * (len(YEARS) + 1)))

    # ================================================================ CSV
    def pmean(raster, m):
        v = m & ~np.isnan(raster)
        return float(raster[v].mean()) if v.any() else float("nan"), int(v.sum())

    def pmean_core(raster, m):
        v = m & core & ~np.isnan(raster)
        return float(raster[v].mean()) if v.any() else float("nan"), int(v.sum())

    def w(path, header, rows):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            c = csv.writer(f); c.writerow(header); c.writerows(rows)

    # --- 表1: 逐公园四季均值
    rows = []
    for i, nm in enumerate(PN):
        m = masks[i]
        vals, cores = [], []
        for q in SEASONS:
            v, _ = pmean(season_r[q], m); vals.append(v)
            cv, _ = pmean_core(season_r[q], m); cores.append(cv)
        amp = max(vals) - min(vals)
        ratio = max(vals) / min(vals)
        rows.append([nm, int(m.sum()),
                     round(vals[0], 2), round(vals[1], 2), round(vals[2], 2), round(vals[3], 2),
                     round(amp, 1), round(ratio, 3),
                     round(cores[0], 2), round(cores[1], 2), round(cores[2], 2), round(cores[3], 2)])
    w(os.path.join(OUT_ROOT, "park_seasonal_means.csv"),
      ["park", "pixel_count", "spring_Q1_mean", "summer_Q2_mean", "autumn_Q3_mean", "winter_Q4_mean",
       "seasonal_amplitude", "peak_to_trough_ratio", "Q1_core_panel_mean", "Q2_core_panel_mean", "Q3_core_panel_mean", "Q4_core_panel_mean"],
      rows)

    # --- 表1b: 逐公园 逐年x逐季 明细
    rows = []
    for i, nm in enumerate(PN):
        m = masks[i]
        for y in YEARS:
            r = []
            for q in SEASONS:
                k = periods.index("%d%s" % (y, q))
                v = m & valid[k]
                r.append(round(float(stack[k][v].mean()), 2) if v.any() else "")
            rows.append([nm, y] + r)
    w(os.path.join(OUT_ROOT, "park_seasonal_means_by_year.csv"),
      ["park", "seasonal_year", "spring_Q1_mean", "summer_Q2_mean", "autumn_Q3_mean", "winter_Q4_mean"], rows)

    # --- 表2: 逐公园逐年均值
    rows = []
    for i, nm in enumerate(PN):
        m = masks[i]
        vals = [pmean(year_r[y], m)[0] for y in YEARS]
        mv, nv = pmean(allr, m)
        rows.append([nm, int(m.sum())] + [round(v, 2) for v in vals] +
                    [round(mv, 2), round(float(np.nanstd(vals)), 2),
                     round(min(vals), 2), round(max(vals), 2),
                     round(max(vals) - min(vals), 2)])
    w(os.path.join(OUT_ROOT, "park_annual_means.csv"),
      ["park", "pixel_count"] + [str(y) for y in YEARS] +
      ["multiyear_mean", "standard_deviation", "minimum", "maximum", "range"], rows)

    # --- 表2b: 年度稳健性(核心面板) 与有效率
    rows = []
    for i, nm in enumerate(PN):
        m = masks[i]
        cv = [pmean_core(year_r[y], m)[0] for y in YEARS]
        mv, nv = pmean_core(allr, m)
        ncore = int((m & core).sum())
        rows.append([nm, int(m.sum()), ncore, round(100.0 * ncore / m.sum(), 1)] +
                    [round(v, 2) for v in cv] + [round(mv, 2)])
    w(os.path.join(OUT_ROOT, "park_annual_means_core_panel.csv"),
      ["park", "pixel_count", "core_pixel_count", "core_fraction_pct"] + [str(y) for y in YEARS] + ["multiyear_mean"], rows)

    print()
    print("=" * 80); print("CSV 已写出"); print("=" * 80)
    for f in ["park_seasonal_means.csv", "park_seasonal_means_by_year.csv",
              "park_annual_means.csv", "park_annual_means_core_panel.csv"]:
        print("  %s" % os.path.join(OUT_ROOT, f))
    print()
    print("季节均值 (ions/cm3):")
    print("%-22s %9s %9s %9s %9s %9s" % ("park", "春Q1", "夏Q2", "秋Q3", "冬Q4", "peak_to_trough_ratio"))
    print("-" * 74)
    for r in rows:
        pass
    for i, nm in enumerate(PN):
        m = masks[i]
        vs = [pmean(season_r[q], m)[0] for q in SEASONS]
        print("%-22s %9.1f %9.1f %9.1f %9.1f %9.3f"
              % (nm, vs[0], vs[1], vs[2], vs[3], max(vs) / min(vs)))

    write_amplitude_products(season_r, PN, masks, OUT1, write_tif, bbox)


if __name__ == "__main__":
    main()
