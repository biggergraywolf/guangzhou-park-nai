# -*- coding: utf-8 -*-
"""
用 9_urban_parks.shp 裁剪 NAI_masked 中全部栅格 -> NAI_urban_parks_clip

要点:
  - 栅格 EPSG:32649 (UTM 49N, 100 m), 裁剪面 EPSG:4326 -> 由 GDAL 自动重投影 cutline
  - cropToCutline: 输出范围收窄到 shp 外包框
  - dstNodata = NaN: 公园外像元置为 NaN, 与源数据 nodata 约定一致
  - 不重采样, 输出网格与源完全一致
"""
import os
import sys
import csv
import glob

from package_paths import PACKAGE_ROOT, PARKS, FROZEN, REPRODUCED, input_path, output_path, preferred_stage, configure_gdal_data
configure_gdal_data()
from osgeo import gdal, ogr, osr
import numpy as np

import argparse


def main():
    ap = argparse.ArgumentParser(description='Clip seasonal outputs to the seven park polygons.')
    ap.add_argument('--in-dir', default=str(preferred_stage('NAI_masked')))
    ap.add_argument('--out-dir', default=str(REPRODUCED / 'NAI_urban_parks_clip'))
    ap.add_argument('--parks', default=str(PARKS))
    ap.add_argument('--only', default=None, help='Comma-separated season filter.')
    args = ap.parse_args()

    gdal.UseExceptions()
    ogr.UseExceptions()

    BASE = str(PACKAGE_ROOT)
    SRC_DIR = str(input_path(args.in_dir))
    SHP = str(input_path(args.parks))
    OUT_DIR = str(output_path(args.out_dir))

    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 78)
    print("1) 检查裁剪面")
    print("=" * 78)

    ds_shp = ogr.Open(SHP)
    lyr = ds_shp.GetLayer(0)
    shp_srs = lyr.GetSpatialRef()
    print("图层      : %s" % lyr.GetName())
    print("要素数    : %d" % lyr.GetFeatureCount())
    print("几何类型  : %s" % ogr.GeometryTypeToName(lyr.GetGeomType()))
    print("坐标系    : EPSG:%s" % shp_srs.GetAuthorityCode(None))

    names = []
    invalid = 0
    for f in lyr:
        g = f.GetGeometryRef()
        nm = f.GetField("name")
        names.append(nm)
        if not g.IsValid():
            invalid += 1
            print("  [无效几何] %s" % nm)
    print("公园列表  : %s" % ", ".join(str(n) for n in names))
    print("无效几何  : %d" % invalid)

    # 若存在无效几何, 生成修复后的临时 shp 再裁剪
    CUTLINE = SHP
    if invalid:
        fixed = os.path.join(OUT_DIR, "_tmp_parks_fixed.shp")
        gdal.VectorTranslate(
            fixed, SHP, options=gdal.VectorTranslateOptions(
                layerName="parks", makeValid=True))
        CUTLINE = fixed
        print("已生成修复后裁剪面: %s" % fixed)
    else:
        print("几何全部有效, 直接使用原 shp")

    # 裁剪面在栅格坐标系下的范围
    lyr.ResetReading()
    src_srs = osr.SpatialReference()
    src_srs.ImportFromEPSG(32649)
    ct = osr.CoordinateTransformation(shp_srs, src_srs)
    xs, ys = [], []
    for f in lyr:
        g = f.GetGeometryRef().Clone()
        g.Transform(ct)
        xmin, xmax, ymin, ymax = g.GetEnvelope()
        xs += [xmin, xmax]
        ys += [ymin, ymax]
    print("裁剪面范围(UTM49N): X %.0f ~ %.0f, Y %.0f ~ %.0f" % (min(xs), max(xs), min(ys), max(ys)))

    print()
    print("=" * 78)
    print("2) 检查源栅格")
    print("=" * 78)

    tifs = sorted(glob.glob(os.path.join(SRC_DIR, "*.tif")))
    tifs = [t for t in tifs if not t.lower().endswith(".ovr")]
    if args.only:
        selected = set(args.only.split(","))
        tifs = [t for t in tifs if os.path.basename(t).replace("GZ_NAI_", "").replace(".tif", "") in selected]
    print("待裁剪栅格: %d 个" % len(tifs))

    ref = None
    problems = []
    for t in tifs:
        ds = gdal.Open(t)
        b = ds.GetRasterBand(1)
        sig = (ds.GetProjection(), ds.GetGeoTransform(), ds.RasterXSize,
               ds.RasterYSize, b.DataType, ds.RasterCount)
        if ref is None:
            ref = sig
            print("参考网格  : %d x %d, %s, %s" % (
                ds.RasterXSize, ds.RasterYSize,
                gdal.GetDataTypeName(b.DataType), os.path.basename(t)))
        elif sig != ref:
            problems.append(os.path.basename(t))
        ds = None
    if problems:
        print("[警告] 与参考网格不一致: %s" % ", ".join(problems))
    else:
        print("全部栅格网格/类型一致: 维度 %d x %d, Float32, nodata=NaN" % (ref[2], ref[3]))

    print()
    print("=" * 78)
    print("3) 开始裁剪")
    print("=" * 78)

    opts = gdal.WarpOptions(
        format="GTiff",
        cutlineDSName=CUTLINE,
        cutlineLayer="parks" if invalid else lyr.GetName(),
        cropToCutline=True,
        dstNodata=float("nan"),
        dstSRS="EPSG:32649",
        resampleAlg=gdal.GRA_NearestNeighbour,
        creationOptions=[
            "COMPRESS=DEFLATE", "PREDICTOR=3", "ZLEVEL=6",
            "TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256",
            "BIGTIFF=IF_SAFER",
        ],
        multithread=True,
        warpMemoryLimit=512,
    )

    rows = []
    ok = 0
    fail = 0
    for i, t in enumerate(tifs, 1):
        base = os.path.splitext(os.path.basename(t))[0]
        out = os.path.join(OUT_DIR, base + ".tif")
        try:
            res = gdal.Warp(out, t, options=opts)
            if res is None:
                raise RuntimeError("Warp 返回 None")
            b = res.GetRasterBand(1)
            arr = b.ReadAsArray()
            valid = int(np.count_nonzero(~np.isnan(arr)))
            tot = arr.size
            if valid > 0:
                vals = arr[~np.isnan(arr)]
                vmin, vmax, vmean = float(vals.min()), float(vals.max()), float(vals.mean())
            else:
                vmin = vmax = vmean = float("nan")
            rows.append([base, res.RasterXSize, res.RasterYSize, valid, tot,
                         round(100.0 * valid / tot, 1) if tot else 0.0,
                         round(vmin, 2), round(vmax, 2), round(vmean, 2)])
            res = None
            ok += 1
            print("[%2d/%d] %-18s -> %4d x %4d, 公园内像元 %7d (%.1f%%), mean %.1f"
                  % (i, len(tifs), base, rows[-1][1], rows[-1][2], valid, rows[-1][5], vmean))
        except Exception as e:
            fail += 1
            print("[%2d/%d] %-18s -> 失败: %s" % (i, len(tifs), base, e))

    print()
    print("=" * 78)
    print("4) 汇总")
    print("=" * 78)
    print("成功: %d   失败: %d" % (ok, fail))

    if rows:
        csv_path = os.path.join(OUT_DIR, "_clip_summary.csv")
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["file", "width", "height", "valid_px", "total_px",
                        "valid_pct", "min", "max", "mean"])
            w.writerows(rows)
        print("汇总表: %s" % csv_path)

        vp = [r[5] for r in rows]
        print("公园内像元占比: %.1f%% ~ %.1f%% (均值 %.1f%%)" % (min(vp), max(vp), sum(vp) / len(vp)))
        mn = [r[8] for r in rows if r[8] == r[8]]
        if mn:
            print("各期公园均值范围: %.1f ~ %.1f ions/cm3" % (min(mn), max(mn)))

    if CUTLINE != SHP and os.path.exists(CUTLINE):
        for ext in (".shp", ".shx", ".dbf", ".prj"):
            p = os.path.splitext(CUTLINE)[0] + ext
            if os.path.exists(p):
                os.remove(p)

    print("输出目录: %s" % OUT_DIR)


if __name__ == "__main__":
    main()
