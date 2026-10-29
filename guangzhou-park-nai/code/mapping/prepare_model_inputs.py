# -*- coding: utf-8 -*-
"""
prepare_model_inputs.py
=======================
把 GEE 导出的栅格转换成 M4 城市公园模型可以直接读取的输入。

为什么需要
----------
GEE 脚本 (GZ_NAI_raster_01_quarterly_features.js) 导出的是【多波段、多分辨率】文件：

    GZ_NAI_FINE_<年Q季>_<season>.tif    4 波段 @100 m
        [1] hls_model_evi2                     ← 模型输入
        [2] hls_model_ndmi                     ← 模型输入
        [3] hls_obs_count                      ← QA，丢弃
        [4] hls_source                         ← QA，丢弃
    GZ_NAI_GPM_<年Q季>_<season>.tif     3 波段 @11132 m
        [1] gpm_precip_uncal_d0_mm             ← 模型输入
        [2] gpm_precip_uncal_prev7d_mm         ← 模型输入
        [3] gpm_halfhour_obs                   ← QA，丢弃
    GZ_NAI_MASK_static.tif             4 波段 @100 m（仅掩膜，不进模型）

而模型 (apply_m4_urban_raster_v2.py) 要求：**每个特征一个单波段 tif，且 6 个文件完全同网格**：

    <season_dir>/doy_sin.tif
    <season_dir>/doy_cos.tif
    <season_dir>/hls_model_evi2.tif
    <season_dir>/hls_model_ndmi.tif
    <season_dir>/gpm_precip_uncal_d0_mm.tif
    <season_dir>/gpm_precip_uncal_prev7d_mm.tif

本脚本完成 3 件事：
    ① 按【波段名】拆分多波段文件为单波段 tif（按名而非按序号，避免波段顺序变化出错）
    ② 把 GPM 用【最近邻】重采样到 FINE 的 100 m 网格（降水是面状平均量，
       用双线性会造出虚假空间梯度，模型会当成真实信号）
    ③ 可选：用城市公园边界矢量裁剪，大幅减小体积并去除无关像元
并输出 seasons.csv（manifest），可直接传给 apply_m4_urban_raster_v2.py。

用法
----
# 最基本：自动扫描 gee-dir 下所有季度对，转换到 out-dir
python prepare_model_inputs.py --gee-dir ./GEE_GZ_NAI_raster --out-dir ./model_inputs

# 用公园边界裁剪（强烈建议）
python prepare_model_inputs.py --gee-dir ./GEE_GZ_NAI_raster --out-dir ./model_inputs \
    --parks ./gz_urban_parks.shp --park-buffer 100

# 同时生成 doy_sin/doy_cos（调用 prepare_model_inputs.py 的逻辑）
python prepare_model_inputs.py --gee-dir ./GEE_GZ_NAI_raster --out-dir ./model_inputs \
    --parks ./gz_urban_parks.shp --with-doy

# 只转换指定季度
python prepare_model_inputs.py --gee-dir ./GEE_GZ_NAI_raster --out-dir ./model_inputs \
    --only 2024Q2,2024Q3
"""
from __future__ import print_function

import argparse
from package_paths import input_path, output_path, DEFAULT_MODEL, REPRODUCED
import calendar
import datetime as dt
import glob
import io
import json
import os
import re
import sys

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.mask import mask as rio_mask
from rasterio.warp import reproject, calculate_default_transform

# 模型需要的 6 个特征（与 m4_urban_model.joblib 的 anom_feats 一致）
MODEL_FEATS = ['doy_sin', 'doy_cos', 'hls_model_evi2', 'hls_model_ndmi',
               'gpm_precip_uncal_d0_mm', 'gpm_precip_uncal_prev7d_mm']
# GEE 导出文件里需要拆出来的波段（特征名 -> 来源）
FROM_FINE = {'hls_model_evi2': None, 'hls_model_ndmi': None}
FROM_GPM = {'gpm_precip_uncal_d0_mm': None, 'gpm_precip_uncal_prev7d_mm': None}
QA_BANDS = ['hls_obs_count', 'hls_source', 'gpm_halfhour_obs']

SEASON_MONTHS = {1: (3, 4, 5), 2: (6, 7, 8), 3: (9, 10, 11), 4: (12, 1, 2)}
SEASON_EN = {1: 'spring', 2: 'summer', 3: 'autumn', 4: 'winter'}


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def band_names(path):
    with rasterio.open(path) as src:
        return [d or '' for d in src.descriptions]


def find_band(src, want):
    """按波段名找索引（1-based）；找不到则返回 None。"""
    for i, d in enumerate(src.descriptions or [], 1):
        if d and d.strip() == want:
            return i
    return None


def resolve_by_band(path, wanted, fallback_idx=None, label=''):
    """从一个多波段 tif 中按名取出所有需要的波段，返回 {name: array}, prof。"""
    out = {}
    with rasterio.open(path) as src:
        desc = [d or '' for d in src.descriptions]
        prof = dict(src.profile)
        nodata = src.nodata
        for j, name in enumerate(wanted):
            idx = find_band(src, name)
            tag = 'by name'
            if idx is None:
                if fallback_idx is not None:
                    idx = fallback_idx[j]
                    tag = 'by index %d (fallback)' % idx
                else:
                    print('    [WARN] %s: band "%s" not found, descriptions=%s'
                          % (os.path.basename(path), name, desc))
                    continue
            arr = src.read(idx).astype('float64')
            if nodata is not None and np.isfinite(nodata):
                arr = np.where(arr == nodata, np.nan, arr)
            out[name] = arr
            print('    %-30s <- %s (%s)' % (name, os.path.basename(path), tag))
    return out, prof


def write_single(path, arr, prof, dtype='float32'):
    p = dict(prof)
    p.update(count=1, dtype=dtype, nodata=np.nan, compress='deflate')
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with rasterio.open(path, 'w', **p) as dst:
        dst.write(arr.astype(dtype), 1)


def resample_to_match(arr, src_prof, ref_prof, resampling=Resampling.nearest):
    """把 arr 从 src_prof 网格最近邻重采样到 ref_prof 网格。"""
    dst = np.full((ref_prof['height'], ref_prof['width']), np.nan, dtype='float64')
    reproject(
        source=arr, destination=dst,
        src_transform=src_prof['transform'], src_crs=src_prof['crs'],
        dst_transform=ref_prof['transform'], dst_crs=ref_prof['crs'],
        src_nodata=np.nan, dst_nodata=np.nan,
        resampling=resampling)
    return dst


def same_grid(prof_a, prof_b, tol=1e-9):
    if (prof_a['height'], prof_a['width']) != (prof_b['height'], prof_b['width']):
        return False, 'shape'
    if str(prof_a['crs']) != str(prof_b['crs']):
        return False, 'crs'
    ta, tb = prof_a['transform'], prof_b['transform']
    for k in range(6):
        if abs(ta[k] - tb[k]) > tol * max(1.0, abs(ta[k])):
            return False, 'transform'
    return True, ''


# ---------------------------------------------------------------------------
# 季节项（与 prepare_model_inputs.py 保持一致：季度窗口逐日均值）
# ---------------------------------------------------------------------------
def season_doy_values(year, q, mode='daily_mean'):
    months = SEASON_MONTHS[q]
    dates = []
    for m in months:
        y = year if (q != 4 or m == 12) else year + 1
        nd = calendar.monthrange(y, m)[1]
        for dd in range(1, nd + 1):
            dates.append(dt.date(y, m, dd))
    if mode == 'midpoint':
        mid = dates[len(dates) // 2]
        doy = mid.timetuple().tm_yday
        return (float(np.sin(2 * np.pi * doy / 365.0)),
                float(np.cos(2 * np.pi * doy / 365.0)))
    doys = np.array([x.timetuple().tm_yday for x in dates], float)
    return (float(np.sin(2 * np.pi * doys / 365.0).mean()),
            float(np.cos(2 * np.pi * doys / 365.0).mean()))


def season_ndays(year, q):
    """该季度窗口的天数（Q4 跨年）。"""
    n = 0
    for m in SEASON_MONTHS[q]:
        y = year if (q != 4 or m == 12) else year + 1
        n += calendar.monthrange(y, m)[1]
    return n


# ---------------------------------------------------------------------------
# 降水尺度换算  【关键修正】
# ---------------------------------------------------------------------------
# 模型训练时 gpm_precip_uncal_d0_mm 是【日累计降水】(mm/d)：
#     训练分布：中位 0.23、p75 3.78、p99 49.5、max 88.98 mm/d
# 而 GEE 脚本在季度尺度导出的是【季度累计降水】(mm/季，实测 600—1000 mm)。
# 若直接送入 → 100% 像元超出训练域（实测确认），降水项被极端外推。
#
# --precip-mode:
#   daily_equiv (默认) : 季度累计 / 季度天数 → 日等效降水，落在训练域内
#   raw                : 直接用季度累计（100% 超域，仅作对照）
#   clamp              : 裁到训练上界 88.975（保留"雨季=上界"的单调信号）
def season_scale_precip(arr, ndays, mode):
    if mode == 'raw':
        return arr
    if mode == 'daily_equiv':
        return arr / float(ndays)
    if mode == 'clamp':
        return np.minimum(arr, 88.975)
    raise ValueError('unknown --precip-mode: %s' % mode)


def scale_precip_season(arr, ndays, mode, name):
    """d0 与 prev7d 分别处理。

    d0      : 季度累计 mm -> 日等效 mm/d（除以天数）
    prev7d  : 季度末前 7 日累计 mm —— 训练时本身就是"7 日累计"，
              量纲一致（训练中位 14.9、max 266.6 mm），**不需要换算**。
    """
    if name == 'gpm_precip_uncal_prev7d_mm':
        return arr
    return season_scale_precip(arr, ndays, mode)


# ---------------------------------------------------------------------------
# 主转换
# ---------------------------------------------------------------------------
def discover_seasons(gee_dir):
    """扫描 GEE 导出目录，找出所有 (season_tag, fine_path, gpm_path)。"""
    fine = {}
    gpm = {}
    for p in glob.glob(os.path.join(gee_dir, '**', '*.tif'), recursive=True):
        b = os.path.basename(p)
        m = re.search(r'FINE_(\d{4})Q(\d)_(\w+)', b)
        if m:
            fine['%sQ%s' % (m.group(1), m.group(2))] = p
            continue
        m = re.search(r'GPM_(\d{4})Q(\d)_(\w+)', b)
        if m:
            gpm['%sQ%s' % (m.group(1), m.group(2))] = p
    tags = sorted(set(fine) & set(gpm))
    only_f = sorted(set(fine) - set(gpm))
    only_g = sorted(set(gpm) - set(fine))
    return [(t, fine[t], gpm[t]) for t in tags], only_f, only_g


def load_parks(shp_path, buffer_m):
    """读取公园边界，返回 GeoJSON-like geometry 列表（EPSG:4326）。"""
    try:
        import geopandas as gpd
    except Exception as e:
        print('  [WARN] 需要 geopandas 才能用 --parks：%s' % e)
        return None
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None:
        print('  [WARN] 公园矢量无 CRS，假定为 EPSG:4326')
        gdf = gdf.set_crs('EPSG:4326')
    geoms = list(gdf.geometry)
    if buffer_m and buffer_m > 0:
        # 先投影到米制再缓冲
        gdf_m = gdf.to_crs(gdf.estimate_utm_crs())
        geoms_m = [g.buffer(buffer_m) for g in gdf_m.geometry]
        gdf_b = gpd.GeoDataFrame(geometry=geoms_m, crs=gdf_m.crs).to_crs('EPSG:4326')
        geoms = list(gdf_b.geometry)
    print('  公园矢量: %d 个要素，缓冲 %s m，CRS=%s'
          % (len(geoms), buffer_m, gdf.crs))
    return geoms


def process_season(tag, fine_path, gpm_path, out_dir, ref_prof=None,
                   parks_geoms=None, with_doy=True, doy_mode='daily_mean',
                   keep_qa=False, resampling=Resampling.nearest,
                   precip_mode='daily_equiv'):
    year = int(tag[:4]); q = int(tag[-1])
    sdir = os.path.join(out_dir, tag)
    os.makedirs(sdir, exist_ok=True)
    report = {'season': tag, 'fine': fine_path, 'gpm': gpm_path, 'steps': []}

    print('  [1/4] 从 FINE 拆分波段')
    fine_data, fine_prof = resolve_by_band(
        fine_path, list(FROM_FINE.keys()), fallback_idx=[1, 2], label='FINE')

    print('  [2/4] 从 GPM 拆分波段')
    gpm_data, gpm_prof = resolve_by_band(
        gpm_path, list(FROM_GPM.keys()), fallback_idx=[1, 2], label='GPM')

    # 参考网格
    if ref_prof is None:
        ref_prof = fine_prof
        report['ref_grid'] = 'first FINE (%dx%d, %s)'
        report['steps'].append('参考网格取自本季度 FINE')
    ok, why = same_grid(fine_prof, ref_prof)
    if not ok:
        print('    [INFO] FINE 与本季度参考网格不一致(%s)，重采样到参考网格' % why)
        for k in list(fine_data):
            fine_data[k] = resample_to_match(fine_data[k], fine_prof, ref_prof,
                                             resampling)
        report['steps'].append('FINE 重采样到参考网格 (%s)' % why)

    # GPM 重采样（几乎一定需要：11 km -> 100 m）
    ok_g, why_g = same_grid(gpm_prof, ref_prof)
    if not ok_g:
        print('    GPM(%s m) -> 参考网格：最近邻重采样（%s）'
              % (int(abs(gpm_prof['transform'][0])),
                 '网格不一致: %s' % why_g))
        for k in list(gpm_data):
            gpm_data[k] = resample_to_match(gpm_data[k], gpm_prof, ref_prof,
                                            Resampling.nearest)
        report['steps'].append('GPM 最近邻重采样到参考网格（%s）' % why_g)
    else:
        print('    GPM 已在同一网格，无需重采样')
        report['steps'].append('GPM 无需重采样')

    print('  [3/4] 写单波段 tif')
    written = []
    for name in ['hls_model_evi2', 'hls_model_ndmi']:
        if name in fine_data:
            write_single(os.path.join(sdir, name + '.tif'), fine_data[name], ref_prof)
            written.append(name)
    ndays = season_ndays(year, q)
    for name in ['gpm_precip_uncal_d0_mm', 'gpm_precip_uncal_prev7d_mm']:
        if name in gpm_data:
            a = scale_precip_season(gpm_data[name], ndays, precip_mode, name)
            write_single(os.path.join(sdir, name + '.tif'), a, ref_prof)
            written.append(name)
    print('    降水尺度换算: mode=%s, 季度天数=%d' % (precip_mode, ndays))
    report['precip_mode'] = precip_mode
    report['season_ndays'] = ndays

    if with_doy:
        s_sin, s_cos = season_doy_values(year, q, doy_mode)
        h, w = ref_prof['height'], ref_prof['width']
        # 季节项继承 EVI2 的有效像元掩膜，保证与遥感特征空间范围一致
        base = fine_data.get('hls_model_evi2')
        msk = np.isfinite(base) if base is not None else np.ones((h, w), bool)
        for nm, val in [('doy_sin', s_sin), ('doy_cos', s_cos)]:
            a = np.full((h, w), val, dtype='float64')
            a[~msk] = np.nan
            write_single(os.path.join(sdir, nm + '.tif'), a, ref_prof)
            written.append(nm)
        print('    doy_sin=%.4f  doy_cos=%.4f  (mode=%s)' % (s_sin, s_cos, doy_mode))
        report['steps'].append('生成季节项 mode=%s' % doy_mode)

    if keep_qa:
        # QA 波段单独存，供质量分级
        for name in QA_BANDS:
            src_data = fine_data if name in fine_data else gpm_data
            if name in src_data:
                write_single(os.path.join(sdir, '_qa_' + name + '.tif'),
                             src_data[name], ref_prof)

    # 报告被丢弃的 QA
    dropped = [b for b in QA_BANDS]
    report['dropped_qa_bands'] = dropped
    print('    已写出 %d 个特征：%s' % (len(written), ', '.join(written)))
    print('    丢弃 QA 波段：%s' % ', '.join(dropped))

    print('  [4/4] 校验 6 个文件同网格')
    profs = {}
    for f in MODEL_FEATS:
        p = os.path.join(sdir, f + '.tif')
        if not os.path.exists(p):
            print('    [ERROR] 缺失 %s' % p)
            continue
        with rasterio.open(p) as src:
            profs[f] = dict(src.profile)
    if len(profs) == len(MODEL_FEATS):
        bad = []
        ref = profs[MODEL_FEATS[0]]
        for f in MODEL_FEATS[1:]:
            okk, whyy = same_grid(profs[f], ref)
            if not okk:
                bad.append('%s(%s)' % (f, whyy))
        if bad:
            print('    [ERROR] 网格不一致：%s' % ', '.join(bad))
            report['grid_check'] = 'FAIL: ' + ','.join(bad)
        else:
            print('    [OK] 6 个文件 CRS/transform/shape 完全一致 (%dx%d)'
                  % (ref['height'], ref['width']))
            report['grid_check'] = 'OK'
    else:
        print('    [ERROR] 只有 %d/%d 个特征文件'
              % (len(profs), len(MODEL_FEATS)))
        report['grid_check'] = 'MISSING'

    # 裁剪
    if parks_geoms is not None:
        print('    裁剪到公园边界（invert=False）')
        n_before = None
        for f in MODEL_FEATS:
            p = os.path.join(sdir, f + '.tif')
            if not os.path.exists(p):
                continue
            try:
                with rasterio.open(p) as src:
                    if n_before is None:
                        n_before = src.height * src.width
                    out_img, out_tr = rio_mask(src, parks_geoms, crop=True,
                                               nodata=np.nan, filled=True)
                pr = dict(src.profile)
                pr.update(height=out_img.shape[1], width=out_img.shape[2],
                          transform=out_tr, nodata=np.nan, compress='deflate')
                with rasterio.open(p, 'w', **pr) as dst:
                    dst.write(out_img.astype('float32'))
            except Exception as e:
                print('    [WARN] 裁剪 %s 失败：%s' % (f, e))
        report['steps'].append('已用公园边界裁剪')

    return report


def write_manifest(out_dir, seasons):
    p = os.path.join(out_dir, 'seasons.csv')
    with io.open(p, 'w', encoding='utf-8') as f:
        f.write('season,feat_dir,out_name\n')
        for t in seasons:
            f.write('%s,%s,GZ_NAI_%s.tif\n'
                    % (t, os.path.relpath(os.path.join(out_dir, t), out_dir).replace('\\', '/'), t))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gee-dir', required=True, help='GEE 导出的 tif 所在目录')
    ap.add_argument('--out-dir', required=True, help='输出目录（每季度一个子目录）')
    ap.add_argument('--parks', default=None, help='城市公园边界矢量（shp/geojson）')
    ap.add_argument('--park-buffer', type=float, default=0.0, help='边界缓冲（米）')
    ap.add_argument('--with-doy', action='store_true', default=True,
                    help='同时生成 doy_sin/doy_cos（默认开启）')
    ap.add_argument('--no-doy', dest='with_doy', action='store_false')
    ap.add_argument('--doy-mode', default='daily_mean',
                    choices=['daily_mean', 'midpoint'])
    ap.add_argument('--keep-qa', action='store_true', help='额外保存 QA 波段')
    ap.add_argument('--precip-mode', default='daily_equiv',
                    choices=['daily_equiv', 'raw', 'clamp'],
                    help='降水尺度换算：daily_equiv=季度累计/季度天数（默认，落在训练域内）；'
                         'raw=直接用季度累计（100%% 超域，仅对照）；'
                         'clamp=裁到训练上界 88.975')
    ap.add_argument('--only', default=None, help='只处理这些季度，逗号分隔，如 2024Q2,2024Q3')
    ap.add_argument('--resampling', default='nearest',
                    choices=['nearest', 'bilinear', 'cubic'],
                    help='重采样方法；降水务必用 nearest')
    args = ap.parse_args()

    # Package-only output protection; input resolution is relative to package root.
    args.gee_dir = str(input_path(args.gee_dir))
    args.out_dir = str(output_path(args.out_dir))
    if args.parks:
        args.parks = str(input_path(args.parks))

    rs = {'nearest': Resampling.nearest, 'bilinear': Resampling.bilinear,
          'cubic': Resampling.cubic}[args.resampling]

    os.makedirs(args.out_dir, exist_ok=True)
    print('=' * 84)
    print('GEE 导出 -> 模型输入  转换器')
    print('=' * 84)
    print('GEE 目录 :', args.gee_dir)
    print('输出目录 :', args.out_dir)
    print('')

    pairs, only_f, only_g = discover_seasons(args.gee_dir)
    if only_f:
        print('[WARN] 只有 FINE 没有 GPM 的季度（将被跳过）：%s' % ', '.join(only_f))
    if only_g:
        print('[WARN] 只有 GPM 没有 FINE 的季度（将被跳过）：%s' % ', '.join(only_g))
    if args.only:
        want = set(x.strip() for x in args.only.split(','))
        pairs = [(t, f, g) for t, f, g in pairs if t in want]
    print('待处理季度：%d 个  %s' % (len(pairs), ', '.join(t for t, _, _ in pairs)))
    print('')

    parks = load_parks(args.parks, args.park_buffer) if args.parks else None
    print('')

    reports = []
    for tag, fp, gp in pairs:
        print('-' * 84)
        print('处理 %s' % tag)
        print('-' * 84)
        rep = process_season(tag, fp, gp, args.out_dir, ref_prof=None,
                            parks_geoms=parks, with_doy=args.with_doy,
                            doy_mode=args.doy_mode, keep_qa=args.keep_qa,
                            resampling=rs, precip_mode=args.precip_mode)
        reports.append(rep)
        print('')

    mani = write_manifest(args.out_dir, [t for t, _, _ in pairs])
    print('=' * 84)
    print('完成：%d 个季度已转换' % len(pairs))
    print('manifest ->', mani)
    print('')
    print('下一步（批量推理）：')
    print('  python apply_m4_urban_raster_v2.py --manifest %s --out-dir <NAI输出目录>' % mani)
    print('=' * 84)

    with io.open(os.path.join(args.out_dir, '_prepare_report.json'), 'w',
                 encoding='utf-8') as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)
    print('转换报告 ->', os.path.join(args.out_dir, '_prepare_report.json'))


if __name__ == '__main__':
    main()
