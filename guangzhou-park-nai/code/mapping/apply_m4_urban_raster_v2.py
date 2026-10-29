# -*- coding: utf-8 -*-
"""
apply_m4_urban_raster_v2.py
===========================
M4 城市公园模型 → 季度尺度 NAI 浓度栅格的**修正版**推理脚本。

相对 apply_m4_urban_raster.py 修正了 4 个问题
---------------------------------------------
[FIX-1] 季节项口径：原脚本把 doy_sin/doy_cos 取季度中点单值。模型是在日样本上
        训练的，季度尺度应取**季度窗口内逐日均值**，否则系统性放大季节振幅。
        本脚本提供 --season-mode {daily_mean,midpoint}，默认 daily_mean。
        并支持直接读取 prepare_model_inputs.py 生成的 DOY tif（更省事）。

[FIX-2] 逐像元适用域掩膜：原脚本只打印"多少像元超范围"的警告，不落盘。
        本脚本输出 *适用域掩膜* 栅格与超范围计数栅格，可直接用于论文的
        applicability domain 图。

[FIX-3] 输入网格不一致时的行为：原脚本在网格不匹配时只有 transform 与 shape
        比较，未检查 CRS，且直接 sys.exit，批量处理时中断整条流水线。
        本脚本统一检查 CRS/transform/shape，并给出可读错误。

[FIX-4] 批量与元数据：原脚本一次只能处理一个月，且不写任何过程元数据。
        本脚本支持 --manifest（CSV 列出 各季度 × 特征目录 × 输出），
        并为每个输出写同名 .json 记录输入栅格、模型版本、掩膜统计。

输入（每个季度一个目录）
------------------------
    doy_sin.tif / doy_cos.tif                     可由 prepare_model_inputs.py 生成
    hls_model_evi2.tif
    hls_model_ndmi.tif
    gpm_precip_uncal_d0_mm.tif
    gpm_precip_uncal_prev7d_mm.tif

输出
----
    <out>.tif         NAI 浓度（ions/cm3, float32, 与输入同网格）
    <out>_mask.tif    0=无效 1=域内 2=超训练域（按超范围特征数分级）
    <out>.json        运行元数据

用法
----
# 单季度
python apply_m4_urban_raster_v2.py --season 2024Q2 --feat-dir ./raster/2024Q2 ^
    --out ./nai/GZ_NAI_2024Q2.tif

# 批量（推荐）
python apply_m4_urban_raster_v2.py --manifest seasons.csv --out-dir ./nai
"""
from __future__ import print_function

import argparse
from package_paths import input_path, output_path, DEFAULT_MODEL, REPRODUCED
import calendar
import datetime as dt
import json
import os
import sys

import numpy as np
import joblib
import rasterio

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = str(DEFAULT_MODEL)

SEASON_MONTHS = {1: (3, 4, 5), 2: (6, 7, 8), 3: (9, 10, 11), 4: (12, 1, 2)}


# --------------------------------------------------------------------------
# 季节项（与 prepare_model_inputs.py 保持一致）
# --------------------------------------------------------------------------
def season_doy_values(year, q, mode='daily_mean'):
    months = SEASON_MONTHS[q]
    dates = []
    for m in months:
        y = year if (q != 4 or m == 12) else year + 1
        nd = calendar.monthrange(y, m)[1]
        for d in range(1, nd + 1):
            dates.append(dt.date(y, m, d))
    if mode == 'midpoint':
        mid = dates[len(dates) // 2]
        doy = mid.timetuple().tm_yday
        return (float(np.sin(2 * np.pi * doy / 365.0)),
                float(np.cos(2 * np.pi * doy / 365.0)))
    doys = np.array([d.timetuple().tm_yday for d in dates], float)
    return (float(np.sin(2 * np.pi * doys / 365.0).mean()),
            float(np.cos(2 * np.pi * doys / 365.0).mean()))


def parse_season(tag):
    """'2024Q2' -> (2024, 2)"""
    year = int(tag[:4])
    q = int(tag[-1])
    if q not in (1, 2, 3, 4):
        raise ValueError('bad season tag: %s' % tag)
    return year, q


# --------------------------------------------------------------------------
def read_band(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype('float64')
        prof = dict(src.profile)
        nodata = src.nodata
        crs = src.crs
        transform = src.transform
        shape = arr.shape
    if nodata is not None and np.isfinite(nodata):
        arr = np.where(arr == nodata, np.nan, arr)
    return arr, prof, crs, transform, shape


def read_band_by_name(path, want=None):
    """按波段名读取；找不到名字时退回第 1 个波段。"""
    with rasterio.open(path) as src:
        names = [n or '' for n in src.descriptions]
        idx = 1
        if want and want in names:
            idx = names.index(want) + 1
        arr = src.read(idx).astype('float64')
        prof = dict(src.profile)
        nodata = src.nodata
        crs, transform, shape = src.crs, src.transform, src.shape
    if nodata is not None and np.isfinite(nodata):
        arr = np.where(arr == nodata, np.nan, arr)
    return arr, prof, crs, transform, shape


def resolve_doy(feat_dir, year, q, doy_dir=None):
    """定位 prepare_model_inputs.py 生成的 DOY tif。

    脚本生成的命名是 GZ_NAI_DOY_<year>Q<q>_<season>.tif（双波段 doy_sin/doy_cos），
    不一定放在 feat_dir 里，因此按以下顺序查找：
      1) doy_dir 参数
      2) feat_dir 内的 doy_sin.tif / doy_cos.tif
      3) feat_dir 及其同级 doy/ 子目录中的 GZ_NAI_DOY_<year>Q<q>*.tif
    """
    import glob
    tag = '%dQ%d' % (year, q)
    # 2) 单独波段的常规命名
    p_sin = os.path.join(feat_dir, 'doy_sin.tif')
    p_cos = os.path.join(feat_dir, 'doy_cos.tif')
    if os.path.exists(p_sin) and os.path.exists(p_cos):
        return ('bands', p_sin, p_cos)
    # 1)/3) 双波段命名，允许在 doy_dir 或 feat_dir(/../doy) 中
    cands = []
    if doy_dir:
        cands.append(doy_dir)
    cands.append(feat_dir)
    cands.append(os.path.join(os.path.dirname(os.path.abspath(feat_dir)), 'doy'))
    for c in cands:
        for pat in ('GZ_NAI_DOY_%s*.tif' % tag, '*DOY*%s*.tif' % tag):
            hits = sorted(glob.glob(os.path.join(c, pat)))
            if hits:
                return ('dual', hits[0], None)
    return (None, None, None)


def apply_one(art, feat_dir, year, q, out_path, season_mode='daily_mean',
              site=None, doy_from_tif=False, doy_dir=None):
    feats = list(art['anom_feats'])
    data = {}
    ref = None

    # 先解析 DOY 来源（若走 tif）
    doy_mode, doy_p1, doy_p2 = (None, None, None)
    if doy_from_tif:
        doy_mode, doy_p1, doy_p2 = resolve_doy(feat_dir, year, q, doy_dir)
        if doy_mode is None:
            return None, ('--doy-from-tif 已开启，但未找到 DOY 栅格。'
                          '请用 --doy-dir 指定目录，或先用 prepare_model_inputs.py 生成 '
                          'GZ_NAI_DOY_%dQ%d_*.tif' % (year, q))

    for f in feats:
        if f in ('doy_sin', 'doy_cos'):
            if not doy_from_tif:
                continue
            if doy_mode == 'bands':
                p = doy_p1 if f == 'doy_sin' else doy_p2
                arr, prof, crs, transform, shape = read_band(p)
            else:                                   # 双波段文件
                arr, prof, crs, transform, shape = read_band_by_name(doy_p1, want=f)
            if ref is None:
                ref = (prof, crs, transform, shape)
            else:
                rp, rcrs, rtr, rsh = ref
                if shape != rsh:
                    return None, 'shape mismatch for %s: %s vs %s' % (f, shape, rsh)
            data[f] = arr
            continue

        p = os.path.join(feat_dir, f + '.tif')
        if not os.path.exists(p):
            return None, 'missing feature tif: %s' % p
        arr, prof, crs, transform, shape = read_band(p)
        if ref is None:
            ref = (prof, crs, transform, shape)
        else:
            rp, rcrs, rtr, rsh = ref
            if shape != rsh:
                return None, 'shape mismatch for %s: %s vs %s' % (f, shape, rsh)
            if str(crs) != str(rcrs):
                return None, 'CRS mismatch for %s: %s vs %s' % (f, crs, rcrs)
            if transform != rtr:
                return None, 'transform mismatch for %s' % f
        data[f] = arr

    if ref is None:
        return None, 'no reference raster resolved'
    prof, crs, transform, (H, W) = ref

    # 季节项：仅当不使用 DOY tif 时，由季度窗口解析计算
    if not doy_from_tif:
        s_sin, s_cos = season_doy_values(year, q, season_mode)
        data['doy_sin'] = np.full((H, W), s_sin)
        data['doy_cos'] = np.full((H, W), s_cos)
    else:
        for f in ('doy_sin', 'doy_cos'):
            if f not in data:
                return None, 'DOY tif resolved but band %s not found' % f

    # 组装矩阵
    X = np.empty((H * W, len(feats)), dtype='float64')
    valid = np.ones((H, W), dtype=bool)
    for j, f in enumerate(feats):
        a = data[f]
        if a.shape != (H, W):
            return None, 'feature %s has shape %s, expected %s' % (f, a.shape, (H, W))
        valid &= np.isfinite(a)
        X[:, j] = a.ravel()

    # 逐像元适用域检查
    n_out = np.zeros(H * W, dtype='int16')
    out_detail = {}
    for j, f in enumerate(feats):
        lo, hi = art['feature_ranges_train'][f]
        col = X[:, j]
        bad = np.isfinite(col) & ((col < lo) | (col > hi))
        n_out += bad.astype('int16')
        out_detail[f] = int(bad.sum())

    # 本底项
    if site:
        if site not in art['site_mean_log1p_urban']:
            return None, 'site not in monitored urban list: %s' % site
        base_log = art['site_mean_log1p_urban'][site]
        base_kind = 'site_mean:%s' % site
    else:
        base_log = art['group_mean_log1p_urban']
        base_kind = 'group_mean_urban'

    pred = np.full(H * W, np.nan)
    if valid.any():
        resid = art['anomaly_model'].predict(X[valid.ravel()])
        pred[valid.ravel()] = np.clip(
            np.expm1(base_log + resid) * art['smear'], 0, None)
    out = pred.reshape(H, W).astype('float32')

    # 掩膜：0 无效 / 1 域内 / 2 超域
    mask = np.zeros((H, W), dtype='uint8')
    mask[valid] = 1
    mask[valid & (n_out.reshape(H, W) > 0)] = 2

    # 写盘
    out_path = str(output_path(out_path))
    op = dict(prof)
    op.update(count=1, dtype='float32', nodata=np.nan, compress='deflate')
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with rasterio.open(out_path, 'w', **op) as dst:
        dst.write(out, 1)

    mp = dict(prof)
    mp.update(count=1, dtype='uint8', nodata=0, compress='deflate')
    mask_path = os.path.splitext(out_path)[0] + '_mask.tif'
    with rasterio.open(mask_path, 'w', **mp) as dst:
        dst.write(mask, 1)

    fin = out[np.isfinite(out)]
    meta = {
        'season': '%dQ%d' % (year, q),
        'model': art.get('model_name'),
        'model_version': art.get('version'),
        'base': base_kind,
        'base_linear': float(np.expm1(base_log)),
        'smear': float(art['smear']),
        'season_mode': season_mode,
        'doy_from_tif': bool(doy_from_tif),
        'n_valid_px': int(valid.sum()),
        'n_out_of_domain_px': int((mask == 2).sum()),
        'out_of_domain_by_feature': out_detail,
        'inputs': {f: os.path.join(feat_dir, f + '.tif') for f in feats},
    }
    if fin.size:
        meta['pred_stats'] = {
            'mean': float(fin.mean()), 'p05': float(np.percentile(fin, 5)),
            'p50': float(np.percentile(fin, 50)), 'p95': float(np.percentile(fin, 95)),
            'min': float(fin.min()), 'max': float(fin.max()),
        }
    with open(os.path.splitext(out_path)[0] + '.json', 'w') as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=2)

    return meta, None


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--season', help='形如 2024Q2（单季度模式）')
    ap.add_argument('--feat-dir', help='该季度的特征栅格目录')
    ap.add_argument('--manifest', help='批量清单 CSV: season,feat_dir[,out_name]')
    ap.add_argument('--out', help='单季度输出 tif 路径')
    ap.add_argument('--out-dir', help='批量输出目录')
    ap.add_argument('--season-mode', default='daily_mean',
                    choices=['daily_mean', 'midpoint'],
                    help='季节项口径；daily_mean 为季度窗口逐日均值（推荐）')
    ap.add_argument('--doy-from-tif', action='store_true',
                    help='若 doy_sin/doy_cos 已由 prepare_model_inputs.py 生成为 tif，'
                         '用此开关直接读取而不重算')
    ap.add_argument('--site', default=None,
                    help='若目标公园是被监测的 9 个城市公园之一，填其名称以使用站点均值本底')
    ap.add_argument('--doy-dir', default=None,
                    help='prepare_model_inputs.py 的输出目录（--doy-from-tif 时用于定位 DOY 栅格）')
    args = ap.parse_args()

    # Package-only output protection; input resolution is relative to package root.
    args.model = str(input_path(args.model))
    for attr in ('feat_dir', 'manifest', 'doy_dir'):
        if getattr(args, attr):
            setattr(args, attr, str(input_path(getattr(args, attr))))
    if args.out:
        args.out = str(output_path(args.out))
    if args.out_dir:
        args.out_dir = str(output_path(args.out_dir))

    art = joblib.load(args.model)
    print('model  : %s (%s)' % (art.get('model_name'), art.get('version')))
    print('features:', list(art['anom_feats']))
    print('base   : group_mean_urban = %.1f ions/cm3 (linear)'
          % np.expm1(art['group_mean_log1p_urban']))
    print('')

    jobs = []
    if args.manifest:
        import csv as _csv
        with open(args.manifest, 'r', encoding='utf-8-sig') as fp:
            for row in _csv.DictReader(fp):
                feature_dir = row['feat_dir'].strip()
                if not os.path.isabs(feature_dir):
                    feature_dir = os.path.normpath(os.path.join(os.path.dirname(args.manifest), feature_dir))
                jobs.append((row['season'].strip(), feature_dir,
                             row.get('out_name', '').strip()))
    elif args.season and args.feat_dir:
        jobs.append((args.season, args.feat_dir, ''))
    else:
        ap.error('need --season/--feat-dir or --manifest')

    n_ok = 0
    for season, feat_dir, out_name in jobs:
        year, q = parse_season(season)
        if args.out and len(jobs) == 1:
            out_path = args.out
        else:
            if not args.out_dir:
                ap.error('need --out-dir for batch mode')
            if not out_name:
                out_name = 'GZ_NAI_%dQ%d.tif' % (year, q)
            out_path = os.path.join(args.out_dir, out_name)
        meta, err = apply_one(art, feat_dir, year, q, out_path,
                              season_mode=args.season_mode, site=args.site,
                              doy_from_tif=args.doy_from_tif,
                              doy_dir=args.doy_dir)
        if err:
            print('[FAIL] %s : %s' % (season, err))
            continue
        n_ok += 1
        ps = meta.get('pred_stats', {})
        print('[OK]   %s  valid=%d  超域=%d  mean=%.0f p05=%.0f p50=%.0f p95=%.0f'
              % (season, meta['n_valid_px'], meta['n_out_of_domain_px'],
                 ps.get('mean', float('nan')), ps.get('p05', float('nan')),
                 ps.get('p50', float('nan')), ps.get('p95', float('nan'))))
    print('')
    print('完成 %d / %d 个季度' % (n_ok, len(jobs)))
    if n_ok:
        print('提示：*_mask.tif 中 2 = 至少一个特征超出训练域，论文中应作为低可信度区标注。')


if __name__ == '__main__':
    main()
