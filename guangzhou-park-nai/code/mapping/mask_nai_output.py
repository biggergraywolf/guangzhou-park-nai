# -*- coding: utf-8 -*-
"""
mask_nai_output.py
==================
用 *_mask.tif 把 NAI 反演结果裁剪为"可信像元"。

掩膜取值：
    0 = 无效（输入特征含 NaN）
    1 = 可信（6 个输入特征全部落在模型训练域内）
    2 = 超训练域（至少一个特征超出模型见过的范围，预测属外推）

本脚本把 NAI 栅格中【非可信像元】置为 NaN，只保留可信范围。

用法
----
# 默认：只保留 mask==1（最严格）
python mask_nai_output.py --in-dir ../NAI_output --out-dir ../NAI_masked

# 放宽：保留 mask 在 {1,2}（只去掉真正无效的像元）
python mask_nai_output.py --in-dir ../NAI_output --out-dir ../NAI_masked --keep 1,2

# 最大放宽：mask 只要 >0
python mask_nai_output.py --in-dir ../NAI_output --out-dir ../NAI_masked --keep-min 1
"""
from __future__ import print_function

import argparse
from package_paths import input_path, output_path, DEFAULT_MODEL, REPRODUCED
import glob
import io
import json
import os

import numpy as np
import rasterio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-dir', required=True, help='NAI_output 目录')
    ap.add_argument('--out-dir', required=True, help='输出目录')
    ap.add_argument('--only', default=None, help='Comma-separated season filter; wrapper defaults to the paper36 seasons.')
    ap.add_argument('--keep', default='1',
                    help='保留哪些掩膜值，逗号分隔，默认 "1"')
    ap.add_argument('--keep-min', type=int, default=None,
                    help='保留 mask >= 该值的像元（与 --keep 二选一）')
    args = ap.parse_args()

    # Package-only output protection; input resolution is relative to package root.
    args.in_dir = str(input_path(args.in_dir))
    args.out_dir = str(output_path(args.out_dir))

    keep = set(int(x) for x in str(args.keep).split(',')) if args.keep_min is None else None
    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.in_dir, 'GZ_NAI_*.tif')))
    files = [f for f in files if not f.endswith('_mask.tif')]
    if args.only:
        selected = set(args.only.split(','))
        files = [f for f in files if os.path.basename(f).replace('GZ_NAI_', '').replace('.tif', '') in selected]

    print('=' * 88)
    print('掩膜提取：用 *_mask.tif 裁剪 NAI 反演结果')
    print('=' * 88)
    print('输入目录 :', args.in_dir)
    print('输出目录 :', args.out_dir)
    if keep is not None:
        print('保留掩膜值 :', sorted(keep))
    else:
        print('保留条件 : mask >= %d' % args.keep_min)
    print('待处理 : %d 个季度' % len(files))
    print('')

    log = []
    def p(*a):
        s = ' '.join(str(x) for x in a)
        log.append(s); print(s)

    p('%-12s %11s %11s %8s %10s %10s %10s' %
      ('season', 'px_全部', 'px_保留', '保留率', 'mean', 'p50', 'p95'))

    rows = []
    for f in files:
        tag = os.path.basename(f).replace('GZ_NAI_', '').replace('.tif', '')
        mp = f.replace('.tif', '_mask.tif')
        if not os.path.exists(mp):
            p('%-12s  [跳过] 缺少掩膜文件' % tag)
            continue

        with rasterio.open(f) as s:
            a = s.read(1).astype('float64')
            prof = dict(s.profile)
        with rasterio.open(mp) as s:
            m = s.read(1)

        if keep is not None:
            sel = np.isin(m, list(keep))
        else:
            sel = m >= args.keep_min

        out = np.where(sel & np.isfinite(a), a, np.nan).astype('float32')

        op = dict(prof)
        op.update(count=1, dtype='float32', nodata=np.nan, compress='deflate')
        dst = os.path.join(args.out_dir, os.path.basename(f))
        with rasterio.open(dst, 'w', **op) as d:
            d.write(out, 1)

        v = out[np.isfinite(out)]
        n_all = int(np.isfinite(a).sum())
        rows.append(dict(season=tag, n_all=n_all, n_kept=int(v.size),
                         keep_ratio=100.0 * v.size / max(n_all, 1),
                         mean=float(v.mean()) if v.size else np.nan,
                         p50=float(np.percentile(v, 50)) if v.size else np.nan,
                         p95=float(np.percentile(v, 95)) if v.size else np.nan))
        p('%-12s %11d %11d %7.1f%% %10.0f %10.0f %10.0f'
          % (tag, n_all, v.size, rows[-1]['keep_ratio'],
             rows[-1]['mean'], rows[-1]['p50'], rows[-1]['p95']))

    import pandas as pd
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(args.out_dir, '_masked_summary.csv'),
             index=False, encoding='utf-8-sig')

    p('')
    p('=' * 88)
    p('汇总')
    p('=' * 88)
    p('  平均保留率 : %.1f%%' % T['keep_ratio'].mean())
    p('  保留率范围 : %.1f%% — %.1f%%' % (T['keep_ratio'].min(), T['keep_ratio'].max()))
    p('  各季度均值范围 : %.0f — %.0f ions/cm3'
      % (T['mean'].min(), T['mean'].max()))
    p('')
    # 季节节律（掩膜后）
    T['q'] = T['season'].str[-1].astype(int)
    g = T.groupby('q')['mean'].agg(['mean', 'size'])
    SE = {1: 'Q1 春(3-5月)', 2: 'Q2 夏(6-8月)', 3: 'Q3 秋(9-11月)', 4: 'Q4 冬(12-2月)'}
    p('  掩膜后的季节节律：')
    for q in [1, 2, 3, 4]:
        if q in g.index:
            p('    %-16s %.0f ions/cm3  (%d 年)' % (SE[q], g.loc[q, 'mean'], int(g.loc[q, 'size'])))
    amp = g['mean'].max() - g['mean'].min()
    p('    季节振幅 %.0f（峰谷比 %.2f）'
      % (amp, g['mean'].max() / g['mean'].min()))
    p('')
    p('  文件已写入 :', args.out_dir)
    p('  汇总表     :', os.path.join(args.out_dir, '_masked_summary.csv'))

    with io.open(os.path.join(args.out_dir, '_masked_summary.txt'), 'w',
                 encoding='utf-8') as fp:
        fp.write('\n'.join(log))


if __name__ == '__main__':
    main()
