// =============================================================================
// GZ_NAI_raster_01_quarterly_features.js   (v2, corrected)
// -----------------------------------------------------------------------------
// 目标：为「城市公园负离子浓度反演」导出 2015—2025 年逐季度的**逐像元栅格**特征
//
// 模型输入（m4_urban_model.joblib 的 anom_feats，共 6 个）与本脚本的对应关系：
//   1) doy_sin                    <- 不导出；由 prepare_model_inputs.py 按季内逐日正余弦均值生成常量栅格
//   2) doy_cos                    <- 同上
//   3) hls_model_evi2             <- 本脚本导出
//   4) hls_model_ndmi             <- 本脚本导出
//   5) gpm_precip_uncal_d0_mm     <- 本脚本导出（季度累计降水 mm）
//   6) gpm_precip_uncal_prev7d_mm <- 本脚本导出（季度末前 7 日累计降水 mm）
//
// 季节划分（中国气象季节，与论文一致）：
//   Q1 春季 3—5 月 | Q2 夏季 6—8 月 | Q3 秋季 9—11 月 | Q4 冬季 12 月—次年 2 月
//
// 数据源与年份覆盖（关键）：
//   NASA/HLS/HLSL30/v002（Landsat 协调产品）2013-04 起
//   NASA/HLS/HLSS30/v002（Sentinel-2 协调产品）2015-11 起
//   => HLS 单用会缺 2015—2016 的春夏季；故提供 MODE：
//        'auto'（默认）：该季度 HLS 有观测则用 HLS，否则回退 L8/L9 + S2
//        'hls'         ：只用 HLS（建议 2017 年起）
//        'harm'        ：只用 L8/L9 + S2（2015 年起全覆盖）
//   回退模式的 EVI2/NDMI 与 HLS 有系统偏差，须用 2024—2025 重叠期交叉标定。
//
// 输出（每个「季度×类型」一个 GeoTIFF，Float32，EPSG:32649）：
//   GZ_NAI_FINE_<Y>Q<q>_<season> : hls_model_evi2, hls_model_ndmi,
//                                  hls_obs_count, hls_source
//   GZ_NAI_GPM_<Y>Q<q>_<season>  : gpm_precip_uncal_d0_mm,
//                                  gpm_precip_uncal_prev7d_mm, gpm_halfhour_obs
//   GZ_NAI_MASK_static           : dem_elevation_m, dw_baseline2023_trees_prob,
//                                  not_water, park_candidate
// =============================================================================

// ------------------------------ 参数 ----------------------------------------
var MODE = 'auto';                 // 'auto' | 'hls' | 'harm'
// HLS 可用起始：HLSL30 (Landsat) 2013-04；HLSS30 (Sentinel-2) 2015-11
// => 本项目的季节窗口 Q1(3-5月)/Q2(6-8月)/Q3(9-11月)/Q4(12-2月) 从 2016 年起
//    全部落在 HLSS30 的覆盖期内，因此 2016—2025 可以完整使用 HLS，无需回退。
var YEAR_START = 2016;
var YEAR_END = 2025;
var LAST_QUARTER_2025 = 2;         // 2025 年只做到 Q2；设为 4 可跑满
var SCALE_FINE = 100;              // 米（训练为 300 m 缓冲区均值，100 m 出图更细）
var SCALE_GPM = 11132;             // GPM IMERG 原生 ~11 km
var CRS = 'EPSG:32649';            // UTM 49N
var EXPORT_FOLDER = 'GZ_NAI_raster';

var ROI = ee.Geometry.Rectangle([112.90, 22.50, 114.10, 23.95], null, false);

// 若已有「城市公园边界」矢量，替换此处可显著减小导出体积（推荐）
// var PARKS = ee.FeatureCollection('projects/xxx/assets/gz_parks');
var USE_PARK_ASSET = false;
var PARKS = null;

// ------------------------------ 季节工具 ------------------------------------
function seasonWindow(year, q) {
  if (q === 1) return [year + '-03-01', year + '-06-01'];
  if (q === 2) return [year + '-06-01', year + '-09-01'];
  if (q === 3) return [year + '-09-01', year + '-12-01'];
  return [year + '-12-01', (year + 1) + '-03-01'];
}
var SEASON_EN = ['', 'spring', 'summer', 'autumn', 'winter'];

function emptyImg(bands) {
  return ee.Image.constant(ee.List.repeat(0, bands.length))
    .rename(bands).updateMask(ee.Image.constant(0));
}

// ------------------------------ 指数计算 ------------------------------------
// 统一公式：EVI2 = 2.5(NIR-Red)/(NIR+2.4Red+1) ; NDMI = (NIR-SWIR1)/(NIR+SWIR1)
function addIndices(refl) {
  var nir = refl.select('nir');
  var red = refl.select('red');
  var swir1 = refl.select('swir1');
  var den = nir.add(red.multiply(2.4)).add(1);
  var evi2raw = nir.subtract(red).multiply(2.5).divide(den);
  var evi2 = evi2raw.updateMask(den.abs().gt(0.02))
    .updateMask(evi2raw.gte(-1).and(evi2raw.lte(2.5)))
    .rename('hls_model_evi2');
  var ndmiRaw = nir.subtract(swir1).divide(nir.add(swir1));
  var ndmi = ndmiRaw.updateMask(nir.add(swir1).abs().gt(0.02))
    .updateMask(ndmiRaw.gte(-1).and(ndmiRaw.lte(1)))
    .rename('hls_model_ndmi');
  return ee.Image.cat([evi2, ndmi]);
}

// ---- HLS -------------------------------------------------------------------
function prepHlsL30(img) {
  var clear = img.select('Fmask').bitwiseAnd(30).eq(0);
  var refl = img.select(['B4', 'B5', 'B6'], ['red', 'nir', 'swir1'])
    .updateMask(clear);
  refl = refl.updateMask(refl.reduce(ee.Reducer.min()).gte(-0.2)
    .and(refl.reduce(ee.Reducer.max()).lte(1.6)));
  return addIndices(refl).copyProperties(img, ['system:time_start']);
}
function prepHlsS30(img) {
  var clear = img.select('Fmask').bitwiseAnd(30).eq(0);
  var refl = img.select(['B4', 'B8A', 'B11'], ['red', 'nir', 'swir1'])
    .updateMask(clear);
  refl = refl.updateMask(refl.reduce(ee.Reducer.min()).gte(-0.2)
    .and(refl.reduce(ee.Reducer.max()).lte(1.6)));
  return addIndices(refl).copyProperties(img, ['system:time_start']);
}
var hls = ee.ImageCollection('NASA/HLS/HLSL30/v002')
  .filterBounds(ROI).map(prepHlsL30)
  .merge(ee.ImageCollection('NASA/HLS/HLSS30/v002')
    .filterBounds(ROI).map(prepHlsS30));

// ---- Landsat 8/9 C02 L2 回退 ------------------------------------------------
function prepLandsat(img) {
  var qa = img.select('QA_PIXEL');
  // bit1 扩张云, bit2 卷云, bit3 云, bit4 云影
  var bad = qa.bitwiseAnd(2).neq(0)
    .or(qa.bitwiseAnd(4).neq(0))
    .or(qa.bitwiseAnd(8).neq(0))
    .or(qa.bitwiseAnd(16).neq(0));
  var clear = bad.not();
  var refl = img.select(['SR_B4', 'SR_B5', 'SR_B6'], ['red', 'nir', 'swir1'])
    .multiply(0.0000275).add(-0.2).updateMask(clear);
  refl = refl.updateMask(refl.reduce(ee.Reducer.min()).gte(-0.2)
    .and(refl.reduce(ee.Reducer.max()).lte(1.6)));
  return addIndices(refl).copyProperties(img, ['system:time_start']);
}
var landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .filterBounds(ROI).filter(ee.Filter.eq('PROCESSING_LEVEL', 'L2SP'))
  .map(prepLandsat)
  .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
    .filterBounds(ROI).filter(ee.Filter.eq('PROCESSING_LEVEL', 'L2SP'))
    .map(prepLandsat));

// ---- Sentinel-2 SR Harmonized 回退 -----------------------------------------
var s2cloud = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
  .filterBounds(ROI);
function prepS2(img) {
  var scl = img.select('SCL');
  var clear = img.select('cs_cdf').gte(0.60)
    .and(scl.neq(1)).and(scl.neq(3)).and(scl.neq(8))
    .and(scl.neq(9)).and(scl.neq(10)).and(scl.neq(11));
  var refl = img.select(['B4', 'B8A', 'B11'], ['red', 'nir', 'swir1'])
    .multiply(0.0001).updateMask(clear);
  refl = refl.updateMask(refl.reduce(ee.Reducer.min()).gte(0)
    .and(refl.reduce(ee.Reducer.max()).lte(1.6)));
  return addIndices(refl).copyProperties(img, ['system:time_start']);
}
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(ROI).linkCollection(s2cloud, ['cs_cdf']).map(prepS2);

// ---- 回退集合（统一波段名） -------------------------------------------------
var harm = landsat.merge(s2);

// ------------------------------ GPM -----------------------------------------
var gpm = ee.ImageCollection('NASA/GPM_L3/IMERG_V07').filterBounds(ROI);
function gpmAccum(start, end, name) {
  var col = gpm.filterDate(start, end).select('precipitationUncal');
  return ee.Image(ee.Algorithms.If(
    col.size().gt(0), col.sum().multiply(0.5).rename(name), emptyImg([name])
  ));
}

// ------------------------------ 静态掩膜 ------------------------------------
function staticMask() {
  var demCol = ee.ImageCollection('COPERNICUS/DEM/GLO30_2024_1').filterBounds(ROI);
  var demProj = ee.Image(demCol.first()).select('DEM').projection();
  var dem = demCol.mosaic().setDefaultProjection(demProj)
    .select('DEM').rename('dem_elevation_m');
  var trees = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
    .filterBounds(ROI).filterDate('2023-01-01', '2024-01-01')
    .select(['trees'], ['dw_baseline2023_trees_prob']).median();
  var occ = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
    .select('occurrence').unmask(0);
  var notWater = occ.lte(50).rename('not_water');
  // 候选城市公园像元：非水体 且 树木概率在训练域内
  var cand = notWater.and(trees.gte(0.02)).and(trees.lte(0.80))
    .rename('park_candidate');
  return ee.Image.cat([dem, trees, notWater, cand]).toFloat();
}

// ------------------------------ 主流程 --------------------------------------
// 关键修正：用 ee.Number 条件 + 单层 ee.Algorithms.If，
// 避免把 ee.Image 当作 JS 三目运算符的条件（GEE 反模式，会报错或得到错误的图）。
function seasonFeatures(year, q) {
  var win = seasonWindow(year, q);
  var s = ee.Date(win[0]);
  var e = ee.Date(win[1]);

  var hlsCol = hls.filterDate(s, e);
  var harmCol = harm.filterDate(s, e);

  var nHls = hlsCol.size();
  var nHarm = harmCol.size();

  // 是否使用 HLS：一个 ee.Number 条件
  var useHls;
  if (MODE === 'hls') {
    useHls = ee.Number(1);
  } else if (MODE === 'harm') {
    useHls = ee.Number(0);
  } else {
    useHls = ee.Number(ee.Algorithms.If(nHls.gt(0), 1, 0));
  }

  var hlsIdx = ee.Image(ee.Algorithms.If(
    nHls.gt(0), hlsCol.mean(), emptyImg(['hls_model_evi2', 'hls_model_ndmi'])
  )).select(['hls_model_evi2', 'hls_model_ndmi']);
  var harmIdx = ee.Image(ee.Algorithms.If(
    nHarm.gt(0), harmCol.mean(), emptyImg(['hls_model_evi2', 'hls_model_ndmi'])
  )).select(['hls_model_evi2', 'hls_model_ndmi']);

  var hlsCnt = ee.Image(ee.Algorithms.If(
    nHls.gt(0), hlsCol.select('hls_model_evi2').count(), emptyImg(['count'])
  )).rename('hls_obs_count');
  var harmCnt = ee.Image(ee.Algorithms.If(
    nHarm.gt(0), harmCol.select('hls_model_evi2').count(), emptyImg(['count'])
  )).rename('hls_obs_count');

  // 最终选择：一个 If 同时决定指数、计数、来源码
  var fine = ee.Image(ee.Algorithms.If(
    useHls.eq(1),
    ee.Image.cat([hlsIdx, hlsCnt, ee.Image.constant(1).rename('hls_source')]),
    ee.Image.cat([harmIdx, harmCnt, ee.Image.constant(2).rename('hls_source')])
  )).toFloat();

  var pTot = gpmAccum(s, e, 'gpm_precip_uncal_d0_mm');
  var pPrev7 = gpmAccum(e.advance(-7, 'day'), e, 'gpm_precip_uncal_prev7d_mm');
  var gpmCnt = ee.Image(ee.Algorithms.If(
    gpm.filterDate(s, e).size().gt(0),
    gpm.filterDate(s, e).select('precipitationUncal').count(),
    emptyImg(['count'])
  )).rename('gpm_halfhour_obs');
  var gpmImg = ee.Image.cat([pTot, pPrev7, gpmCnt]).toFloat();

  return {fine: fine, gpm: gpmImg, nHls: nHls, nHarm: nHarm, tag: year + 'Q' + q + '_' + SEASON_EN[q]};
}

function exportSeason(year, q) {
  var f = seasonFeatures(year, q);
  Export.image.toDrive({
    image: f.fine,
    description: 'GZ_NAI_FINE_' + f.tag,
    folder: EXPORT_FOLDER,
    fileNamePrefix: 'GZ_NAI_FINE_' + f.tag,
    region: ROI,
    scale: SCALE_FINE,
    crs: CRS,
    maxPixels: 1e13
  });
  Export.image.toDrive({
    image: f.gpm,
    description: 'GZ_NAI_GPM_' + f.tag,
    folder: EXPORT_FOLDER,
    fileNamePrefix: 'GZ_NAI_GPM_' + f.tag,
    region: ROI,
    scale: SCALE_GPM,
    crs: CRS,
    maxPixels: 1e13
  });
  print('  task:', f.tag, '| HLS scenes =', f.nHls, '| harm scenes =', f.nHarm);
}

// 静态掩膜（一次）
Export.image.toDrive({
  image: staticMask(),
  description: 'GZ_NAI_MASK_static',
  folder: EXPORT_FOLDER,
  fileNamePrefix: 'GZ_NAI_MASK_static',
  region: ROI,
  scale: SCALE_FINE,
  crs: CRS,
  maxPixels: 1e13
});

// 逐季度任务
var n = 0;
for (var y = YEAR_START; y <= YEAR_END; y++) {
  var qmax = (y === 2025) ? LAST_QUARTER_2025 : 4;
  for (var q = 1; q <= qmax; q++) {
    exportSeason(y, q);
    n += 1;
  }
}

print('==================== 任务概览 ====================');
print('MODE        :', MODE);
print('年份范围    :', YEAR_START, '-', YEAR_END, '（2025 年到 Q' + LAST_QUARTER_2025 + '）');
print('季节划分    : Q1 春(3-5) Q2 夏(6-8) Q3 秋(9-11) Q4 冬(12-2)');
print('HLS 覆盖    : HLSL30 自 2013-04，HLSS30 自 2015-11');
print('              => 2016Q1 起所有季度窗口均在 HLS 覆盖期内，无需回退');
print('季度任务数  :', n, '× 2 (FINE+GPM) + 1 (MASK) =', n * 2 + 1, '个 GeoTIFF 任务');
print('FINE 尺度   :', SCALE_FINE, 'm  CRS', CRS);
print('GPM  尺度   :', SCALE_GPM, 'm（IMERG 原生）');
print('导出目录    :', EXPORT_FOLDER);
print('');
print('【务必阅读】');
print('1) doy_sin / doy_cos 不在本脚本导出，由 prepare_model_inputs.py 生成季内逐日正余弦均值常量栅格。');
print('2) 每个季度的 HLS 场景数已在下方逐条打印（HLS scenes = ...）。');
print('   若某季度 HLS scenes = 0（例如 2016 年以前），才会回退到 L8/L9+S2，');
print('   回退产品的 EVI2/NDMI 与 HLS 有尺度差异，必须先用 GZ_NAI_raster_02_crosscalib.js 标定。');
print('3) 建议用城市公园边界矢量裁剪 ROI，可大幅减小文件体积（84 个 tif）。');
print('4) gpm_precip_uncal_prev7d_mm 是「季度末前 7 日」累计，');
print('   与模型训练时的日尺度定义不同，属季度尺度近似，须在论文中声明。');
print('==================================================');

Map.centerObject(ROI, 9);
Map.addLayer(ROI, {color: 'red'}, 'ROI');
Map.addLayer(staticMask().select('dem_elevation_m'),
  {min: 0, max: 900, palette: ['blue', 'green', 'red']}, 'DEM', false);
