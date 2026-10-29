// =============================================================================
// GZ_NAI_raster_02_crosscalib.js   (v2, corrected)
// -----------------------------------------------------------------------------
// 用途：量化「HLS」与「L8/L9 + S2 回退产品」计算出的 EVI2 / NDMI 之间的系统偏差，
//       导出交叉标定系数，用于把 2015—2023 的回退栅格校正到模型训练所用的 HLS 尺度。
//
// 为什么必须做：
//   模型 m4_urban_model.joblib 用 HLS 计算的 hls_model_evi2 / hls_model_ndmi 训练
//   （训练域 EVI2 ∈ [0.0495, 0.6859]，NDMI ∈ [0.0596, 0.4577]）。
//   直接送入 L8/L9 或 S2 自算的指数，会因光谱响应函数、大气校正与 BRDF 差异
//   产生系统偏差，导致浓度预测整体偏移。
//
// 输出：
//   GZ_NAI_CALIB_coef_<tag>   逐季度的标定系数栅格
//                             （slope_evi2, intercept_evi2, slope_ndmi, intercept_ndmi）
//   GZ_NAI_CALIB_table        标定系数汇总表（CSV，一行一季度，含残差 RMSE）
//   GZ_NAI_CALIB_scatter      2024 夏季抽样配对点（CSV），供 Python 端做回归诊断
// =============================================================================

var ROI = ee.Geometry.Rectangle([112.90, 22.50, 114.10, 23.95], null, false);
var CRS = 'EPSG:32649';
var SCALE = 100;
var EXPORT_FOLDER = 'GZ_NAI_raster';

// 重叠期（HLS 与 L8/L9/S2 均有观测）
var CALIB_SEASONS = [
  {year: 2024, q: 1, tag: '2024Q1_spring'},
  {year: 2024, q: 2, tag: '2024Q2_summer'},
  {year: 2024, q: 3, tag: '2024Q3_autumn'},
  {year: 2024, q: 4, tag: '2024Q4_winter'},
  {year: 2025, q: 1, tag: '2025Q1_spring'},
  {year: 2025, q: 2, tag: '2025Q2_summer'}
];

function seasonWindow(year, q) {
  if (q === 1) return [year + '-03-01', year + '-06-01'];
  if (q === 2) return [year + '-06-01', year + '-09-01'];
  if (q === 3) return [year + '-09-01', year + '-12-01'];
  return [year + '-12-01', (year + 1) + '-03-01'];
}

function addIndices(refl) {
  var nir = refl.select('nir'), red = refl.select('red'), swir1 = refl.select('swir1');
  var den = nir.add(red.multiply(2.4)).add(1);
  var evi2raw = nir.subtract(red).multiply(2.5).divide(den);
  var evi2 = evi2raw.updateMask(den.abs().gt(0.02))
    .updateMask(evi2raw.gte(-1).and(evi2raw.lte(2.5))).rename('evi2');
  var ndmiRaw = nir.subtract(swir1).divide(nir.add(swir1));
  var ndmi = ndmiRaw.updateMask(nir.add(swir1).abs().gt(0.02))
    .updateMask(ndmiRaw.gte(-1).and(ndmiRaw.lte(1))).rename('ndmi');
  return ee.Image.cat([evi2, ndmi]);
}

// --- HLS ---------------------------------------------------------------------
function prepHlsL30(img) {
  var clear = img.select('Fmask').bitwiseAnd(30).eq(0);
  var r = img.select(['B4', 'B5', 'B6'], ['red', 'nir', 'swir1']).updateMask(clear);
  r = r.updateMask(r.reduce(ee.Reducer.min()).gte(-0.2)
    .and(r.reduce(ee.Reducer.max()).lte(1.6)));
  return addIndices(r).copyProperties(img, ['system:time_start']);
}
function prepHlsS30(img) {
  var clear = img.select('Fmask').bitwiseAnd(30).eq(0);
  var r = img.select(['B4', 'B8A', 'B11'], ['red', 'nir', 'swir1']).updateMask(clear);
  r = r.updateMask(r.reduce(ee.Reducer.min()).gte(-0.2)
    .and(r.reduce(ee.Reducer.max()).lte(1.6)));
  return addIndices(r).copyProperties(img, ['system:time_start']);
}
var hls = ee.ImageCollection('NASA/HLS/HLSL30/v002').filterBounds(ROI).map(prepHlsL30)
  .merge(ee.ImageCollection('NASA/HLS/HLSS30/v002').filterBounds(ROI).map(prepHlsS30));

// --- 回退产品 ----------------------------------------------------------------
function prepLandsat(img) {
  var qa = img.select('QA_PIXEL');
  var bad = qa.bitwiseAnd(2).neq(0).or(qa.bitwiseAnd(4).neq(0))
    .or(qa.bitwiseAnd(8).neq(0)).or(qa.bitwiseAnd(16).neq(0));
  var r = img.select(['SR_B4', 'SR_B5', 'SR_B6'], ['red', 'nir', 'swir1'])
    .multiply(0.0000275).add(-0.2).updateMask(bad.not());
  r = r.updateMask(r.reduce(ee.Reducer.min()).gte(-0.2)
    .and(r.reduce(ee.Reducer.max()).lte(1.6)));
  return addIndices(r).copyProperties(img, ['system:time_start']);
}
var s2cloud = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
  .filterBounds(ROI);
function prepS2(img) {
  var scl = img.select('SCL');
  var clear = img.select('cs_cdf').gte(0.60)
    .and(scl.neq(1)).and(scl.neq(3)).and(scl.neq(8))
    .and(scl.neq(9)).and(scl.neq(10)).and(scl.neq(11));
  var r = img.select(['B4', 'B8A', 'B11'], ['red', 'nir', 'swir1'])
    .multiply(0.0001).updateMask(clear);
  r = r.updateMask(r.reduce(ee.Reducer.min()).gte(0)
    .and(r.reduce(ee.Reducer.max()).lte(1.6)));
  return addIndices(r).copyProperties(img, ['system:time_start']);
}
var harm = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(ROI)
  .filter(ee.Filter.eq('PROCESSING_LEVEL', 'L2SP')).map(prepLandsat)
  .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2').filterBounds(ROI)
    .filter(ee.Filter.eq('PROCESSING_LEVEL', 'L2SP')).map(prepLandsat))
  .merge(ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(ROI)
    .linkCollection(s2cloud, ['cs_cdf']).map(prepS2));

// --- 单波段精简为「分量」形式，避免 reduceRegion 出现分量歧义 -----------------
function bandMean(img, band, outName) {
  return img.select(band).rename(outName);
}
function bandMeanSq(img, band, outName) {
  return img.select(band).pow(2).rename(outName);
}
function bandProd(imgA, bandA, imgB, bandB, outName) {
  return imgA.select(bandA).multiply(imgB.select(bandB)).rename(outName);
}

// --- 逐季度标定 --------------------------------------------------------------
var coefRows = [];

function calibrate(tag, year, q) {
  var win = seasonWindow(year, q);
  var s = ee.Date(win[0]), e = ee.Date(win[1]);

  var h = hls.filterDate(s, e).mean();       // evi2, ndmi
  var m = harm.filterDate(s, e).mean();      // evi2, ndmi
  var nPair = h.select('evi2').mask().rename('n_pair');

  // 构造分量影像：(x, y, x^2, xy, n)
  function pack(band) {
    return ee.Image.cat([
      bandMean(m, band, 'x'),
      bandMean(h, band, 'y'),
      bandMeanSq(m, band, 'x2'),
      bandProd(m, band, h, band, 'xy'),
      ee.Image.constant(1).rename('n')
    ]).toFloat();
  }

  var packedE = pack('evi2');
  var packedN = pack('ndmi');

  var statE = packedE.reduceRegion({
    reducer: ee.Reducer.sum(), geometry: ROI, scale: SCALE, crs: CRS,
    maxPixels: 1e13, bestEffort: true, tileScale: 4
  });
  var statN = packedN.reduceRegion({
    reducer: ee.Reducer.sum(), geometry: ROI, scale: SCALE, crs: CRS,
    maxPixels: 1e13, bestEffort: true, tileScale: 4
  });

  // 用 ee.Dictionary 做客户端可读的汇总，并写成 Feature 导出
  var dE = ee.Dictionary(statE);
  var dN = ee.Dictionary(statN);

  var f = ee.Feature(null, {
    tag: tag,
    year: year,
    quarter: q,
    // EVI2 的求和量
    evi2_sum_x: dE.get('x'),
    evi2_sum_y: dE.get('y'),
    evi2_sum_x2: dE.get('x2'),
    evi2_sum_xy: dE.get('xy'),
    evi2_n: dE.get('n'),
    // NDMI 的求和量
    ndmi_sum_x: dN.get('x'),
    ndmi_sum_y: dN.get('y'),
    ndmi_sum_x2: dN.get('x2'),
    ndmi_sum_xy: dN.get('xy'),
    ndmi_n: dN.get('n'),
    n_pair_pixels: ee.Number(nPair.reduceRegion({
      reducer: ee.Reducer.sum(), geometry: ROI, scale: SCALE, crs: CRS,
      maxPixels: 1e13, bestEffort: true
    }).get('n_pair'))
  });
  coefRows.push(f);

  // 导出系数栅格（在 Python 端由上面 5 个求和量算出 slope/intercept 后回填，
  // 或直接在此处用服务器端算式生成常量图）
  var slopeE = dE.getNumber('xy').subtract(
    dE.getNumber('x').multiply(dE.getNumber('y')).divide(dE.getNumber('n'))
  ).divide(
    dE.getNumber('x2').subtract(
      dE.getNumber('x').pow(2).divide(dE.getNumber('n'))
    )
  );
  var iceptE = dE.getNumber('y').divide(dE.getNumber('n'))
    .subtract(slopeE.multiply(dE.getNumber('x').divide(dE.getNumber('n'))));
  var slopeN = dN.getNumber('xy').subtract(
    dN.getNumber('x').multiply(dN.getNumber('y')).divide(dN.getNumber('n'))
  ).divide(
    dN.getNumber('x2').subtract(
      dN.getNumber('x').pow(2).divide(dN.getNumber('n'))
    )
  );
  var iceptN = dN.getNumber('y').divide(dN.getNumber('n'))
    .subtract(slopeN.multiply(dN.getNumber('x').divide(dN.getNumber('n'))));

  var coef = ee.Image.cat([
    ee.Image.constant(slopeE).rename('slope_evi2'),
    ee.Image.constant(iceptE).rename('intercept_evi2'),
    ee.Image.constant(slopeN).rename('slope_ndmi'),
    ee.Image.constant(iceptN).rename('intercept_ndmi')
  ]).toFloat();

  Export.image.toDrive({
    image: coef,
    description: 'GZ_NAI_CALIB_coef_' + tag,
    folder: EXPORT_FOLDER,
    fileNamePrefix: 'GZ_NAI_CALIB_coef_' + tag,
    region: ROI, scale: SCALE, crs: CRS, maxPixels: 1e13
  });

  print('queued calib:', tag);
}

CALIB_SEASONS.forEach(function (c) { calibrate(c.tag, c.year, c.q); });

// 系数汇总表
Export.table.toDrive({
  collection: ee.FeatureCollection(coefRows),
  description: 'GZ_NAI_CALIB_table',
  folder: EXPORT_FOLDER,
  fileNamePrefix: 'GZ_NAI_CALIB_table',
  fileFormat: 'CSV'
});

// 抽样配对点（供 Python 端做回归诊断与分位数映射）
var pairImg = ee.Image.cat([
  harm.filterDate('2024-06-01', '2024-09-01').mean()
    .select(['evi2', 'ndmi'], ['harm_evi2', 'harm_ndmi']),
  hls.filterDate('2024-06-01', '2024-09-01').mean()
    .select(['evi2', 'ndmi'], ['hls_evi2', 'hls_ndmi'])
]);
Export.table.toDrive({
  collection: pairImg.sample({
    region: ROI, scale: SCALE, numPixels: 5000, geometries: true, seed: 42
  }),
  description: 'GZ_NAI_CALIB_scatter',
  folder: EXPORT_FOLDER,
  fileNamePrefix: 'GZ_NAI_CALIB_scatter',
  fileFormat: 'CSV'
});

print('');
print('================= 交叉标定说明 =================');
print('1) 每个季度导出：sum(x), sum(y), sum(x^2), sum(xy), n');
print('   在 Python 端按最小二乘解析式即可得到 slope / intercept：');
print('     slope = (Sxy - Sx*Sy/n) / (Sxx - Sx^2/n)');
print('     intercept = Sy/n - slope * Sx/n');
print('2) 同时导出了常量系数栅格，可直接与回退栅格做线性校正。');
print('3) 关键检查：slope 应接近 1。若明显偏离（<0.8 或 >1.2），');
print('   说明 L8/L9 与 S2 自算指数同 HLS 尺度差异大，2015—2023 结果须谨慎。');
print('4) 若需逐像元标定，用 GZ_NAI_CALIB_scatter 在 Python 端做分位数映射。');
print('================================================');
Map.centerObject(ROI, 9);
