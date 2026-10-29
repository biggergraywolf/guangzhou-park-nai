// Guangzhou negative-air-ion study: one-row-per-monitoring-record core feature table.
// Exact-day variables use the Guangzhou local calendar day. HLS additionally
// provides explicitly labelled nearest-valid auxiliary variables within ±66 d.
// The six exports have identical columns and should be concatenated vertically.

var MONITOR_ASSET = 'projects/ee-hcqwhu/assets/guangzhou_anion_monitoring';
var BUFFER_M = 300;  // 0 = table coordinate pixel; 300 = recommended for current POI coordinates.
var CLOUD_SCORE_THRESHOLD = 0.60;
var HLS_MAX_GAP_DAYS = 66;
var EXPORT_FOLDER = 'GEE_Guangzhou_Anion';
var EXPORT_PREFIX = 'GZ_anion_09_master_core';
var ROI = ee.Geometry.Rectangle([112.5, 22.7, 114.2, 23.95], null, false);

var BASE_FIELDS = [
  'row_id', 'area', 'date', 'average_value', 'grade', 'remark',
  'source_id_text', 'least_value', 'max_value', 'longitude', 'latitude',
  'local_day_start_utc_ms', 'local_day_end_utc_ms', 'spatial_support_m',
  'coarse_sampling_mode'
];

function loadMonitor() {
  return ee.FeatureCollection(MONITOR_ASSET).map(function (f) {
    var coordinates = ee.List(f.geometry().coordinates());
    return ee.Feature(f.geometry(), f.toDictionary()).set({
      longitude: ee.Number(coordinates.get(0)),
      latitude: ee.Number(coordinates.get(1))
    });
  });
}

function spatialSupport(f) {
  var geom = f.geometry();
  if (BUFFER_M > 0) geom = geom.buffer(BUFFER_M);
  return ee.Feature(geom).copyProperties(f).set('spatial_support_m', BUFFER_M);
}

// Rebuild center points before sampling 1 km and 11 km products. This prevents
// a small high-resolution buffer from missing the center of a coarse pixel.
function coarsePointSupport(f) {
  var lon = ee.Number.parse(ee.String(f.get('longitude')));
  var lat = ee.Number.parse(ee.String(f.get('latitude')));
  return ee.Feature(ee.Geometry.Point([lon, lat]), f.toDictionary())
    .set('coarse_sampling_mode', 'center_pixel_at_site');
}

function emptyImage(names) {
  return ee.Image.constant(ee.List.repeat(0, names.length))
    .rename(names).updateMask(ee.Image.constant(0));
}

function safeDailyMedian(col, start, end, bands, countName) {
  var dayCol = col.filterDate(start, end);
  var image = ee.Image(ee.Algorithms.If(
    dayCol.size().gt(0), dayCol.median().select(bands), emptyImage(bands)
  ));
  var valid = ee.Image(ee.Algorithms.If(
    dayCol.size().gt(0), dayCol.select(bands[0]).count().rename(countName),
    emptyImage([countName])
  ));
  return image.addBands(valid);
}

function reduceMean(image, rows, scale) {
  return image.reduceRegions({
    collection: rows,
    reducer: ee.Reducer.mean(),
    scale: scale,
    tileScale: 8,
    maxPixelsPerRegion: 10000000
  });
}

// -----------------------------------------------------------------------------
// Static terrain, surface water and study-period land-cover baseline.
// -----------------------------------------------------------------------------

var demCollection = ee.ImageCollection('COPERNICUS/DEM/GLO30_2024_1')
  .filterBounds(ROI);
var demProjection = ee.Image(demCollection.first()).select('DEM').projection();
var dem = demCollection.mosaic().setDefaultProjection(demProjection);
var elevation = dem.select('DEM').rename('dem_elevation_m');
var slope = ee.Terrain.slope(elevation).rename('dem_slope_deg');
var aspectRad = ee.Terrain.aspect(elevation).multiply(Math.PI / 180);
var aspectSin = aspectRad.sin().rename('dem_aspect_sin');
var aspectCos = aspectRad.cos().rename('dem_aspect_cos');
var tpi300 = elevation.subtract(elevation.focalMean({radius: 300, units: 'meters'}))
  .rename('dem_tpi_300m');
var relief500 = elevation.focalMax({radius: 500, units: 'meters'})
  .subtract(elevation.focalMin({radius: 500, units: 'meters'}))
  .rename('dem_relief_500m');

var gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater');
var waterOccurrence = gsw.select('occurrence').unmask(0)
  .rename('jrc_water_occurrence_pct');
var waterSeasonality = gsw.select('seasonality').unmask(0)
  .rename('jrc_water_seasonality_months');
var permanentWater = waterSeasonality.gte(10).rename('jrc_permanent_water');
var distanceToWater = permanentWater.selfMask()
  .distance(ee.Kernel.euclidean(5000, 'meters')).unmask(5000)
  .rename('jrc_distance_to_water_m');

var STATIC_TERRAIN_BANDS = [
  'dem_elevation_m', 'dem_slope_deg', 'dem_aspect_sin', 'dem_aspect_cos',
  'dem_tpi_300m', 'dem_relief_500m', 'jrc_water_occurrence_pct',
  'jrc_water_seasonality_months', 'jrc_permanent_water',
  'jrc_distance_to_water_m'
];
var staticTerrain = ee.Image.cat([
  elevation, slope, aspectSin, aspectCos, tpi300, relief500,
  waterOccurrence, waterSeasonality, permanentWater, distanceToWater
]).select(STATIC_TERRAIN_BANDS);

var DW_PERIOD_SOURCE = ['water', 'trees', 'grass', 'built', 'bare'];
var STATIC_DW_BANDS = [
  'dw_baseline2023_water_prob', 'dw_baseline2023_trees_prob',
  'dw_baseline2023_grass_prob', 'dw_baseline2023_built_prob',
  'dw_baseline2023_bare_prob'
];
var staticDw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
  .filterBounds(ROI).filterDate('2023-01-01', '2024-01-01')
  .select(DW_PERIOD_SOURCE, STATIC_DW_BANDS).median();

// Static variables are calculated once for the 14 unique sites and joined to
// all monitoring rows. Recomputing them for every date wastes memory.
var monitorBase = loadMonitor();
var staticSites = monitorBase.distinct(['area']).map(spatialSupport);
var staticSiteStats = reduceMean(staticTerrain, staticSites, 30);
staticSiteStats = reduceMean(staticDw, staticSiteStats, 10);
var staticJoin = ee.Join.saveFirst('static_match').apply({
  primary: monitorBase,
  secondary: staticSiteStats,
  condition: ee.Filter.equals({leftField: 'area', rightField: 'area'})
});
var monitor = ee.FeatureCollection(staticJoin).map(function (f) {
  var match = ee.Feature(f.get('static_match'));
  return f.set(match.toDictionary(
    STATIC_TERRAIN_BANDS.concat(STATIC_DW_BANDS)
  )).set('spatial_support_m', BUFFER_M);
});

// -----------------------------------------------------------------------------
// HLS optical core: harmonized Landsat + Sentinel-2 at 30 m.
// -----------------------------------------------------------------------------

var HLS_BANDS = [
  'hls_ndvi', 'hls_evi2', 'hls_nirv', 'hls_ndmi',
  'hls_mndwi', 'hls_ndbi', 'hls_bsi'
];

var HLS_EXACT_BANDS = HLS_BANDS.map(function (name) {
  return name.replace('hls_', 'hls_exact_');
});
var HLS_NEAR_BANDS = HLS_BANDS.map(function (name) {
  return name.replace('hls_', 'hls_near_');
});

function stableHlsRatio(numerator, denominator, name) {
  var raw = numerator.divide(denominator);
  return raw.updateMask(denominator.abs().gt(0.02))
    .updateMask(raw.gte(-1).and(raw.lte(1))).rename(name);
}

function hlsIndices(refl) {
  var blue = refl.select('blue');
  var green = refl.select('green');
  var red = refl.select('red');
  var nir = refl.select('nir');
  var swir1 = refl.select('swir1');
  var ndvi = stableHlsRatio(nir.subtract(red), nir.add(red), 'hls_ndvi');
  var eviDen = nir.add(red.multiply(2.4)).add(1);
  var eviRaw = nir.subtract(red).multiply(2.5).divide(eviDen);
  var evi2 = eviRaw.updateMask(eviDen.abs().gt(0.02))
    .updateMask(eviRaw.gte(-1).and(eviRaw.lte(2.5))).rename('hls_evi2');
  var nirvRaw = ndvi.multiply(nir);
  var nirv = nirvRaw.updateMask(nirvRaw.gte(-0.2).and(nirvRaw.lte(1.6)))
    .rename('hls_nirv');
  var ndmi = stableHlsRatio(nir.subtract(swir1), nir.add(swir1), 'hls_ndmi');
  var mndwi = stableHlsRatio(green.subtract(swir1), green.add(swir1), 'hls_mndwi');
  var ndbi = stableHlsRatio(swir1.subtract(nir), swir1.add(nir), 'hls_ndbi');
  var bsi = stableHlsRatio(
    swir1.add(red).subtract(nir.add(blue)),
    swir1.add(red).add(nir).add(blue), 'hls_bsi'
  );
  return ee.Image.cat([ndvi, evi2, nirv, ndmi, mndwi, ndbi, bsi]);
}

function prepHlsL30(img) {
  var clear = img.select('Fmask').bitwiseAnd(30).eq(0);
  var refl = img.select(
    ['B2', 'B3', 'B4', 'B5', 'B6'],
    ['blue', 'green', 'red', 'nir', 'swir1']
  ).updateMask(clear);
  var physical = refl.reduce(ee.Reducer.min()).gte(-0.2)
    .and(refl.reduce(ee.Reducer.max()).lte(1.6));
  refl = refl.updateMask(physical);
  return hlsIndices(refl).copyProperties(img, ['system:time_start']);
}

function prepHlsS30(img) {
  var clear = img.select('Fmask').bitwiseAnd(30).eq(0);
  var refl = img.select(
    ['B2', 'B3', 'B4', 'B8A', 'B11'],
    ['blue', 'green', 'red', 'nir', 'swir1']
  ).updateMask(clear);
  var physical = refl.reduce(ee.Reducer.min()).gte(-0.2)
    .and(refl.reduce(ee.Reducer.max()).lte(1.6));
  refl = refl.updateMask(physical);
  return hlsIndices(refl).copyProperties(img, ['system:time_start']);
}

var hls = ee.ImageCollection('NASA/HLS/HLSL30/v002')
  .filterBounds(ROI).map(prepHlsL30)
  .merge(ee.ImageCollection('NASA/HLS/HLSS30/v002')
    .filterBounds(ROI).map(prepHlsS30));

function hlsForTargetDay(start, end) {
  var exactCol = hls.filterDate(start, end);
  var exact = ee.Image(ee.Algorithms.If(
    exactCol.size().gt(0),
    exactCol.median().select(HLS_BANDS).rename(HLS_EXACT_BANDS),
    emptyImage(HLS_EXACT_BANDS)
  ));
  var exactCount = ee.Image(ee.Algorithms.If(
    exactCol.size().gt(0),
    exactCol.select('hls_ndvi').count().rename('hls_exact_valid_obs'),
    emptyImage(['hls_exact_valid_obs'])
  ));

  var targetMidpoint = start.advance(12, 'hour');
  var candidates = hls
    .filterDate(
      start.advance(-HLS_MAX_GAP_DAYS, 'day'),
      end.advance(HLS_MAX_GAP_DAYS, 'day')
    )
    .map(function (img) {
      var signedGap = ee.Number(ee.Date(img.get('system:time_start'))
        .difference(targetMidpoint, 'day'));
      var absGap = signedGap.abs();
      var validMask = img.select('hls_ndvi').mask();
      var signedBand = ee.Image.constant(signedGap).toFloat()
        .rename('hls_near_signed_gap_days').updateMask(validMask);
      var absBand = ee.Image.constant(absGap).toFloat()
        .rename('hls_near_abs_gap_days').updateMask(validMask);
      var quality = absBand.multiply(-1).rename('hls_near_time_quality');
      return img.addBands(ee.Image.cat([signedBand, absBand, quality]));
    });

  var sourceNames = HLS_BANDS.concat([
    'hls_near_signed_gap_days', 'hls_near_abs_gap_days'
  ]);
  var nearNames = HLS_NEAR_BANDS.concat([
    'hls_near_signed_gap_days_mean', 'hls_near_abs_gap_days_mean'
  ]);
  var nearest = ee.Image(ee.Algorithms.If(
    candidates.size().gt(0),
    candidates.qualityMosaic('hls_near_time_quality')
      .select(sourceNames, nearNames),
    emptyImage(nearNames)
  ));
  var nearCount = ee.Image(ee.Algorithms.If(
    candidates.size().gt(0),
    candidates.select('hls_ndvi').count().rename('hls_near_candidate_obs'),
    emptyImage(['hls_near_candidate_obs'])
  ));
  return exact.addBands(exactCount).addBands(nearest).addBands(nearCount);
}

// -----------------------------------------------------------------------------
// Sentinel-2 red-edge core and same-day Dynamic World probabilities.
// -----------------------------------------------------------------------------

var S2_BANDS = ['s2_ndre1', 's2_ndre2', 's2_ci_rededge', 's2_mtci'];
var DW_DAY_SOURCE = ['water', 'trees', 'built', 'bare'];
var DW_DAY_BANDS = [
  'dw_day_water_prob', 'dw_day_trees_prob',
  'dw_day_built_prob', 'dw_day_bare_prob'
];

function prepS2(img) {
  var scl = img.select('SCL');
  var clear = img.select('cs_cdf').gte(CLOUD_SCORE_THRESHOLD)
    .and(scl.neq(1)).and(scl.neq(3)).and(scl.neq(8)).and(scl.neq(9))
    .and(scl.neq(10)).and(scl.neq(11));
  var r = img.select(
    ['B4', 'B5', 'B6', 'B8A'], ['red', 're1', 're2', 'nirn']
  ).multiply(0.0001).updateMask(clear);
  var reflectanceOk = r.reduce(ee.Reducer.min()).gte(0)
    .and(r.reduce(ee.Reducer.max()).lte(1.6));
  r = r.updateMask(reflectanceOk);
  var red = r.select('red');
  var re1 = r.select('re1');
  var re2 = r.select('re2');
  var nirn = r.select('nirn');
  var ndre1 = stableHlsRatio(nirn.subtract(re1), nirn.add(re1), 's2_ndre1');
  var ndre2 = stableHlsRatio(nirn.subtract(re2), nirn.add(re2), 's2_ndre2');
  var cireRaw = nirn.divide(re1).subtract(1);
  var cire = cireRaw.updateMask(re1.abs().gt(0.02))
    .updateMask(cireRaw.gte(-1).and(cireRaw.lte(15)))
    .rename('s2_ci_rededge');
  var mtciDen = re1.subtract(red);
  var mtciRaw = re2.subtract(re1).divide(mtciDen);
  var mtci = mtciRaw.updateMask(mtciDen.abs().gt(0.02))
    .updateMask(mtciRaw.gte(-20).and(mtciRaw.lte(20)))
    .rename('s2_mtci');
  return ee.Image.cat([ndre1, ndre2, cire, mtci])
    .copyProperties(img, ['system:time_start']);
}

var s2Raw = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(ROI);
var cloudScore = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
  .filterBounds(ROI);
var s2 = s2Raw.linkCollection(cloudScore, ['cs_cdf']).map(prepS2);
var dynamicWorld = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1').filterBounds(ROI);

function dwForDay(start, end) {
  var col = dynamicWorld.filterDate(start, end)
    .select(DW_DAY_SOURCE, DW_DAY_BANDS);
  var image = ee.Image(ee.Algorithms.If(
    col.size().gt(0), col.mean().select(DW_DAY_BANDS), emptyImage(DW_DAY_BANDS)
  ));
  var valid = ee.Image(ee.Algorithms.If(
    col.size().gt(0),
    col.select(DW_DAY_BANDS[0]).count().rename('dw_day_valid_obs'),
    emptyImage(['dw_day_valid_obs'])
  ));
  return image.addBands(valid);
}

// -----------------------------------------------------------------------------
// Sentinel-1 dual-polarization SAR, ascending/descending kept separate.
// -----------------------------------------------------------------------------

var S1_ASC_BANDS = [
  's1_vv_db_asc', 's1_vh_db_asc', 's1_vh_minus_vv_db_asc',
  's1_rvi_dualpol_asc'
];
var S1_DESC_BANDS = [
  's1_vv_db_desc', 's1_vh_db_desc', 's1_vh_minus_vv_db_desc',
  's1_rvi_dualpol_desc'
];

function prepS1(img, suffix) {
  var angle = img.select('angle');
  var valid = angle.gt(30).and(angle.lt(46))
    .and(img.select('VV').gt(-35)).and(img.select('VV').lt(5))
    .and(img.select('VH').gt(-40)).and(img.select('VH').lt(0));
  var vvDb = img.select('VV').updateMask(valid).rename('s1_vv_db_' + suffix);
  var vhDb = img.select('VH').updateMask(valid).rename('s1_vh_db_' + suffix);
  var ratioDb = img.select('VH').subtract(img.select('VV')).updateMask(valid)
    .rename('s1_vh_minus_vv_db_' + suffix);
  var vv = ee.Image.constant(10).pow(img.select('VV').divide(10))
    .updateMask(valid);
  var vh = ee.Image.constant(10).pow(img.select('VH').divide(10))
    .updateMask(valid);
  var rvi = vh.multiply(4).divide(vv.add(vh))
    .rename('s1_rvi_dualpol_' + suffix);
  return ee.Image.cat([vvDb, vhDb, ratioDb, rvi])
    .copyProperties(img, ['system:time_start']);
}

var s1Base = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(ROI)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.eq('resolution_meters', 10))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'));

function s1OrbitForDay(start, end, pass, suffix, bands, countName) {
  var col = s1Base.filterDate(start, end)
    .filter(ee.Filter.eq('orbitProperties_pass', pass))
    .map(function (img) { return prepS1(img, suffix); });
  var image = ee.Image(ee.Algorithms.If(
    col.size().gt(0), col.median().select(bands), emptyImage(bands)
  ));
  var valid = ee.Image(ee.Algorithms.If(
    col.size().gt(0), col.select(bands[0]).count().rename(countName),
    emptyImage([countName])
  ));
  return image.addBands(valid);
}

// -----------------------------------------------------------------------------
// Landsat surface temperature at its 100 m thermal information scale.
// -----------------------------------------------------------------------------

var LANDSAT_BANDS = [
  'landsat_lst_c', 'landsat_lst_uncertainty_k', 'landsat_emissivity'
];

function prepLandsat(img) {
  var clear = img.select('QA_PIXEL').bitwiseAnd(63).eq(0)
    .and(img.select('QA_RADSAT').eq(0));
  var lst = img.select('ST_B10').multiply(0.00341802).add(149)
    .subtract(273.15).rename('landsat_lst_c');
  var uncertainty = img.select('ST_QA').multiply(0.01)
    .rename('landsat_lst_uncertainty_k');
  var emissivity = img.select('ST_EMIS').multiply(0.0001)
    .rename('landsat_emissivity');
  var physical = lst.gte(-80).and(lst.lte(70))
    .and(uncertainty.gte(0)).and(uncertainty.lte(20))
    .and(emissivity.gte(0.6)).and(emissivity.lte(1));
  return ee.Image.cat([lst, uncertainty, emissivity])
    .updateMask(clear.and(physical))
    .copyProperties(img, ['system:time_start']);
}

var landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .filterBounds(ROI).filter(ee.Filter.eq('PROCESSING_LEVEL', 'L2SP'))
  .map(prepLandsat)
  .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
    .filterBounds(ROI).filter(ee.Filter.eq('PROCESSING_LEVEL', 'L2SP'))
    .map(prepLandsat));

// -----------------------------------------------------------------------------
// MAIAC AOD/water vapor, MODIS day/night LST and daylight shortwave radiation.
// -----------------------------------------------------------------------------

var MAIAC_BANDS = ['maiac_aod_055', 'maiac_column_wv_cm'];
var MODIS_LST_BANDS = [
  'modis_terra_lst_day_c', 'modis_terra_lst_night_c',
  'modis_aqua_lst_day_c', 'modis_aqua_lst_night_c'
];
var MODIS_DSR_BANDS = ['modis_dsr_daylight_mean_wm2'];

function prepMaiac(img) {
  var qa = img.select('AOD_QA');
  var clearBest = qa.bitwiseAnd(7).eq(1)
    .and(qa.rightShift(8).bitwiseAnd(15).eq(0));
  var aod = img.select('Optical_Depth_055').multiply(0.001)
    .rename('maiac_aod_055');
  var wv = img.select('Column_WV').multiply(0.001)
    .rename('maiac_column_wv_cm');
  var physical = aod.gte(-0.1).and(aod.lte(8))
    .and(wv.gte(0)).and(wv.lte(30));
  return ee.Image.cat([aod, wv]).updateMask(clearBest.and(physical))
    .copyProperties(img, ['system:time_start']);
}

function modisLstMask(qa) {
  return qa.bitwiseAnd(3).lte(1)
    .and(qa.rightShift(2).bitwiseAnd(3).eq(0))
    .and(qa.rightShift(6).bitwiseAnd(3).lte(1));
}

function prepModisLst(img, prefix) {
  var day = img.select('LST_Day_1km').multiply(0.02).subtract(273.15)
    .updateMask(modisLstMask(img.select('QC_Day')))
    .rename(prefix + '_lst_day_c');
  var night = img.select('LST_Night_1km').multiply(0.02).subtract(273.15)
    .updateMask(modisLstMask(img.select('QC_Night')))
    .rename(prefix + '_lst_night_c');
  day = day.updateMask(day.gte(-80).and(day.lte(70)));
  night = night.updateMask(night.gte(-80).and(night.lte(70)));
  return ee.Image.cat([day, night]).copyProperties(img, ['system:time_start']);
}

function prepDsr(img) {
  var quality = img.select('DSR_Quality').bitwiseAnd(3);
  var daylight = img.select([
    'GMT_0000_DSR', 'GMT_0300_DSR', 'GMT_0600_DSR', 'GMT_0900_DSR'
  ]).updateMask(quality.neq(0));
  daylight = daylight.updateMask(daylight.gte(0).and(daylight.lte(1400)));
  return daylight.reduce(ee.Reducer.mean())
    .rename('modis_dsr_daylight_mean_wm2')
    .copyProperties(img, ['system:time_start']);
}

var maiac = ee.ImageCollection('MODIS/061/MCD19A2_GRANULES')
  .filterBounds(ROI).map(prepMaiac);
var terra = ee.ImageCollection('MODIS/061/MOD11A1').filterBounds(ROI)
  .map(function (img) { return prepModisLst(img, 'modis_terra'); });
var aqua = ee.ImageCollection('MODIS/061/MYD11A1').filterBounds(ROI)
  .map(function (img) { return prepModisLst(img, 'modis_aqua'); });
var dsr = ee.ImageCollection('MODIS/062/MCD18A1').filterBounds(ROI).map(prepDsr);

// -----------------------------------------------------------------------------
// GPM IMERG V07. precipitationUncal is used for a strictly satellite-only model.
// -----------------------------------------------------------------------------

var GPM_BANDS = [
  'gpm_precip_uncal_d0_mm', 'gpm_precip_uncal_prev1d_mm',
  'gpm_precip_uncal_prev3d_mm', 'gpm_precip_uncal_prev7d_mm',
  'gpm_halfhour_obs_d0'
];
var gpm = ee.ImageCollection('NASA/GPM_L3/IMERG_V07').filterBounds(ROI);

function gpmAccum(start, end, outputName) {
  var col = gpm.filterDate(start, end).select('precipitationUncal');
  return ee.Image(ee.Algorithms.If(
    col.size().gt(0), col.sum().multiply(0.5).rename(outputName),
    emptyImage([outputName])
  ));
}

function gpmForDay(start, end) {
  var obs = gpm.filterDate(start, end).select('precipitationUncal');
  var obsCount = ee.Image(ee.Algorithms.If(
    obs.size().gt(0), obs.count().rename('gpm_halfhour_obs_d0'),
    emptyImage(['gpm_halfhour_obs_d0'])
  ));
  return gpmAccum(start, end, 'gpm_precip_uncal_d0_mm')
    .addBands(gpmAccum(start.advance(-1, 'day'), start,
      'gpm_precip_uncal_prev1d_mm'))
    .addBands(gpmAccum(start.advance(-3, 'day'), start,
      'gpm_precip_uncal_prev3d_mm'))
    .addBands(gpmAccum(start.advance(-7, 'day'), start,
      'gpm_precip_uncal_prev7d_mm'))
    .addBands(obsCount);
}

// -----------------------------------------------------------------------------
// Per-date multi-resolution extraction. Only HLS has a clearly labelled
// nearest-valid auxiliary group; every other dynamic product remains exact-day.
// -----------------------------------------------------------------------------

var HLS_OUT = HLS_EXACT_BANDS.concat(['hls_exact_valid_obs'])
  .concat(HLS_NEAR_BANDS)
  .concat([
    'hls_near_signed_gap_days_mean', 'hls_near_abs_gap_days_mean',
    'hls_near_candidate_obs', 'hls_exact_available', 'hls_near_available',
    'hls_gapfill_used', 'hls_near_mean_gap_within_16d',
    'hls_near_mean_gap_within_32d', 'hls_max_gap_days_setting'
  ]);
var S2_OUT = S2_BANDS.concat(['s2_valid_obs']);
var DW_DAY_OUT = DW_DAY_BANDS.concat(['dw_day_valid_obs']);
var S1_OUT = S1_ASC_BANDS.concat(['s1_valid_obs_asc'])
  .concat(S1_DESC_BANDS).concat(['s1_valid_obs_desc']);
var LANDSAT_OUT = LANDSAT_BANDS.concat(['landsat_valid_obs']);
var MAIAC_OUT = MAIAC_BANDS.concat(['maiac_valid_obs']);
var MODIS_LST_OUT = MODIS_LST_BANDS
  .concat(['modis_terra_valid_obs', 'modis_aqua_valid_obs']);
var MODIS_DSR_OUT = MODIS_DSR_BANDS.concat(['modis_dsr_valid_obs']);
var ALL_FEATURE_FIELDS = STATIC_TERRAIN_BANDS.concat(STATIC_DW_BANDS)
  .concat(HLS_OUT).concat(S2_OUT).concat(DW_DAY_OUT).concat(S1_OUT)
  .concat(LANDSAT_OUT).concat(MAIAC_OUT).concat(MODIS_LST_OUT)
  .concat(MODIS_DSR_OUT).concat(GPM_BANDS).concat(['gpm_day_complete_48']);

function processDate(dateString) {
  dateString = ee.String(dateString);
  var rowsForDate = monitor.filter(ee.Filter.eq('date', dateString));
  var first = ee.Feature(rowsForDate.first());
  var start = ee.Date(ee.Number.parse(ee.String(first.get('local_day_start_utc_ms'))));
  var end = ee.Date(ee.Number.parse(ee.String(first.get('local_day_end_utc_ms'))));

  var rows = rowsForDate.map(spatialSupport);
  rows = reduceMean(hlsForTargetDay(start, end), rows, 30);
  rows = rows.map(function (f) {
    var exactMissing = ee.Algorithms.IsEqual(f.get('hls_exact_ndvi'), null);
    var nearMissing = ee.Algorithms.IsEqual(f.get('hls_near_ndvi'), null);
    var exactAvailable = ee.Number(ee.Algorithms.If(exactMissing, 0, 1));
    var nearAvailable = ee.Number(ee.Algorithms.If(nearMissing, 0, 1));
    var gap = ee.Number(ee.Algorithms.If(
      nearMissing, 9999, f.get('hls_near_abs_gap_days_mean')
    ));
    return f.set({
      hls_exact_available: exactAvailable,
      hls_near_available: nearAvailable,
      hls_gapfill_used: nearAvailable.multiply(ee.Number(1).subtract(exactAvailable)),
      hls_near_mean_gap_within_16d: nearAvailable.multiply(gap.lte(16)),
      hls_near_mean_gap_within_32d: nearAvailable.multiply(gap.lte(32)),
      hls_max_gap_days_setting: HLS_MAX_GAP_DAYS
    });
  });
  rows = reduceMean(
    safeDailyMedian(s2, start, end, S2_BANDS, 's2_valid_obs'), rows, 20
  );
  rows = reduceMean(dwForDay(start, end), rows, 10);

  var s1Asc = s1OrbitForDay(
    start, end, 'ASCENDING', 'asc', S1_ASC_BANDS, 's1_valid_obs_asc'
  );
  var s1Desc = s1OrbitForDay(
    start, end, 'DESCENDING', 'desc', S1_DESC_BANDS, 's1_valid_obs_desc'
  );
  rows = reduceMean(s1Asc.addBands(s1Desc), rows, 10);

  rows = reduceMean(
    safeDailyMedian(landsat, start, end, LANDSAT_BANDS, 'landsat_valid_obs'),
    rows, 100
  );
  rows = rows.map(coarsePointSupport);
  rows = reduceMean(
    safeDailyMedian(maiac, start, end, MAIAC_BANDS, 'maiac_valid_obs'),
    rows, 1000
  );
  rows = reduceMean(
    safeDailyMedian(terra, start, end,
      ['modis_terra_lst_day_c', 'modis_terra_lst_night_c'],
      'modis_terra_valid_obs'),
    rows, 1000
  );
  rows = reduceMean(
    safeDailyMedian(aqua, start, end,
      ['modis_aqua_lst_day_c', 'modis_aqua_lst_night_c'],
      'modis_aqua_valid_obs'),
    rows, 1000
  );
  rows = reduceMean(
    safeDailyMedian(dsr, start, end, MODIS_DSR_BANDS, 'modis_dsr_valid_obs'),
    rows, 1000
  );
  rows = reduceMean(gpmForDay(start, end), rows, 11132);
  return rows.map(function (f) {
    var obs = f.get('gpm_halfhour_obs_d0');
    var obsNumber = ee.Number(ee.Algorithms.If(
      ee.Algorithms.IsEqual(obs, null), -1, obs
    ));
    return f.set('gpm_day_complete_48', obsNumber.eq(48));
  });
}

// End dates are exclusive. These six tasks cover every date in the uploaded table.
var CHUNKS = [
  {start: '2024-02-03', end: '2024-04-01', tag: '2024Q1_partial'},
  {start: '2024-04-01', end: '2024-07-01', tag: '2024Q2'},
  {start: '2024-07-01', end: '2024-10-01', tag: '2024Q3'},
  {start: '2024-10-01', end: '2025-01-01', tag: '2024Q4'},
  {start: '2025-01-01', end: '2025-04-01', tag: '2025Q1'},
  {start: '2025-04-01', end: '2025-05-16', tag: '2025Q2_partial'}
];

print('All input records (expected 5998)', monitor.size());
print('HLS nearest-valid maximum gap (days)', HLS_MAX_GAP_DAYS);
print('This script creates six quarterly CSV tasks.');
Map.centerObject(monitor.distinct(['area']), 9);
Map.addLayer(monitor.distinct(['area']), {color: 'red'}, 'Monitoring locations');

CHUNKS.forEach(function (chunk) {
  var subset = monitor.filter(ee.Filter.gte('date', chunk.start))
    .filter(ee.Filter.lt('date', chunk.end));
  var dates = ee.List(subset.aggregate_array('date')).distinct().sort();
  var output = ee.FeatureCollection(dates.map(processDate)).flatten();
  var exportName = EXPORT_PREFIX + '_' + chunk.tag + '_b' + BUFFER_M + 'm';
  Export.table.toDrive({
    collection: output,
    description: exportName,
    folder: EXPORT_FOLDER,
    fileNamePrefix: exportName,
    fileFormat: 'CSV',
    selectors: BASE_FIELDS.concat(ALL_FEATURE_FIELDS)
  });
});
