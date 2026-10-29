# Trend analysis of historical NAI estimates

This directory contains exploratory pixel-level trends in model-estimated negative air ion (NAI) concentrations for seven urban parks. The estimates are historical reconstructions from a model fitted to 2024–2025 observations, not independent measurements of NAI concentrations during 2016–2024.

## Temporal design and pixel coverage

The analysis uses nine seasonal years labelled 2016–2024. Each seasonal year runs from March through the following February: Q1 is March–May, Q2 June–August, Q3 September–November, and Q4 December–February. The complete period therefore spans March 2016–February 2025.

Trend tests use the 2,865 pixels with finite annual estimates in all nine seasonal years, giving nine values per tested pixel. This is the balanced annual panel. It does not require complete coverage of all 36 underlying seasons: the annual aggregation uses the available seasonal estimates.

The separate core-panel summaries use pixels with valid estimates in at least 90% of the 36 seasons. This coverage criterion differs from the nine-year balanced annual panel and must not be treated as the same mask.

## Trend estimation and multiple testing

The Theil–Sen slope is the median of the 36 pairwise slopes between annual estimates, expressed in ions cm^-3 year^-1. The Mann–Kendall test uses a two-sided normal approximation, with variance correction for ties and continuity correction of the test statistic. The slope confidence-limit rasters retain the original rank-based calculations.

Benjamini–Hochberg (BH) adjustment is applied across the tested pixels at a nominal level of 0.05. For m tests, sort their p-values as p(1) through p(m), and choose the largest rank k satisfying p(k) <= 0.05 k/m. The hypotheses at ranks 1 through k are then rejected. Comparing only the smallest p-value with 0.05/m is insufficient: failure at rank 1 does not exclude rejection at a larger rank.

No pixel in the current outputs remains significant after BH adjustment. This is a non-detection in a short, model-derived series; it does not establish that true historical NAI concentrations were constant. Unadjusted significant-pixel counts are descriptive. Binomial count comparisons without an adequate treatment of spatial dependence, previously reported detection thresholds, and assertions that the slope field represents noise are not used here as inferential evidence.

## Files and preservation

- `all_parks/` contains the slope, lower and upper 95% confidence limits, Mann–Kendall S, Z and p, and unadjusted and BH-adjusted trend-class rasters. `by_park/` contains their park-specific counterparts.
- `park_trend_summary.csv`, `park_pixel_slope_distribution.csv` and `all_parks_trend_summary.csv` provide the corresponding descriptive summaries.
- Associated tables are `../park_seasonal_means.csv`, `../park_seasonal_means_by_year.csv`, `../park_annual_means.csv`, `../park_annual_means_core_panel.csv` and `../seasonal_2016_2024/derived/park_seasonal_amplitude.csv`.

The derived seasonal-amplitude rasters are the Q2 mean minus the Q4 mean. In the amplitude table, `amplitude_relative_to_2400_pct` is 100 times the mean amplitude divided by a fixed reference of 2,400 ions cm^-3; it is not divided by each park's own annual mean.

The archived park-specific amplitude rasters for Baiyun Mountain, Dafushan, Guangzhou Haizhu, Luhu, Yuexiu and Zhujiang retain an older full-domain grid origin that differs from their corresponding seasonal source rasters. Their pixel values equal Q2 minus Q4, but these six archived files should not be used for spatial overlay without regeneration. The core `aggregate` command regenerates them with the corresponding seasonal raster's correct transform in the external reproduction directory. The archived raster bytes and transforms were preserved during filename translation. This issue does not affect the supplied annual-mean and trend rasters, which were checked against the current clipping workflow.

Trend classes are -1 for a significant decrease, 0 for a non-significant result, and 1 for a significant increase; classification NoData is -99. Raster CRS, transforms, pixel values, masks and NoData must be preserved. A read-only audit of all 222 GeoTIFFs in the clipped-results archive found no Chinese in persistent dataset or band metadata, so English filenames require no raster-data rewriting. Numerical CSV values are unchanged.

This English methods note, deployed as `README_trend_analysis.md`, replaces the earlier Chinese internal note to correct its BH interpretation. The original note is retained outside the submission directory for provenance.
