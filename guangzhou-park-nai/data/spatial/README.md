# Park boundary data

`urban_parks.shp` (and its `.shx`, `.dbf`, `.prj`, `.sbn`, `.sbx` companions) contains the seven urban park polygons used to clip the NAI raster products.

- **CRS**: WGS 84 (EPSG:4326), geographic coordinates.
- **Attributes**: `osm_id`, `code`, `fclass`, `name` — the schema of an OpenStreetMap land-use extract.
- **Use in the study**: clipping and summarising `results/raster/` to park boundaries; see `code/mapping/run_mapping.py clip`.

## Attribution and licence

This layer is derived from OpenStreetMap and is distributed under the **Open Database License (ODbL)**:

> © OpenStreetMap contributors, available under the Open Database License (ODbL).
> https://www.openstreetmap.org/copyright

If you redistribute this layer or a derived database, keep this attribution and release the derived database under the same licence. The CC BY 4.0 terms in `LICENSE-DATA` do **not** apply to this shapefile.

## Note on metadata

The ArcGIS item metadata file (`urban_parks.shp.xml`) is not included: it embedded a local file path rather than information about the data. Geometry, attributes and projection are unchanged; only that auxiliary metadata file was dropped.
