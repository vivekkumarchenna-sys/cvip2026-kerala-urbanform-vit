# Data Access — verified working (2026-08-08, no authentication required)

## 1. Sentinel-2 L2A (imagery)
- **Source:** AWS Earth Search STAC — `https://earth-search.aws.element84.com/v1`
- **Collection:** `sentinel-2-l2a`
- **Access:** anonymous S3 (COGs); read via `pystac-client` + `rasterio`
  (`AWS_NO_SIGN_REQUEST=YES`).
- **Verified:** 5 low-cloud (<8%) scenes over Kochi, Jan–Apr 2024; read a 256×256
  window of B04 (10980×10980, EPSG:32643, uint16). Best: S2A_43PFM_20240311 (0.8% cloud).
- **Bands used:** blue(B2) green(B3) red(B4) rededge1-3(B5-7) nir(B8) nir08(B8A)
  swir16(B11) swir22(B12) + scl (cloud mask).

## 2. ESA WorldCover 2021 v200 (labels)
- **Source:** AWS open bucket `esa-worldcover` (eu-central-1), anonymous.
- **Tile for Kerala cities:** `ESA_WorldCover_10m_2021_v200_N09E075_Map.tif`
  (covers ~9–12°N, 75–78°E → Kochi, TVM, Kozhikode).
- **Verified:** read Kochi window; class mix Water 46.1%, Built-up 27.4%,
  Tree 18.8%, Grass 3.4%, Crop 1.6%, Wetland 1.2%, Mangrove 1.0%, Bare 0.2%.
- **Classes:** 10 Tree, 20 Shrub, 30 Grass, 40 Crop, 50 Built-up, 60 Bare,
  70 Snow, 80 Water, 90 Wetland, 95 Mangrove, 100 Moss.

## 3. EuroSAT (benchmark, labeled)
- **Source:** HuggingFace `blanchon/EuroSAT_MSI` (13-band) and `blanchon/EuroSAT_RGB`.
- **Verified:** dataset metadata reachable (11 / 6 files).
- **Content:** 27,000 Sentinel-2 patches (64×64), 10 LULC classes.

## City bounding boxes (lon_min, lat_min, lon_max, lat_max)
- Kochi:              76.20, 9.88, 76.38, 10.05
- Thiruvananthapuram: 76.88, 8.44, 77.02, 8.58
- Kozhikode:          75.74, 11.20, 75.86, 11.33

## Notes
- Dry season (Dec–Apr) required for low cloud in Kerala (SW monsoon Jun–Sep).
- For 2016 baseline, use `sentinel-2-l1c`/`l2a` availability from ~2016; if L2A
  sparse, fall back to 2017 or L1C + simple atmospheric handling.
