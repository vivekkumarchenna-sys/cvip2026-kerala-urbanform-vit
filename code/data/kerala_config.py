"""Shared config for the Kerala Sentinel-2 pipeline."""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
KERALA_DIR = os.path.join(ROOT, "data", "kerala")
PROC_DIR = os.path.join(ROOT, "data", "processed")
os.makedirs(KERALA_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)

STAC_URL = "https://earth-search.aws.element84.com/v1"

# lon_min, lat_min, lon_max, lat_max  (urban + peri-urban extent)
CITIES = {
    "kochi":     [76.20, 9.88, 76.38, 10.05],
    "trivandrum":[76.88, 8.44, 77.02, 8.58],
    "kozhikode": [75.74, 11.20, 75.90, 11.35],
}

# Canonical display names — use the official spelling consistently everywhere.
CITY_DISP = {"kochi": "Kochi", "trivandrum": "Thiruvananthapuram", "kozhikode": "Kozhikode"}

# Sentinel-2 L2A asset names (earth-search v1) we pull, in fixed order.
# 10 spectral bands used for classification + SCL for cloud masking.
BANDS = ["blue", "green", "red", "rededge1", "rededge2",
         "rededge3", "nir", "nir08", "swir16", "swir22"]
BAND_CODES = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
SCL = "scl"

# Dry-season windows (SW monsoon Jun-Sep => use Dec-Apr for low cloud).
YEAR_WINDOWS = {
    2024: "2023-12-01/2024-04-30",
    2018: "2017-12-01/2018-04-30",
    2017: "2016-12-01/2017-04-30",
    2016: "2015-12-01/2016-04-30",
}

# SCL classes to KEEP (valid surface). Exclude: 0 nodata,1 sat,3 cloud-shadow,
# 8 cloud-med,9 cloud-high,10 cirrus.  Keep: 4 veg,5 bare,6 water,7 unclass,11 snow,2 dark.
SCL_KEEP = [2, 4, 5, 6, 7, 11]

# ESA WorldCover — tiles are 3x3 deg, named by SW corner (e.g. N09E075).
WORLDCOVER_BASE = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
WC_RAW = {10:"Tree",20:"Shrub",30:"Grass",40:"Crop",50:"Built-up",60:"Bare",
          70:"Snow",80:"Water",90:"Wetland",95:"Mangrove",100:"Moss"}


def worldcover_tiles(bbox):
    """Return WorldCover tile URLs covering a lon/lat bbox (3-deg tiles)."""
    import math
    lon0, lat0, lon1, lat1 = bbox
    urls = []
    for lat in {math.floor(la/3)*3 for la in (lat0, lat1)}:
        for lon in {math.floor(lo/3)*3 for lo in (lon0, lon1)}:
            ns = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
            ew = f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"
            urls.append(f"{WORLDCOVER_BASE}ESA_WorldCover_10m_2021_v200_{ns}{ew}_Map.tif")
    return urls


# Collapse WorldCover -> 4 planning-relevant, well-supported classes.
# 0 Built-up | 1 Tree | 2 Cropland/Grass (crop+grass+shrub+bare/open) | 3 Water (incl. wetland/mangrove)
# Wetland/Mangrove (WC 90/95) is folded into Water: it occurs almost only in Kochi
# (~0.4% overall) and as a separate class would be untrainable in leave-one-city-out.
WC_TO_CLASS = {50:0, 10:1, 40:2, 30:2, 20:2, 60:2, 70:2, 100:2, 80:3, 90:3, 95:3}
CLASS_NAMES = ["Built-up", "Tree", "Cropland/Grass", "Water"]
CLASS_COLORS = ["#e31a1c", "#1a9850", "#a6d96a", "#2c7fb8"]
