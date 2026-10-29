"""Resolve mapping inputs and keep every new output outside the study package."""
from pathlib import Path
import importlib.util
import os

CODE_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = CODE_DIR.parents[1]
# Raster products are archived separately; set NAI_RASTER_DIR to their location if
# they are not restored under results/raster/.
FROZEN = Path(os.environ.get("NAI_RASTER_DIR", PACKAGE_ROOT / "results" / "raster")).expanduser().resolve()
PARKS = PACKAGE_ROOT / "data" / "spatial" / "urban_parks.shp"
DEFAULT_MODEL = PACKAGE_ROOT / "models" / "production" / "m4_urban_model.joblib"
SEASONS = tuple(f"{year}Q{quarter}" for year in range(2016, 2025) for quarter in range(1, 5))


def input_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PACKAGE_ROOT / path).resolve()


def is_within(path, parent):
    def normalized(value):
        text = os.path.normcase(str(Path(value).resolve()))
        if text.startswith("\\\\?\\unc\\"):
            text = "\\\\" + text[8:]
        elif text.startswith("\\\\?\\"):
            text = text[4:]
        return Path(text)
    try:
        normalized(path).relative_to(normalized(parent))
        return True
    except ValueError:
        return False


def output_path(value):
    path = input_path(value)
    if is_within(path, PACKAGE_ROOT):
        raise ValueError("New outputs must be outside the study package: " + str(path))
    return path


def reproduction_root(value=None):
    given = value or os.environ.get("NAI_REPRODUCED_DIR")
    return output_path(given) if given else PACKAGE_ROOT.parent / "NAI_reproduced"


REPRODUCED = reproduction_root() / "mapping"


def preferred_stage(name):
    fresh = REPRODUCED / name
    return fresh if fresh.is_dir() and any(fresh.glob("*.tif")) else FROZEN / name


def configure_gdal_data():
    spec = importlib.util.find_spec("osgeo")
    if spec is None or not spec.submodule_search_locations:
        return
    osgeo_dir = Path(next(iter(spec.submodule_search_locations)))
    candidates = [osgeo_dir / "data", osgeo_dir.parent.parent.parent / "share"]
    for variable, folder, required in [
        ("PROJ_DATA", "proj", "proj.db"),
        ("PROJ_LIB", "proj", "proj.db"),
        ("GDAL_DATA", "gdal", "gdalvrt.xsd"),
    ]:
        if os.environ.get(variable):
            continue
        for base in candidates:
            location = base / folder
            if (location / required).is_file():
                os.environ[variable] = str(location)
                break


def select_seasons(value=None):
    wanted = SEASONS if value is None else tuple(x.strip() for x in value.split(",") if x.strip())
    invalid = [x for x in wanted if x not in SEASONS]
    if invalid or not wanted:
        raise ValueError("Select seasons only from 2016Q1 through 2024Q4: " + repr(invalid))
    return tuple(dict.fromkeys(wanted))


PARK_NAMES = {'珠江公园': 'Zhujiang Park', '越秀公园': 'Yuexiu Park', '麓湖公园': 'Luhu Park', '广东花都湖国家湿地公园': 'Huadu Lake National Wetland Park', '海珠湿地公园': 'Guangzhou Haizhu National Wetland Park', '白云山': 'Baiyun Mountain Scenic Area', '大夫山森林公园': 'Dafushan Forest Park'}


def park_label(value):
    if value in PARK_NAMES:
        return PARK_NAMES[value]
    if value in PARK_NAMES.values():
        return value
    raise ValueError("Unknown study park name: " + str(value))


def park_slug(value):
    return park_label(value).replace(" ", "_")
