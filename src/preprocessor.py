""" preprocessor.py
"""

from __future__ import annotations

import logging.config
import os
from pathlib import Path
from typing import Optional

import cv2
import netCDF4 as nc
import numpy as np
import yaml

logging.config.fileConfig(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "logging.conf"),
    disable_existing_loggers=False,
)
logger = logging.getLogger(__name__)
CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "config", "config.yml"
)

with open(CONFIG_PATH, "r") as file:
    config = yaml.safe_load(file)["postprocessor"]

N_LINES_PER_SCAN = config["N_LINES_PER_SCAN"]
DNB_BASE_PATH = "/observation_data"
DNB_OBERVATIONS_PATH = f"{DNB_BASE_PATH}/DNB_observations"
DNB_QUALITY_PATH = f"{DNB_BASE_PATH}/DNB_quality_flags"

GEO_BASE_PATH = "/geolocation_data"
LATITUDE_PATH = f"{GEO_BASE_PATH}/latitude"
LONGITUDE_PATH = f"{GEO_BASE_PATH}/longitude"
LAND_WATER_MASK_PATH = f"{GEO_BASE_PATH}/land_water_mask"
MOONLIGHT_PATH = f"{GEO_BASE_PATH}/moon_illumination_fraction"

CLD_BASE_PATH = "/geophysical_data"
CLOUD_PATH = f"{CLD_BASE_PATH}/Clear_Sky_Confidence"

MODRAW_M10 = f"{DNB_BASE_PATH}/M10"
MOD_LAT = f"{GEO_BASE_PATH}/latitude"
MOD_LON = f"{GEO_BASE_PATH}/longitude"
MOD_SOLAR = f"{GEO_BASE_PATH}/solar_zenith"

NAV_BASE_PATH = "/navigation_data"
SCAN_ANGLE_PATH = f"{NAV_BASE_PATH}/att_ang_mid"  # pitch, yaw roll


def extract_data(
    dnb_file: Path,
    geo_file: Path,
    cloud_mask: Optional[Path] = None,
    modraw_path: Optional[Path] = None,
    modgeo_path: Optional[Path] = None,
) -> dict:
    """extracts data from nc file and builds dictionary of numpy arrays for each layer

    Note on the resizing of the scan line data:
    Scan Angle:
    att_ang_mid	Attitude angles at mid-time	float32(number_of_scans, vector_elements)
                    SDS Attributes:
                    Attribute Name	Format		Example
                    --------------	------		-------
                    long_name	string		"Attitude angles (roll, pitch, yaw) at EV mid-times"
                    units		string		"degrees"
                    _FillValue	float32		-999.9
                    valid_min	float32		-180.0
                    valid_max	float32		180.0

    Parameters
    ----------
    dnb_file : Path
        _description_
    geo_file : Path
        _description_
    cloud_mask : Path, optional
        cloud mask path, by default Path("not_available")

    Returns
    -------
    dict
        dictionary containing all data required for processing
    """
    dnb_array, dnb_metadata = get_layer(dnb_file, DNB_OBERVATIONS_PATH)
    dnb_quality_array, _ = get_layer(dnb_file, DNB_QUALITY_PATH)
    latitude_array, _ = get_layer(geo_file, LATITUDE_PATH)
    longitude_array, _ = get_layer(geo_file, LONGITUDE_PATH)
    land_sea_array, _ = get_layer(geo_file, LAND_WATER_MASK_PATH)
    moonlight_array, _ = get_layer(geo_file, MOONLIGHT_PATH)
    scan_angle_array, _ = get_layer(geo_file, SCAN_ANGLE_PATH)

    scan_angle = np.repeat(scan_angle_array, N_LINES_PER_SCAN).reshape(
        -1, scan_angle_array.shape[1]
    )  # reshape to match dimensionality of other pixel based arrays
    if cloud_mask:
        try:
            cloud_array_raw, _ = get_layer(cloud_mask, CLOUD_PATH)
            height, width = dnb_array.shape
            # Note that cloud array needs to be resized to dimensions of DNB data
            resized_cloud_array = cv2.resize(
                cloud_array_raw, (width, height), interpolation=cv2.INTER_AREA
            )
        except Exception:
            logger.exception(
                "Unable to read cloud data, creating zerod array, assuming frame is all clouds",
                exc_info=True,
            )
            resized_cloud_array = np.zeros(
                dnb_array.shape
            )  # == this is the equivalent of 100% cloud cover
    else:
        logger.warning(
            "Cloud data not provided, creating zerod array, assuming frame is all clouds",
            exc_info=True,
        )
        resized_cloud_array = np.zeros(
            dnb_array.shape
        )  # == this is the equivalent of 100% cloud cover

    m10_data = None
    if modraw_path:
        try:
            m10_data = {
                "m10": get_layer(modraw_path, MODRAW_M10),
                "lat": get_layer(modgeo_path, MOD_LAT),
                "lon": get_layer(modgeo_path, MOD_LON),
                "solar_zenith": get_layer(modgeo_path, MOD_SOLAR),
            }
        except Exception:
            logger.exception(
                "Exception mod data, defaulting to zero array", exc_info=True
            )

    return {
        "dnb": {
            "data": dnb_array,
            "metadata": dnb_metadata,
            "quality": dnb_quality_array,
        },
        "latitude": latitude_array,
        "longitude": longitude_array,
        "land_sea_mask": land_sea_array,
        "moonlight": moonlight_array,
        "cloud_mask": resized_cloud_array,
        "m10_band": m10_data,
        "scan_angle": scan_angle,
    }


def get_layer(filename: Optional[Path], layer: str) -> tuple[np.ndarray, dict]:
    """gets layer from netcdf dataset
    Note this function is applied to netcdf specifically

    Parameters
    ----------
    filename : Path
        nc filepath
    layer : str
        desired layer (see gdalinfo if you are not sure)

    Returns
    -------
    Tuple[np.ndarray, dict]
        data array, metadata for array
    """
    dataset = nc.Dataset(filename)
    layer_data = dataset[layer]
    data_array = layer_data[:].data
    layer_metadata = layer_data.__dict__
    global_metadata = dataset.__dict__
    global_metadata.update(layer_metadata)

    return data_array, global_metadata
