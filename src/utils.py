import io
import json
import logging.config
import math
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import requests
import yaml
from fastapi.openapi.utils import get_openapi
from google.cloud.storage import Client as StorageClient
from matplotlib import cm
from PIL import Image, ImageFilter
from pydantic import BaseModel
from skimage import draw

logging.config.fileConfig(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "logging.conf"),
    disable_existing_loggers=False,
)
logger = logging.getLogger(__name__)
CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "config", "config.yml"
)
with open(CONFIG_PATH, "r") as file:
    config = yaml.safe_load(file)

utils_config = config["utils"]
model_config = config["model"]

VIIRS_PIXEL_SIZE = utils_config["VIIRS_PIXEL_SIZE"]
CHIP_IMG_DIM = (utils_config["CHIP_IMG_DIM_HEIGHT"], utils_config["CHIP_IMG_DIM_WIDTH"])
OCEAN_VALS = utils_config["OCEAN_VALS"]
IMAGE_CHIP_SIZE = utils_config["IMAGE_CHIP_SIZE"]
AURORA_LAT_THRESHOLD = utils_config["AURORA_LAT_THRESHOLD"]
OUTLIER_THRESHOLD_NW = model_config["OUTLIER_THRESHOLD_NW"]
LIGHTNING_WIDTH = utils_config["LIGHTNING_WIDTH"]
LIGHTNING_HEIGHT = utils_config["LIGHTNING_HEIGHT"]
MAX_LIGHTNING_LENGTH = utils_config["MAX_LIGHTNING_LENGTH"]
MIN_LIGHTNING_LENGTH = utils_config["MIN_LIGHTNING_LENGTH"]
AURORA_BLUR_KERNEL_SIZE = utils_config["AURORA_BLUR_KERNEL_SIZE"]
DILATION_KERNEL_SIZE = utils_config["DILATION_KERNEL_SIZE"]
HIGH_AURORA = utils_config["HIGH_AURORA"]
MID_AURORA = utils_config["MID_AURORA"]
LOW_AURORA = utils_config["LOW_AURORA"]
GAS_FLARE_THRESHOLD = utils_config["GAS_FLARE_THRESHOLD"]
CLOUD_EROSION_KERNEL_DIM = utils_config["CLOUD_EROSION_KERNEL_DIM"]

TOKEN = os.environ.get("EARTHDATA_TOKEN")


class GeoPoint(BaseModel):
    """A class to represent a point on the earth's surface"""

    lat: float
    lon: float


class CorrelatedDetection(TypedDict):
    id: str
    ts: str
    lon: float
    lat: float
    length: Optional[str]


def upload_image(
    gcp_bucket: str,
    chips_dict: dict,
    destination_path: str,
    image_name: Path,
) -> dict:
    """uploads image to cloud

    Parameters
    ----------
    gcp_bucket : str
    chips_dict : Dict
    destination_path : str
    image_name: str

    Returns
    -------
    chips_dict
        updated chips dict with locations of remote storage path
    """
    storage_client = StorageClient()
    bucket = storage_client.bucket(gcp_bucket)
    incomplete_chips = []
    for idx, chip_info in chips_dict.items():
        try:
            lat = chip_info["latitude"]
            lon = chip_info["longitude"]
            filename = f"{lat}_{lon}.jpeg"
            destination_blob_name = os.path.join(
                destination_path,
                image_name.stem,
                "image_chips",
                os.path.basename(filename),
            )
            blob = bucket.blob(destination_blob_name)

            normalized_chip = cv2.normalize(
                chip_info["chip"], chip_info["chip"], 0, 255, cv2.NORM_MINMAX
            )

            resized_img = cv2.resize(
                normalized_chip, CHIP_IMG_DIM, interpolation=cv2.INTER_AREA
            )

            img_encoded = cv2.imencode(".jpeg", resized_img)[1].tobytes()
            blob.upload_from_string(img_encoded)
            chip_info["meters_per_pixel"] = (
                normalized_chip.shape[0] / resized_img.shape[0]
            ) * VIIRS_PIXEL_SIZE

            logger.debug(f"File {filename} uploaded to {destination_blob_name}.")
            chip_info["path"] = destination_blob_name

        except Exception as e:
            incomplete_chips.append(idx)
            chip_info["path"] = "None"
            logger.exception(str(e), exc_info=True)

    [chips_dict.pop(idx) for idx in incomplete_chips]

    return chips_dict


def get_average_moonlight(dnb_dataset: Dict) -> float:
    """average moonlight within frame

    Parameters
    ----------
    dnb_dataset : Dict

    Returns
    -------
    float
        average moonlight, ranges from 0 to 100
    """

    try:
        average_moonlight = np.nanmean(dnb_dataset["moonlight"])
    except Exception:
        logger.exception("Exception extracting average moonlight", exc_info=True)
        average_moonlight = np.nan
    return average_moonlight


def calculate_e2e_cog(
    start_point: GeoPoint, end_point: GeoPoint
) -> Tuple[float, float]:
    """Calculate great circle distance, forward and backward azimuth

    Parameters
    ----------
    start_point : GeoPoint
    end_point : GeoPoint

    Returns
    -------
    Tuple[float, float]
        fwd_azimuth, distance in km
    """
    geodesic = pyproj.Geod(ellps="WGS84")
    fwd_azimuth, _, meters = geodesic.inv(
        start_point.lon, start_point.lat, end_point.lon, end_point.lat
    )
    kilometers = meters * 1e-3
    return fwd_azimuth, kilometers


def get_frame_extents(dnb_dataset: dict) -> List[List[float]]:
    """gets corner coordinates of the frame from lat and lon arrays

    Parameters
    ----------
    dnb_dataset : dict
        _description_

    Returns
    -------
    List[List[float]]
        4 corner coordinates of frame + origin ordered lon, lat
    """
    latitude = dnb_dataset["latitude"]
    longitude = dnb_dataset["longitude"]
    lon_corners = longitude[[0, 0, -1, -1, 0], [0, -1, -1, 0, 0]]
    lat_corners = latitude[[0, 0, -1, -1, 0], [0, -1, -1, 0, 0]]
    frame_extents = [[lon, lat] for lon, lat in zip(lon_corners, lat_corners)]
    return frame_extents


def download_dnb_image(
    gcp_bucket: str, filename: str, image_dir: str, dest_dir: str
) -> Path:
    """Downloads VIIRS image from GCP bucket to local storage

    Parameters
    ----------
    info : VVDRequest
        Request containing source image info
    destination_file_name : str
        path to local directory to store downloaded image

    Returns
    -------
    Path
        path to downloaded image
    """
    src_path = os.path.join(image_dir, filename)
    dest_path = os.path.join(dest_dir, filename)

    storage_client = StorageClient()
    bucket = storage_client.bucket(gcp_bucket)
    blob = bucket.blob(src_path)
    blob.download_to_filename(dest_path)
    logger.debug(f"Copied {os.path.join(gcp_bucket, src_path)} to {dest_path}.")

    return Path(dest_path)


def download_phys_image(
    gcp_bucket: str, filename: str, image_dir: str, dest_dir: str
) -> Path:
    """Downloads geo images from GCP bucket to local storage

    This function identifies the name of the supporting files from the DNB source file
    The supporting files have the same time of collection but a different collection
    number (CCC) as well as a different prefix. Note this function uses list_blobs.

    file_naming_convention = "CLDMSK_L2_VIIRS_SNPP.A2023008.2124.001.nrt.nc

    Parameters
    ----------
    info : VVDRequest
        Request containing source image info
    destination_file_name : str
        path to local directory to store downloaded image

    Returns
    -------
    Path
        path to downloaded geo image
    """
    SNPP_PRODUCT_NAME = "VNP02DNB_NRT"
    NOAA20_PRODUCT_NAME = "VJ102DNB_NRT"
    if "VNP02" in filename:
        temp = filename.replace(SNPP_PRODUCT_NAME, "CLDMSK_L2_VIIRS_SNPP")
        phys_dir = image_dir.replace(SNPP_PRODUCT_NAME, "CLDMSK_L2_VIIRS_SNPP_NRT")

    elif "VJ102" in filename:
        temp = filename.replace(NOAA20_PRODUCT_NAME, "CLDMSK_L2_VIIRS_NOAA20")
        phys_dir = image_dir.replace(NOAA20_PRODUCT_NAME, "CLDMSK_L2_VIIRS_NOAA20_NRT")

    temp_list = temp.rsplit(".")
    phys_filename_prefix = ".".join(temp_list[0:3])

    storage_client = StorageClient()
    bucket = storage_client.bucket(gcp_bucket)
    phys_path_prefix = os.path.join(phys_dir, phys_filename_prefix)
    for blob in bucket.list_blobs(prefix=phys_path_prefix):
        src_path = blob.name
        phys_filename = src_path.rsplit("/")[-1]
        blob = bucket.blob(src_path)
        dest_path = os.path.join(dest_dir, phys_filename)
        blob.download_to_filename(dest_path)
    logger.debug(f"Copied {os.path.join(gcp_bucket, src_path)} to {dest_path}")

    return Path(dest_path)


def download_mod_images(
    gcp_bucket: str, dnb_file: str, geo_file: str, image_dir: str, dest_dir: str
) -> Tuple[Path, Path]:
    """Downloads geo images from GCP bucket to local storage

    This function identifies MOD data from the DNB filename.
    Note this function uses list_blobs which is not free.

    # filename is VJ102DNB_NRT.A2023031.0130.021.2023031034239.nc
    # mod filenames are:


    file_naming_convention = "*MOD_NRT.AYYYYDDD.HHMM.CCC.nc"

    Parameters
    ----------
    info : VVDRequest
        Request containing source image info
    destination_file_name : str
        path to local directory to store downloaded image

    Returns
    -------
    Path
        path to downloaded geo image
    """
    modraw_file = dnb_file.replace("DNB", "MOD")
    modraw_dir = image_dir.replace("DNB", "MOD")
    modgeo_file = geo_file.replace("DNB", "MOD")
    modgeo_dir = image_dir.replace("02DNB", "03MOD")

    modraw_list = modraw_file.rsplit(".")
    modraw_filename_prefix = ".".join(modraw_list[0:3])
    modgeo_list = modgeo_file.rsplit(".")
    modgeo_filename_prefix = ".".join(modgeo_list[0:3])

    storage_client = StorageClient()
    bucket = storage_client.bucket(gcp_bucket)
    modraw_path_prefix = os.path.join(modraw_dir, modraw_filename_prefix)
    for blob in bucket.list_blobs(prefix=modraw_path_prefix):
        src_path = blob.name
        geo_filename = src_path.rsplit("/")[-1]
        blob = bucket.blob(src_path)
        dest_modraw_path = os.path.join(dest_dir, geo_filename)
        blob.download_to_filename(dest_modraw_path)
    logger.debug(f"Copied {os.path.join(gcp_bucket, src_path)} to {dest_modraw_path}.")
    modgeo_path_prefix = os.path.join(modgeo_dir, modgeo_filename_prefix)
    for blob in bucket.list_blobs(prefix=modgeo_path_prefix):
        src_path = blob.name
        geo_filename = src_path.rsplit("/")[-1]
        blob = bucket.blob(src_path)
        dest_modgeo_path = os.path.join(dest_dir, geo_filename)
        blob.download_to_filename(dest_modgeo_path)
    logger.debug(f"Copied {os.path.join(gcp_bucket, src_path)} to {dest_modgeo_path}.")

    return Path(dest_modraw_path), Path(dest_modgeo_path)


def download_geo_image(
    gcp_bucket: str, filename: str, image_dir: str, dest_dir: str
) -> Path:
    """Downloads geo images from GCP bucket to local storage

    This function identifies the name of the supporting files from the DNB source file
    The supporting files have the same time of collection but a different
    collection number (CCC) as well as a different prefix. Note this function uses
    list_blobs.

    file_naming_convention = "*DNB_NRT.AYYYYDDD.HHMM.CCC.nc"

    Parameters
    ----------
    info : VVDRequest
        Request containing source image info
    destination_file_name : str
        path to local directory to store downloaded image

    Returns
    -------
    Path
        path to downloaded geo image
    """
    if "VNP02" in filename:
        temp = filename.replace("VNP02", "VNP03")
        geo_dir = image_dir.replace("VNP02", "VNP03")

    elif "VJ102" in filename:
        temp = filename.replace("VJ102", "VJ103")
        geo_dir = image_dir.replace("VJ102", "VJ103")

    temp_list = temp.rsplit(".")
    geo_filename_prefix = ".".join(temp_list[0:3])

    storage_client = StorageClient()
    bucket = storage_client.bucket(gcp_bucket)
    geo_path_prefix = os.path.join(geo_dir, geo_filename_prefix)
    for blob in bucket.list_blobs(prefix=geo_path_prefix):
        src_path = blob.name
        geo_filename = src_path.rsplit("/")[-1]
        blob = bucket.blob(src_path)
        dest_path = os.path.join(dest_dir, geo_filename)
        blob.download_to_filename(dest_path)
    logger.debug(f"Copied {os.path.join(gcp_bucket, src_path)} to {dest_path}.")

    return Path(dest_path)


def image_is_daytime(metadata: dict) -> bool:
    """reads Day Night Flag from metadata
    Parameters
    ----------
    dnb_dataset : dict
        _description_
    """
    day_night_flag = metadata["DayNightFlag"]

    return (day_night_flag == "Day") or (day_night_flag == "Both")


def image_contains_ocean(dnb_dataset: dict) -> bool:
    """checks whether image contains any ocean

    Parameters
    ----------
    land_sea_array : np.ndarray

    Returns
    -------
    bool
        whether land mask contains any pixels on ocean
    """
    land_sea_array_values = set(np.unique(dnb_dataset["land_sea_mask"]))

    contains_ocean = False

    if len(set(OCEAN_VALS).intersection(land_sea_array_values)) > 0:
        contains_ocean = True

    return contains_ocean


def get_chips(image: np.ndarray, detections: dict, dnb_dataset: Dict) -> dict:
    """extracts the context from original image surrounding a vessel
    # Check with product if they prefer something else here, like black pixels.

    Parameters
    ----------
    image : np.ndarray
        image to crop
    coordinates : List[List[float]]
        pixel (array) based coordinates
    dnb_dataset : Dict
        dnb and supporting data
    bounding_box_width : int
        desired bounding box width in pixels
    bounding_box_height : int
        desired bounding box height in pixels

    Returns
    -------
    dict
        {1: {"coords": [x,y], "chip": np.ndarray},...,}
    """

    if dnb_dataset["dnb"]["metadata"]["startDirection"] == "Ascending":
        chip_image = cv2.rotate(image, cv2.ROTATE_180)
        chip_latitude = cv2.rotate(dnb_dataset["latitude"], cv2.ROTATE_180)
        chip_longitude = cv2.rotate(dnb_dataset["longitude"], cv2.ROTATE_180)
    else:
        chip_image = np.copy(image)
        chip_latitude = dnb_dataset["latitude"]
        chip_longitude = dnb_dataset["longitude"]

    chip_half_width = round(IMAGE_CHIP_SIZE / 2)

    chips_dict = {}
    x_pixels, y_pixels = image.shape
    padded_image = np.pad(
        chip_image, pad_width=IMAGE_CHIP_SIZE, mode="constant", constant_values=0
    )
    if detections is not None:
        for idx, detection in detections.items():
            # pixel based coordinate system
            x0, y0 = detection["coords"]

            # for chip creation, center need to be adjusted by amount image was padded
            padded_center_x = x0 + IMAGE_CHIP_SIZE
            padded_center_y = y0 + IMAGE_CHIP_SIZE

            # set the boundaries of the image chip
            top = int(padded_center_x - chip_half_width)
            bottom = int(padded_center_x + chip_half_width)
            left = int(padded_center_y - chip_half_width)
            right = int(padded_center_y + chip_half_width)

            chip_dnb = padded_image[top:bottom, left:right]

            fwd_azimuth = get_chip_azimuth(
                x0,
                y0,
                chip_latitude,
                chip_longitude,
                chip_half_width,
                x_pixels,
                y_pixels,
            )

            chips_dict[idx] = {
                "coords_pix": [detection["coords"][0], detection["coords"][1]],
                "latitude": detection["latitude"],
                "longitude": detection["longitude"],
                "chip": chip_dnb,
                "fwd_azimuth": fwd_azimuth,
                "orientation": 360 - fwd_azimuth,
                "moonlight_illumination": detection["moonlight_illumination"],
                "max_nanowatts": detection["max_nanowatts"],
                "clear_sky_confidence": detection["clear_sky_confidence"],
            }
    return chips_dict


def draw_detections(img: np.ndarray, detections: dict) -> np.ndarray:
    """Draws circles centered on detections on img

    Used for visualizing output/debugging

    Parameters
    ----------
    img : np.ndarray
    detections : dict
    Returns
    -------
    _type_
        img with circles centered on detections
    """
    img_copy = np.copy(img)
    max_watts = np.max(img_copy) + 1
    for idx, detection in detections.items():
        x0, y0 = detection["coords"]

        rr, cc = draw.circle_perimeter(int(x0), int(y0), radius=3, shape=img_copy.shape)
        img_copy[rr, cc] = max_watts

    return img_copy


def get_provider_name(dnb_dataset: Dict) -> str:
    """gets the name of the provider"""
    return dnb_dataset["dnb"]["metadata"]["platform"]


def get_acquisition_time(dnb_dataset: Dict) -> Tuple[str, str]:
    """gets time the frame was acquired

    Parameters
    ----------
    dnb_dataset : Dict
        _description_

    Returns
    -------
    Tuple[str, str]
        start and end time of data acquisition
    """
    start_time = dnb_dataset["dnb"]["metadata"]["time_coverage_start"]
    end_time = dnb_dataset["dnb"]["metadata"]["time_coverage_end"]

    return start_time, end_time


def format_detections(chips_dict: dict) -> List:
    """format detections for response

    Parameters
    ----------
    remote_chip_paths : _type_
        _description_

    Returns
    -------
    List
        list of formatted predictions

    """
    predictions = []
    for idx, chip_info in chips_dict.items():
        predictions.append(
            {
                "latitude": chip_info["latitude"],
                "longitude": chip_info["longitude"],
                "chip_path": chip_info["path"],
                "orientation": chip_info["orientation"],
                "meters_per_pixel": chip_info["meters_per_pixel"],
                "moonlight_illumination": chip_info["moonlight_illumination"],
                "nanowatts": chip_info["max_nanowatts"],
                "clear_sky_confidence": chip_info["clear_sky_confidence"],
            }
        )
    return predictions


def land_water_mask(
    img_to_mask: np.ndarray, land_mask: np.ndarray, kernel_size: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """create boolean mask of land and water
    The land mask has the following values:
        0 Shallow Ocean (0-160 ft)
        1 Land (drop)
        2 Shoreline (drop)
        3 Inland Water (drop) (>160 ft)
        4 Deep Inland Water (drop)
        5 Ephemeral Water (drop)
        6 Moderate Ocean (161-400 ft)
        7 Deep Ocean (>400 ft)

    Mask Documentation:
    https://landweb.modaps.eosdis.nasa.gov/QA_WWW/forPage/MODIS_C6_Water_Mask_v3.pdf

    We retain only shallow, moderate, and deep ocean (0, 6, 7) as we are not interested
    in lights on land

    Parameters
    ----------
    img_to_mask : np.ndarray
    land_mask : np.ndarray
    kernel_size: int
        optional kernel size to erode land mask (bigger = more conservative)

    Returns
    -------
    np.ndarray
       masked image
    """

    mask = np.copy(land_mask)
    bool_mask = np.logical_or(mask == 0, mask >= 6, mask)
    if kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        bool_mask = cv2.dilate(bool_mask, kernel)

    masked_img = bool_mask * img_to_mask

    return masked_img, bool_mask


def clear_sky_mask(
    img_to_mask: np.ndarray, cld_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """_summary_

    Parameters
    ----------
    img_to_mask : np.ndarray
    cld_mask : np.ndarray
    kernel_size : int, optional
        _description_, by default 5
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]

    """

    mask = np.copy(cld_mask)
    height, width = img_to_mask.shape
    # resize image

    resized_mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_AREA)

    if CLOUD_EROSION_KERNEL_DIM > 1:
        kernel = np.ones((CLOUD_EROSION_KERNEL_DIM, CLOUD_EROSION_KERNEL_DIM), np.uint8)
        resized_mask = cv2.erode(resized_mask, kernel)

    bool_mask = np.logical_or(resized_mask > 0.99, resized_mask > 0.99)
    clear_sky = bool_mask * img_to_mask
    cloudy_sky = ~bool_mask * img_to_mask
    return clear_sky, bool_mask, cloudy_sky


def detection_near_mask(
    bool_mask: np.ndarray, detection: Tuple[int, int], distance_threshold: float = 4000
) -> bool:
    """determines whether a detection is near shore

    Parameters
    ----------
    bool_mask : np.ndarray
        _description_
    detection : Tuple[int, int]
        _description_
    distance_threshold : float, optional
        _description_, by default 2000 in meters
    Returns
    -------
    bool
        whether detection is too close to shore
    """
    x, y = detection
    x_pixels, y_pixels = bool_mask.shape
    radius = math.ceil(distance_threshold / VIIRS_PIXEL_SIZE)

    rr, cc = draw.disk((int(x), int(y)), radius)
    xs = []
    ys = []
    for x, y in zip(rr, cc):
        if x < x_pixels and y < y_pixels:
            xs.append(x)
            ys.append(y)

    return 0 in bool_mask[xs, ys]


def save_chips_locally(
    chips_dict: dict, destination_path: str, chip_features: dict
) -> None:
    """saves image of each detection

    Parameters
    ----------
    chips_dict : Dict
        _description_
    destination_path : str
        _description_
    image_name: str
    """
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    for idx, chip_info in chips_dict.items():
        features = chip_features[idx]

        lat = chip_info["latitude"]
        long = chip_info["longitude"]

        img_filename = f"{lat}_{long}.jpeg"
        dest_img = os.path.join(
            destination_path,
            os.path.basename(img_filename),
        )

        csv_filename = f"{lat}_{long}.csv"
        dest_csv = os.path.join(
            destination_path,
            os.path.basename(csv_filename),
        )

        normalized_chip = cv2.normalize(
            chip_info["chip"], chip_info["chip"], 0, 255, cv2.NORM_MINMAX
        )

        resized_img = cv2.resize(
            normalized_chip, CHIP_IMG_DIM, interpolation=cv2.INTER_AREA
        )
        chip_info["meters_per_pixel"] = (
            normalized_chip.shape[0] / resized_img.shape[0]
        ) * VIIRS_PIXEL_SIZE

        chip_info["path"] = dest_img

        feature_dict = {
            "xmin": chip_info["coords_pix"][0],
            "ymin": chip_info["coords_pix"][1],
            "lat": chip_info["latitude"],
            "lon": chip_info["latitude"],
            "area": features["area"],
            "perimeter": features["perimeter"],
            "max_nanowatts": features["max_nanowatts"],
            "min_nanowatts": features["min_nanowatts"],
            "moonlight_illumination": features["moonlight_illumination"],
            "clear_sky_confidence": features["clear_sky_confidence"],
            "mean_nanowatts": features["mean_nanowatts"],
        }
        pd.DataFrame(feature_dict, index=[0]).to_csv(dest_csv)
        cv2.imwrite(dest_img, resized_img)


def numpy_nms(detections: dict, thresh: float = 0.1) -> dict:
    """numpy implementation of NMS, uses wattage in place of confidence score


    Parameters
    ----------
    detections : dict
    thresh : float, optional
        area overlap threshold, by default 0.1 -- yes this is a small value. Note that
        the bounding boxes are 2 just pixels so even very small overlaps may be worth
        supressing

    Returns
    -------
    dict
        NMSed detections
    """
    x1s = []
    y1s = []
    x2s = []
    y2s = []
    scores = []
    indices = []
    update_detections = {}
    count = 0
    for idx, detection in detections.items():
        update_detections[count] = detection
        count += 1

    for idx, detection in update_detections.items():
        indices.append(idx)
        x1, y1 = detection["coords"]
        x1s.append(x1 - 2)
        y1s.append(y1 - 2)
        x2s.append(x1 + 2)
        y2s.append(y1 + 2)
        if ~np.isnan(detection["max_nanowatts"]):
            scores.append(detection["max_nanowatts"])
        else:
            scores.append(0)

    x1 = np.array(x1s)
    y1 = np.array(y1s)
    x2 = np.array(x2s)
    y2 = np.array(y2s)
    scores_array = np.array(scores)
    dets = pd.DataFrame.from_dict(
        {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "scores": scores}
    ).to_numpy()

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores_array.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]  # pick maxmum iou box
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    nms_detections = {}
    for idx in keep:
        nms_detections[idx] = update_detections[idx]

    return nms_detections


def download_earthdata_url(file_urls: List, local_dir: str) -> None:
    """Download test files from ladsweb.modaps.eosdis@nasa.gov

    BASE_URL = "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/5200/"
    EXAMPLE_URL_LIST = [
        f"{BASE_URL}VNP02DNB/2022/348/VNP02DNB.A2022348.1142.002.2022348173537.nc",
    ]
    """

    for url in file_urls:
        headers = dict()

        headers["Authorization"] = f"Bearer {TOKEN}"

        response = requests.get(url, headers=headers, timeout=600)
        local_path = Path(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                local_dir,
                Path(url).name,
            )
        )
        with open(local_path, "w+b") as fh:
            fh.write(response.content)


def moonlit_clouds_irradiance(
    dnb_dataset: dict, could_percentile: float = 99.73
) -> Tuple[float, float]:
    """estimates the background cloud illumination

    Parameters
    ----------
    dnb_dataset : dict
    could_percentile : float, optional
        capture this percentile of the total distribution, by default 99.73

    Returns
    -------
    Tuple[float, float]
    """

    # erode the land water mask to make sure that clouds are not over land lights
    dnb_observations, _ = land_water_mask(
        dnb_dataset["dnb"]["data"], dnb_dataset["land_sea_mask"], kernel_size=20
    )
    mask = np.copy(dnb_dataset["cloud_mask"])
    bool_mask = np.logical_or(mask == 0, mask == 0.0)
    cloudy_sky = bool_mask * dnb_observations
    cloudy_skies = np.clip(cloudy_sky * 1e9, 0, 10000)
    # reject outliers
    cloudy_skies[cloudy_skies > 1000] = 0
    # areas that do not contain any clouds should not contribute to calculation
    cloudy_skies[cloudy_skies == 0.0] = np.nan
    percentile = np.nanpercentile(cloudy_skies, could_percentile)
    median = np.nanmedian(cloudy_skies[cloudy_skies <= percentile])
    return percentile, median


def check_image_metadata(dnb_dataset: dict) -> List:
    """checks whether image is daytime, land only or should be processed"""
    status = []
    metadata = dnb_dataset["dnb"]["metadata"]
    if image_is_daytime(metadata):
        status.append("daytime")

    if not image_contains_ocean(dnb_dataset):
        status.append("land_only")

    if not image_is_daytime(metadata) and image_contains_ocean(dnb_dataset):
        status = ["processed"]

    return status


def aurora_mask(dnb_dataset: dict) -> np.ndarray:
    """
    Consider erosion -> dilation to retain detections around blue hole but remove
    aurora when close to edge.
        opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

    This function removes detections close to the aurora (but not within the aurora)
    which are a source of false postives.

    from collections import Counter
    from sklearn.cluster import KMeans
    # may be useful to know center of aurora if there are multiple clusters
    points = [detection["coords"] for idx, detection in detections.items()]
    kmeans = KMeans(n_clusters=10)
    kmeans.fit(points)
    counter = Counter(kmeans.labels_)
    largest_cluster_idx = np.argmax(counter.values())
    aurora_center = kmeans.cluster_centers_[largest_cluster_idx]
    """

    image_contains_aurora = False
    # check if frame is within auroral zones
    frame_lat = dnb_dataset["latitude"]
    metadata = dnb_dataset["dnb"]["metadata"]
    valid_min = float(metadata["valid_min"])
    valid_max = float(metadata["valid_max"])
    dnb_observations = dnb_dataset["dnb"]["data"]
    dnb_observations = np.clip(dnb_observations, valid_min, valid_max)
    dnb_observations *= 1e9
    mask = np.zeros(dnb_observations.shape)
    if np.abs(np.min(frame_lat)) >= AURORA_LAT_THRESHOLD:
        b = dnb_observations.copy()

        # Get the high noise values within aurora
        b[b < 1000] = 0
        b[b >= 1000] = 1

        h, w = b.shape
        half_height = int(h / 2)
        half_width = int(w / 2)

        top_left_counts = np.count_nonzero(b[0:half_height, 0:half_width])
        top_right_counts = np.count_nonzero(b[0:half_height, half_width:])
        bot_left_counts = np.count_nonzero(b[half_height:, 0:half_width])
        bot_right_counts = np.count_nonzero(b[half_height:, half_width:])

        # likely that frame contains auroral ring data
        if (
            np.max(
                [top_left_counts, top_right_counts, bot_left_counts, bot_right_counts]
            )
            > 1000
        ):
            image_contains_aurora = True

            b = dnb_observations.copy()
            b[b <= HIGH_AURORA] = 0
            b[b > HIGH_AURORA] = 1
            mask = np.zeros(dnb_observations.shape)
        else:
            # check if the frame is on the edge of the aurora
            b = dnb_observations.copy()

            # Get the noisy values on the edge of the aurora
            b[b < MID_AURORA] = 0
            b[b >= MID_AURORA] = 1

            top_left_counts = np.count_nonzero(b[0:half_height, 0:half_width])
            top_right_counts = np.count_nonzero(b[0:half_height, half_width:])
            bot_left_counts = np.count_nonzero(b[half_height:, 0:half_width])
            bot_right_counts = np.count_nonzero(b[half_height:, half_width:])

            # likely that frame clips edge of aurora
            if (
                np.max(
                    [
                        top_left_counts,
                        top_right_counts,
                        bot_left_counts,
                        bot_right_counts,
                    ]
                )
                > 10000
            ):
                image_contains_aurora = True

                b = dnb_observations.copy()
                # using 50 to be more conservative than
                b[b <= LOW_AURORA] = 0
                b[b > LOW_AURORA] = 1
                opening = cv2.morphologyEx(
                    b, cv2.MORPH_OPEN, kernel=np.ones((10, 10), np.uint8)
                )  # removes salt and pepper noise from within aurora
                blurred_aurora = cv2.blur(
                    opening, (AURORA_BLUR_KERNEL_SIZE, AURORA_BLUR_KERNEL_SIZE)
                )
                kernel = np.ones((DILATION_KERNEL_SIZE, DILATION_KERNEL_SIZE), np.uint8)
                mask = cv2.dilate(blurred_aurora, kernel)
                mask[mask > 0] = 1
                mask = 1 - mask

    return image_contains_aurora, mask


def quality_flag_mask(
    raw_data: np.ndarray, quality_flag_data: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """boolean mask for quality flags
        0: (No identified issues)
        1: (LSB) Substitute_Cal Granule-average has been substituted for SV and/or BB
        2: Out_of_Range Calibrated radiance out of range (see Appendix A)
        4: Saturation L1A Earth view counts ≥ 4095.
        8: Temp_not_Nominal Measured temperatures outside nominal range
        16: Low_Gain One or more low-gain samples (most common image artifact)
        32: Mixed_Gain Mix of high- and low-gain samples
        64: DG_Anomaly Dual-gain anomaly
        128: Some_Saturation One or more samples in aggregation were saturated
        256: Bowtie_Deleted Data excluded by VIIRS for bowtie deletion
        512: Missing_EV Packet missing or corrupted in transmission
        1024: Cal_Fail Calibration failure
        2048: Dead_Detector Detector is not producing valid data


    Consider special handling of low gain samples to salvage some true positive
    detections in center of the frame even with bad edge data
        quality_flag_data_copy = quality_flag_data.copy()
        low_gain_samples = (
            quality_flag_data_copy[:, 1:] == 16
        )  # ignore first column bowtie deletions.
        low_gain_rows = np.where(low_gain_samples.any(axis=1))[0]
        temp = np.zeros(quality_flag_data.shape)
        if len(low_gain_rows) > 0:
            min_r = np.min(low_gain_rows)
            max_r = np.max(low_gain_rows) + 1  # buffer row
            temp[0:max_r, 0:200] = 1
            temp[0:max_r, -200:] = 1

    Parameters
    ----------
    raw_data : np.ndarray
        DNB data
    quality_flag_data : np.ndarray

    Returns
    -------
    np.ndarray
       masked image
    np.ndarray
        mask
    """

    quality_flag_data_copy = quality_flag_data.copy()
    data = np.copy(raw_data)
    bool_mask = np.logical_or(
        quality_flag_data_copy == 0, quality_flag_data_copy == 0, quality_flag_data_copy
    )
    if not np.all(bool_mask):
        logger.info("Found quality flag issues")

    masked_img = bool_mask * data

    return masked_img, bool_mask


def format_detections_df(detections: dict, filename: str) -> pd.DataFrame:
    """formats detections into a pandas dataframe

    Parameters
    ----------
    detections : dict

    filename : str


    Returns
    -------
    pd.DataFrame

    """
    xmins = []
    ymins = []
    area = []
    perimeter = []
    max_nanowatts = []
    min_nanowatts = []
    moonlight_illumination = []
    clear_sky_confidence = []
    mean_nanowatts = []
    img_name = []
    lats = []
    lons = []
    for idx, detection in detections.items():
        xmin, ymin = detection["coords"]
        xmins.append(int(xmin))
        ymins.append(int(ymin))
        lats.append(detection["latitude"])
        lons.append(detection["longitude"])
        area.append(detection["area"])
        perimeter.append(detection["perimeter"])
        max_nanowatts.append(detection["max_nanowatts"])
        min_nanowatts.append(detection["min_nanowatts"])
        moonlight_illumination.append(detection["moonlight_illumination"])
        clear_sky_confidence.append(detection["clear_sky_confidence"])
        mean_nanowatts.append(detection["mean_nanowatts"])
        img_name.append(filename)

    detections_df = pd.DataFrame.from_dict(
        {
            "xmin": xmins,
            "ymin": ymins,
            "latitude": lats,
            "longitude": lons,
            "area": area,
            "perimeter": perimeter,
            "max_nanowatts": max_nanowatts,
            "min_nanowatts": min_nanowatts,
            "moonlight_illumination": moonlight_illumination,
            "clear_sky_confidence": clear_sky_confidence,
            "mean_nanowatts": mean_nanowatts,
            "img_name": img_name,
        }
    )

    return detections_df


def viirs_annotate_pipeline(
    dnb_filename: str,
    geo_filename: str,
    input_dir: str,
    output_dir: str,
    **optional_files: str,
) -> Tuple[dict, List]:
    """viirs debugging pipeline

    Parameters
    ----------
    dnb_filename : str
    geo_filename : str
    input_dir : str
    output_dir : str
    cloud_filename: str

    Returns
    -------
    Tuple[dict, List]
        _description_
    """
    from pipeline import VIIRSVesselDetection

    VVD = VIIRSVesselDetection()
    dnb_path = Path(os.path.join(input_dir, dnb_filename))
    geo_path = Path(os.path.join(input_dir, geo_filename))
    if "cloud_filename" in optional_files:
        phys_path = Path(
            os.path.join(
                input_dir,
                str(optional_files.get("cloud_filename")),
            )
        )
    else:
        phys_path = None

    if "modraw" in optional_files:
        modraw_path = Path(
            os.path.join(
                input_dir,
                str(optional_files.get("modraw")),
            )
        )
    else:
        modraw_path = None

    if "modgeo" in optional_files:
        modgeo_path = Path(
            os.path.join(
                input_dir,
                str(optional_files.get("modgeo")),
            )
        )
    else:
        modgeo_path = None

    all_detections, img_array, dnb_dataset, status = VVD.run_pipeline(
        dnb_path, geo_path, TemporaryDirectory(), phys_path, modraw_path, modgeo_path
    )
    filtered_detections = all_detections["vessel_detections"]
    output_dir = os.path.join(
        output_dir,
        Path(dnb_filename).stem,
    )
    chip_dir = os.path.join(output_dir, "image_chips")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(chip_dir, exist_ok=True)
    chips_dict = get_chips(img_array, filtered_detections, dnb_dataset)
    save_chips_locally(chips_dict, chip_dir, filtered_detections)
    _, land_mask = land_water_mask(
        dnb_dataset["dnb"]["data"], dnb_dataset["land_sea_mask"]
    )
    all_detections_csv = format_detections_df(
        filtered_detections, f"{dnb_path.stem}.npy"
    )
    annotation_csv_path = os.path.join(output_dir, "detections.csv")
    all_detections_csv.to_csv(annotation_csv_path)
    logger.debug(f"Wrote {len(all_detections_csv)} detections to {annotation_csv_path}")

    # save image as numpy array
    img_array, _, _ = preprocess_raw_data(dnb_dataset)
    np.save(
        os.path.join(output_dir, f"{dnb_path.stem}.npy"),
        img_array,
    )
    plt.imsave(
        os.path.join(output_dir, "detections.jpg"),
        draw_detections(np.clip(img_array, 0, 100), filtered_detections),
        cmap=cm.gray,
    )

    if phys_path:
        clear_skies, cld_mask, _ = clear_sky_mask(
            dnb_dataset["dnb"]["data"], dnb_dataset["cloud_mask"]
        )
        _, _, cloudy_skies = clear_sky_mask(
            dnb_dataset["dnb"]["data"], dnb_dataset["cloud_mask"]
        )

        dnb_observations, _, _ = preprocess_raw_data(dnb_dataset)
        all_channels = np.stack(
            [
                dnb_observations,
                dnb_dataset["land_sea_mask"],
                dnb_dataset["moonlight"],
                dnb_dataset["cloud_mask"],
            ],
            axis=0,
        )
        for chip_idx, chip in chips_dict.items():
            x, y = chip["coords_pix"]
            lat = chip["latitude"]
            lon = chip["longitude"]

            chip_all_channels, skip = get_chip_from_all_channels(all_channels, x, y)
            if not skip:
                out_filename = os.path.join(
                    output_dir, "image_chips", f"{lat}_{lon}.npy"
                )

                np.save(out_filename, chip_all_channels)

    return filtered_detections, status


def preprocess_raw_data(dnb_dataset: dict) -> Tuple[np.ndarray, float, float]:
    """converts raw dnb data to usable array of nanowatts

    Parameters
    ----------
    dnb_dataset : dict

    Returns
    -------
    Tuple[np.ndarray, float, float]
    """
    metadata = dnb_dataset["dnb"]["metadata"]
    valid_min = float(metadata["valid_min"])
    valid_max = float(metadata["valid_max"])

    dnb_observations, _ = land_water_mask(
        dnb_dataset["dnb"]["data"], dnb_dataset["land_sea_mask"]
    )

    dnb_observations, _ = quality_flag_mask(
        dnb_observations, dnb_dataset["dnb"]["quality"]
    )

    dnb_observations = np.clip(dnb_observations, valid_min, valid_max)
    dnb_observations *= 1e9
    dnb_observations = remove_outliers(dnb_observations)
    return dnb_observations, valid_min, valid_max


def get_chip_azimuth(
    x0: int,
    y0: int,
    chip_latitude: np.ndarray,
    chip_longitude: np.ndarray,
    chip_half_width: int,
    xpixels: int,
    ypixels: int,
) -> float:
    """calculates the azimuth of the chip

    Parameters
    ----------
    x0 : int
    y0 : int
    chip_latitude : np.ndarray
    chip_longitude : np.ndarray
    chip_half_width : int
    xpixels : int
    ypixels : int

    Returns
    -------
    float

    """
    xmin = np.max([0, x0 - chip_half_width])
    xmax = np.min([x0 + chip_half_width - 1, xpixels - 2])
    ymin = np.max([0, y0 - chip_half_width])
    ymax = np.min([ypixels - 2, y0 + chip_half_width - 1])

    chip_latitude_crop = chip_latitude[xmin:xmax, ymin:ymax]
    chip_longitude_crop = chip_longitude[xmin:xmax, ymin:ymax]

    fwd_azimuth, _ = calculate_e2e_cog(
        GeoPoint(lat=chip_latitude_crop[-1, 0], lon=chip_longitude_crop[-1, 0]),
        GeoPoint(lat=chip_latitude_crop[0, 0], lon=chip_longitude_crop[0, 0]),
    )
    if fwd_azimuth < 0:
        fwd_azimuth += 360

    return fwd_azimuth


def remove_outliers(dnb_observations: np.ndarray) -> np.ndarray:
    """Ionospheric noise is defined to be greater than 1000 nano watts

    Parameters
    ----------
    dnb_observations : np.ndarray
        dnb in nanowatts

    Returns
    -------
    np.ndarray
        dnb without outliers
    """
    dnb_observations[dnb_observations > OUTLIER_THRESHOLD_NW] = 0
    return dnb_observations


def threshold_image(norm: np.ndarray) -> np.ndarray:
    """thresholds image

    Parameters
    ----------
    norm : _type_
        _description_
    Returns
    -------
    _type_
        _description_
    """
    threshold = cv2.adaptiveThreshold(
        norm,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        3,
        15,
    )
    return threshold


def lightning_detector(image: np.ndarray) -> np.ndarray:
    """lightning detector for DNB images

    Because DNB data are collected 16 lines at a time, lightning detections
    create a characteristic 16-line horizontal stripe exactly parallel to the
    scan. Such lines are unlikely to be vessels or associated with vessels

    Method is: normalization->threshold->edge detection->horizontal kernel->
    if properties of resulting contours match empirical lightning critiera,
    those pixels are classified as lightning.

    We identify these sources in order to remove false positive
    detections near them.

    Parameters
    ----------
    image : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    normalized_image = cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )

    thresh = threshold_image(normalized_image)

    image = Image.fromarray(thresh)
    image = image.filter(ImageFilter.FIND_EDGES)
    image = np.array(image)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    detect_horizontal = cv2.morphologyEx(
        image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2
    )

    cnts = cv2.findContours(
        detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    x_pixels, y_pixels = normalized_image.shape
    lightning_mask = np.ones(normalized_image.shape)
    lightning_detected = 0
    for contour in cnts:
        for i in range(0, len(contour) - 1):
            first_corner = contour[i]
            second_corner = contour[i + 1]
            y_start, x_start = first_corner[0][0], first_corner[0][1]
            y_end, x_end = second_corner[0][0], second_corner[0][1]
            if x_start == x_end and x_start != x_pixels - 1:
                line_length = abs(y_start - y_end)
                if (
                    line_length > MIN_LIGHTNING_LENGTH
                    and line_length < MAX_LIGHTNING_LENGTH
                ):
                    x_start = min([x_start, x_end])
                    x_end = max([x_start, x_end])
                    lower_bound = np.min(
                        [x_start + 2, x_pixels]
                    )  # lightning is offset from the x value of the edge (so  +2)
                    upper_bound = np.min([x_start + 18, x_pixels])
                    if lower_bound == upper_bound:
                        lower_bound = upper_bound - 1
                    below_line_brightness = np.mean(
                        normalized_image[
                            lower_bound:upper_bound,
                            int((y_start + y_end) / 2),
                        ]
                    )
                    lower_bound = np.max([x_start - 15, 0])
                    upper_bound = np.min([x_start + 1, x_pixels])
                    if lower_bound == upper_bound:
                        upper_bound = lower_bound + 1
                    above_line_brightness = np.mean(
                        normalized_image[
                            lower_bound:upper_bound,
                            int((y_start + y_end) / 2),
                        ]
                    )

                    if above_line_brightness > below_line_brightness:
                        lightning_mask_xmin, lightning_mask_xmax = (
                            np.max([x_start - LIGHTNING_HEIGHT, 0]),
                            np.min([x_start + 5, x_pixels]),
                        )
                    else:
                        lightning_mask_xmin, lightning_mask_xmax = (
                            np.max([x_start - 5, 0]),
                            np.min([x_start + LIGHTNING_HEIGHT, x_pixels]),
                        )

                    lightning_mask_xmin = np.max([0, lightning_mask_xmin])
                    lightning_mask_xmax = np.min([x_pixels, lightning_mask_xmax])

                    lightning_mask_ymin = np.max([0, y_start - LIGHTNING_WIDTH])
                    lightning_mask_ymax = np.min([y_end + LIGHTNING_WIDTH, y_pixels])

                    lightning_mask[
                        lightning_mask_xmin:lightning_mask_xmax,
                        lightning_mask_ymin:lightning_mask_ymax,
                    ] = 0
                    lightning_detected += 1

    return lightning_mask, lightning_detected


def download_from_gcp(bucket: str, filename: str, input_dir: str, dir: str) -> Tuple:
    """downloads filename from a given bucket/input_dir and copies to dir

    Parameters
    ----------
    bucket : str
    filename : str
    input_dir : str
    dir : str

    Returns
    -------
    Tuple

    """
    logger.debug(f"Downloading data from cloud storage: {filename}")
    dnb_path = download_dnb_image(bucket, filename, input_dir, dir)

    geo_path = download_geo_image(bucket, filename, input_dir, dir)

    try:
        modraw_path, modgeo_path = download_mod_images(
            bucket, filename, geo_path.name, input_dir, dir
        )
    except Exception:
        modraw_path = None
        modgeo_path = None

    try:
        phys_path = download_phys_image(bucket, filename, input_dir, dir)
    except Exception:
        phys_path = None

    return dnb_path, geo_path, modraw_path, modgeo_path, phys_path


def copy_local_files(
    info: Any, dir: str
) -> Tuple[str, str, Optional[str], Optional[str], Optional[str]]:
    """Copy local files to a temporary directory

    Parameters
    ----------
    info : _type_
    tmpdir : _type_

    Returns
    -------
    Tuple

    """
    logger.debug(f"Copying local files to {dir}")

    dnb_path = os.path.join(dir, info.dnb_filename)
    geo_path = os.path.join(dir, info.geo_filename)
    shutil.copy2(os.path.join(info.input_dir, info.dnb_filename), dir)
    shutil.copy2(os.path.join(info.input_dir, info.geo_filename), dir)

    if info.modraw_filename is not None:
        modraw_path = os.path.join(dir, info.modraw_filename)
        shutil.copy2(os.path.join(info.input_dir, info.modraw_filename), modraw_path)
    else:
        modraw_path = None

    if info.modgeo_filename is not None:
        modgeo_path = os.path.join(dir, info.modgeo_filename)
        shutil.copy2(os.path.join(info.input_dir, info.modgeo_filename), modgeo_path)
    else:
        modgeo_path = None

    if info.phys_filename is not None:
        phys_path = os.path.join(dir, info.phys_filename)
        shutil.copy2(os.path.join(info.input_dir, info.phys_filename), phys_path)
    else:
        phys_path = None

    return dnb_path, geo_path, modraw_path, modgeo_path, phys_path


def create_earth_data_url(
    source_url: str, product_name: str, year: str, doy: str, time: str
) -> str:
    """Create the URL for the Earth Data website

    Parameters
    ----------
    source_url : str
    product_name : str
    year : str
    doy : str
    time : str

    Returns
    -------
    str
        _description_
    """
    logger.debug(f"Retrieving: {source_url}/{product_name}/{year}/{doy}/{time}")
    df = pd.read_html(
        f"{source_url}/{product_name}/{year}/{doy}", extract_links="body"
    )[0]

    df[["filename", "url"]] = pd.DataFrame(
        df["Select All  Name"].tolist(), index=df.index
    )
    df = df.iloc[1:]
    filename = df.loc[df["url"].str.contains(f"A{year}{doy}.{time}", case=False)][
        "url"
    ].values[0]
    base = ("/").join(filename.split("/")[4:])
    return f"{source_url}/{base}"


def get_cld_filename(product_name: str, year: str, doy: str, time: str) -> str:
    """_summary_

    Parameters
    ----------
    product_name : str
        _description_
    year : str
        _description_
    doy : str
        _description_
    time : str
        _description_

    Returns
    -------
    str
        _description_
    """
    source_url = "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/5110"

    if "VNP02" in product_name:
        cld_product = "CLDMSK_L2_VIIRS_SNPP"
    elif "VJ102" in product_name:
        cld_product = "CLDMSK_L2_VIIRS_NOAA20"
    cld_url = create_earth_data_url(source_url, cld_product, year, doy, time)

    return cld_url


def get_geo_filename(product_name: str, year: str, doy: str, time: str) -> str:
    """_summary_

    Parameters
    ----------
    product_name : str
        _description_
    year : str
        _description_
    doy : str
        _description_
    time : str
        _description_

    Returns
    -------
    str
        _description_
    """
    source_url = "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/5200"

    if "VNP02" in product_name:
        geo_product = product_name.replace("VNP02", "VNP03")
    elif "VJ102" in product_name:
        geo_product = product_name.replace("VJ102", "VJ103")

    return create_earth_data_url(source_url, geo_product.split("_")[0], year, doy, time)


def get_dnb_filename(product_name: str, year: str, doy: str, time: str) -> str:
    """_summary_

    Parameters
    ----------
    product_name : str
        _description_
    year : str
        _description_
    doy : str
        _description_
    time : str
        _description_

    Returns
    -------
    str
        _description_
    """
    source_url = "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/5200"
    return create_earth_data_url(
        source_url, product_name.split("_")[0], year, doy, time
    )


def parse_event_id(event_id: str) -> Tuple[str, str, str, str, str, str]:
    """_summary_

    Parameters
    ----------
    event_id : str
        _description_

    Returns
    -------
    Tuple[str, str, str, str, str, str]
        _description_
    """
    import re

    chip = event_id[event_id.rfind("/") + 1 : event_id.rfind("?")]
    event_id_parts = chip.split("_")
    lat = event_id_parts[2]
    lon = re.sub("[^0-9.-]", "", event_id_parts[3])
    img_name = event_id_parts[0] + "_" + event_id_parts[1]
    date = img_name.split(".")[1]
    year = date[1:5]
    doy = date[5:]
    time = img_name.split(".")[2]
    product_name = img_name.split(".")[0]
    return product_name, year, doy, time, lat, lon


def download_url(
    url: str, bearer_token: str, out: io.BufferedRandom
) -> Optional[io.BufferedRandom]:
    """Download a URL to a file

    Parameters
    ----------
    url : str
    bearer_token : str
    out : io.BufferedRandom

    Returns
    -------
    Optional[io.BufferedRandom]

    """
    headers = dict()
    if bearer_token is not None:
        headers["Authorization"] = bearer_token
    try:
        response = requests.get(url, headers=headers, timeout=600)
        if response.status_code == requests.codes.ok:
            if out is None:
                return response.content.decode("utf-8")
            else:
                out.write(response.content)
                return out
        else:
            print(f"HTTP error: {response.status_code}, reason: {response.reason}")
    except Exception as ex:
        print(f"Unexpected exception trying to download url: {url}. Error: {str(ex)}")
    return None


def get_chip_from_all_channels(all_channels: np.ndarray, x: int, y: int) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    all_channels : np.ndarray
    x : int
    y : int

    Returns
    -------
    np.ndarray

    """
    chip_channels = all_channels[:, x - 10 : x + 10, y - 10 : y + 10]
    skip = False
    if chip_channels.shape != (4, 20, 20):
        chip_channels = np.zeros((4, 20, 20))
        skip = True

    return chip_channels, skip


def autogen_api_schema() -> None:
    """_summary_"""

    from main import app

    with open("docs/openapi.json", "w") as f:
        json.dump(
            get_openapi(
                title=app.title,
                version=app.version,
                openapi_version=app.openapi_version,
                description=app.description,
                routes=app.routes,
            ),
            f,
        )


def get_all_times_from_date() -> List[str]:
    """This function returns a datetime for every 6 minutes in a day

    Returns
    -------
    _type_
        satellite collects imagery every 6 minutes
    """
    start_date_string = "08/03/2023 00:00"
    end_date_string = "08/03/2023 23:59"
    index = pd.date_range(start=start_date_string, end=end_date_string, freq="6T")
    times = []
    for time in index:
        times.append(time.strftime("%H:%M:%S")[0:2] + time.strftime("%H:%M:%S")[3:5])
    return times


def get_detections_from_one_frame(
    product_name: str,
    year: str,
    doy: str,
    time: str,
    tmpdir: str,
    test_file_output_dir: str,
) -> dict:
    """_summary_

    Parameters
    ----------
    product_name : _type_
        _description_
    year : _type_
        _description_
    doy : _type_
        _description_
    time : _type_
        _description_
    tmpdir : _type_
        _description_
    TEST_FILE_OUTPUT_DIR : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    dnb_url = get_dnb_filename(product_name, year, doy, time)
    geo_url = get_geo_filename(product_name, year, doy, time)
    phys_url = get_cld_filename(product_name, year, doy, time)

    dnb_path = os.path.join(tmpdir, dnb_url.split("/")[-1])
    geo_path = os.path.join(tmpdir, geo_url.split("/")[-1])
    phys_path = os.path.join(tmpdir, phys_url.split("/")[-1][0:40] + ".nrt.nc")

    detections, _ = viirs_annotate_pipeline(
        dnb_path,
        geo_path,
        tmpdir,
        test_file_output_dir,
        cloud_filename=phys_path,
    )
    return detections


def gas_flare_locations(dnb_dataset: dict) -> List:
    """_summary_

    Parameters
    ----------
    dnb_dataset : dict
        _description_

    Returns
    -------
    List
        _description_
    """
    m10data = dnb_dataset["m10_band"]["m10"][0]
    m10data[m10data > 100] = 0
    gas_flares_locations = np.argwhere(m10data > GAS_FLARE_THRESHOLD)
    mod_lats = dnb_dataset["m10_band"]["lat"][0]
    mod_lons = dnb_dataset["m10_band"]["lon"][0]
    return lat_lon_from_pix(gas_flares_locations, mod_lats, mod_lons)


def lat_lon_from_pix(
    location: List[Tuple], mod_lats: np.ndarray, mod_lons: np.ndarray
) -> List[GeoPoint]:
    """_summary_

    Parameters
    ----------
    location : List[Tuple]
        _description_
    mod_lats : np.ndarray
        _description_
    mod_lons : np.ndarray
        _description_

    Returns
    -------
    List[Tuple]
        _description_
    """
    coordinates = []
    for pixel in location:
        latitude = mod_lats[pixel[0], pixel[1]]
        longitude = mod_lons[pixel[0], pixel[1]]
        coordinates.append(GeoPoint(lat=latitude, lon=longitude))
    return coordinates


def format_dets_for_correlation(
    detections_json_path: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """formats detections for use in correlation service

    Parameters
    ----------
    detections_json_path : json
        _description_

    Returns
    -------
    Tuple[List[dict], dict]
        _description_
    """

    with open(detections_json_path, "r") as f:
        detections = json.load(f)
    formatted_detections = []
    timestamp = detections["acquisition_time"]
    for _id, detection in enumerate(detections["predictions"]):
        formatted_detections.append(
            {
                "id": _id,
                "ts": timestamp,
                "lat": detection["latitude"],
                "lon": detection["longitude"],
            }
        )

    frame = {
        "ts": detections["acquisition_time"],
        "polygon_points": detections["frame_extents"],
    }

    return formatted_detections, frame
