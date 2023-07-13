"""Main VVD model module"""
import logging.config
import os
from typing import Dict, Tuple

import cv2
import numpy as np
import yaml
from skimage.measure import label, regionprops
from utils import (
    clear_sky_mask,
    land_water_mask,
    moonlit_clouds_irradiance,
    quality_flag_mask,
)

logging.config.fileConfig(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "logging.conf"),
    disable_existing_loggers=False,
)
logger = logging.getLogger(__name__)
CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "config", "config.yml"
)

with open(CONFIG_PATH, "r") as file:
    config = yaml.safe_load(file)["model"]


STRUCTURING_ELEMENT_SIZE = (config["KERNEL_DIM_1"], config["KERNEL_DIM_2"])
IMG_MAX_VALUE = config["IMG_MAX_VALUE"]
BLOCK_SIZE = config["BLOCK_SIZE"]
ADAPTIVE_CONSTANT = config["ADAPTIVE_CONSTANT"]
CLIP_MAX = config["CLIP_MAX"]
OUTLIER_THRESHOLD_NW = config["OUTLIER_THRESHOLD_NW"]
MOONLIGHT_ILLUMINATION_PERCENT = config["MOONLIGHT_ILLUMINATION_PERCENT"]
VESSEL_CONNECTIVITY = config["VESSEL_CONNECTIVITY"]
ADAPTIVE_METHOD = cv2.ADAPTIVE_THRESH_MEAN_C
THRESHOLD_TYPE = cv2.THRESH_BINARY
VESSEL_KERNEL = np.array([[0, -2.0, 0], [-2.0, 10, -2.0], [0, -2.0, 0]])
VESSEL_THRESHOLD = 10


def vvd_cv_model(dnb_dataset: Dict) -> Tuple[dict, np.ndarray]:
    """main vvd model method


    Parameters
    ----------
    dnb_dataset : dnb dataset defined in preprocess.py

    Returns
    -------
    Tuple[dict, np.ndarray]
        dict of detections and formatted output image
    """

    vessel_detections = {}
    try:
        dnb_observations, valid_min, valid_max = preprocess_raw_data(dnb_dataset)

        # If moon is bright, compute cloud glow
        if np.mean(dnb_dataset["moonlight"]) > MOONLIGHT_ILLUMINATION_PERCENT:
            if "cloud_mask" in dnb_dataset:
                clear_sky_confidence_array = dnb_dataset.get("cloud_mask")
                (cloud_illumination, _) = moonlit_clouds_irradiance(dnb_dataset)
                if np.isnan(cloud_illumination):
                    cloud_illumination = 0
            else:
                clear_sky_confidence_array = np.zeros(dnb_observations.shape)
                cloud_illumination = CLIP_MAX

            dnb_observations, cld_mask, cloudy_observations = clear_sky_mask(
                dnb_observations, clear_sky_confidence_array
            )

        else:
            cloud_illumination = 0

        logger.debug(f"Moonlit cloud irradiance threshold set to {cloud_illumination}")

        vessel_detections = detect_vessels(dnb_observations, cloud_illumination)

        formatted_image = np.clip(dnb_dataset["dnb"]["data"] * 1e9, valid_min, CLIP_MAX)

    except Exception as e:
        logger.exception("Exception computing vessel detections", str(e), exc_info=True)

    return vessel_detections, formatted_image


def components_to_detections(label_im: np.ndarray) -> dict:
    """image to vessel coordinates

    Parameters
    ----------
    label_im : np.ndarray
        preprocessed masked image with lighted vessels as connected components

    Returns
    -------
    dict
    """
    regions = regionprops(label_im)

    x_pixels, y_pixels = label_im.shape
    detections = {}
    for idx, reg in enumerate(regions):
        x0, y0 = reg.centroid
        x0 = max(0, x0)  # avoid negatives in chip generation
        y0 = max(0, y0)  # avoid negatives in chip generation
        x0 = int(min(x_pixels, x0))
        y0 = int(min(y_pixels, y0))
        detections[idx] = {
            "coords": [x0, y0],
            "bbox": reg.bbox,
            "area": reg.area,
            "perimeter": reg.perimeter,
        }
    return detections


def remove_outliers(dnb_observations: np.ndarray) -> np.ndarray:
    """Ionospheric noise is > 1000 nano watts

    Parameters
    ----------
    dnb_observations : np.ndarray
        dnb in nanowatts

    Returns
    -------
    np.ndarray
        dnb without iono noise
    """

    logger.debug(f"Removing high intensity outliers > {OUTLIER_THRESHOLD_NW=} ")
    dnb_observations[dnb_observations > OUTLIER_THRESHOLD_NW] = 0
    return dnb_observations


def threshold_image(norm: np.ndarray) -> np.ndarray:
    """apply adaptive threshold to image

    Parameters
    ----------
    norm : _type_
        _description_

    Returns
    -------
    np.ndarray
    """
    threshold = cv2.adaptiveThreshold(
        norm,
        IMG_MAX_VALUE,
        ADAPTIVE_METHOD,
        THRESHOLD_TYPE,
        BLOCK_SIZE,
        ADAPTIVE_CONSTANT,
    )
    return threshold


def detect_vessels(dnb_observations: np.ndarray, cloud_illumination: float) -> dict:
    """core vessel detection logic using 2d kernel

    This function applies several image processing steps, followed by a
    2d kernel defined by VESSEL_KERNEL.

    1. norm
    2. adapative threshold
    3. arithmetic on thresholded image to make it compataible with a 2d kernel
    4. kernel
    5. connected components

    Parameters
    ----------
    dnb_observations : np.ndarray
    cloud_illumination : float

    Returns
    -------
    dict
    """
    clipped_masked_img = np.clip(dnb_observations, cloud_illumination, CLIP_MAX)
    norm = (clipped_masked_img - np.min(clipped_masked_img)) / (
        np.max(clipped_masked_img) - np.min(clipped_masked_img)
    )

    norm = np.array(IMG_MAX_VALUE * norm, dtype=np.uint8)
    threshold = threshold_image(norm)
    temp = np.clip(threshold, 0, 1) + 1
    img = cv2.filter2D(temp, -1, VESSEL_KERNEL)
    img = img >= VESSEL_THRESHOLD
    connected_regions = label(img, connectivity=VESSEL_CONNECTIVITY)

    return components_to_detections(connected_regions)


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
