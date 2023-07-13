""" Post Processing pipeline for VIIRS Vessel Detection"""
import logging.config
import os
from typing import List, Tuple

import numpy as np
import yaml
from scipy.stats import chisquare

from feedback_model.nets import NightLightsNet
from model import MOONLIGHT_ILLUMINATION_PERCENT
from utils import (
    IMAGE_CHIP_SIZE,
    GeoPoint,
    aurora_mask,
    calculate_e2e_cog,
    detection_near_mask,
    gas_flare_locations,
    get_chip_from_all_channels,
    land_water_mask,
    lightning_detector,
    moonlit_clouds_irradiance,
    numpy_nms,
    preprocess_raw_data,
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
    config = yaml.safe_load(file)["postprocessor"]


TRIM_DETECTIONS_EDGE = config["TRIM_DETECTIONS_EDGE"]
TRIM_DETECTIONS_EDGE_THRESHOLD = config["TRIM_DETECTIONS_EDGE_THRESHOLD"]
CHIP_HALF_WIDTH = round(IMAGE_CHIP_SIZE / 2)
N_LINES_PER_SCAN = config["N_LINES_PER_SCAN"]
MAX_DETECTIONS = config["MAX_DETECTIONS"]
NMS_THRESHOLD = config["NMS_THRESHOLD"]
NEAR_SHORE_THRESHOLD = config["NEAR_SHORE_THRESHOLD"]
FLARE_DISTANCE_THRESHOLD = config["FLARE_DISTANCE_THRESHOLD"]
CONFIDENCE_THRESHOLD = config["CONFIDENCE_THRESHOLD"]
EVAL_BATCH_SIZE = config["EVAL_BATCH_SIZE"]
SAA_BOUNDS = config["SOUTHERN_ATLANTIC_ANOMALY_BOUNDS"]


class VVDPostProcessor:
    """Class for postprocessing VIIRS vessel detections"""

    @staticmethod
    def run_pipeline(
        detections: dict, dnb_dataset: dict, filters: List, image_array: np.ndarray
    ) -> dict:
        """Apply postprocessing filters to detections and return filtered detections

        Returns
        -------
        Tuple[dict, List]
        """
        if "aurora" in filters:
            detections = remove_aurora_artifacts(detections, dnb_dataset)
        if "edge" in filters:
            detections = remove_edge_detections(detections, dnb_dataset)
        if "artifacts" in filters:
            detections = remove_image_artifacts(detections, dnb_dataset)
        if "bowtie" in filters:
            detections = remove_bowtie_artifacts(detections, dnb_dataset)
        if "near_shore" in filters:
            detections = remove_detections_near_shore(detections, dnb_dataset)

        # Avoid computing features until it is necesssary
        detections = get_detection_attributes(detections, dnb_dataset)
        detections = remove_non_local_maximum(detections)
        detections = remove_outliers(detections)
        lightning_count, gas_flare_count = 0, 0
        if "nms" in filters:
            detections = non_max_suppression(detections)
        if "south_atlantic_anomaly" in filters:
            detections = remove_noise_particles_from_saa(detections, dnb_dataset)
        if "moonlight" in filters:
            detections = remove_moonlit_clouds(detections, dnb_dataset)
        if "lightning" in filters:
            detections, lightning_count = lightning_filter(detections, image_array)
        if "gas_flares" in filters:
            detections, gas_flare_count = remove_gas_flares(detections, dnb_dataset)
        if "feedback_cnn" in filters:
            detections = feedback_cnn(detections, dnb_dataset)
        if len(detections) > MAX_DETECTIONS:
            logger.warning(f"({len(detections)}) > {MAX_DETECTIONS=}")
            detections = {}

        all_detections = {
            "vessel_detections": detections,
            "lightning_count": lightning_count,
            "gas_flare_count": gas_flare_count,
        }
        return all_detections


def feedback_cnn(detections: dict, dnb_dataset: dict) -> dict:
    """Classifies chips and removes non-vessel classifications

    Parameters
    ----------
    detections : dict
    dnb_dataset : dict
    Returns
    -------
    dict

    """
    import torch

    model = NightLightsNet()

    MODEL_PATH = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "feedback_model", "model.pt"
    )
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    chips = []

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
    skips = []
    idxs = []
    if detections is not None and len(detections) > 0:
        for chip_idx, chip in detections.items():
            x, y = chip["coords"]
            chip_data, skip = get_chip_from_all_channels(all_channels, x, y)
            chips.append(chip_data)
            skips.append(skip)
            idxs.append(chip_idx)
        all_chips = torch.tensor(np.array(chips).astype(np.float32))
        batch_predictions = []
        for batch in torch.split(all_chips, EVAL_BATCH_SIZE):
            probs = torch.nn.functional.softmax(model(batch), dim=1)
            supra_threshold = probs > CONFIDENCE_THRESHOLD
            incorrect_predictions = supra_threshold[:, 0]
            batch_predictions.append(incorrect_predictions)
        all_predictions = np.concatenate(batch_predictions)

        [
            detections.pop(idx)
            for idx, prediction in zip(idxs, all_predictions)
            if prediction
        ]
        logger.info(
            f"Removing {len(np.where(all_predictions is True)[0])} misclassifications"
        )

    return detections


def remove_noise_particles_from_saa(detections: dict, dnb_dataset: dict) -> dict:
    """removes false positive detections that are in the South Atlantic Anomaly

    Note that we already filter high energy particles by default. This is an additional
    step for false postiive detections that are not high energy but still puttative
    false positives based on the detection/chip characteristics.

    Parameters
    ----------
    detections : dict
    dnb_dataset : dict

    Returns
    -------
    dict

    """
    saa_anomalies = []
    if detections is not None and len(detections) > 0:
        for chip_idx, chip in detections.items():
            lat = chip["latitude"]
            lon = chip["longitude"]
            if (SAA_BOUNDS["SOUTH"] < lat < SAA_BOUNDS["NORTH"]) and (
                SAA_BOUNDS["WEST"] < lon < SAA_BOUNDS["EAST"]
            ):
                if chip["mean_nanowatts"] < 1:
                    saa_anomalies.append(chip_idx)
        logger.info(f"Removing {len(saa_anomalies)} south atlantic anomaly detections")
        [detections.pop(idx) for idx in saa_anomalies]
    return detections


def remove_gas_flares(detections: dict, dnb_dataset: dict) -> Tuple[dict, int]:
    """Removes a vessel detection if colocated with a known gas flare

    Obvious hot pixels are excluded from the background pixel set by
    screening out digital values over 100.

    See: https://www.mdpi.com/2072-4292/5/9/4423 for method details


    Parameters
    ----------
    detections : dict
    dnb_dataset : dict

    Returns
    -------
    detections
        detections with gas flares removed

    """
    gas_flares = []
    try:
        if dnb_dataset["m10_band"]:
            gas_flares_coordinates = gas_flare_locations(dnb_dataset)

            if detections is not None and len(detections) > 0:
                for chip_idx, chip in detections.items():
                    detection_coords = GeoPoint(
                        lat=chip["latitude"], lon=chip["longitude"]
                    )
                    distances_km = []
                    for gas_flare in gas_flares_coordinates:
                        _, distance_km = calculate_e2e_cog(detection_coords, gas_flare)
                        distances_km.append(distance_km)
                    if np.min(np.array(distances_km)) < FLARE_DISTANCE_THRESHOLD:
                        gas_flares.append(chip_idx)

                logger.debug(
                    f"Removing {len(gas_flares)} detections that overlap with flares"
                )
                [detections.pop(idx) for idx in gas_flares]
        else:
            logger.warning("M10 band not available for flare removal")
    except Exception:
        logger.exception("exception processing M10 band", exc_info=True)

    return detections, len(gas_flares)


def remove_outliers(detections: dict) -> dict:
    """removes detections that are not sufficiently different from background

    Parameters
    ----------
    detections : dict

    Returns
    -------
    dict
    """
    outliers = []
    if detections is not None and len(detections) > 0:
        for chip_idx, chip in detections.items():
            if chip["max_nanowatts"] > 1000:
                outliers.append(chip_idx)
        logger.info(f"Removing {len(outliers)} outliers")
        [detections.pop(idx) for idx in outliers]
    return detections


def remove_non_local_maximum(detections: dict, ndev: int = 4) -> dict:
    """removes detections that are not sufficiently different from background

    Parameters
    ----------
    detections : dict
    ndev : int, optional
         by default 4

    Returns
    -------
    dict
    """
    non_local_maxima = []
    if detections is not None and len(detections) > 0:
        for chip_idx, chip in detections.items():
            if chip["max_nanowatts"] <= (ndev * chip["mean_nanowatts"]):
                non_local_maxima.append(chip_idx)

        logger.info(f"Removing {len(non_local_maxima)} weak detections")
        [detections.pop(idx) for idx in non_local_maxima]
    return detections


def remove_moonlit_clouds(detections: dict, dnb_dataset: dict, ndev: int = 2) -> dict:
    """removes the moonlit clouds that are not > ndev*median cloud illumination
    Note that the base model already factors in cloud illimunation.

    Note that not every single false positive will be removed at all times,
    because the cloud mask is not perfect, and detections can resemble small
    wispy clouds in some cases.

    Parameters
    ----------
    detections : dict
    dnb_dataset : dict

    Returns
    -------
    dict
        detections removing moonlit clouds
    """
    if np.mean(dnb_dataset["moonlight"]) > MOONLIGHT_ILLUMINATION_PERCENT:
        if "cloud_mask" in dnb_dataset and len(detections) > 0:
            (_, cloud_median_illumination) = moonlit_clouds_irradiance(dnb_dataset)

            moonlit_detections = []
            try:
                for chip_idx, chip in detections.items():
                    if chip["max_nanowatts"] < cloud_median_illumination * ndev:
                        moonlit_detections.append(chip_idx)
            except Exception as e:
                moonlit_detections.append(chip_idx)
                logger.exception(str(e), exc_info=True)

            logger.debug(f"Removing {len(moonlit_detections)} edge detections")
            [detections.pop(idx) for idx in moonlit_detections]

    return detections


def non_max_suppression(detections: dict) -> dict:
    """applied non max suppression to overlapping detections

    Parameters
    ----------
    detections : dict


    Returns
    -------
    dict
        detections after non max suppression
    """
    return numpy_nms(detections, thresh=NMS_THRESHOLD)


def remove_edge_detections(detections: dict, dnb_dataset: dict) -> dict:
    """Removes detections on edge of frame

    This is an optional filter that can be applied to remove detections alongside
    the edge of the frame. There are several reasons to apply this filter.
    1. The edge of the frame is often noisy and often causes false positives
    2. If a detection is on the edge of the frame, it is not possible to examine the
    surrounding context which is needed for 1) the CNN to make a decision within the
    feedback layer (i.e. if we simply padded the data, we would not know if we were
    padding over land, for example).

    Parameters
    ----------
    detections : dict
    dnb_dataset : dict

    Returns
    -------
    dict
    """
    x_pixels, y_pixels = dnb_dataset["dnb"]["data"].shape
    if len(detections) > 0:
        edge_detections = []
        for chip_idx, chip in detections.items():
            x0, y0 = chip["coords"]
            if (
                (y0 < TRIM_DETECTIONS_EDGE)
                or (x0 < TRIM_DETECTIONS_EDGE)
                or (y0 > (y_pixels - TRIM_DETECTIONS_EDGE))
                or (x0 > (x_pixels - TRIM_DETECTIONS_EDGE))
            ):
                edge_detections.append(chip_idx)
        logger.debug(f"Removing {len(edge_detections)} detections on edge of frame")
        [detections.pop(idx) for idx in edge_detections]

    return detections


def remove_detections_near_shore(detections: dict, dnb_dataset: dict) -> dict:
    """_summary_

    Parameters
    ----------
    detections : dict
        _description_
    dnb_dataset : dict
        _description_

    Returns
    -------
    dict
        _description_
    """
    _, mask = land_water_mask(dnb_dataset["dnb"]["data"], dnb_dataset["land_sea_mask"])
    if len(detections) > 0:
        near_shore_detections = []
        for chip_idx, chip in detections.items():
            x0, y0 = chip["coords"]
            if detection_near_mask(
                mask, (x0, y0), distance_threshold=NEAR_SHORE_THRESHOLD
            ):
                near_shore_detections.append(chip_idx)
        n_near_shore_detections = len(near_shore_detections)
        logger.debug(
            f"Removing {n_near_shore_detections=} within {NEAR_SHORE_THRESHOLD=} meters"
        )
        [detections.pop(idx) for idx in near_shore_detections]
    return detections


def remove_aurora_artifacts(detections: dict, dnb_dataset: dict) -> dict:
    """
    This filter only runs at latitude greater than AURORA_LAT_THRESHOLD
    to avoid the possibility of removing detections that are not false positives
    outside of the Auroral Ring.

    See: https://www.swpc.noaa.gov/products/aurora-30-minute-forecast

    """
    img_contains_aurora, mask = aurora_mask(dnb_dataset)
    if img_contains_aurora and len(detections) > 0:
        near_aurora_detections = []
        for chip_idx, chip in detections.items():
            x0, y0 = chip["coords"]
            if detection_near_mask(mask, (x0, y0), 10):
                near_aurora_detections.append(chip_idx)
        logger.debug(f"Removing {len(near_aurora_detections)} near aurora detections")
        [detections.pop(idx) for idx in near_aurora_detections]
    return detections


def remove_image_artifacts(detections: dict, dnb_dataset: dict) -> dict:
    """_summary_

    Parameters
    ----------
    detections : dict
        _description_
    dnb_dataset : dict
        _description_

    Returns
    -------
    dict
        _description_
    """
    false_positives = []
    left_detections = []
    x_pixels, y_pixels = dnb_dataset["dnb"]["data"].shape
    if detections is not None and len(detections) > 0:
        for chip_idx, chip in detections.items():
            x0, y0 = chip["coords"]
            if int(y0) < (0.25 * y_pixels):
                left_detections.append(x0)
                false_positives.append(chip_idx)

        if (
            is_uniform(left_detections, x_pixels)
            and len(false_positives) > 20
            and len(false_positives) / len(detections) > 0.85
        ):
            logger.debug(f"Removing {len(false_positives)} due to moonlight artifacts")
            [detections.pop(idx) for idx in false_positives]

    return detections


def remove_bowtie_artifacts(detections: dict, dnb_dataset: dict) -> dict:
    """Remove detections that are likely false positives due to bowtie artifacts

    Checks uniformity of detections across rows/x which is a statistically unlikely
    presentation of legitimate detections.

    Parameters
    ----------
    detections : dict

    dnb_dataset : dict


    Returns
    -------
    dict

    """
    left_false_positives = []
    right_false_positives = []
    left_detections = []
    right_detections = []
    x_pixels, y_pixels = dnb_dataset["dnb"]["data"].shape
    if detections is not None and len(detections) > 0:
        for chip_idx, chip in detections.items():
            x0, y0 = chip["coords"]
            if int(y0) < (0.25 * y_pixels):
                left_detections.append(x0)
                left_false_positives.append(chip_idx)
            elif int(y0) > (0.75 * y_pixels):
                right_detections.append(x0)
                right_false_positives.append(chip_idx)
        if len(left_false_positives + right_false_positives) / len(detections) > 0.85:
            if is_uniform(left_detections, x_pixels):
                logger.debug(
                    f"Removing {len(left_false_positives)=} due to bowtie artifacts"
                )
                [detections.pop(idx) for idx in left_false_positives]

            if is_uniform(right_detections, x_pixels):
                logger.debug(
                    f"Removing {len(right_false_positives)=} due to bowtie artifacts"
                )
                [detections.pop(idx) for idx in right_false_positives]

    return detections


def is_uniform(counts: List, pixels: int) -> bool:
    """tests whether counts appear uniform from dist pixels

    Note on the retry logic below:

    Occasionally image artifacts results in tens of thousands of detections. The
    the chi square test will always return p-value ~ 0.0 even if the results look
    reasonably uniform for this test. For those edge cases, we resample some small
    percentage, and reattempt the test.

    Parameters
    ----------
    counts : List
    x_pixels : int

    Returns
    -------
    bool
        whether data is uniform
    """
    attempts = 0
    while attempts < 10:
        if len(counts) > 1000:
            counts = np.random.choice(counts, 100)
        inds = np.digitize(counts, np.linspace(0, pixels, num=7))
        unique, counts = np.unique(inds, return_counts=True)
        stat, pval = chisquare(counts)
        attempts += 1
        if pval > 0.05:
            return True

    return False


def get_detection_attributes(detections: dict, dnb_dataset: dict) -> dict:
    """_summary_

    Parameters
    ----------
    detections : dict
        _description_
    dnb_dataset : dict
        _description_

    Returns
    -------
    dict
        _description_
    """
    if detections is not None and len(detections) > 0:
        metadata = dnb_dataset["dnb"]["metadata"]
        raw_data = dnb_dataset["dnb"]["data"]
        valid_min = float(metadata["valid_min"])
        valid_max = float(metadata["valid_max"])
        logger.debug(f"DNB metadata: valid min: {valid_min}, valid max: {valid_max}")
        raw_watts = np.clip(raw_data, valid_min, valid_max)
        raw_nanowatts = raw_watts * 1e9
        latitude_array = dnb_dataset["latitude"]
        longitude_array = dnb_dataset["longitude"]

        for chip_idx, chip in detections.items():
            xmin, ymin, xmax, ymax = chip["bbox"]
            x0, y0 = chip["coords"]
            try:
                xmin_chip = int(x0 - CHIP_HALF_WIDTH)
                xmax_chip = int(x0 + CHIP_HALF_WIDTH)
                ymin_chip = int(y0 - CHIP_HALF_WIDTH)
                ymax_chip = int(y0 + CHIP_HALF_WIDTH)
                chip["latitude"] = latitude_array[x0, y0]
                chip["longitude"] = longitude_array[x0, y0]
                chip["mean_nanowatts"] = np.mean(
                    raw_nanowatts[xmin_chip:xmax_chip, ymin_chip:ymax_chip]
                )
                chip["median_nanowatts"] = np.median(
                    raw_nanowatts[xmin_chip:xmax_chip, ymin_chip:ymax_chip]
                )
                chip["max_nanowatts"] = np.max(raw_nanowatts[xmin:xmax, ymin:ymax])
                chip["nanowatt_variance"] = np.var(
                    raw_nanowatts[xmin_chip:xmax_chip, ymin_chip:ymax_chip]
                )
                chip["min_nanowatts"] = np.min(
                    raw_nanowatts[xmin_chip:xmax_chip, ymin_chip:ymax_chip]
                )
                chip["moonlight_illumination"] = dnb_dataset["moonlight"][x0, y0]
                chip["clear_sky_confidence"] = dnb_dataset["cloud_mask"][x0, y0]

            except Exception:
                chip["mean_nanowatts"] = np.nan
                chip["median_nanowatts"] = np.nan
                chip["max_nanowatts"] = np.nan
                chip["nanowatt_variance"] = np.nan
                chip["min_nanowatts"] = np.nan
                chip["moonlight_illumination"] = np.nan
                chip["clear_sky_confidence"] = np.nan
    return detections


def lightning_filter(detections: dict, image_array: np.ndarray) -> Tuple[dict, int]:
    """_summary_

    Parameters
    ----------
    detections : dict
        _description_
    image_array : np.ndarray
        _description_

    Returns
    -------
    dict
        _description_
    """
    lightning_detections = []
    try:
        lightning_mask, lightning_count = lightning_detector(image_array)
    except Exception:
        logger.exception(
            "Exception computing lightning mask, not able to detect lightning",
            exc_info=True,
        )
        lightning_mask = np.ones(image_array.shape)
        lightning_count = 0

    if lightning_count:
        logger.debug("Detected lightning in this frame, creating lightning mask")
    if detections is not None and len(detections) > 0:
        for chip_idx, chip in detections.items():
            x0, y0 = chip["coords"]
            detection_near_lightning = detection_near_mask(
                lightning_mask, (x0, y0), 1000
            )

            if detection_near_lightning:
                lightning_detections.append(chip_idx)
        logger.debug(f"Removing {len(lightning_detections)} due to lightning")
        [detections.pop(idx) for idx in lightning_detections]

    return detections, lightning_count
