""" generate a dataset of annotated images from event id urls
(this is used as part of a continuous retraining loop) -- we train on correct
labels (from the gen_ob_detection_dataset output) and incorrect labels (from production)
This script generates the training data for the incorrect classification from
production. A sample of incorrect predictions is provided under
feedback/incorrect_detections.txt from which the training data is created.
"""
import logging
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import utils
from preprocessor import extract_data

TOKEN = f"Bearer {os.environ.get('EARTHDATA_TOKEN')}"
logger = logging.getLogger(__name__)
MODEL_DIR = "feedback_model"
IMG_DIR = Path(os.path.join(MODEL_DIR, "vvd_annotations")).resolve()
IMG_DIR.mkdir(parents=True, exist_ok=True)
INCORRECT_OUTPUT_DIR = Path(os.path.join(IMG_DIR, "incorrect")).resolve()
INCORRECT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INCORRECT_DETECTIONS = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    MODEL_DIR,
    "vvd_annotations",
    "list_incorrect_detections.txt",
)


def main(event_id: str) -> None:
    """_summary_

    Parameters
    ----------
    event_id : str
        _description_
    """
    product_name, year, doy, time, lat, lon = utils.parse_event_id(event_id)

    dnb_url = utils.get_dnb_filename(product_name, year, doy, time)
    geo_url = utils.get_geo_filename(product_name, year, doy, time)
    phys_url = utils.get_cld_filename(product_name, year, doy, time)

    dnb_path = os.path.join(IMG_DIR, dnb_url.split("/")[-1])

    if not os.path.exists(dnb_path):
        with open(dnb_path, "w+b") as fh:
            utils.download_url(dnb_url, TOKEN, fh)

    geo_path = os.path.join(IMG_DIR, geo_url.split("/")[-1])
    if not os.path.exists(geo_path):
        with open(geo_path, "w+b") as fh:
            utils.download_url(geo_url, TOKEN, fh)

    phys_path = os.path.join(IMG_DIR, phys_url.split("/")[-1])
    if not os.path.exists(phys_path):
        with open(phys_path, "w+b") as fh:
            utils.download_url(phys_url, TOKEN, fh)

    dnb_dataset = extract_data(dnb_path, geo_path, phys_path)

    lon = round(float(lon), 6)

    coords = np.where(dnb_dataset["latitude"] == float(lat))
    try:
        if len(coords) > 1:
            xs = coords[0]
            ys = coords[1]
            for x_candidate, y_candidate in zip(xs, ys):
                if math.isclose(
                    dnb_dataset["longitude"][x_candidate, y_candidate],
                    lon,
                    rel_tol=1e-5,
                ):
                    x = x_candidate
                    y = y_candidate
                    break

        else:
            x_candidate = coords[0][0]
            y_candidate = coords[1][0]
            if math.isclose(
                dnb_dataset["longitude"][x_candidate, y_candidate], lon, rel_tol=1e-5
            ):
                x = x_candidate
                y = y_candidate

        dnb_observations, _, _ = utils.preprocess_raw_data(dnb_dataset)
        all_channels = np.stack(
            [
                dnb_observations,
                dnb_dataset["land_sea_mask"],
                dnb_dataset["moonlight"],
                dnb_dataset["cloud_mask"],
            ],
            axis=0,
        )

        chip, _ = utils.get_chip_from_all_channels(all_channels, x, y)
        if chip.any():
            out_arr_filename = f"{product_name}.{year}.{doy}.{time}.{lat}_{lon}.npy"
            out_img_filename = f"{product_name}.{year}.{doy}.{time}.{lat}_{lon}.jpeg"
            out_arr_path = os.path.join(INCORRECT_OUTPUT_DIR, out_arr_filename)
            out_img_path = os.path.join(INCORRECT_OUTPUT_DIR, out_img_filename)
            np.save(out_arr_path, chip)

            plt.imsave(  # for visualization
                out_img_path,
                np.clip(chip[0, :, :], 0, 100),
            )
        os.remove(dnb_path)
        os.remove(geo_path)
        os.remove(phys_path)
    except Exception as e:
        logger.exception(e)


def event_ids_from_file() -> None:
    """_summary_"""

    event_id_file = open(
        INCORRECT_DETECTIONS,
        "r",
    )

    for line in event_id_file.readlines():
        event_id = line.strip()
        main(event_id)


def event_ids_from_arg(event_id: str) -> None:
    """_summary_

    Parameters
    ----------
    event_id : str
        _description_
    """
    main(event_id)


if __name__ == "__main__":
    event_ids_from_file()
