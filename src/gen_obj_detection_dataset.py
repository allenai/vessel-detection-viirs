"""
This module downloads a VIIRS dataset, annotates it, and saves the detections to disk
along with the corresponding imagery (in .npy format). Each image will be annotated
with the following data:
.
├── EXAMPLE_IMAGE.npy (4 channel image used for model training)
├── detections.csv (all detections in csv format )
├── detections.jpg (all detections drawn on original DNB image)
└── image_chips # cropped detection (jpeg, image array, and csv file)

Note that to run this script you will need to have a valid Earthdata token stored as
an environment variable. See the README for more details.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging.config
import os
import os.path
from datetime import date, datetime
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple

import numpy as np

import utils
from utils import viirs_annotate_pipeline

logging.config.fileConfig(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "logging.conf"),
    disable_existing_loggers=False,
)

logger = logging.getLogger(__name__)
TOKEN = f"Bearer {os.environ.get('EARTHDATA_TOKEN')}"

YEAR = 2022
DAYS_IN_YEAR = 365
NUMBER_OF_DAYS = 1  # in the random sample
N_CORES = 2

FULL_MOONS_2022 = [
    (YEAR, 1, 17),
    (YEAR, 2, 16),
    (YEAR, 3, 18),
    (YEAR, 4, 16),
    (YEAR, 5, 16),
    (YEAR, 6, 14),
    (YEAR, 7, 13),
    (YEAR, 8, 11),
    (YEAR, 9, 10),
    (YEAR, 10, 9),
    (YEAR, 11, 8),
    (YEAR, 12, 7),
]


def random_sample_days(days: List[str], n_days: int) -> List:
    """Generates a random sample of n_days from a list of days

    Parameters
    ----------
    days : List[str]
    n_days : int

    Returns
    -------
    List
    """
    return np.random.choice(a=days, size=n_days, replace=False)


def get_dark_days(full_moons: List[Tuple[int, int, int]]) -> List[str]:
    """Defines a period of darkness around new moon"""

    FULL_MOONS_DOY = [date(*full_moon).timetuple().tm_yday for full_moon in full_moons]
    start = np.array(FULL_MOONS_DOY) - 7
    end = np.array(FULL_MOONS_DOY) + 8
    bright_times = [[beg, end] for beg, end in zip(start, end)]

    bright_days = []
    for doy in range(1, 365):
        for bright_period in bright_times:
            if doy in range(bright_period[0], bright_period[1]):
                bright_days.append(doy)

    dark_days = list(set(range(1, 365)) - set(bright_days))
    dark_days = list(doy for doy in dark_days)
    dark_days_str = list(map(str, dark_days))
    return dark_days_str


def download_and_detect_one_frame(
    product_name: str,
    year: str,
    doy: str,
    time: str,
    image_dir: str,
    annotation_dir: str,
) -> None:
    """Downloads dataset from one area/time and runs inference pipeline on it

    Parameters
    ----------
    product_name : str
    year : str
    doy : str
    time : str
    image_dir : str
    annotation_dir : str
    """
    try:
        dnb_url = utils.get_dnb_filename(product_name, year, doy, time)
        geo_url = utils.get_geo_filename(product_name, year, doy, time)
        phys_url = utils.get_cld_filename(product_name, year, doy, time)

        dnb_path = os.path.join(image_dir, dnb_url.split("/")[-1])

        if not os.path.exists(dnb_path):
            with open(dnb_path, "w+b") as fh:
                utils.download_url(dnb_url, TOKEN, fh)

        geo_path = os.path.join(image_dir, geo_url.split("/")[-1])
        if not os.path.exists(geo_path):
            with open(geo_path, "w+b") as fh:
                utils.download_url(geo_url, TOKEN, fh)

        phys_path = os.path.join(image_dir, phys_url.split("/")[-1])
        if not os.path.exists(phys_path):
            with open(phys_path, "w+b") as fh:
                utils.download_url(phys_url, TOKEN, fh)

        viirs_annotate_pipeline(
            Path(dnb_path).name,
            Path(geo_path).name,
            input_dir=image_dir,
            output_dir=annotation_dir,
            cloud_filename=phys_path,
        )

    except Exception as e:
        logger.exception(
            f"Error processing: {product_name=}, {year=}, {doy=}, {time=}: {e}"
        )
    finally:
        try:
            os.remove(dnb_path)
        except UnboundLocalError:
            logger.exception(f"Error removing {dnb_path}")
        try:
            os.remove(geo_path)
        except UnboundLocalError:
            logger.exception(f"Error removing {dnb_path}")
        try:
            os.remove(phys_path)
        except UnboundLocalError:
            logger.exception(f"Error removing {dnb_path}")


def generate_annotated_data() -> None:
    """Runs the inference pipeline against a random sample"""
    # datetime object containing current date and time

    dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

    dataset_dir = Path(f"viirs-dataset-{dt_string}").resolve()
    images_dir = os.path.join(dataset_dir, "images")
    annotation_dir = os.path.join(dataset_dir, "annotations")
    Path(images_dir).mkdir(parents=True, exist_ok=True)
    Path(annotation_dir).mkdir(parents=True, exist_ok=True)

    with Pool(N_CORES) as par_pool:
        dark_days = random_sample_days(get_dark_days(FULL_MOONS_2022), NUMBER_OF_DAYS)
        logger.debug(f"Downloading days: {dark_days}")

        for product_name in ["VNP02DNB", "VJ102DNB"]:
            for day in dark_days:
                times = utils.get_all_times_from_date()
                download_and_detect_args = zip(
                    repeat(product_name),
                    repeat(YEAR),
                    repeat(day),
                    times,
                    repeat(images_dir),
                    repeat(annotation_dir),
                )
                par_pool.starmap(
                    download_and_detect_one_frame, download_and_detect_args
                )


if __name__ == "__main__":
    generate_annotated_data()
