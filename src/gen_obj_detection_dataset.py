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
from datetime import datetime
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path

import click
import numpy as np
from skyfield.almanac import find_discrete, phases
from skyfield.api import load

import utils
from utils import viirs_annotate_pipeline

# Initialize logging
logging.config.fileConfig(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "logging.conf"),
    disable_existing_loggers=False,
)
logger = logging.getLogger(__name__)
TOKEN = f"Bearer {os.environ.get('EARTHDATA_TOKEN')}"

DAYS_IN_YEAR = 365
NUMBER_OF_DAYS = 10  # Number of days to randomly sample from a given year.


def full_moons_in_doy(year: int) -> list[int]:
    """
    Returns a list of Days of Year (DOY) for each full moon in the specified year.

    Args:
    year (int): The year for which to calculate full moon DOYs.

    Returns:
    list of int: DOYs for each full moon in the specified year.
    """
    # Load ephemeris data for planetary and lunar positions
    ts = load.timescale()
    eph = load("de421.bsp")

    # Start and end times for the year
    t0 = ts.utc(year, 1, 1)
    t1 = ts.utc(year + 1, 1, 1)

    # Find times of full moons
    times, _ = find_discrete(t0, t1, phases(eph, "moon"))

    # Convert times to DOY
    full_moon_doy = [t.utc_datetime().timetuple().tm_yday for t in times]

    return full_moon_doy


# Example usage
# print(full_moons_in_doy(2023))


def list_all_days(year: int) -> list[str]:
    """
    Generates a list of all days in the year as strings.

    Parameters:
    year: int - The year for which to generate the day list.

    Returns:
    list[str] - A list of all days in the year, formatted as strings.
    """
    return [str(day) for day in range(1, DAYS_IN_YEAR + 1)]


def random_sample_days(days: list[str], n_days: int) -> list:
    """Generates a random sample of n_days from a list of days

    Parameters
    ----------
    days : list[str]
    n_days : int

    Returns
    -------
    list
    """
    return np.random.choice(a=days, size=n_days, replace=False)


def get_dark_days(year: int) -> list[str]:
    """Defines a period of darkness around new moon"""
    FULL_MOONS_DOY = full_moons_in_doy(year)
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


def get_default_cores() -> int:
    """Calculate the default number of cores: total cores minus 2, but at least 1."""
    total_cores = os.cpu_count() or 4  # Fallback to 4 if os.cpu_count() returns None
    return max(1, total_cores - 2)


@click.command()
@click.option(
    "--all-days", is_flag=True, help="Process data for every day of the specified year."
)
@click.option(
    "--year",
    default=2023,
    help="The year for which to process the data.",
    show_default=True,
)
def main(all_days: bool, year: int) -> None:
    def generate_annotated_data(all_days: bool) -> None:
        dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        dataset_dir = Path(f"viirs-dataset-{dt_string}").resolve()
        images_dir = os.path.join(dataset_dir, "images")
        annotation_dir = os.path.join(dataset_dir, "annotations")
        Path(images_dir).mkdir(parents=True, exist_ok=True)
        Path(annotation_dir).mkdir(parents=True, exist_ok=True)

        days = (
            list_all_days(year)
            if all_days
            else random_sample_days(get_dark_days(year), NUMBER_OF_DAYS)
        )
        logger.debug(f"Processing days: {days}")

        with Pool(40) as par_pool:
            for product_name in ["VNP02DNB", "VJ102DNB"]:
                for day in days:
                    times = utils.get_all_times_from_date()
                    download_and_detect_args = zip(
                        repeat(product_name),
                        repeat(year),
                        repeat(day),
                        times,
                        repeat(images_dir),
                        repeat(annotation_dir),
                    )
                    par_pool.starmap(
                        download_and_detect_one_frame, download_and_detect_args
                    )

    generate_annotated_data(all_days)


if __name__ == "__main__":
    main()
