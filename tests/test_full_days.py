"""These tests evaluate a full day of imagery (~50 GB) under new moon and full moon
conditions. Because of the file size and time to inference, these tests use
parallel processing and are excluded except when env variable VIIRS_TEST_LEVEL
is set to "dev"
"""
import logging
import os
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import pytest

from src.utils import get_detections_from_one_frame

logger = logging.getLogger(__name__)

TEST_FILE_INPUT_DIR = os.path.abspath("tests/test_files")
TEST_FILE_OUTPUT_DIR = os.path.abspath("tests/test_outputs")
DATE = "15/02/2023"
N_CORES = 10


@pytest.mark.skipif(
    os.getenv("VIIRS_TEST_LEVEL", "default_not_set") != "dev",
    reason="this test inferences a full day of imagery, and excluded by default",
)
def test_new_moon() -> None:
    """tests whole planet coverage under new moon"""
    product_name = "VJ102DNB_NRT"
    year = "2023"
    doy = "078"
    new_moon_dir = os.path.join(TEST_FILE_INPUT_DIR, "new_moon", "images")
    new_moon_output_dir = os.path.join(TEST_FILE_INPUT_DIR, "new_moon", "outputs")
    files = Path(new_moon_dir).rglob("VJ102DNB.A2023078.*.nc")
    times = [file.name.split(".")[2] for file in files]

    download_and_detect_args = zip(
        repeat(product_name),
        repeat(year),
        repeat(doy),
        times,
        repeat(new_moon_dir),
        repeat(new_moon_output_dir),
    )

    with Pool(N_CORES) as par_pool:
        par_pool.starmap(get_detections_from_one_frame, download_and_detect_args)
    detections = sum(
        [
            len(pd.read_csv(file))
            for file in Path(new_moon_output_dir).rglob("detections.csv")
        ]
    )
    assert detections == 13075


@pytest.mark.skipif(
    os.getenv("VIIRS_TEST_LEVEL", "default_not_set") != "dev",
    reason="this test inferences a full day of imagery, and excluded by default",
)
def test_full_moon() -> None:
    """tests whole planet coverage under full moon"""

    product_name = "VJ102DNB_NRT"
    year = "2023"
    doy = "067"
    full_moon_dir = os.path.join(TEST_FILE_INPUT_DIR, "full_moon", "images")
    full_moon_output_dir = os.path.join(TEST_FILE_OUTPUT_DIR, "full_moon", "images")
    files = Path(full_moon_dir).rglob("VJ102DNB.A2023067.*.nc")
    times = [file.name.split(".")[2] for file in files]

    download_and_detect_args = zip(
        repeat(product_name),
        repeat(year),
        repeat(doy),
        times,
        repeat(full_moon_dir),
        repeat(full_moon_output_dir),
    )

    with Pool(N_CORES) as par_pool:
        par_pool.starmap(get_detections_from_one_frame, download_and_detect_args)
    detections = sum(
        [
            len(pd.read_csv(file))
            for file in Path(full_moon_output_dir).rglob("detections.csv")
        ]
    )

    assert detections == 1010
