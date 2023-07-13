import logging
import os
from pathlib import Path

import pytest

from src.utils import format_detections_df, viirs_annotate_pipeline

logger = logging.getLogger(__name__)

TEST_FILE_INPUT_DIR = os.path.abspath("tests/test_files")
TEST_FILE_OUTPUT_DIR = os.path.abspath("tests/test_outputs")

DNB_FILES_LIST = [
    "VJ102DNB.A2023018.1342.021.2023018164614.nc",
    "VJ102DNB.A2023001.0742.021.2023001094409.nc",
    "VJ102DNB.A2022354.2130.021.2022355000855.nc",
    "VNP02DNB.A2023053.1900.002.2023053213251.nc",
]
GEO_FILES_LIST = [
    "VJ103DNB.A2023018.1342.021.2023018162103.nc",
    "VJ103DNB.A2023001.0742.021.2023001090617.nc",
    "VJ103DNB.A2022354.2130.021.2022354234249.nc",
    "VNP03DNB.A2023053.1900.002.2023053211409.nc",
]
IDS = [
    "Hudson Bay with some image artifacts and mild aurora",
    "North Pacific, multiple fleets, deep ocean, some noise artifacts)",
    "Arabian Sea, new moon, few clouds, squid fishing fleet)",
    "South east Asia, Bay of Bengal, Gulf of Thailand, Andaman sea, South China Sea",
]
N_DETECTIONS = [5, 54, 637, 2586]


@pytest.mark.parametrize(
    "dnb_filename, geo_filename, n_detections",
    zip(DNB_FILES_LIST, GEO_FILES_LIST, N_DETECTIONS),
    ids=IDS,
)
def test_true_positives(
    dnb_filename: str, geo_filename: str, n_detections: int
) -> None:
    detections, status = viirs_annotate_pipeline(
        dnb_filename, geo_filename, TEST_FILE_INPUT_DIR, TEST_FILE_OUTPUT_DIR
    )

    assert len(detections) == n_detections


def test_elvidge_et_al_2015_frame() -> None:
    """VIIRS validation frame from Elvidge et. al 2015

    C. Elvidge, M. Zhizhin, K. Baugh, and F.-C. Hsu,
    “Automatic Boat Identification System for VIIRS Low Light Imaging Data,”
    Remote Sensing, vol. 7, no. 3, pp. 3020--3036, Mar. 2015.

    """
    dnb_filename = "VNP02DNB.A2014270.1836.002.2021028152649.nc"
    geo_filename = "VNP03DNB.A2014270.1836.002.2021028132210.nc"

    all_detections, status = viirs_annotate_pipeline(
        dnb_filename,
        geo_filename,
        TEST_FILE_INPUT_DIR,
        TEST_FILE_OUTPUT_DIR,
    )
    xmin = 600
    xmax = 970
    ymin = 2700
    ymax = 3225
    detections_df = format_detections_df(
        all_detections, f"{Path(dnb_filename).stem}.npy"
    )
    n_detections_in_crop = len(
        detections_df[
            (detections_df["xmin"] < xmax)
            & (detections_df["xmin"] > xmin)
            & (detections_df["ymin"] < ymax)
            & (detections_df["ymin"] > ymin)
        ]
    )

    assert len(all_detections) == 1320
    assert n_detections_in_crop == 527


def test_gas_flare_removal_north_sea() -> None:
    """multiple fleets, deep ocean, oil platforms with gas flares"""
    dnb_filename = "VJ102DNB.A2022362.0154.021.2022362055600.nc"
    geo_filename = "VJ103DNB.A2022362.0154.021.2022362052511.nc"
    modraw_filename = "VJ102MOD.A2022362.0154.002.2022362115107.nc"
    modgeo_filename = "VJ103MOD.A2022362.0154.002.2022362095104.nc"
    detections, status = viirs_annotate_pipeline(
        dnb_filename,
        geo_filename,
        TEST_FILE_INPUT_DIR,
        TEST_FILE_OUTPUT_DIR,
        modraw=modraw_filename,
        modgeo=modgeo_filename,
    )
    assert len(detections) == 77


def test_south_atlantic_anomaly() -> None:
    """South atlantic anomaly produces many false positives"""
    dnb_filename = "VNP02DNB.A2023083.0254.002.2023083104946.nc"
    geo_filename = "VNP03DNB.A2023083.0254.002.2023083103206.nc"
    detections, status = viirs_annotate_pipeline(
        dnb_filename, geo_filename, TEST_FILE_INPUT_DIR, TEST_FILE_OUTPUT_DIR
    )
    assert len(detections) == 98


def test_lightning_removal() -> None:
    """lightning causes false positive detections if not identified and removed"""
    dnb_filename = "VJ102DNB.A2023031.0130.021.2023031034239.nc"
    geo_filename = "VJ103DNB.A2023031.0130.021.2023031025754.nc"
    detections, status = viirs_annotate_pipeline(
        dnb_filename, geo_filename, TEST_FILE_INPUT_DIR, TEST_FILE_OUTPUT_DIR
    )
    assert len(detections) == 23


def test_edge_artifacts() -> None:
    """edge artifacts are present in a lot of VIIRS images"""
    dnb_filename = "VJ102DNB.A2023020.1306.021.2023020150928.nc"
    geo_filename = "VJ103DNB.A2023020.1306.021.2023020144457.nc"
    detections, status = viirs_annotate_pipeline(
        dnb_filename, geo_filename, TEST_FILE_INPUT_DIR, TEST_FILE_OUTPUT_DIR
    )

    assert len(detections) == 8


def test_corner_artifacts() -> None:
    """corner artifacts are typically caused by Low_Gain samples

    This test case is  centered on the blue hole with squid fishing,
    Aurora, satellite artifacts and low gain samples are also present
    """
    dnb_filename = "VJ102DNB.A2023020.0512.021.2023020064541.nc"
    geo_filename = "VJ103DNB.A2023020.0512.021.2023020062326.nc"
    detections, status = viirs_annotate_pipeline(
        dnb_filename, geo_filename, TEST_FILE_INPUT_DIR, TEST_FILE_OUTPUT_DIR
    )

    assert len(detections) == 0


def test_moonlit_clouds() -> None:
    """moonlit clouds may show false positives around full moons"""
    dnb_filename = "VJ102DNB.A2023009.1018.021.2023009135632.nc"
    geo_filename = "VJ103DNB.A2023009.1018.021.2023009133026.nc"
    cloud_mask = "CLDMSK_L2_VIIRS_NOAA20.A2023009.1018.001.nrt.nc"
    detections, status = viirs_annotate_pipeline(
        dnb_filename,
        geo_filename,
        TEST_FILE_INPUT_DIR,
        TEST_FILE_OUTPUT_DIR,
        cloud_filename=cloud_mask,
    )

    assert len(detections) == 0


def test_retain_cloud_free_detections() -> None:
    """vessels near but not under moonlit clouds should be detected"""
    dnb_filename = "VNP02DNB.A2023008.2124.002.2023009045328.nc"
    geo_filename = "VNP03DNB.A2023008.2124.002.2023009042918.nc"
    cloud_mask = "CLDMSK_L2_VIIRS_SNPP.A2023008.2124.001.nrt.nc"
    detections, status = viirs_annotate_pipeline(
        dnb_filename,
        geo_filename,
        TEST_FILE_INPUT_DIR,
        TEST_FILE_OUTPUT_DIR,
        cloud_filename=cloud_mask,
    )

    assert len(detections) == 153


def test_return_status_land_only() -> None:
    """Model should not return detections on land."""
    dnb_filename = "VNP02DNB.A2022348.1142.002.2022348173537.nc"
    geo_filename = "VNP03DNB.A2022348.1142.002.2022348171655.nc"
    _, status = viirs_annotate_pipeline(
        dnb_filename, geo_filename, TEST_FILE_INPUT_DIR, TEST_FILE_OUTPUT_DIR
    )
    assert "land_only" in status


def test_return_status_during_daytime() -> None:
    """Model should not return detections during daytime."""
    dnb_filename = "VNP02DNB.A2022348.1142.002.2022348173537.nc"
    geo_filename = "VNP03DNB.A2022348.1142.002.2022348171655.nc"
    _, status = viirs_annotate_pipeline(
        dnb_filename, geo_filename, TEST_FILE_INPUT_DIR, TEST_FILE_OUTPUT_DIR
    )
    assert "daytime" in status


def test_aurora_filter_removal_false_positives() -> None:
    """Aurora filter should remove false positives caused by auroral glow"""
    dnb_filename = "VJ102DNB.A2022360.2318.021.2022361013428.nc"
    geo_filename = "VJ103DNB.A2022360.2318.021.2022361011214.nc"
    filtered_detections, _ = viirs_annotate_pipeline(
        dnb_filename, geo_filename, TEST_FILE_INPUT_DIR, TEST_FILE_OUTPUT_DIR
    )
    assert len(filtered_detections) == 0


def test_aurora_filter_retain_true_positives() -> None:
    """Aurora filter should retain true positives outside of auroral zones"""
    dnb_filename = "VJ102DNB.A2022365.0054.021.2022365043024.nc"
    geo_filename = "VJ103DNB.A2022365.0054.021.2022365034714.nc"
    filtered_detections, _ = viirs_annotate_pipeline(
        dnb_filename, geo_filename, TEST_FILE_INPUT_DIR, TEST_FILE_OUTPUT_DIR
    )
    assert len(filtered_detections) == 87


def test_remove_image_artifacts() -> None:
    """Image artifacts should be removed from the model"""
    dnb_filename = "VNP02DNB.A2022365.0854.002.2022365110350.nc"
    geo_filename = "VNP03DNB.A2022365.0854.002.2022365104212.nc"
    filtered_detections, _ = viirs_annotate_pipeline(
        dnb_filename, geo_filename, TEST_FILE_INPUT_DIR, TEST_FILE_OUTPUT_DIR
    )
    assert len(filtered_detections) == 0
