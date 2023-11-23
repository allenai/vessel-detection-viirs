"""tests for the main model method
"""
import numpy as np

from src import model
from src.utils import IMAGE_CHIP_SIZE, get_chips


def create_single_detection_dataset() -> dict:
    """"""
    x_pixels, y_pixels = (3216, 4064)

    dnb_array = np.zeros((x_pixels, y_pixels))
    # nanowatts (ultimately converted to watts)
    dnb_array[int(x_pixels / 2), int(y_pixels / 2)] = 1e-9
    dnb_metadata = {
        "valid_min": "0",
        "valid_max": "100000",
        "startDirection": "Descending",
    }
    dnb_quality_array = np.zeros((x_pixels, y_pixels))
    latitude_array = np.random.random_sample((x_pixels, y_pixels))
    longitude_array = np.random.random_sample((x_pixels, y_pixels))
    moonlight_array = np.zeros((x_pixels, y_pixels))
    # put this in deep ocean with no land
    land_sea_array = np.ones((x_pixels, y_pixels)) * 7

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
    }


def create_moonlight_dataset() -> dict:
    """ """
    x_pixels, y_pixels = (3216, 4064)

    dnb_array = np.zeros((x_pixels, y_pixels))

    dnb_array[1, 10] = 1e-9
    dnb_array[1, -10] = 1e-9
    dnb_array[-1, 10] = 1e-9
    dnb_array[-1, -10] = 1e-9

    dnb_metadata = {
        "valid_min": "0",
        "valid_max": "100000",
        "startDirection": "Descending",
    }
    dnb_quality_array = np.zeros((x_pixels, y_pixels))
    latitude_array = np.random.random_sample((x_pixels, y_pixels))
    longitude_array = np.random.random_sample((x_pixels, y_pixels))
    moonlight_array = np.ones((x_pixels, y_pixels)) * 100  # i.e. full moon
    land_sea_array = np.ones((x_pixels, y_pixels)) * 7

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
    }


def create_edge_detection_dataset() -> dict:
    """
    This synthetic dataset has 4 vessels at each corner
    """
    x_pixels, y_pixels = (3216, 4064)

    dnb_array = np.zeros((x_pixels, y_pixels))

    dnb_array[3, 10] = 1e-9  # avoid edge that is cropped to 8 pixels
    dnb_array[3, -10] = 1e-9
    dnb_array[-3, 10] = 1e-9  # avoid edge that is cropped to 8 pixels
    dnb_array[-3, -10] = 1e-9

    dnb_metadata = {
        "valid_min": "0",
        "valid_max": "100000",
        "DayNightFlag": "Night",
        "startDirection": "Descending",
    }
    dnb_quality_array = np.zeros((x_pixels, y_pixels))
    latitude_array = np.random.random_sample((x_pixels, y_pixels))
    longitude_array = np.random.random_sample((x_pixels, y_pixels))
    moonlight_array = np.zeros((x_pixels, y_pixels))
    # put this in deep ocean with no land
    land_sea_array = np.ones((x_pixels, y_pixels)) * 7

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
    }


def test_vvd_cv_model_e2e() -> None:
    """tests that detection is returned at correct location"""
    dnb_dataset = create_single_detection_dataset()
    detections, input_image = model.vvd_cv_model(dnb_dataset)
    detection = detections[0]["coords"]
    assert np.abs(detection[0] - 1608) < 1
    assert np.abs(detection[1] - 2032) < 1


def test_vvd_cv_model_e2e_n_detections() -> None:
    """tests that a single detection is returned"""
    dnb_dataset = create_single_detection_dataset()
    detections, input_image = model.vvd_cv_model(dnb_dataset)
    assert len(detections) == 1


def test_vvd_cv_model_edge_detections() -> None:
    """Ensure that chips on border of image are correct size"""
    dnb_dataset = create_edge_detection_dataset()
    from src.pipeline import VVDPostProcessor

    detections, input_image = model.vvd_cv_model(dnb_dataset)
    all_detections = VVDPostProcessor.run_pipeline(
        detections, dnb_dataset, filters=[], image_array=input_image
    )
    detections = all_detections["vessel_detections"]
    chips_dict = get_chips(input_image, detections, dnb_dataset)

    for idx, chip_info in chips_dict.items():
        assert chip_info["chip"].shape == (IMAGE_CHIP_SIZE, IMAGE_CHIP_SIZE)
    assert len(detections) == 4


def test_full_moon_no_cloud_mask() -> None:
    """Ensure under bright moonlight results will not be returned if no cloud mask"""
    dnb_dataset = create_moonlight_dataset()
    detections, input_image = model.vvd_cv_model(dnb_dataset)
    chips_dict = get_chips(input_image, detections, dnb_dataset)

    for idx, chip_info in chips_dict.items():
        assert chip_info["chip"].shape == (IMAGE_CHIP_SIZE, IMAGE_CHIP_SIZE)
    assert len(detections) == 0
