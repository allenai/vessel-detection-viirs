# mypy: ignore-errors
import json
import os
import unittest

import pytest
import requests

OUTPUT_FILENAME = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "sample_response.json"
)

VVD_ENDPOINT = "http://0.0.0.0:5555/detections"
DNB_FILENAME = "VNP02DNB.A2023053.1900.002.2023053213251.nc"
GEO_FILENAME = "VNP03DNB.A2023053.1900.002.2023053211409.nc"


def api_request_cloud_files() -> requests.Response:
    """Requests VIIRS detections from running server

    Returns
    -------
    requests.Response

    """
    GCP_BUCKET = "YOUR_GCP_BUCKET"
    SAMPLE_INPUT_DIR = "vessel-detection/viirs/tests/test_files/"
    SAMPLE_OUTPUT_DIR = "vessel-detection/viirs/tests/test_outputs/"

    REQUEST_BODY = {
        "gcp_bucket": GCP_BUCKET,
        "filename": DNB_FILENAME,
        "input_dir": SAMPLE_INPUT_DIR,
        "output_dir": SAMPLE_OUTPUT_DIR,
    }
    response = requests.post(VVD_ENDPOINT, json=REQUEST_BODY, timeout=600)

    with open(OUTPUT_FILENAME, "w") as outfile:
        json_response = response.json()
        json.dump(json_response, outfile)

    return response


def api_request_local_files() -> requests.Response:
    """Requests VIIRS detections from running server

    Returns
    -------
    requests.Response

    """

    SAMPLE_INPUT_DIR = "/src/tests/test_files/"
    SAMPLE_OUTPUT_DIR = "/src/tests/output"
    DNB_FILENAME = "VNP02DNB.A2023053.1900.002.2023053213251.nc"
    GEO_FILENAME = "VNP03DNB.A2023053.1900.002.2023053211409.nc"

    REQUEST_BODY = {
        "input_dir": SAMPLE_INPUT_DIR,
        "output_dir": SAMPLE_OUTPUT_DIR,
        "dnb_filename": DNB_FILENAME,
        "geo_filename": GEO_FILENAME,
    }

    response = requests.post(VVD_ENDPOINT, json=REQUEST_BODY)

    with open(OUTPUT_FILENAME, "w") as outfile:
        json_response = response.json()
        json.dump(json_response, outfile)

    return response


class TestApiLocalFiles(unittest.TestCase):
    """Tests for the API"""

    response = NotImplemented

    @classmethod
    def setUpClass(cls) -> None:
        """Run once before all tests in this class."""
        cls.response = api_request_local_files()

    def test_response_code(self) -> None:
        """The response code should be 200."""

        assert self.response.ok

    def test_status(self) -> None:
        """The status should be processed."""
        assert self.response.json()["status"] == ["processed"]

    def test_acquisition_time(self) -> None:
        """The acquisition time should be present in the response object."""
        assert self.response.json()["acquisition_time"] == "2023-02-22T19:00:00+00:00"

    def test_filename(self) -> None:
        """The filename time should be present in the response object."""
        assert self.response.json()["filename"] == DNB_FILENAME

    def test_predictions_count(self) -> None:
        """There should be predictions for this frame."""
        assert len(self.response.json()["predictions"]) >= 1

    def test_frame_extents(self) -> None:
        """The frame extents should be present in the response object."""

        FRAME_EXTENTS = [
            [85.80496978759766, 28.292612075805664],
            [115.91454315185547, 23.634376525878906],
            [109.68016815185547, 3.395270824432373],
            [82.32111358642578, 7.638208389282227],
            [85.80496978759766, 28.292612075805664],
        ]
        assert self.response.json()["frame_extents"] == FRAME_EXTENTS

    def test_moonlight_illumination(self) -> None:
        """The moonlight illumination should be present in the response object."""
        assert self.response.json()["average_moonlight"] <= 100.0

    def test_response_size_under_one_megabyte(self) -> None:
        """The response size should be under 1MB."""
        assert float(self.response.headers["content-length"]) < 1000000 * 0.9


@pytest.mark.skipif(
    os.getenv("VIIRS_TEST_LEVEL", "default_not_set") != "dev",
    reason="this test inferences a full day of imagery, and excluded by default",
)
class TestApiCloudFiles(unittest.TestCase):
    """Tests for the API"""

    response = NotImplemented

    @classmethod
    def setUpClass(cls) -> None:
        """Run once before all tests in this class."""
        cls.response = api_request_cloud_files()

    def test_response_code(self) -> None:
        """The response code should be 200."""

        assert self.response.ok

    def test_status(self) -> None:
        """The status should be processed."""
        assert self.response.json()["status"] == ["processed"]

    def test_acquisition_time(self) -> None:
        """The acquisition time should be present in the response object."""
        assert self.response.json()["acquisition_time"] == "2023-02-22T19:00:00+00:00"

    def test_filename(self) -> None:
        """The filename time should be present in the response object."""
        assert self.response.json()["filename"] == DNB_FILENAME

    def test_predictions_count(self) -> None:
        """There should be predictions for this frame."""
        assert len(self.response.json()["predictions"]) >= 1

    def test_frame_extents(self) -> None:
        """The frame extents should be present in the response object."""

        FRAME_EXTENTS = [
            [85.80496978759766, 28.292612075805664],
            [115.91454315185547, 23.634376525878906],
            [109.68016815185547, 3.395270824432373],
            [82.32111358642578, 7.638208389282227],
            [85.80496978759766, 28.292612075805664],
        ]
        assert self.response.json()["frame_extents"] == FRAME_EXTENTS

    def test_moonlight_illumination(self) -> None:
        """The moonlight illumination should be present in the response object."""
        assert self.response.json()["average_moonlight"] <= 100.0

    def test_response_size_under_one_megabyte(self) -> None:
        """The response size should be under 1MB."""
        assert float(self.response.headers["content-length"]) < 1000000 * 0.9
