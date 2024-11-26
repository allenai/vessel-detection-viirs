""" Use this script to inference the API with locally stored data"""

import json
import os

import requests

PORT = os.getenv("VVD_PORT", default=5555)
VVD_ENDPOINT = f"http://localhost:{PORT}/detections"
SAMPLE_INPUT_DIR = "tests/test_files/"
SAMPLE_OUTPUT_DIR = "tests/test_files/chips/"
TIMEOUT_SECONDS = 600


def sample_request() -> None:
    """Sample request for files stored locally"""

    REQUEST_BODY = {
        "input_dir": SAMPLE_INPUT_DIR,
        "output_dir": SAMPLE_OUTPUT_DIR,
        "dnb_filename": "VNP02DNB_NRT.A2023300.1136.002.2023300154339.nc",
        "geo_filename": "VNP03DNB_NRT.A2023300.1136.002.2023300145841.nc",
    }

    response = requests.post(VVD_ENDPOINT, json=REQUEST_BODY, timeout=TIMEOUT_SECONDS)
    output_filename = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "sample_response.json"
    )
    if response.ok:
        with open(output_filename, "w") as outfile:
            json.dump(response.json(), outfile)


if __name__ == "__main__":
    sample_request()
