""" Use this script to inference the API with locally stored data"""
import json
import os
import time

import requests

PORT = os.getenv("VVD_PORT", default=5555)
VVD_ENDPOINT = f"http://localhost:{PORT}/detections"
SAMPLE_INPUT_DIR = "/example/"
SAMPLE_OUTPUT_DIR = "/example/chips/"
TIMEOUT_SECONDS = 600
DNB_FILENAME = "VJ102DNB_NRT_2023_310_VJ102DNB_NRT.A2023310.0606.021.2023310104322.nc"
GEO_FILENAME = "VJ103DNB_NRT_2023_310_VJ103DNB_NRT.A2023310.0606.021.2023310093233.nc"
def sample_request() -> None:
    """Sample request for files stored locally"""
    start = time.time()

    REQUEST_BODY = {
        "input_dir": SAMPLE_INPUT_DIR,
        "output_dir": SAMPLE_OUTPUT_DIR,
        "dnb_filename": DNB_FILENAME,
        "geo_filename": GEO_FILENAME

    }

    response = requests.post(VVD_ENDPOINT, json=REQUEST_BODY, timeout=TIMEOUT_SECONDS)
    output_filename = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "sample_response.json"
    )
    if response.ok:
        with open(output_filename, "w") as outfile:
            json.dump(response.json(), outfile)
    end = time.time()
    print(f"elapsed time: {end-start}")


if __name__ == "__main__":
    sample_request()
