"""Runs a sample request for VIIRS detections from running server for images in cloud
"""
import json
import os
import time

import requests

PORT = os.getenv("VVD_PORT", default=5555)
VVD_ENDPOINT = f"http://localhost:{PORT}/detections"
GCP_BUCKET = "YOUR_GCP_BUCKET"
SAMPLE_INPUT_DIR = "input/"
SAMPLE_OUTPUT_DIR = "output/"
TIMEOUT_SECONDS = 600


def sample_request(sample_image_data: str) -> None:
    """Requests VIIRS detections from running server for images in cloud

    Parameters
    ----------
    sample_image_data : str

    """
    start = time.time()

    REQUEST_BODY = {
        "gcp_bucket": GCP_BUCKET,
        "filename": sample_image_data,
        "input_dir": SAMPLE_INPUT_DIR,
        "output_dir": SAMPLE_OUTPUT_DIR,
    }

    response = requests.post(VVD_ENDPOINT, json=REQUEST_BODY, timeout=TIMEOUT_SECONDS)
    output_filename = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "sample_response.json"
    )
    if response.ok:
        with open(output_filename, "w") as outfile:
            json.dump(response.json(), outfile)
    end = time.time()
    print(f"elapsed time for {sample_image_data} is: {end-start}")


if __name__ == "__main__":
    filename = "VJ102DNB.A2022362.0154.021.2022362055600.nc"
    sample_request(filename)
