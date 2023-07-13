This repository contains a computer vision model along with a containerized restful API (FastAPI) for serving streaming detections of vessels in near real time. See [docs/model_card.md](./docs/model_card.md) for information about the model and [docs/openapi.json](./docs/openapi.json) for the API specification. This model was built for [Skylight](https://www.skylight.global/), a product of AI2 that supports maritime transparency through actionable intelligence in order to help protect our oceans.

<p float="left">
  <img src="images/summary.png" >
</p>

---

# Getting started

Note that the model and API are designed to run in resource constrained environments. The hardware requirements are a CPU and at least 4 GB of RAM. Note that a GPU is not needed including for fast inference.

## Prerequisites

- Python 3.10
- Docker: https://docs.docker.com/get-docker/
- Docker Compose: https://docs.docker.com/compose/install/ (note that Docker compose may already be installed depending on how you installed docker)
- git-lfs: https://git-lfs.com/ Test files used for development are stored on GitHub with git-lfs.

## Installation

Pull the latest package from [GitHub](https://github.com/allenai/vessel-detection-viirs/pkgs/container/vessel-detection-viirs)

```bash
docker pull ghcr.io/vulcanskylight/vessel-detection-viirs
```

Once the package is downloaded, start the service with:

```bash
docker run -d -p 5555:5555 vvd-service
```

You may override the default port by passing in your preferred port in the docker run command e.g. `-e VVD_PORT=PORT`

Or clone this repository and build the container with

```bash
docker compose up
```

The service will now be running on port 5555 (verify with `docker ps -a`).
To query the API with an example request, install `requirements/requirements-inference.txt` on the host.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements/requirements-inference.txt
```

## Usage

```bash
$ python examples/sample_request.py
```

## Tests

Unit and integration tests (see tests/) are run as part of CICD via GitHub actions. To manually run these tests, execute:

```bash
$ pytest tests -vv
```

Test files are stored on GitHub (test/test_files/) using git-lfs (retrieve these files via `git lfs fetch`). Since these are large files they are excluded from the inference docker container.

## Development notes

There are many parameters that can be modified to control precision and recall and tune the model to other desired use cases. See src/config/config.yml for the parameters that can be modified and how to do so.

### Performance

- Real-time latency is measured from the time that the light is emitted by a vessel and when we ultimately show the detected vessel to our users. In our plaftorm, we obvserve an average latency of 2 hours from a ship emitting light to when we surface that data to our users. The latency is determined primarily by the time required to downlink the data to NASA's servers. Our processing time is < 1 second.

## Model architecture

<p float="left">
  <img src="images/model_arch.png">
</p>

For details see docs/model_card.md

## Acknowledgements

We are very grateful to NASA for making the raw satellite data freely accessible from earthdata: https://www.earthdata.nasa.gov/. Thanks to NOAA and NASA for launching the satellites. Thanks to https://sips.ssec.wisc.edu/#/ for creating cloud masks. Thanks to the Earth Observation Group at the Colorado School of Mines for extensive research on VIIRS and their work on vessel detection (https://www.mdpi.com/2072-4292/7/3/3020).

## Contributing

We are grateful for your feedback and contributions are appreciated. Please see CONTRIBUTING.md for details on contributing.

## Limitations

While we do our best to ensure high precision and recall across the planet every night, the model does not get everything right. The largest source of error occurs around full moons due to the interaction of moonlight and clouds. We control for that source of error by measuring the background glow of clouds and only surfacing detections that are not underneath clouds and above the background glow of clouds. This conditional processing only occurs on and around full moons (+/- 2 days).

Note that the repository only contains the model and service to create streaming vessel detections from raw VIIRS data. There are tools within this repository to download the raw data from NASA's servers but this application does not do so automatically. To create a fully automated streaming service of vessel detections, one would need to add logic to poll NASA's servers, copy new data, and inference that data (using this service).

## Contact

support@skylight.org
