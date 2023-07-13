"""VIIRS Vessel Detection Service
"""
from __future__ import annotations

import logging.config
import os
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, Response
from pydantic import BaseModel
from typing_extensions import TypedDict

import utils
from monitoring import instrumentator
from pipeline import VIIRSVesselDetection

app = FastAPI()
instrumentator.instrument(app).expose(app)
logger = logging.getLogger(__name__)

HOST = "0.0.0.0"  # nosec B104
PORT = os.getenv("VVD_PORT", default=5555)
VVD: VIIRSVesselDetection
MODEL_VERSION = datetime.today()  # concatenate with git hash


class FormattedPrediction(TypedDict):
    """Formatted prediction for a single vessel detection"""

    latitude: float
    longitude: float
    chip_path: str
    orientation: float
    meters_per_pixel: int
    moonlight_illumination: float
    nanowatts: float
    clear_sky_confidence: float


class VVDResponse(BaseModel):
    """Response object for vessel detections"""

    status: List[str]
    filename: str
    gcp_bucket: Optional[str] = None
    acquisition_time: datetime  # ISO 8601 format
    satellite_name: str
    model_version: datetime  # ISO 8601 format
    predictions: List[FormattedPrediction]
    frame_extents: List[List[float]]  # [[lon, lat],...,]
    average_moonlight: float


class VVDRequest(BaseModel):
    """Request object for vessel detections"""

    input_dir: str
    filename: Optional[str] = None  # dnb_filename
    output_dir: str
    gcp_bucket: Optional[str] = None
    dnb_filename: Optional[str] = None
    geo_filename: Optional[str] = None
    modraw_filename: Optional[str] = None
    modgeo_filename: Optional[str] = None
    phys_filename: Optional[str] = None

    class Config:
        """example configuration for a request where files are stored in cloud"""

        schema_extra = {
            "example": {
                "input_dir": "input",
                "output_dir": "output",
                "filename": "VJ102DNB.A2022362.0154.021.2022362055600.nc",
                "geo_filename": "VJ103DNB.A2022362.0154.021.2022362052511.nc",
                "modraw_filename": "VJ102MOD.A2022362.0154.002.2022362115107.nc",
                "modgeo_filename": "VJ103MOD.A2022362.0154.002.2022362095104.nc",
            },
        }


@app.on_event("startup")
async def vvd_init() -> None:
    """VIIRS Vessel Service Initialization"""
    logger.info("Initializing")
    global VVD
    VVD = VIIRSVesselDetection()


@app.get("/")
async def home() -> dict:
    return {"message": "VIIRS Vessel Detection App"}


@app.post("/detections", response_model=VVDResponse)
async def get_detections(info: VVDRequest, response: Response) -> VVDResponse:
    """Returns vessel detections Response object for a given Request object"""
    start = perf_counter()

    with TemporaryDirectory() as tmpdir:
        if info.gcp_bucket is not None:
            (
                dnb_path,
                geo_path,
                modraw_path,
                modgeo_path,
                phys_path,
            ) = utils.download_from_gcp(
                info.gcp_bucket, info.filename, info.input_dir, tmpdir
            )
        else:
            (
                dnb_path,
                geo_path,
                modraw_path,
                modgeo_path,
                phys_path,
            ) = utils.copy_local_files(info, tmpdir)
        logger.info(f"Starting VVD inference on {dnb_path}")
        all_detections, image, dnb_dataset, status = VVD.run_pipeline(
            dnb_path,
            geo_path,
            tmpdir,
            phys_path=phys_path,
            modraw_path=modraw_path,
            modgeo_path=modgeo_path,
        )
        ves_detections = all_detections["vessel_detections"]

        satellite_name = utils.get_provider_name(dnb_dataset)
        acquisition_time, end_time = utils.get_acquisition_time(dnb_dataset)
        chips_dict = utils.get_chips(image, ves_detections, dnb_dataset)

        if info.gcp_bucket is not None:
            chips_dict = utils.upload_image(
                info.gcp_bucket, chips_dict, info.output_dir, dnb_path
            )
        else:
            utils.save_chips_locally(
                chips_dict,
                destination_path=info.output_dir,
                chip_features=ves_detections,
            )

        average_moonlight = utils.get_average_moonlight(dnb_dataset)

        frame_extents = utils.get_frame_extents(dnb_dataset)

    predictions = utils.format_detections(chips_dict)
    elapsed_time = perf_counter() - start
    logger.info(f"VVD {elapsed_time=}, found {len(chips_dict)} detections)")
    response.headers["n_detections"] = str(len(chips_dict))
    response.headers["avg_moonlight"] = str(average_moonlight)
    response.headers["lightning_count"] = str(all_detections["lightning_count"])
    response.headers["gas_flare_count"] = str(all_detections["gas_flare_count"])
    response.headers["inference_time"] = str("elapsed_time")

    return VVDResponse(
        status=status,
        filename=str(Path(dnb_path).name),
        gcp_bucket=info.gcp_bucket,
        acquisition_time=acquisition_time,
        satellite_name=satellite_name,
        model_version=MODEL_VERSION,
        predictions=predictions,
        frame_extents=frame_extents,
        average_moonlight=average_moonlight,
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT, proxy_headers=True)
