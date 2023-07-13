"""Pipeline for identifying vessels in nighttime VIIRS imagery
"""
import logging.config
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

from model import vvd_cv_model
from postprocessor import VVDPostProcessor
from preprocessor import extract_data
from utils import check_image_metadata

logging.config.fileConfig(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "logging.conf"),
    disable_existing_loggers=False,
)
logger = logging.getLogger(__name__)
CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "config", "config.yml"
)

with open(CONFIG_PATH, "r") as file:
    config = yaml.safe_load(file)["pipeline"]

DEFAULT_FILTERS = config["filters"]


class VIIRSVesselDetection:
    """Class for identifying vessels in nighttime VIIRS imagery
    This class defines a pipeline to produce detections from a VIIRS image.
    """

    def __init__(self) -> None:
        """Typical place to load model weights/other large files"""

        self.preprocessor = extract_data
        self.model = vvd_cv_model
        self.postprocessor = VVDPostProcessor

    def run_pipeline(
        self,
        image_path: Path,
        geo_path: Path,
        tmpdir: TemporaryDirectory,
        phys_path: Optional[Path] = None,
        modraw_path: Optional[Path] = None,
        modgeo_path: Optional[Path] = None,
        filters: List = DEFAULT_FILTERS,
    ) -> Tuple[Dict, np.ndarray, Dict, List[str]]:
        """raw data > preprocessed dataset > model > postprocessing > vessel detections


        Parameters
        ----------
        image_path : Path
        geo_path : Path
        tmpdir : TemporaryDirectory

        Returns
        -------
        Tuple[Dict, np.ndarray, Dict, List[str]]

        """
        self.dest_dir = tmpdir
        self.image_path = image_path
        self.geo_path = geo_path
        self.phys_path = phys_path
        self.modraw_path = modraw_path
        self.modgeo_path = modgeo_path

        dnb_dataset = self.preprocessor(
            self.image_path,
            self.geo_path,
            self.phys_path,
            self.modraw_path,
            self.modgeo_path,
        )
        self.detections, self.image_array = self.model(dnb_dataset)

        self.status = check_image_metadata(dnb_dataset)
        if self.status != ["processed"]:
            self.all_detections = {
                "vessel_detections": {},
                "lightning_count": 0,
                "gas_flare_count": 0,
            }
        else:
            self.all_detections = self.postprocessor.run_pipeline(
                self.detections, dnb_dataset, filters, self.image_array
            )

        return self.all_detections, self.image_array, dnb_dataset, self.status
