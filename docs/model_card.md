---

---

# Model Card for VIIRS Vessel Detection

This model detects lighted vessels in night time imagery sourced from two [VIIRS](https://www.earthdata.nasa.gov/sensors/viirs) equipped satellites.


#  Table of Contents

- [Table of Contents](#table-of-contents)
- [Model Details](#model-details)
  - [Model Description](#model-description)
- [Uses](#uses)
  - [Direct Use](#direct-use)
  - [Downstream Use](#downstream-use-optional)
  - [Out-of-Scope Use](#out-of-scope-use)
- [Bias, Risks, and Limitations](#bias-risks-and-limitations)
  - [Recommendations](#recommendations)
- [Training Details](#training-details)
  - [Training Data](#training-data)
  - [Training Procedure](#training-procedure)
    - [Preprocessing](#preprocessing)
    - [Speeds, Sizes, Times](#speeds-sizes-times)
- [Evaluation](#evaluation)
  - [Testing Data, Factors & Metrics](#testing-data-factors--metrics)
    - [Testing Data](#testing-data)
    - [Factors](#factors)
    - [Metrics](#metrics)
  - [Results](#results)
- [Model Examination](#model-examination)
- [Environmental Impact](#environmental-impact)
- [Technical Specifications [optional]](#technical-specifications-optional)
  - [Model Architecture and Objective](#model-architecture-and-objective)
  - [Compute Infrastructure](#compute-infrastructure)
    - [Hardware](#hardware)
    - [Software](#software)
- [Citation](#citation)
- [Glossary [optional]](#glossary-optional)
- [More Information [optional]](#more-information-optional)
- [Model Card Authors [optional]](#model-card-authors-optional)
- [Model Card Contact](#model-card-contact)
- [How to Get Started with the Model](#how-to-get-started-with-the-model)


# Model Details

## Model Description
This VIIRS vessel detection model detects lighted vessels in night time imagery. First the model detects bright objects on the water and secondly excludes non-vessel lighted objected through a variety of post-processing layers resulting in the detections of lighted vessels (especially those using light to attract catch).


- **Developed by:** Skylight-ML
- **Shared by:** Skylight
- **Model type:** Object Detection

# Uses

## Direct Use

```python
import os

from app.utils import viirs_annotate_pipeline

INPUT_DIR = 'sample_input'
OUTPUT_DIR =  'sample_output'

dnb_filename = "VNP02DNB.A2023053.1900.002.2023053213251.nc"
geo_filename = "VNP03DNB.A2023053.1900.002.2023053211409.nc"

detections, status = viirs_annotate_pipeline(
    dnb_filename,
    geo_filename,
    TEST_FILE_INPUT_DIR,
    TEST_FILE_OUTPUT_DIR,
    cloud_filename=phys_file,
)
```


## Downstream Use
```python
import json
import os
import time

import requests

VVD_ENDPOINT = "http://localhost:5555/detections"
SAMPLE_INPUT_DIR = "/test_files/"
SAMPLE_OUTPUT_DIR = "/test_files/chips/"


def sample_request() -> None:
    start = time.time()

    REQUEST_BODY = {
        "input_dir": SAMPLE_INPUT_DIR,
        "output_dir": SAMPLE_OUTPUT_DIR,
        "dnb_filename": "VNP02DNB.A2023053.1900.002.2023053213251.nc",
        "geo_filename": "VNP03DNB.A2023053.1900.002.2023053211409.nc",
    }

    response = requests.post(VVD_ENDPOINT, json=REQUEST_BODY)

    output_filename = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "sample_response.json"
    )
    if response.ok:
        with open(output_filename, "w") as outfile:
            json.dump(response.json(), outfile)
    end = time.time()
    print(f"elasped time: {end-start}")


if __name__ == "__main__":
    sample_request()
```

# Bias, Risks, and Limitations
Due to the large pixel resolution of VIIRS (750m), VIIRS vessel detections are not conclusive evidence that a vessel is present. It is also not possible to conclusively establish how many vessels correspond to a particular lighted pixel.

## Recommendations

It is recommended that analysts carefully examine the chips of each detection -- crops of raw data centered on vessels -- to determine if the detection is likely a vessel.


# Training Details

## Training Data

The current model uses a two part archiecture. The first part used unsupervised feature extraction (two dimensional light detection kernel); the second part uses a CNN with 4 input channels trained on correct and incorrect classifications that were chosen from the outputs of the model across February through March 2023. The second part (supervised model) is trained on ~3000 correct examples and ~500 incorrect examples.


## Training Procedure
Random train (90%) and validation (10%) splits. WANDB was used to track experiments and tune the parameters.
Tuned parameters:
  * N_EPOCHS = 10
  * TRAIN_BATCH_SIZE = 24
  * SGD_MOMENTUM = 0.8
  * LEARNING_RATE = 0.0001
  * OPTIMIZER = SGD


### Preprocessing

Raw data are downloaded from earthdata as `*.netcdf` files. Numpy arrays are extracted from each layer of those netcdf files. The raw data is then clipped to the valid min and valid max of the arrays (these are provided within metadata of each file), normalized, then converted to nanowatts (multiplied by 1e9). Nanowatts are used by convention due to the small values. The chips of each detection are created from slices of the numpy array (center +/- 10 pixels), which results in an approximately 14x14 kilometer context for each vessel detection.

### Speeds, Sizes, Times

Total image processing time may take anywhere from 1 second to 1 minute. Information processing time takes ~ 1 second. Image io to and from the cloud or local storage comprises the bulk (90\%) of total processing time. Code was profiled via cprofile.

# Evaluation
Evaluation consisted of several steps.

1. The primary method of evaluation was daily observation of all of the vessel detections spanning the entire planet across January - March 2023. During this period we noticed failure modes (such as the aurora and lightning) and updated the model to address these cases.
2. Through (1) we created a variety of ~15 test cases (see testing data below) that are integrated with CICD covering known failure modes.
3. Comparison against previous VIIRS validation frames.
4. The second part of the model (supervised CNN) is evaluated using conventional image clsasification metrics (F1) against known correct and incorrect detections.

## Testing Data, Factors & Metrics

### Test Cases

  * VJ102DNB.A2023018.1342.021.2023018164614.nc   -- Hudson Bay w/ corner artifact and mild aurora
  * VNP02DNB.A2014270.1836.002.2021028152649.nc   -- Java sea (Elvidge et. al. validation frame)
  * VJ102DNB.A2023001.0742.021.2023001094409.nc   -- North Pacific (multiple fleets, deep ocean, some noise artifacts)
  * VJ102DNB.A2022354.2130.021.2022355000855.nc   -- Arabian Sea (clear night, squid fishing fleet)
  * VNP02DNB.A2022355.0336.002.2022355062247.nc  -- South Atlantic (includes portion of South Atlantic Anamoly with noise particles)
  * VNP02DNB.A2023053.1900.002.2023053213251.nc  -- SE Asia, Bay of Bengal/Gulf of Thailand, Andaman sea, S. China Sea
  * VJ102DNB.A2022362.0154.021.2022362055600.nc -- Gas flares
  * VJ102DNB.A2023031.0130.021.2023031034239.nc -- Lightning
  * VJ102DNB.A2023020.1306.021.2023020150928.nc -- Image artifacts
  * VJ102DNB.A2023020.0512.021.2023020064541.nc -- Image quality issues:
  * VJ102DNB.A2023009.1018.021.2023009135632.nc -- Moonlit clouds (false positives)
  * VNP02DNB.A2023008.2124.002.2023009045328.nc -- Moonlit clouds mixed with true positives
  *  "VNP02DNB.A2022348.1142.002.2022348173537.nc -- land only image
  * VNP02DNB.A2022348.1142.002.2022348173537.nc -- day time image
  * VJ102DNB.A2022360.2318.021.202236101342 nc -- image containing aurora
  * VJ102DNB.A2022365.0054.021.2022365043024.nc -- image containing aurora and true positives
  * VNP02DNB.A2022365.0854.002.2022365110350.nc -- image containing artifacts



### Factors



### Metrics



## Results
1. We provide sample screenshots from several days through the month showing the detections in each frame.
2. Test cases are treated as pass/fail.
3. We have agreement of around 90% with the validation frame.
4. Correct vs. Incorrect: F1 98% (See WANDB logs)

See logs here: https://wandb.ai/skylight-ml/VIIRS-Feedback-Model/table

# Model Examination



# Environmental Impact

- **Hardware Type:** V100
- **Cloud Provider:** GCP
- **Compute Region:** us-west
- **Carbon Emitted:** 0.01 kg**

** Carbon emissions were estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).


# Technical Specifications

## Model Architecture and Objective

The model is a 2 part architecture consisting of an unsupervised layer with all of the following components:
1. Convolutional 2d kernel to extract lighted pixels of interest
2. Lightning  detected via an edge detection algorithm
3. Aurora detection via clusters of especially bright clouds
4. Gas flares via spike detection (M10 band)
5. Satellite artifact detection using a combination of statistical modeling and rules.
The second part is a supervised CNN trained on correct and incorrect labels that is intended to discard false positives that were included incorrectly in the outputs.
The four channels are:
* Nanowatts
* Moonlight Illumination Factor
* Land-Sea Mask
* Clear sky confidence

## Compute Infrastructure

GCP

### Hardware

V100

### Software

Python, PyTorch, Docker, FastAPI

# Citation
NA

# Glossary [optional]

VIIRS: Visible Infrared Imaging-Radiometer Suite


# Model Card Authors
Patrick Beukema

# Model Card Contact

patrickb@allenai.org

# How to Get Started with the Model

See #usage
