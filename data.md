## Data for inference

There are two required datasets for inference, the light intensity data (\*DNB_NRT) and supporting data including geolocation, moonlight illumination, and other files used during inference. In addition to these two data sources, there are several optional datasets that are used to improve the quality of the detections. The optional datasets are cloud masks (CLDMSK_NRT) and additional bands (MOD_NRT) used for gas flare identification and removal. The DNB and MOD datasets are provided in near real time through [earthdata](https://www.earthdata.nasa.gov/learn/find-data/near-real-time/viirs) and the cloud masks are provided in near real time through [sips-data](https://sips-data.ssec.wisc.edu/nrt/). The urls for each dataset and satellite is below. Note that downloads require a token, if using the API. Register for the API and create a token at [earthdata](https://urs.earthdata.nasa.gov/).
Suomi NPP (NOAA/NASA Suomi National Polar-orbiting Partnership)
| File | SUOMI-NPP | NOAA-20 | NOAA-21 | 
|-------------------------------|-----------------------------------------------------------------------|----------|----------|
| Day/Night Band (DNB) | [url](https://nrt3.modaps.eosdis.nasa.gov/archive/allData/5200/VNP02DNB_NRT) | [url](https://nrt3.modaps.eosdis.nasa.gov/archive/allData/5200/VJ102DNB_NRT) | [url](https://nrt3.modaps.eosdis.nasa.gov/archive/allData/5200/VJ202DNB_NRT/)
| Terrain Corrected Geolocation (DNB) | [url](https://nrt3.modaps.eosdis.nasa.gov/archive/allData/5200/VNP03DNB_NRT)| [url](https://nrt3.modaps.eosdis.nasa.gov/archive/allData/5200/VJ103DNB_NRT)| [url]
(https://nrt3.modaps.eosdis.nasa.gov/archive/allData/5200/VJ203DNB_NRT/)| Clear sky confidence | [url](https://sips-data.ssec.wisc.edu/nrt/CLDMSK_L2_VIIRS_SNPP_NRT) | [url](https://sips-data.ssec.wisc.edu/nrt/CLDMSK_L2_VIIRS_NOAA20_NRT)| TBD 
| Gas Flares Band | [url](https://nrt3.modaps.eosdis.nasa.gov/archive/allData/5200/VNP02MOD_NRT/) | [url](https://nrt3.modaps.eosdis.nasa.gov/archive/allData/5200/VJ102MOD_NRT/)| [url](https://nrt3.modaps.eosdis.nasa.gov/archive/allData/5200/VJ203MOD_NRT/)
| Terrain Corrected Geolocation (MOD) | [url](https://nrt3.modaps.eosdis.nasa.gov/archive/allData/5200/VNP03MOD_NRT/)| [url](https://nrt3.modaps.eosdis.nasa.gov/archive/allData/5200/VJ103DNB_NRT/)| [url](https://nrt3.modaps.eosdis.nasa.gov/archive/allData/5200/VJ202MOD_NRT/)

## Downloading data

1. Register an account on earthdata and download a token: https://www.earthdata.nasa.gov/learn/find-data
2. Set this token in your environment e.g. (export EARTHDATA_TOKEN=$DOWNLOADED_TOKEN)
3. Download data for each img_path (DNB, GEO data, and cloud masks are required with the default configuration on and around full moons)

```python
TOKEN = f"{os.environ.get('EARTHDATA_TOKEN')}"
with open(dnb_path, "w+b") as fh:
    utils.download_url(img_path, TOKEN, fh)
```

Sample data can be found in the test_files directory. The example requests reference data within test_files.

## API documentation

The API schema is automatically generated from src.utils.autogen_api_schema. The schema is written to docs/openapi.json (open in openapi editor such as swagger: https://editor.swagger.io/). Documentation and additional examples are available at http://0.0.0.0:5555/redoc after starting server. Example data is located in test_files.

To regenerate the schema:

```bash
python -c 'from src import utils; utils.autogen_api_schema()'
```

## Tuning the model

Parameters are defined in src/config/config.yml. Within that config, there are in line comments for the most important parameters, along with recommendations on appropriate ranges to tune those values in order to achieve higher precision or higher recall.

By default, the model filters out a variety of light sources and image artifacts that cause false positive detections. These filters are defined in pipeline section, and can be turned off or on within the config. By default, there are filters for auroral lit clouds, moonlit clouds, image artifacts (bowtie/noise smiles, edge noise), near shore detections, non-max suppression, lightning, and gas flares.

## Generate a labeled dataset

There are two types of training datasets. The first contains bounding box annotations for each detection in a frame. The second contains image level labels (crops of detected vessels) for training the supervised CNN referenced in src/postprocessor.

To generate a new object detection dataset:

1. Create account at https://nrt3.modaps.eosdis.nasa.gov/
2. Download earthdata token by clicking on profile icon and "Download token"
3. Build and run docker container with an an optional mounted volume:

```bash
docker run -d -m="50g" --cpus=120 --mount type=bind,source="$(pwd)"/target,target=/src/raw_data ghcr.io/allenai/vessel-detection-viirs:latest
```

4. Set this token in your environment: e.g. `export EARTHDATA_TOKEN=YOUR_DOWNLOADED_TOKEN_FROM_STEP_2`
5. Annotate the data from within the docker container using `python src/gen_object_detection_dataset.py`

To generate a new image label dataset:

1. Use src/gen_image_labeled_dataset.py. Sample imagery to train the feedback model is contained within the feedback_model/viirs_classifier folder

Note that a sample dataset of ~1000 detections (<1 GB) has been provided within this repository.
