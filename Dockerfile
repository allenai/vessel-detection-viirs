FROM ubuntu:22.04@sha256:67211c14fa74f070d27cc59d69a7fa9aeff8e28ea118ef3babc295a0428a6d21

RUN apt-get update -y
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN apt-get install libhdf5-serial-dev netcdf-bin libnetcdf-dev -y

RUN apt-get update && apt-get install -y \
    python3-pip

COPY requirements/requirements.txt  requirements.txt

RUN pip3 install --no-cache-dir --upgrade -r requirements.txt

WORKDIR /src

COPY ./src /src
COPY ./tests /src/tests

CMD ["python3", "main.py"]
