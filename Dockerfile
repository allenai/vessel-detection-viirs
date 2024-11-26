# Use an official Python runtime as a parent image with SHA for reproducibility
# hadolint ignore=DL3008
FROM python:3.12-slim@sha256:2a6386ad2db20e7f55073f69a98d6da2cf9f168e05e7487d2670baeb9b7601c5

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/src

# Install all required system packages in one RUN statement to reduce image layers
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Original required packages
    ffmpeg \
    libsm6 \
    libxext6 \
    libhdf5-dev \
    netcdf-bin \
    libnetcdf-dev \
    # Additional geospatial packages
    gdal-bin \
    libgdal-dev \
    libproj-dev \
    libgeos-dev \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*  # Clean up to reduce image size

# Copy requirements to leverage Docker cache
COPY requirements/requirements.txt /tmp/requirements.txt

# Fix urllib3 version specifier and install watchdog instead of pathtools
RUN pip install --no-cache-dir --upgrade -r /tmp/requirements.txt

# Set the working directory
WORKDIR /src

# Copy the source code in one layer
COPY ./src /src
COPY ./tests /src/tests

# Specify the default command to run
CMD ["python", "main.py"]
