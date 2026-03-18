# get OS
FROM python:3.11

# get additional packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    time \
    zlib1g-dev \
    libjpeg-dev \
    libxcb-xinerama0 \
    libegl1 \
    libgl1 \
    libfontconfig1 && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# upgrade pip
RUN pip install --upgrade pip

# 
RUN mkdir -p /app/fsqc

#
COPY pyproject.toml /app/fsqc
COPY setup.py /app/fsqc
COPY requirements.txt /app/fsqc
COPY DESCRIPTION.md /app/fsqc
COPY LICENSE /app/fsqc
COPY VERSION /app/fsqc
COPY fsqc /app/fsqc/fsqc
COPY run_fsqc /app/fsqc

# install fsqc as a package
RUN pip install /app/fsqc

# Add other environment variables
ENV OMP_NUM_THREADS=1
ENV MPLCONFIGDIR=/tmp/mplconfigdir

# Suppress Mesa's shader-cache warning ("Failed to create //.cache …") that
# appears when running as a non-standard user inside Docker where $HOME is
# unset or points to a non-writable directory.
ENV MESA_SHADER_CACHE_DISABLE=1

# In order to find NVIDIA GPUs (--gpus all)
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

# Register the NVIDIA EGL ICD so libEGL finds the GPU driver
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
    > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Set the working directory to /app
WORKDIR /app

# Set entrypoint
ENTRYPOINT ["run_fsqc"]
