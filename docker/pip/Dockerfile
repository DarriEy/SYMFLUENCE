# ============================================================================
# SYMFLUENCE pip install (Method B from README.md / installation.html)
#
#   pip install symfluence
#   symfluence binary install
#
# Build: docker build -f docker/pip/Dockerfile -t symfluence:pip .
# Run:   docker run --rm symfluence:pip --help
# ============================================================================

FROM python:3.11-slim-bookworm

# Build toolchain (docs/SYSTEM_REQUIREMENTS.md L156-160) +
# GDAL build prereqs (README.md L107) + git/curl/ca-certificates
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        cmake \
        libopenmpi-dev \
        openmpi-bin \
        libopenblas-dev \
        liblapack-dev \
        libnetcdf-dev \
        libnetcdff-dev \
        libhdf5-dev \
        gdal-bin \
        libgdal-dev \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# README.md L108-109: GDAL Python wheel needs these to find headers.
# OMPI_ALLOW_RUN_AS_ROOT: compiled hydrology models invoke OpenMPI inside the
# container, which by default refuses to run as root.
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal \
    C_INCLUDE_PATH=/usr/include/gdal \
    HDF5_USE_FILE_LOCKING=FALSE \
    OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir symfluence
RUN symfluence binary install

WORKDIR /workspace

ENTRYPOINT ["symfluence"]
CMD ["--help"]
