# ============================================================================
# SYMFLUENCE source / bootstrap install (Method G from README.md /
# installation.html "Development Installation")
#
#   git clone https://github.com/symfluence-org/SYMFLUENCE.git
#   cd SYMFLUENCE
#   ./scripts/symfluence-bootstrap --install
#   source venv/bin/activate
#
# Build: docker build -f docker/source/Dockerfile -t symfluence:source .
# Run:   docker run --rm symfluence:source --help
#
# Manual-setup variant (installation.html "Manual Setup"): builds via editable
# install of a local checkout instead of using the bootstrap script.
#   docker build --target manual -f docker/source/Dockerfile -t symfluence:source-manual .
# ============================================================================

# ----------------------------------------------------------------------------
# Default target: bootstrap from upstream clone
# ----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS bootstrap

# Build toolchain (docs/SYSTEM_REQUIREMENTS.md L156-160) +
# GDAL build prereqs (README.md L107) + git (needed by bootstrap clone)
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

# README.md L60-64 / installation.html "Development Installation":
#   git clone https://github.com/symfluence-org/SYMFLUENCE.git
#   cd SYMFLUENCE
#   ./scripts/symfluence-bootstrap --install
WORKDIR /opt
RUN git clone https://github.com/symfluence-org/SYMFLUENCE.git
WORKDIR /opt/SYMFLUENCE
RUN ./scripts/symfluence-bootstrap --install

# bootstrap creates a venv at /opt/SYMFLUENCE/venv but doesn't add it to PATH.
ENV PATH=/opt/SYMFLUENCE/venv/bin:$PATH

WORKDIR /workspace

ENTRYPOINT ["symfluence"]
CMD ["--help"]

# ----------------------------------------------------------------------------
# Manual variant: editable install of local checkout (installation.html
# "Manual Setup") — `docker build --target manual ...`
# ----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS manual

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

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal \
    C_INCLUDE_PATH=/usr/include/gdal \
    HDF5_USE_FILE_LOCKING=FALSE \
    OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    PYTHONUNBUFFERED=1

# installation.html "Manual Setup":
#   python3 -m venv venv
#   source venv/bin/activate
#   pip install -e .
COPY . /opt/SYMFLUENCE
WORKDIR /opt/SYMFLUENCE
RUN python3 -m venv venv \
    && ./venv/bin/pip install --no-cache-dir -e .

ENV PATH=/opt/SYMFLUENCE/venv/bin:$PATH

WORKDIR /workspace

ENTRYPOINT ["symfluence"]
CMD ["--help"]
