# ============================================================================
# SYMFLUENCE pipx install (Method E from README.md)
#
#   pipx install symfluence
#   symfluence binary install
#
# pipx puts symfluence in an isolated venv at /root/.local/share/pipx/venvs/
# with a shim at /root/.local/bin/.
#
# Build: docker build -f docker/pipx/Dockerfile -t symfluence:pipx .
# Run:   docker run --rm symfluence:pipx --help
#
# System dependencies follow the canonical registry at
# src/symfluence/resources/system_deps.yml — every apt package below maps to a
# dep_id there (all `required: true` entries, plus optionals `udunits2`, `cdo`,
# `r_base`, `perl` needed by the default `symfluence binary install` tool set).
# Run `symfluence binary doctor` to validate the host against the registry.
# Update the apt list when adding or removing dep_ids in system_deps.yml.
# ============================================================================

FROM python:3.11-slim-bookworm

# Build toolchain (docs/SYSTEM_REQUIREMENTS.md L156-160) +
# GDAL build prereqs (README.md L107) + git/curl/ca-certificates
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        cmake \
        make \
        libopenmpi-dev \
        openmpi-bin \
        libopenblas-dev \
        liblapack-dev \
        libnetcdf-dev \
        libnetcdff-dev \
        libhdf5-dev \
        gdal-bin \
        libgdal-dev \
        libproj-dev \
        libgeos-dev \
        libudunits2-dev \
        cdo \
        r-base \
        r-base-dev \
        perl \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# README.md L108-109: GDAL Python wheel needs these to find headers.
# OMPI_ALLOW_RUN_AS_ROOT: compiled hydrology models invoke OpenMPI inside the
# container, which by default refuses to run as root.
# PATH: pipx shim lives in /root/.local/bin and is otherwise off-PATH.
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal \
    C_INCLUDE_PATH=/usr/include/gdal \
    HDF5_USE_FILE_LOCKING=FALSE \
    OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/root/.local/bin:$PATH

# pipx is not on the slim base; install via pip and let it manage isolated venvs.
# Debian's `apt-get install pipx` ships a stale version on bookworm.
RUN pip install --no-cache-dir pipx \
    && pipx ensurepath

# README.md L48: pipx install symfluence
RUN pipx install symfluence
RUN symfluence binary install

WORKDIR /workspace

ENTRYPOINT ["symfluence"]
CMD ["--help"]
