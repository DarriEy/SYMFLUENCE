# ============================================================================
# SYMFLUENCE conda install (Method F from installation.html — Windows path,
# also documented as macOS Apple Silicon GDAL workaround)
#
#   conda create -n symfluence python=3.11
#   conda activate symfluence
#   conda install -c conda-forge gdal geopandas rasterio netcdf4 hdf5
#   pip install symfluence
#
# Base image matches the existing root Dockerfile.
#
# Build: docker build -f docker/conda/Dockerfile -t symfluence:conda .
# Run:   docker run --rm symfluence:conda --help
# ============================================================================

FROM condaforge/miniforge3:24.11.3-2

# Verbatim conda command sequence from installation.html "Windows" subsection.
RUN conda create -y -n symfluence python=3.11 \
    && conda install -y -n symfluence -c conda-forge \
         gdal geopandas rasterio netcdf4 hdf5 \
    && conda clean -afy

# `conda activate` doesn't work in a `RUN` (the shell isn't interactive);
# put the env's bin on PATH and set CONDA_DEFAULT_ENV instead.
ENV PATH=/opt/conda/envs/symfluence/bin:$PATH \
    CONDA_DEFAULT_ENV=symfluence \
    CONDA_PREFIX=/opt/conda/envs/symfluence

RUN pip install --no-cache-dir symfluence
RUN symfluence binary install

WORKDIR /workspace

ENTRYPOINT ["symfluence"]
CMD ["--help"]
