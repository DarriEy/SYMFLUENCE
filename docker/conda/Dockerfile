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
#
# System dependencies follow the canonical registry at
# src/symfluence/resources/system_deps.yml — every conda package below maps to
# a dep_id there (all `required: true` entries, plus optionals `udunits2`,
# `cdo`, `r_base`, `perl` needed by the default `symfluence binary install`
# tool set). Run `symfluence binary doctor` to validate the env against the
# registry. Update the conda list when adding or removing dep_ids in
# system_deps.yml.
# ============================================================================

FROM condaforge/miniforge3:24.11.3-2

# Conda package list derived from src/symfluence/resources/system_deps.yml.
# Build toolchain + core libraries (all `required: true` entries) + default-
# toolset optionals (udunits2, cdo, r-base, perl). Each name on the right is
# the canonical `packages.conda` value for its dep_id; do not freelance.
#
# Note: `c-compiler` / `cxx-compiler` / `fortran-compiler` are conda-forge
# metapackages that resolve to the right native toolchain per platform
# (gcc_linux-64 on linux-64, gcc_linux-aarch64 on linux-aarch64, clang_osx-*
# on macOS, vs2019_win-64 on Windows) and ship activation scripts that
# expose unprefixed `gcc`/`g++`/`gfortran` plus CC/CXX/FC env vars — which
# is what pip wheel builds and upstream model Makefiles expect.
RUN conda create -y -n symfluence python=3.11 \
    && conda install -y -n symfluence -c conda-forge \
         cmake c-compiler cxx-compiler fortran-compiler make \
         gdal proj geos netcdf4 netcdf-fortran hdf5 openmpi openblas \
         udunits2 cdo r-base perl \
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
