Installation
============

Overview
--------
SYMFLUENCE supports a one-command installer that creates an isolated Python 3.11
environment and installs required packages. HPC environments vary; use the module
recipes on your cluster as needed.

Quick Install
-------------
Run the built-in installer from the project root:

.. code-block:: bash

   ./symfluence --install

Note: The npm package under ``tools/npm/`` is release-only packaging used to
distribute prebuilt binaries. It is not required for local development when
installing via pip.

What this does:
- Creates/updates ``.venv/`` (Python 3.11 recommended)
- Installs Python dependencies with ``pip``
- Reuses the environment on subsequent runs


Manual Setup (Optional)
-----------------------
If you prefer to manage the environment yourself:

.. code-block:: bash

   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

System Prerequisites
--------------------
- Build toolchain: GCC/Clang (C/C++), gfortran; CMake >= 3.20; MPI (OpenMPI/MPICH)
- Core libs: GDAL, HDF5, NetCDF (C + Fortran), BLAS/LAPACK
- Optional tools: R, CDO (when applicable)

CDS API Credentials (CARRA/CERRA)
---------------------------------
If you plan to use CARRA or CERRA forcing datasets, configure CDS API credentials:

1. Register: https://cds.climate.copernicus.eu/
2. Get your API key at https://cds.climate.copernicus.eu/user
3. Create ``~/.cdsapirc``:

.. code-block:: bash

   cat > ~/.cdsapirc << EOF
   url: https://cds.climate.copernicus.eu/api
   key: {UID}:{API_KEY}
   EOF

4. Restrict permissions:

.. code-block:: bash

   chmod 600 ~/.cdsapirc

Verification:

.. code-block:: python

   import cdsapi
   c = cdsapi.Client()
   print("CDS API configured successfully!")

HPC Module Recipes
------------------
Use your site's module system, then run the installer:

Anvil (Purdue RCAC):

.. code-block:: bash

   module load r/4.4.1
   module load gcc/14.2.0
   module load openmpi/4.1.6
   module load gdal/3.10.0
   module load conda/2024.09
   module load openblas/0.3.17
   module load netcdf-fortran/4.5.3
   module load udunits/2.2.28

ARC (University of Calgary):

.. code-block:: bash

   . /work/comphyd_lab/local/modules/spack/2024v5/lmod-init-bash
   module unuse $MODULEPATH
   module use /work/comphyd_lab/local/modules/spack/2024v5/modules/linux-rocky8-x86_64/Core/

   module load gcc/14.2.0
   module load cmake
   module load netcdf-fortran/4.6.1
   module load netcdf-c/4.9.2
   module load openblas/0.3.27
   module load hdf5/1.14.3
   module load gdal/3.9.2
   module load netlib-lapack/3.11.0
   module load openmpi/4.1.6
   module load python/3.11.7
   module load r/4.4.1

FIR (Compute Canada):

.. code-block:: bash

   module load StdEnv/2023
   module load gcc/12.3
   module load python/3.11.5
   module load gdal/3.9.1
   module load r/4.5.0
   module load cdo/2.2.2
   module load mpi4py/4.0.3
   module load netcdf-fortran/4.6.1
   module load openblas/0.3.24

Then run:

.. code-block:: bash

   ./symfluence --install

macOS (Intel or Apple Silicon)
------------------------------

Install Xcode tools and Homebrew:

.. code-block:: bash

   xcode-select --install
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

Install core packages:

.. code-block:: bash

   brew update
   brew install cmake gcc open-mpi gdal hdf5 netcdf netcdf-fortran openblas lapack cdo r

Optional compiler pinning:

.. code-block:: bash

   export CC=$(brew --prefix)/bin/gcc-14
   export CXX=$(brew --prefix)/bin/g++-14
   export FC=$(brew --prefix)/bin/gfortran-14

Then run:

.. code-block:: bash

   ./symfluence --install

Verification
------------
.. code-block:: bash

   ./symfluence --help

Next Steps
----------
- :doc:`getting_started` — your first run
- :doc:`configuration` — YAML structure and options
- :doc:`examples` — progressive tutorials
