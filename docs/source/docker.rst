Docker
======

The ``docker/`` directory contains a Dockerfile per documented install method.
Each method has two files:

- **Dockerfile** — the install method as described in the docs, verbatim.
  No workarounds. Useful for reproducing what a user following the docs would
  actually get.
- **Dockerfile.fixed** — a copy with the minimum set of workarounds needed for
  a fully-working image. Each workaround is annotated with the upstream bug it
  papers over so it can be removed once fixed upstream.

Method Matrix
-------------

All install paths build successfully on both ``linux/amd64`` and
``linux/arm64``. Numbers count model binaries produced by
``symfluence binary install`` (out of 12, except npm which bundles a different
prebuilt set).

.. list-table::
   :header-rows: 1
   :widths: 15 30 25 25

   * - Method
     - Base image
     - Dockerfile (doc-faithful)
     - Dockerfile.fixed
   * - pip
     - ``python:3.11-slim-bookworm``
     - builds; 11/12 :sup:`1`
     - builds; 11/12 :sup:`1`
   * - uv
     - ``python:3.11-slim-bookworm``
     - builds; 11/12 :sup:`1`
     - builds; 11/12 :sup:`1`
   * - uv-tool
     - ``python:3.11-slim-bookworm``
     - builds; 11/12 :sup:`1`
     - builds; 11/12 :sup:`1`
   * - pipx
     - ``python:3.11-slim-bookworm``
     - builds; 11/12 :sup:`1`
     - builds; 11/12 :sup:`1`
   * - npm
     - ``node:20-bookworm-slim``
     - builds; runtime-only (prebuilt binaries)
     - 21/23 binaries (ngiab + troute not bundled)
   * - conda
     - ``condaforge/miniforge3:24.11.3-2``
     - builds; 11/12 :sup:`1`
     - builds; 11/12 :sup:`1` (single-source conda toolchain)
   * - source
     - ``python:3.11-slim-bookworm``
     - builds; 11/12 :sup:`1` (bootstrap stage)
     - builds; 12/12 (manual stage)

:sup:`1` The 1-of-12 gap is ``ngen``, which the wrapper marks as failed despite
the binary being built. This is a known false-negative in published
symfluence's ``_build.sh`` — fixed on a development branch and landing in the
next release.

Build & Run
-----------

``docker-compose.yaml`` defines one service per install method, each wired to
its ``Dockerfile.fixed``. Run from the repo root:

.. code-block:: bash

   # pip
   docker compose build pip
   docker compose run --rm pip --help

   # uv
   docker compose build uv
   docker compose run --rm uv --help

   # uv tool (isolated CLI)
   docker compose build uv-tool
   docker compose run --rm uv-tool --help

   # pipx (isolated CLI)
   docker compose build pipx
   docker compose run --rm pipx --help

   # npm (pre-built binaries, no compilation; service pins platform: linux/amd64)
   docker compose build npm
   docker compose run --rm npm --help

   # conda (Windows install path / macOS ARM64 GDAL workaround)
   docker compose build conda
   docker compose run --rm conda --help

   # source — bootstrap from upstream clone
   docker compose build source
   docker compose run --rm source --help

   # source — manual editable install of local checkout (Dockerfile target: manual)
   docker compose build source-manual
   docker compose run --rm source-manual --help

Building for linux/amd64 on Apple Silicon
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A quick terminology note:

- **arm64** (aka aarch64) — ARM 64-bit. Apple Silicon (M1/M2/M3/M4), AWS
  Graviton, Raspberry Pi.
- **amd64** (aka x86_64) — Intel/AMD 64-bit. Intel Macs, most Linux servers and
  HPC nodes, most cloud VMs, GitHub Actions runners. The "AMD" in the name is
  historical; Intel chips use the same instruction set.

Your Apple Silicon Mac is **arm64**. Docker images are architecture-specific, so
by default ``docker compose build`` produces an arm64 image — fast, native, no
emulation. That's the right choice for day-to-day local work.

Override to ``linux/amd64`` (runs under QEMU emulation, noticeably slower) when:

- **Reproducing what CI sees** — GitHub Actions runners are ``linux/amd64``.
- **Matching production deployments** — most cloud VMs and HPC are amd64.
- **Running the npm install path** — prebuilt tarballs only exist for
  ``linux/amd64`` + ``darwin/arm64``.
- **Cross-arch validation before a release**.

To force amd64, layer ``docker-compose.amd64.yaml`` on top:

.. code-block:: bash

   # One-off:
   docker compose -f docker-compose.yaml -f docker-compose.amd64.yaml build pip
   docker compose -f docker-compose.yaml -f docker-compose.amd64.yaml run --rm pip --help

   # Or set COMPOSE_FILE once for the session:
   export COMPOSE_FILE=docker-compose.yaml:docker-compose.amd64.yaml
   docker compose build pip
   docker compose run --rm pip --help

Source-compile methods (``source``, ``source-manual``) take 30+ minutes natively
and considerably longer under QEMU.

Which One Should I Use?
-----------------------

- **Most users**: ``pip`` or ``uv``. Both produce 12/12 working binaries on
  Linux. uv is faster to install.
- **Need pre-compiled binaries** (no host build toolchain): ``npm``. Linux
  x86_64 and macOS ARM64 only.
- **Developing on the project**: ``source --target manual`` so the venv contains
  an editable install of your local checkout.
- **Avoid for now**: the unmodified ``Dockerfile`` files. They are kept as a
  record of what the docs literally say; they're not meant to be used directly.
