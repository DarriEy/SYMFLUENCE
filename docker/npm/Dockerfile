# ============================================================================
# SYMFLUENCE npm install (Method A from README.md / installation.html)
#
#   npm install -g symfluence
#   symfluence binary info
#   symfluence binary doctor
#
# Distributes PRE-BUILT binaries — no compilation, no `symfluence binary install`.
# Only runtime libraries are needed.
#
# Platform support (README.md L84-86, npm/package.json):
#   - Linux x86_64 (Ubuntu 22.04+, RHEL 9+, Debian 12+)
#   - macOS ARM64 (Apple Silicon)
#
# Build (Linux x86_64, force platform if you're on Apple Silicon):
#   docker build --platform=linux/amd64 -f docker/npm/Dockerfile -t symfluence:npm .
# Run:
#   docker run --rm --platform=linux/amd64 symfluence:npm --help
# ============================================================================

FROM node:20-bookworm-slim

# Runtime-only system libraries (docs/SYSTEM_REQUIREMENTS.md L34-44).
# No build-essential / gfortran / cmake — npm ships compiled binaries.
#
# `curl` is required by npm/install.js: its `tryPixiBootstrap()` path downloads
# a pixi binary from GitHub releases via `curl -fsSL`. Without it, pixi cannot
# bootstrap and the postinstall has no way to install the Python wrapper that
# the npm shim forwards to.
#
# `python3` is required because the npm package is a wrapper around the
# Python CLI — install.js runs `uv pip install symfluence` (or pixi-equivalent)
# during postinstall.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libnetcdf19 \
        libnetcdff7 \
        libhdf5-103 \
        libgdal32 \
        libproj25 \
        libgeos3.11.1 \
        libopenmpi3 \
        python3 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Astral's recommended Docker pattern: copy the static uv binary from the
# upstream image instead of `curl -LsSf https://astral.sh/uv/install.sh | sh`.
# install.js's tryInstallPython() picks this up as its preferred installer
# (strategies order: uv → pip3 → pip). Pinnable to a specific uv version by
# replacing `:latest` with e.g. `:0.5.0`.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# uv 0.5+ refuses to install to system Python without an active venv or
# explicit opt-in, and install.js's tryInstallPython() calls `uv pip install
# symfluence` with no flags. Set the env vars uv reads instead.
#  - UV_SYSTEM_PYTHON=true: install into the system Python (no venv needed)
#  - UV_BREAK_SYSTEM_PACKAGES=true: bypass Debian's PEP-668 externally-managed
#    marker. The plain Dockerfile deliberately avoids a venv (that's a
#    workaround that belongs in Dockerfile.fixed); these env vars are the
#    minimal equivalent that keeps install.js's command line unchanged.
ENV UV_SYSTEM_PYTHON=true \
    UV_BREAK_SYSTEM_PACKAGES=true

# README.md L75: npm install -g symfluence
RUN npm install -g symfluence

# README.md L78-82: bundled-binary sanity checks.
RUN symfluence binary info \
    && symfluence binary doctor

WORKDIR /workspace

ENTRYPOINT ["symfluence"]
CMD ["--help"]
