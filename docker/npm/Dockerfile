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
RUN apt-get update && apt-get install -y --no-install-recommends \
        libnetcdf19 \
        libnetcdff7 \
        libhdf5-103 \
        libgdal32 \
        libproj25 \
        libgeos3.11.1 \
        libopenmpi3 \
        python3 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# README.md L75: npm install -g symfluence
RUN npm install -g symfluence

# README.md L78-82: bundled-binary sanity checks.
RUN symfluence binary info \
    && symfluence binary doctor

WORKDIR /workspace

ENTRYPOINT ["symfluence"]
CMD ["--help"]
