# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Utility for reading GeoTIFF rasters and preparing them for Bokeh image overlays.

Reprojects to Web Mercator (EPSG:3857) and downsamples for browser performance.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def read_tiff_for_bokeh(tiff_path, band=1, max_pixels=1000):
    """Read a GeoTIFF and return data suitable for Bokeh ``fig.image()``.

    Args:
        tiff_path: Path to a .tif / .tiff file.
        band: Band index to read (1-based).
        max_pixels: Maximum dimension (width or height) after downsampling.

    Returns:
        Dict with keys ``{image, x, y, dw, dh, vmin, vmax}`` ready for
        ``fig.image(image=[d['image']], x=d['x'], ...)``, or *None* if the
        file cannot be read.
    """
    try:
        import rasterio
        from rasterio.warp import Resampling, calculate_default_transform, reproject
    except ImportError:
        logger.warning("rasterio is required for raster overlays")
        return None

    tiff_path = Path(tiff_path)
    if not tiff_path.exists():
        logger.warning(f"Raster file not found: {tiff_path}")
        return None

    dst_crs = 'EPSG:3857'

    try:
        with rasterio.open(tiff_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )

            # Downsample if too large
            scale = 1.0
            if max(width, height) > max_pixels:
                scale = max_pixels / max(width, height)
                width = int(width * scale)
                height = int(height * scale)
                transform = transform * transform.scale(1 / scale, 1 / scale)

            kwargs = src.meta.copy()
            kwargs.update(
                crs=dst_crs,
                transform=transform,
                width=width,
                height=height,
            )

            data = np.empty((height, width), dtype=np.float64)
            reproject(
                source=rasterio.band(src, band),
                destination=data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )

        # Mask nodata
        nodata_mask = ~np.isfinite(data)
        if nodata_mask.all():
            logger.warning(f"Raster is entirely nodata: {tiff_path}")
            return None
        data[nodata_mask] = np.nan

        # Compute extent in Web Mercator
        x_min = transform.c
        y_max = transform.f
        x_max = x_min + transform.a * width
        y_min = y_max + transform.e * height  # e is negative

        # Flip vertically: Bokeh image() expects origin at bottom-left
        image = np.flipud(data)

        valid = data[np.isfinite(data)]
        vmin = float(np.nanmin(valid)) if valid.size > 0 else 0.0
        vmax = float(np.nanmax(valid)) if valid.size > 0 else 1.0

        return {
            'image': image,
            'x': x_min,
            'y': y_min,
            'dw': x_max - x_min,
            'dh': y_max - y_min,
            'vmin': vmin,
            'vmax': vmax,
        }

    except Exception as exc:  # noqa: BLE001 — UI resilience
        logger.warning(f"Failed to read raster {tiff_path}: {exc}")
        return None
