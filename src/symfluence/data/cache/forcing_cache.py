"""
Raw forcing data cache manager for SYMFLUENCE.

Provides content-addressable caching of raw forcing data downloads
from CDS API and cloud sources (ERA5, AORC, HRRR, etc.).

Conservative caching approach:
- Strict cache key validation (dataset, bbox, time, variables)
- Version-based invalidation
- Checksum verification on every read
- Limited TTL (30 days default)
- LRU eviction when cache exceeds size limit
"""

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from symfluence.project.logging_manager import get_logger

# Get SYMFLUENCE version for cache invalidation
try:
    from symfluence import __version__ as SYMFLUENCE_VERSION
except ImportError:
    SYMFLUENCE_VERSION = "unknown"

logger = get_logger(__name__)


class RawForcingCache:
    """
    Content-addressable cache for raw forcing data downloads.

    Caches raw NetCDF files from external APIs to avoid redundant downloads.
    Implements conservative validation and eviction policies.

    Parameters
    ----------
    cache_root : Path or str
        Root directory for cache storage
    max_size_gb : float
        Maximum cache size in gigabytes (default: 3.0)
    ttl_days : int
        Time-to-live for cached files in days (default: 30)
    enable_checksum : bool
        Whether to verify checksums (default: True for conservative approach)
    """

    def __init__(
        self,
        cache_root: Union[Path, str],
        max_size_gb: float = 3.0,
        ttl_days: int = 30,
        enable_checksum: bool = True,
    ):
        self.cache_root = Path(cache_root)
        self.max_size_gb = max_size_gb
        self.ttl_days = ttl_days
        self.enable_checksum = enable_checksum

        # Create cache directory if it doesn't exist
        self.cache_root.mkdir(parents=True, exist_ok=True)

        logger.debug(
            f"Initialized RawForcingCache at {self.cache_root} "
            f"(max_size={max_size_gb}GB, ttl={ttl_days}days)"
        )

    def generate_cache_key(
        self,
        dataset: str,
        bbox: Union[tuple, list, str],
        time_start: str,
        time_end: str,
        variables: Optional[List[str]] = None,
    ) -> str:
        """
        Generate stable, version-aware cache key.

        Parameters
        ----------
        dataset : str
            Forcing dataset name (e.g., "ERA5", "CARRA", "CERRA")
        bbox : tuple, list, or str
            Bounding box (lat_min, lon_min, lat_max, lon_max)
        time_start : str
            Start time in ISO format
        time_end : str
            End time in ISO format
        variables : list of str, optional
            List of variables to download

        Returns
        -------
        str
            Cache key (dataset_hash16)
        """
        # Normalize bbox to string
        if isinstance(bbox, (tuple, list)):
            bbox_str = ",".join(str(x) for x in bbox)
        else:
            bbox_str = str(bbox)

        # Create key data dictionary
        key_data = {
            "dataset": dataset.upper(),
            "bbox": bbox_str,
            "time_start": str(time_start),
            "time_end": str(time_end),
            "variables": sorted(variables) if variables else [],
            "version": SYMFLUENCE_VERSION,  # Invalidate on version change
        }

        # Generate hash
        key_json = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_json.encode()).hexdigest()[:16]

        cache_key = f"{dataset.upper()}_{key_hash}"
        logger.debug(f"Generated cache key: {cache_key}")

        return cache_key

    def get(self, cache_key: str) -> Optional[Path]:
        """
        Retrieve cached file with validation.

        Performs conservative validation:
        1. Check file exists
        2. Check metadata exists
        3. Verify TTL
        4. Verify checksum (if enabled)

        Parameters
        ----------
        cache_key : str
            Cache key from generate_cache_key()

        Returns
        -------
        Path or None
            Path to cached file if valid, None otherwise
        """
        cache_path = self.cache_root / f"{cache_key}.nc"
        metadata_path = self.cache_root / f"{cache_key}.meta.json"

        # Check existence
        if not cache_path.exists() or not metadata_path.exists():
            logger.debug(f"Cache miss: {cache_key} (file not found)")
            return None

        # Load and validate metadata
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Cache corruption: Invalid metadata for {cache_key}: {e}")
            self._remove_cache_entry(cache_key)
            return None

        # Check TTL
        try:
            created_at = datetime.fromisoformat(metadata["created_at"])
            age_days = (datetime.now() - created_at).days
            if age_days > self.ttl_days:
                logger.debug(f"Cache expired: {cache_key} (age={age_days} days)")
                self._remove_cache_entry(cache_key)
                return None
        except (KeyError, ValueError) as e:
            logger.warning(f"Cache corruption: Invalid timestamp for {cache_key}: {e}")
            self._remove_cache_entry(cache_key)
            return None

        # Verify checksum (conservative approach)
        if self.enable_checksum:
            expected_hash = metadata.get("checksum")
            if expected_hash:
                actual_hash = self._compute_checksum(cache_path)
                if actual_hash != expected_hash:
                    logger.warning(
                        f"Cache corruption: Checksum mismatch for {cache_key}"
                    )
                    self._remove_cache_entry(cache_key)
                    return None
            else:
                logger.warning(f"Cache metadata missing checksum for {cache_key}")

        logger.info(f"Cache hit: {cache_key} (age={age_days} days)")
        return cache_path

    def put(
        self, cache_key: str, file_path: Union[Path, str], metadata: Dict[str, Any]
    ) -> None:
        """
        Store file in cache with metadata.

        Parameters
        ----------
        cache_key : str
            Cache key from generate_cache_key()
        file_path : Path or str
            Path to file to cache
        metadata : dict
            Metadata to store (dataset, bbox, time_range, etc.)
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Cannot cache non-existent file: {file_path}")

        cache_path = self.cache_root / f"{cache_key}.nc"
        metadata_path = self.cache_root / f"{cache_key}.meta.json"

        # Copy file to cache
        logger.debug(f"Caching file: {file_path} -> {cache_path}")
        shutil.copy(file_path, cache_path)

        # Compute and store checksum
        if self.enable_checksum:
            metadata["checksum"] = self._compute_checksum(cache_path)

        metadata["created_at"] = datetime.now().isoformat()
        metadata["cache_key"] = cache_key
        metadata["original_path"] = str(file_path)
        metadata["file_size_mb"] = cache_path.stat().st_size / (1024 * 1024)

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            f"Cached {metadata['file_size_mb']:.2f}MB: {cache_key} "
            f"({metadata.get('dataset', 'unknown')})"
        )

        # Enforce cache size limit
        self._evict_if_needed()

    def clear(self) -> None:
        """Clear all cached files."""
        count = 0
        for cache_file in self.cache_root.glob("*"):
            if cache_file.is_file():
                cache_file.unlink()
                count += 1
        logger.info(f"Cleared cache: removed {count} files")

    def get_cache_size_gb(self) -> float:
        """
        Get current cache size in gigabytes.

        Returns
        -------
        float
            Cache size in GB
        """
        total_bytes = sum(
            f.stat().st_size for f in self.cache_root.glob("*") if f.is_file()
        )
        return total_bytes / (1024**3)

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns
        -------
        dict
            Cache statistics (size, file count, datasets, etc.)
        """
        cache_files = list(self.cache_root.glob("*.nc"))
        metadata_files = list(self.cache_root.glob("*.meta.json"))

        total_size_gb = self.get_cache_size_gb()

        # Count by dataset
        datasets: Dict[str, int] = {}
        for meta_file in metadata_files:
            try:
                with open(meta_file, "r") as f:
                    meta = json.load(f)
                    dataset = meta.get("dataset", "unknown")
                    datasets[dataset] = datasets.get(dataset, 0) + 1
            except Exception:
                pass

        return {
            "cache_root": str(self.cache_root),
            "total_size_gb": total_size_gb,
            "max_size_gb": self.max_size_gb,
            "file_count": len(cache_files),
            "datasets": datasets,
            "ttl_days": self.ttl_days,
        }

    def _compute_checksum(self, file_path: Path) -> str:
        """
        Compute SHA256 checksum of file.

        Parameters
        ----------
        file_path : Path
            Path to file

        Returns
        -------
        str
            Hex digest of SHA256 hash
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read in 64KB chunks for memory efficiency
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _remove_cache_entry(self, cache_key: str) -> None:
        """
        Remove cache entry (data file + metadata).

        Parameters
        ----------
        cache_key : str
            Cache key to remove
        """
        cache_path = self.cache_root / f"{cache_key}.nc"
        metadata_path = self.cache_root / f"{cache_key}.meta.json"

        cache_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)

        logger.debug(f"Removed cache entry: {cache_key}")

    def _evict_if_needed(self) -> None:
        """
        LRU eviction if cache exceeds max size.

        Removes oldest files (by created_at timestamp) until
        cache size is below max_size_gb.
        """
        current_size = self.get_cache_size_gb()
        if current_size <= self.max_size_gb:
            return

        logger.info(
            f"Cache size {current_size:.2f}GB exceeds limit {self.max_size_gb}GB, "
            "evicting oldest entries"
        )

        # Get all cache entries with timestamps
        entries = []
        for meta_file in self.cache_root.glob("*.meta.json"):
            try:
                with open(meta_file, "r") as f:
                    meta = json.load(f)
                    cache_key = meta.get("cache_key")
                    created_at = datetime.fromisoformat(meta["created_at"])
                    entries.append((created_at, cache_key))
            except Exception:
                # Remove corrupted metadata
                meta_file.unlink(missing_ok=True)

        # Sort by age (oldest first)
        entries.sort(key=lambda x: x[0])

        # Evict oldest entries until size is below limit
        evicted_count = 0
        for created_at, cache_key in entries:
            if self.get_cache_size_gb() <= self.max_size_gb:
                break
            self._remove_cache_entry(cache_key)
            evicted_count += 1

        logger.info(
            f"Evicted {evicted_count} entries, "
            f"new cache size: {self.get_cache_size_gb():.2f}GB"
        )
