# Deprecated Code

This directory contains deprecated code that will be removed in a future version.

## Files

### `attribute_processing.py`
- **Status:** Deprecated as of version 0.5.10
- **Replacement:** Use `attribute_processing_refactored.py` in the parent directory
- **Reason:** The monolithic 250KB file was refactored into a modular system using the `attribute_processors/` directory
- **Migration:** The new system provides the same functionality through individual processor modules (elevation, geology, soil, landcover, climate, hydrology)
- **Removal planned:** Version 1.0.0

## Migration Guide

If you have custom code importing from `attribute_processing.py`, update to use `attribute_processing_refactored.py`:

```python
# Old (deprecated)
from symfluence.utils.data.preprocessing.attribute_processing import AttributeProcessor

# New (current)
from symfluence.utils.data.preprocessing.attribute_processing_refactored import AttributeProcessor
```
