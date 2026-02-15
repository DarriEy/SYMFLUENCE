.PHONY: install install-dev install-gdal clean-constraints

# Auto-detect system GDAL version for Python binding compatibility
GDAL_VERSION := $(shell gdal-config --version 2>/dev/null)
CONSTRAINTS := constraints-gdal.txt

# Generate constraints file pinning GDAL to system version
$(CONSTRAINTS):
ifdef GDAL_VERSION
	@echo "gdal==$(GDAL_VERSION)" > $(CONSTRAINTS)
	@echo "Pinned GDAL Python bindings to system libgdal $(GDAL_VERSION)"
else
	$(error gdal-config not found. Install GDAL first: brew install gdal (macOS) / apt install libgdal-dev (Ubuntu))
endif

# Install GDAL Python bindings matching system library
install-gdal: $(CONSTRAINTS)
	uv pip install "gdal==$(GDAL_VERSION)"

# Install symfluence with GDAL version-matched to system
install: $(CONSTRAINTS)
	uv pip install -c $(CONSTRAINTS) .

# Install in editable mode with dev/test extras
install-dev: $(CONSTRAINTS)
	uv pip install -c $(CONSTRAINTS) -e ".[dev,test]"

clean-constraints:
	rm -f $(CONSTRAINTS)
