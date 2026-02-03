# SYMFLUENCE Paper Section 4.1 - Required Text Changes

Generated: 2026-01-26

This document summarizes all text changes needed to update Section 4.1 (Domain Definition) to match the actual shapefile data.

---

## Summary of Discrepancies Found and Fixed

| Item | Paper Says | Actual Data |
|------|-----------|-------------|
| Paradise ERA5 cells | 4 | **9** |
| Bow semi-distributed | 15 GRUs | **49 GRUs** |
| Bow elevation HRUs | ~85 | **379** (semi-dist), **12** (lumped) |
| Iceland GRUs | 847 | **7,618** |

---

## Section 4.1.1 Paradise Paragraph

**Change:**
- "4 ERA5 grid cells" → "**9 ERA5 grid cells**"

---

## Section 4.1.2 Bow River Paragraph

**Replace the existing discretization description with:**

> SYMFLUENCE supports flexible spatial discretization at multiple levels:
>
> **Lumped configuration:** A single GRU representing the entire basin, subdividable into HRUs by:
> - Elevation bands (200m intervals): 12 HRUs spanning 1,383-3,436 m
> - Elevation + aspect (8 cardinal directions): 94 HRUs
> - Land cover (IGBP classification): 9 HRUs
>
> **Semi-distributed configuration:** 49 sub-basin GRUs derived from river network topology, each further subdividable into:
> - Elevation bands: 379 HRUs (~7.7 per GRU)
> - Elevation + aspect: 2,596 HRUs (~53 per GRU)
>
> **Distributed configuration:** 2,335 grid cells at 1 km resolution.

---

## Section 4.1.3 Iceland Paragraph

**Changes:**
- "847 GRUs" → "**7,618 GRUs**"
- Add: "with optional elevation subdivision yielding **21,474 HRUs**"

---

## New Comprehensive Table for Section 4.1

| Domain | Configuration | GRUs | HRUs | Segments |
|--------|--------------|------|------|----------|
| Paradise | Point | 1 | 1 | 0 |
| | ERA5 forcing grid | - | - | 9 cells |
| Bow | Lumped | 1 | 1 | 1 |
| | + Elevation | 1 | 12 | 1 |
| | + Elev + SD routing | 1 | 12 | 49 |
| | + Landclass | 1 | 9 | 1 |
| | + Elev+Aspect | 1 | 94 | 1 |
| | Semi-distributed | 49 | 49 | 49 |
| | + Elevation | 49 | 379 | 49 |
| | + Elev+Aspect | 49 | 2,596 | 49 |
| | Distributed | 2,335 | 2,335 | 2,335 |
| Iceland | Semi-distributed | 7,618 | 7,618 | 6,606 |
| | + Elevation | 7,618 | 21,474 | 6,606 |

---

## Generated Figures

1. **figure_4_1_domain_definition.png/pdf** - Updated main figure (2-row, 8-panel layout)
   - Row 1: Paradise (9 cells), Iceland (7,618 GRUs)
   - Row 2: Lumped (1), Elevation (12), Land Cover (9), Sub-basins (49), Sub+Elev (379), Distributed (2,335)

4. **figure_4_1c_bow.png/pdf** - Updated Bow River figure (3x3 grid, 9 map panels)
   - Row 1: (a) Single GRU, (b) Sub-basin GRUs, (c) Grid cells (1 km)
   - Row 2: (d) + Elevation bands, (e) + Elevation bands, (f) + Elev. + SD routing [NEW - combines d+i]
   - Row 3: (g) + Land cover, (h) + Elevation + Aspect, (i) Lumped + semi-dist. routing
   - Panel (f) replaces the previous summary stats table with a new map showing
     lumped elevation-band HRUs (12) routed through the semi-distributed river network (49 segments)

2. **figure_4_1_hru_methods.png/pdf** - New supplementary figure (2x3 grid)
   - Shows all HRU subdivision methods from lumped to semi-distributed with aspect

3. **figure_4_1_gru_comparison.png/pdf** - Updated GRU comparison (1x3)
   - Lumped → Sub-basins → Elevation HRUs progression

---

## Verification Commands

To verify feature counts in shapefiles:

```bash
# Paradise ERA5 cells
ogrinfo -so paradise/forcing/forcing_ERA5.shp forcing_ERA5 | grep "Feature Count"
# Output: Feature Count: 9

# Bow configurations
ogrinfo -so bow/catchment/lumped/run_1/*_HRUs_GRUs.shp *_HRUs_GRUs | grep "Feature Count"
# Output: Feature Count: 1 (lumped)

ogrinfo -so bow/catchment/lumped/run_1/*_HRUs_elevation.shp *_HRUs_elevation | grep "Feature Count"
# Output: Feature Count: 12

ogrinfo -so bow/catchment/semidistributed/run_1/*_HRUs_GRUs.shp *_HRUs_GRUs | grep "Feature Count"
# Output: Feature Count: 49

ogrinfo -so bow/catchment/semidistributed/run_1/*_HRUs_elevation.shp *_HRUs_elevation | grep "Feature Count"
# Output: Feature Count: 379

ogrinfo -so bow/catchment/distributed/run_1/*_HRUs_GRUS.shp *_HRUs_GRUS | grep "Feature Count"
# Output: Feature Count: 2335

# Iceland
ogrinfo -so Iceland/river_basins/Iceland_riverBasins_with_coastal.shp Iceland_riverBasins_with_coastal | grep "Feature Count"
# Output: Feature Count: 7618
```

---

## Land Cover Classes Present (IGBP)

The Bow River watershed contains 9 land cover classes:

| Code | Class Name |
|------|------------|
| 1 | Evergreen Needleleaf Forest |
| 8 | Woody Savannas |
| 9 | Savannas |
| 10 | Grasslands |
| 11 | Permanent Wetlands |
| 13 | Urban and Built-up |
| 15 | Snow and Ice |
| 16 | Barren or Sparsely Vegetated |
| 17 | Water Bodies |
