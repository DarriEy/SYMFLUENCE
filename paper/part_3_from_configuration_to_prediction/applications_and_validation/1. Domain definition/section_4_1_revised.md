# 4. Applications and Validation

This section demonstrates SYMFLUENCE capabilities through applications spanning point-scale flux estimation to regional hydrological simulation. Each application illustrates specific framework features while validating the workflow against independent observations.

## 4.1 Domain Definition

SYMFLUENCE supports three fundamental spatial modes -- point, watershed, and regional -- enabling consistent workflows from single-site validation to continental-scale simulation. We demonstrate this flexibility through applications at Paradise SNOTEL (Washington, USA), the Bow River at Banff (Alberta, Canada), and the national domain of Iceland. The spatial discretizations considered in the study are listed in Table 1.

**Table 1.** Spatial domains and discretization configurations considered in the study.

| Domain | Configuration | GRUs | HRUs | River Segments |
|--------|--------------|------|------|----------------|
| Paradise | Point (single GRU) | 1 | 1 | 0 |
| | ERA5 forcing grid | -- | -- | 9 cells |
| Bow River | Lumped | 1 | 1 | 1 |
| | + Elevation bands | 1 | 12 | 1 |
| | + Elevation + SD routing | 1 | 12 | 49 |
| | + Land cover | 1 | 9 | 1 |
| | + Elevation + Aspect | 1 | 94 | 1 |
| | Semi-distributed | 49 | 49 | 49 |
| | + Elevation bands | 49 | 379 | 49 |
| | + Elevation + Aspect | 49 | 2,596 | 49 |
| | Distributed (1 km) | 2,335 | 2,335 | 2,335 |
| Iceland | Semi-distributed | 6,600 | 6,600 | 6,606 |
| | + Coastal GRUs | 7,618 | 7,618 | 6,606 |
| | + Elevation HRUs | 7,618 | 21,474 | 6,606 |

### 4.1.1 Point-Scale: Paradise SNOTEL

Point-scale applications treat the domain as a single computational unit without lateral routing, appropriate for flux tower validation or snow monitoring stations where the primary interest lies in vertical process dynamics. The Paradise SNOTEL station (#679; 46.78°N, 121.75°W; elevation 1,560 m) on Mount Rainier receives approximately 2,500 mm annual precipitation, predominantly as snow, making it an ideal testbed for snow process representation.

Configuration requires only pour point coordinates and a bounding box for forcing data acquisition. The framework automatically generates a single-GRU domain and identifies the forcing grid cells that overlap the station location. Figure 3 illustrates the resulting domain: the underlying 30 m DEM reveals substantial topographic heterogeneity beneath the 3 x 3 ERA5 forcing grid (0.25°, dashed orange), with elevations ranging from ~260 m in the river valleys to 4,390 m at Mount Rainier's summit -- all within a single ERA5 cell. The overlaid AORC grid (0.01°, purple) highlights the resolution contrast between available forcing products: each ERA5 cell contains 625 AORC cells, offering considerably finer representation of orographic precipitation gradients on the volcano's flanks. This subgrid heterogeneity motivates the forcing ensemble experiments presented in Section 4.3, where multiple forcing products are evaluated at the same location. The minimal configuration -- requiring approximately 15 lines of YAML -- produces a complete model setup for snow water equivalent simulation and validation against SNOTEL observations.

### 4.1.2 Watershed-Scale: Bow River at Banff

Watershed applications represent the most common use case, where spatially distributed processes require explicit representation but domain extent remains computationally tractable. The Bow River basin upstream of Banff (51.17°N, 115.57°W; approximately 2,210 km², 20 ERA5 cells) spans elevations from 1,383 m at the gauge to over 3,436 m at the continental divide, exhibiting strong gradients in precipitation, temperature, and snow dynamics that motivate spatial discretization. Figure 4 illustrates SYMFLUENCE's discretization options across three columns (lumped, semi-distributed, distributed) and three rows of increasing complexity.

**Lumped mode** (Figure 4a) treats the entire basin as a single GRU, aggregating all forcing and parameters to basin-average values. This mode minimizes computational cost and parameter dimensionality, appropriate for initial calibration or when spatial data are unavailable. Within this single GRU, the framework supports subdivision into HRUs by elevation bands at 200 m intervals (Figure 4d; 12 HRUs spanning 1,383--3,436 m) or by IGBP land cover classification (Figure 4g; 9 HRUs). Elevation-band subdivision enables lapse-rate corrections for temperature and precipitation without requiring explicit sub-basin delineation.

**Semi-distributed mode** (Figure 4b) partitions the basin into 49 sub-basin GRUs derived from the TDX river network topology, each further subdividable into HRUs. Elevation-band subdivision yields 379 HRUs (~7.7 per GRU; Figure 4e), while combined elevation and 8-class aspect subdivision produces 2,596 HRUs (~53 per GRU; Figure 4h). The semi-distributed river network (49 segments) enables explicit lateral routing through mizuRoute.

**Distributed mode** (Figure 4c) discretizes the basin into 2,335 grid cells at 1 km resolution, each functioning as both a GRU and an HRU, with a corresponding distributed river network of 2,335 segments. This mode provides the highest spatial fidelity but at proportionally increased computational cost.

**Hybrid configurations** demonstrate that spatial discretization choices for hydrological modeling and routing are independent in SYMFLUENCE and can be mixed freely. Figure 4i shows a lumped GRU (single computational unit) paired with the semi-distributed river network (49 segments), where the lumped runoff is distributed across routing segments. Figure 4f extends this concept by combining lumped elevation-band HRUs (12 HRUs) with semi-distributed routing (49 segments). This hybrid captures elevation-dependent processes -- orographic precipitation gradients, snow accumulation and melt timing -- through lapse-rate corrections applied to each elevation band, while routing the resulting runoff through the spatially explicit river network. Such configurations offer a computationally efficient compromise: vertical process heterogeneity is represented without requiring full sub-basin delineation, and the routing network preserves spatial patterns in streamflow generation and travel time.

The progression from lumped to distributed increases computational cost roughly proportionally to HRU count, but enables representation of within-basin heterogeneity critical for snow-dominated systems. Importantly, all configurations derive from the same pour point and underlying DEM, ensuring consistent domain boundaries while varying only internal discretization.

River network delineation uses TauDEM algorithms applied to the MERIT-DEM at 90 m resolution. Stream initiation occurs at cells exceeding a configurable flow accumulation threshold (5,000 cells for Bow, corresponding to approximately 4.5 km² contributing area). The framework automatically generates topology files mapping HRU-to-segment connectivity for mizuRoute lateral routing.

### 4.1.3 Regional-Scale: Iceland

Regional applications extend watershed concepts to multi-basin domains where consistent forcing and parameter treatment across drainage divides enables coherent large-scale simulation. Iceland (approximately 102,000 km²) presents an ideal demonstration domain: island geography provides unambiguous boundaries, diverse hydroclimatology spans glaciated highlands to coastal lowlands, and the LamaH-ICE dataset provides validation streamflow across multiple gauges.

Regional configuration specifies a bounding box rather than a pour point, with automatic delineation of all watersheds draining to the coast. The `DELINEATE_COASTAL_WATERSHEDS` option identifies terminal basins draining directly to ocean boundaries, ensuring complete coverage without manual pour point specification. Figure 5 shows the resulting domain in three panels of increasing complexity.

Panel (a) shows the river basin GRUs obtained from geofabric delineation: 6,600 GRUs connected by 6,606 river segments, covering the major river systems (Thjorsa, Olfusa, Jokulsa) and interior highlands. Panel (b) adds 1,018 coastal GRUs -- small drainage areas along the coastline that drain directly to the ocean rather than through the river network -- bringing the total to 7,618 GRUs and ensuring complete spatial coverage of the island. Panel (c) further subdivides each GRU into elevation-band HRUs, yielding 21,474 HRUs across the domain (approximately 2.8 per GRU on average), with elevations spanning from sea level to over 2,000 m. The ERA5 forcing grid for Iceland comprises 618 cells at 0.25° resolution. All three panels show the river network and ERA5 grid overlay, coloured by mean elevation using a consistent terrain colormap.

### 4.1.4 Discretization Trade-offs

The choice of spatial discretization reflects a trade-off between process representation and computational/parametric complexity. Table 1 summarizes the configurations across the three demonstration domains.

Lumped and semi-distributed modes share parameters across all computational units, maintaining the same calibration dimensionality regardless of HRU count. This means a lumped configuration with 12 elevation-band HRUs has the same number of free parameters as one with a single HRU -- the elevation bands differ only in their forcing (via lapse-rate adjustments) and static attributes (mean elevation, area), not in calibrated parameter values. Distributed modes can optionally enable spatially varying parameters, though this substantially increases calibration complexity and typically requires regionalization or transfer approaches.

The hybrid configurations shown in Figures 4f and 4i illustrate that practitioners need not choose a single discretization paradigm. By decoupling the hydrological response unit structure from the routing network, SYMFLUENCE allows users to independently control vertical complexity (elevation bands, land cover classes, aspect classes) and horizontal complexity (routing network density). For snow-dominated basins like the Bow River, a lumped elevation-band configuration with semi-distributed routing (Figure 4f) may capture the dominant sources of spatial variability at a fraction of the computational cost of a fully distributed setup.

### 4.1.5 Forcing Grid Intersection

Regardless of spatial mode, SYMFLUENCE generates intersection weights mapping forcing grid cells to computational units. For ERA5 (0.25° resolution), the Paradise point domain intersects 9 grid cells; the Bow watershed intersects 20 cells; Iceland requires 618 cells. Intersection weights account for partial overlaps, and the framework supports bilinear interpolation or area-weighted averaging depending on configuration.

The intersection geometry is preserved as a shapefile (e.g., `forcing_ERA5.shp`), enabling visualization of forcing grid coverage relative to basin boundaries (shown as dashed orange lines in Figures 4 and 5). This transparency supports diagnosis of scale mismatches -- for instance, identifying when a small headwater basin falls entirely within a single coarse forcing cell, potentially missing orographic precipitation gradients that would be captured by higher-resolution forcing products.
