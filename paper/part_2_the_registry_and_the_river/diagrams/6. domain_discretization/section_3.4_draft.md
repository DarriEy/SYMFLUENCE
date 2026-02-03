# Section 3.4 — Draft text (GRU vs HRU distinction)

> **Notes for authors:**
> - Figure number (currently "Figure 6") should be verified against final numbering.
> - GRU/HRU counts are from the actual shapefiles (49 GRUs, 11 elevation-only HRUs in GRU 38, 78 elevation×aspect HRUs in GRU 38).
> - This text is intended to clarify the spatial hierarchy concept only; domain delineation workflows, point-scale, and distributed configurations are covered elsewhere.

---

## 3.4 Spatial Discretization Hierarchy

SYMFLUENCE separates spatial discretization into two distinct levels — Grouped Response Units (GRUs) and Hydrological Response Units (HRUs) — decoupling the lateral routing structure from within-catchment process heterogeneity (Figure 6).

**GRUs** define the routing topology. Each GRU corresponds to a subcatchment draining to a single river segment, and the complete set of GRUs tiles the catchment without overlap (Figure 6a). The drainage network connecting GRUs encodes upstream–downstream relationships that govern lateral water transfer through routing models such as mizuRoute. GRU boundaries are derived from DEM-based flow accumulation and remain fixed regardless of how the interior of each subcatchment is further subdivided.

**HRUs** capture sub-grid heterogeneity within each GRU. An HRU is defined by one or more discretizing attributes — elevation bands, aspect classes, land cover types, soil classifications — intersected with the parent GRU boundary. Because HRUs nest strictly within their parent GRU, the mapping from each HRU to its routing segment is unambiguous: all HRUs within a GRU contribute runoff to the same river segment.

The distinction between the two levels is illustrated for GRU 38 of the Bow River at Banff domain in Figures 6b–c. A single-attribute discretization by elevation bands yields 11 HRUs (Figure 6b), each spanning a 200 m elevation interval and receiving lapse-rate-adjusted forcing. Combining elevation with eight aspect classes produces 78 HRUs in the same GRU (Figure 6c), capturing the interaction between elevation-dependent temperature gradients and aspect-dependent radiation receipt. This combined discretization is extensible: additional attributes (land cover, soil type) can be included, with HRU count growing as the geometric intersection of all attribute classes (Figure 6d).

The hierarchical design carries two practical consequences. First, increasing HRU complexity within a GRU does not alter the routing network — practitioners can refine vertical process representation independently of lateral connectivity. Second, all HRUs within a GRU share calibrated parameter values by default; HRUs differ only in their forcing adjustments and static attributes, so adding discretization layers does not increase the calibration parameter space. This property makes it feasible to test progressively finer discretizations without proportional growth in calibration cost.
