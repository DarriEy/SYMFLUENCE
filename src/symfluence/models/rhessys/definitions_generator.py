# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
RHESSys Default Definition Files Generator

Handles generation of RHESSys default parameter files (.def) for each
level of the spatial hierarchy: basin, hillslope, zone, soil (patch),
vegetation (stratum), and landuse.

Extracted from RHESSysPreProcessor for modularity.
"""
import logging

logger = logging.getLogger(__name__)


class RHESSysDefinitionsGenerator:
    """
    Generates RHESSys default parameter definition files.

    Creates .def files that define default parameter values for each
    level of the RHESSys spatial hierarchy. These files are referenced
    by ID in the worldfile.

    Args:
        preprocessor: Parent RHESSysPreProcessor instance providing access
            to configuration, paths, and helper methods.
    """

    def __init__(self, preprocessor):
        self.pp = preprocessor

    def generate_default_files(self):
        """Generate RHESSys default parameter files."""
        logger.info("Generating default files...")

        self._write_default_basin_file()
        self._write_default_hillslope_file()
        self._write_default_zone_file()
        self._write_default_soil_file()
        self._write_default_vegetation_file()
        self._write_default_landuse_file()

        logger.info(f"Default files written to {self.pp.defs_dir}")

    def _write_default_basin_file(self):
        """Write basin default parameter file."""
        # Note: sat_to_gw_coeff and gw_loss_coeff are primarily patch/hillslope params,
        # but RHESSys also reads them at basin level as defaults. Keep them here.
        basin_def = self.pp.defs_dir / "basin.def"
        basin_content = """1    basin_default_ID
-6.0    psi_air_entry
0.12    pore_size_index
0.01    sat_to_gw_coeff
1.0    gw_loss_coeff
0.5    n_routing_power
1.0    m_pai
"""
        basin_def.write_text(basin_content, encoding='utf-8')

    def _write_default_hillslope_file(self):
        """Write hillslope default parameter file."""
        # gw_loss_coeff controls baseflow rate - higher values = faster baseflow drainage
        # gw_loss_fast_threshold: when gw.storage > threshold, excess uses gw_loss_fast_coeff
        #   - Set to positive value (e.g., 0.1m) to enable fast flow component
        #   - Default of -1.0 disables fast flow entirely (all storage uses slow coeff)
        # gw_loss_fast_coeff: coefficient for fast groundwater drainage when storage > threshold
        hillslope_def = self.pp.defs_dir / "hillslope.def"
        hillslope_content = """1    hillslope_default_ID
0.1    gw_loss_coeff
0.1    gw_loss_fast_threshold
0.3    gw_loss_fast_coeff
"""
        hillslope_def.write_text(hillslope_content, encoding='utf-8')

    def _write_default_zone_file(self):
        """Write zone default parameter file."""
        zone_def = self.pp.defs_dir / "zone.def"
        # RHESSys lapse convention: T_zone = T_base - lapse_rate * (z_zone - z_base)
        # Positive values = standard environmental lapse (temp decreases with elevation)
        zone_content = """1    zone_default_ID
0.0    atm_trans_lapse_rate
0.006    dewpoint_lapse_rate
0.0065    lapse_rate_tmax
0.0065    lapse_rate_tmin
6.0    max_effective_lai
2.0    max_snow_temp
-2.0    min_rain_temp
0.7    ndep_NO3
0.004    wet_lapse_rate
-999.0    lapse_rate_precip_default
"""
        zone_def.write_text(zone_content, encoding='utf-8')

    def _write_default_soil_file(self):
        """Write soil (patch) default parameter file."""
        # Note: soil_depth set to 3.0m for reasonable storage without excessive depth
        # m set to 2.0 for moderate Ksat decay with depth
        # Ksat_0 set to allow reasonable infiltration
        # Ksat_0_v set to realistic ratio with Ksat_0 (typically 1-10x, not 200x)
        soil_def = self.pp.defs_dir / "soil.def"
        soil_content = """1    patch_default_ID
-6.0    psi_air_entry
0.12    pore_size_index
0.45    porosity_0
0.45    porosity_decay
0.1    Ksat_0
1.0    Ksat_0_v
2.0    m
1.5    m_z
2.0    N_decay
3.0    soil_depth
0.001    sat_to_gw_coeff
1.0    active_zone_z
0.2    albedo
-100.0    maximum_snow_energy_deficit
4.0    snow_melt_Tcoef
1.0    snow_water_capacity
0.5    wilting_point
0.15    theta_mean_std_p1
0.0    theta_mean_std_p2
0.0    gl_c
0.01    gsurf_slope
0.001    gsurf_intercept
"""
        soil_def.write_text(soil_content, encoding='utf-8')

    def _write_default_vegetation_file(self):
        """Write vegetation (stratum) default parameter file."""
        # Based on evergreen conifer parameters from RHESSys ParameterLibrary
        veg_def = self.pp.defs_dir / "stratum.def"
        veg_content = """1    stratum_default_ID
TREE    epc.veg.type
0.8    K_absorptance
0.1    K_reflectance
0.1    K_transmittance
1.0    PAR_absorptance
0.0    PAR_reflectance
0.0    PAR_transmittance
0.5    epc.ext_coef
0.00024    specific_rain_capacity
0.00024    specific_snow_capacity
0.002    wind_attenuation_coef
1.5    mrc.q10
0.21    mrc.per_N
0.2    epc.gr_perc
1.0    lai_stomatal_fraction
0.1    epc.flnr
0.03    epc.ppfd_coef
15.0    epc.topt
40.0    epc.tmax
0.2    epc.tcoef
-0.65    epc.psi_open
-2.5    epc.psi_close
500.0    epc.vpd_open
3500.0    epc.vpd_close
0.006    epc.gl_smax
0.00006    epc.gl_c
0.01    gsurf_slope
0.001    gsurf_intercept
static    epc.phenology_flag
EVERGREEN    epc.phenology.type
4.0    epc.max_lai
9.0    epc.proj_sla
2.6    epc.lai_ratio
1.4    epc.proj_swa
0.27    epc.leaf_turnover
91    epc.day_leafon
260    epc.day_leafoff
30    epc.ndays_expand
30    epc.ndays_litfall
45.0    epc.leaf_cn
70.0    epc.leaflitr_cn
0.0    min_heat_capacity
0.0    max_heat_capacity
waring    epc.allocation_flag
1.0    epc.storage_transfer_prop
0.27    epc.froot_turnover
0.7    epc.livewood_turnover
0.01    epc.kfrag_base
139.7    epc.froot_cn
200.0    epc.livewood_cn
0.31    epc.leaflitr_flab
0.45    epc.leaflitr_fcel
0.24    epc.leaflitr_flig
0.23    epc.frootlitr_flab
0.41    epc.frootlitr_fcel
0.36    epc.frootlitr_flig
0.52    epc.deadwood_fcel
0.48    epc.deadwood_flig
1.325    epc.alloc_frootc_leafc
0.3    epc.alloc_crootc_stemc
1.62    epc.alloc_stemc_leafc
0.073    epc.alloc_livewoodc_woodc
0.05    epc.maxlgf
0.5    epc.alloc_prop_day_growth
0.0    epc.daily_fire_turnover
0.57    epc.height_to_stem_exp
11.39    epc.height_to_stem_coef
0.00005    epc.max_daily_mortality
0.00003    epc.min_daily_mortality
0.0    epc.daily_mortality_threshold
0    epc.dyn_alloc_prop_day_growth
0.22    epc.min_leaf_carbon
100    epc.max_years_resprout
0.001    epc.resprout_leaf_carbon
0.01    epc.litter_gsurf_slope
0.001    epc.litter_gsurf_intercept
1.0    epc.coef_CO2
0.8    epc.root_growth_direction
8.0    epc.root_distrib_parm
0.6    epc.crown_ratio
-2.0    epc.gs_tmin
5.0    epc.gs_tmax
900.0    epc.gs_vpd_min
4100.0    epc.gs_vpd_max
36000.0    epc.gs_dayl_min
39600.0    epc.gs_dayl_max
-15.0    epc.gs_psi_min
-14.0    epc.gs_psi_max
6.0    epc.gs_ravg_days
0.5    epc.gsi_thresh
1.0    epc.gs_npp_on
187.0    epc.gs_npp_slp
197.0    epc.gs_npp_intercpt
0.2    epc.max_storage_percent
0.27    epc.min_percent_leafg
0.25    epc.dickenson_pa
0.8    epc.waring_pa
2.5    epc.waring_pb
0.0    epc.branch_turnover
0    epc.Tacclim
3.22    epc.Tacclim_intercpt
0.046    epc.Tacclim_slp
30    epc.Tacclim_days
0.002    epc.litter_moist_coef
50.0    epc.litter_density
0    epc.nfix
1    epc.edible
0    epc.psi_curve
-1.0    epc.psi_threshold
0.2    epc.psi_slp
1.0    epc.psi_intercpt
"""
        veg_def.write_text(veg_content, encoding='utf-8')

    def _write_default_landuse_file(self):
        """Write landuse default parameter file."""
        landuse_def = self.pp.defs_dir / "landuse.def"
        # Only write if not already created by world header generation
        if not landuse_def.exists():
            landuse_content = """1    landuse_default_ID
1.0    irrigation_fraction
1.0    septic_water_load
1.0    septic_NO3_load
1.0    fertilizer_NO3_load
1.0    fertilizer_NH4_load
1    fertilizer_day_of_year
0.0    grazing_Closs
0.0    impervious_fraction
"""
            landuse_def.write_text(landuse_content, encoding='utf-8')
