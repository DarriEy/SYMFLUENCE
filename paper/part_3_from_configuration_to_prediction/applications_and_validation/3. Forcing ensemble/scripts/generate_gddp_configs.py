#!/usr/bin/env python3
"""
Generate additional GDDP-CMIP6 ensemble member configurations.

Creates configs for 7 new climate models to complement the existing 3
(ACCESS-CM2, GFDL-ESM4, MRI-ESM2-0), spanning the full range of
equilibrium climate sensitivities available in NEX-GDDP-CMIP6:

  Low ECS:   INM-CM5-0 (1.9 C), NorESM2-LM (2.5 C)
  Mid ECS:   MPI-ESM1-2-HR (3.0 C)
  High ECS:  CNRM-CM6-1 (4.9 C), IPSL-CM6A-LR (4.8 C),
             CanESM5 (5.6 C), UKESM1-0-LL (5.0 C)

Can operate in two modes:
  1. Template mode (default): reads an existing GDDP config and substitutes
     model-specific identifiers.
  2. Standalone mode (--standalone): generates configs from built-in template
     when the existing configs are not accessible (e.g. cloud sync issues).

Usage:
    python generate_gddp_configs.py                # template mode
    python generate_gddp_configs.py --standalone   # standalone mode
"""

import argparse
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parent.parent
CONFIGS_DIR = STUDY_DIR / "configs"

# New GDDP members to add
# key: (NEX-GDDP-CMIP6 model name, short description, ECS)
NEW_MEMBERS = {
    'gddp_ukesm1_0_ll':    ('UKESM1-0-LL',    'UK Met Office',                    '5.0'),
    'gddp_canesm5':        ('CanESM5',         'Canadian Centre',                  '5.6'),
    'gddp_ipsl_cm6a_lr':   ('IPSL-CM6A-LR',   'Institut Pierre-Simon Laplace',    '4.8'),
    'gddp_cnrm_cm6_1':     ('CNRM-CM6-1',     'CNRM/CERFACS France',              '4.9'),
    'gddp_mpi_esm1_2_hr':  ('MPI-ESM1-2-HR',  'Max Planck Institute',             '3.0'),
    'gddp_noresm2_lm':     ('NorESM2-LM',     'Norwegian Climate Centre',         '2.5'),
    'gddp_inm_cm5_0':      ('INM-CM5-0',      'Institute for Numerical Mathematics', '1.9'),
}

# Template substitution identifiers
TEMPLATE_FORCING_KEY = 'gddp_access_cm2'
TEMPLATE_MODEL_NAME = 'ACCESS-CM2'

# ---------------------------------------------------------------------------
# Built-in template for standalone mode
# Derived from the common settings in README.md and analyze_results.py:
#   - Domain: Paradise SNOTEL Station, WA
#   - Model: SUMMA via SYMFLUENCE
#   - Calibration: DDS optimizing RMSE on SWE
# ---------------------------------------------------------------------------
STANDALONE_TEMPLATE = """\
# ============================================================================
# SYMFLUENCE Configuration: Forcing Ensemble Study
# GDDP-CMIP6 Member: {model_name}
# ============================================================================
#
# Auto-generated for the Section 4.3 forcing ensemble experiment.
# This configuration runs SUMMA driven by {model_name} from the
# NEX-GDDP-CMIP6 downscaled climate projections (historical period).
#
# ECS ({model_name}): ~{ecs} C

# --- Experiment identification ---
EXPERIMENT_ID: forcing_ensemble_{forcing_key}
DOMAIN_NAME: paradise_snotel_wa_{forcing_key}

# --- Paths ---
CONFLUENCE_DATA_DIR: /Users/darrieythorsson/compHydro/code/SYMFLUENCE_data
CONFLUENCE_CODE_DIR: /Users/darrieythorsson/compHydro/code/SYMFLUENCE

# --- Domain definition ---
DOMAIN_DEFINITION:
  POINT_NAME: paradise_snotel_wa
  LATITUDE: 46.7862
  LONGITUDE: -121.7474
  ELEVATION: 1561.0
  BOUNDING_BOX:
    lon_min: -121.80
    lat_min: 46.75
    lon_max: -121.70
    lat_max: 46.82

# --- Forcing data ---
FORCING_DATASET: gddp-cmip6
FORCING_TYPE: gddp-cmip6
GDDP_MODEL: {model_name}
GDDP_SCENARIO: historical
FORCING_START_YEAR: 2015
FORCING_END_YEAR: 2020
FORCING_VARIABLES:
  - pr
  - tas
  - tasmin
  - tasmax
  - hurs
  - rsds
  - rlds
  - sfcWind

# --- Time periods ---
EXPERIMENT_TIME_START: "2015-01-01"
EXPERIMENT_TIME_END: "2020-12-31"
CALIBRATION_PERIOD: "2015-10-01, 2018-09-30"
EVALUATION_PERIOD: "2018-10-01, 2020-09-30"

# --- Hydrological model ---
HYDROLOGICAL_MODEL: SUMMA

# --- Calibration settings ---
CALIBRATION_METHOD: DDS
CALIBRATION_METRIC: RMSE
CALIBRATION_TARGET: SWE
NUMBER_OF_ITERATIONS: 10
PARAMS_TO_CALIBRATE: >
  tempCritRain,tempRangeTimestep,frozenPrecipMultip,
  albedoMax,albedoMinWinter,albedoDecayRate,
  constSnowDen,mw_exp,k_snow,z0Snow
BASIN_PARAMS_TO_CALIBRATE: routingGammaScale

# --- Observations ---
OBSERVATIONS:
  SWE:
    SOURCE: snotel
    STATION_ID: paradise
    VARIABLE: swe
    UNITS: inches

# --- Output settings ---
OUTPUT_FREQUENCY: daily
OUTPUT_VARIABLES:
  - scalarSWE
  - mLayerVolFracLiq
  - mLayerDepth
  - scalarRainfall
  - scalarSnowfall
  - scalarSurfaceTemp
"""


def generate_from_template():
    """Generate configs by substituting identifiers in an existing config."""
    template_path = CONFIGS_DIR / f"config_paradise_{TEMPLATE_FORCING_KEY}.yaml"
    if not template_path.exists():
        print(f"Template not found: {template_path.name}")
        return False

    try:
        template_text = template_path.read_text()
    except (OSError, TimeoutError) as e:
        print(f"Cannot read template (file may not be locally cached): {e}")
        return False

    print(f"Using template: {template_path.name}")
    return _generate_configs(template_text, mode='template')


def generate_standalone():
    """Generate configs from built-in template."""
    print("Using built-in standalone template")
    return _generate_configs(None, mode='standalone')


def _generate_configs(template_text, mode='template'):
    """Create config files for all NEW_MEMBERS."""
    created = []
    for forcing_key, (model_name, institute, ecs) in NEW_MEMBERS.items():
        out_name = f"config_paradise_{forcing_key}.yaml"
        out_path = CONFIGS_DIR / out_name

        if out_path.exists():
            print(f"  SKIP (exists): {out_name}")
            continue

        if mode == 'template':
            cfg = template_text
            cfg = cfg.replace(
                f'domain_paradise_snotel_wa_{TEMPLATE_FORCING_KEY}',
                f'domain_paradise_snotel_wa_{forcing_key}')
            cfg = cfg.replace(
                f'forcing_ensemble_{TEMPLATE_FORCING_KEY}',
                f'forcing_ensemble_{forcing_key}')
            cfg = cfg.replace(TEMPLATE_FORCING_KEY, forcing_key)
            cfg = cfg.replace(TEMPLATE_MODEL_NAME, model_name)
        else:
            cfg = STANDALONE_TEMPLATE.format(
                model_name=model_name,
                forcing_key=forcing_key,
                ecs=ecs,
            )

        out_path.write_text(cfg)
        print(f"  CREATED: {out_name}  ({model_name}, ECS={ecs} C, {institute})")
        created.append(out_name)

    if created:
        print(f"\nCreated {len(created)} new GDDP configs.")
    else:
        print("\nNo new configs created (all already exist).")

    print("\nFull GDDP ensemble:")
    for f in sorted(CONFIGS_DIR.glob("config_paradise_gddp_*.yaml")):
        print(f"  {f.name}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Generate GDDP-CMIP6 ensemble member configurations')
    parser.add_argument('--standalone', action='store_true',
                        help='Generate from built-in template instead of '
                             'reading an existing config file')
    args = parser.parse_args()

    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    if args.standalone:
        generate_standalone()
    else:
        # Try template first, fall back to standalone
        if not generate_from_template():
            print("Falling back to standalone mode...\n")
            generate_standalone()


if __name__ == "__main__":
    main()
