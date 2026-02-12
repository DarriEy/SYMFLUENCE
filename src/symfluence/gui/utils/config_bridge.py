"""
Bridge between SymfluenceConfig (Pydantic) and BasicConfigParams (param).

Provides two-way conversion so the GUI widgets stay in sync with the
underlying typed config model.
"""


def config_to_params(config) -> dict:
    """
    Extract GUI-relevant fields from a SymfluenceConfig into a flat dict
    keyed by BasicConfigParams field names.

    Args:
        config: SymfluenceConfig instance

    Returns:
        dict suitable for setting BasicConfigParams attributes
    """
    d = config.domain
    f = config.forcing
    m = config.model
    o = config.optimization

    return {
        'domain_name': d.name or '',
        'experiment_id': d.experiment_id or '',
        'time_start': d.time_start or '',
        'time_end': d.time_end or '',
        'calibration_period': d.calibration_period or '',
        'evaluation_period': d.evaluation_period or '',
        'definition_method': d.definition_method or 'lumped',
        'discretization': d.discretization or 'lumped',
        'pour_point_coords': d.pour_point_coords or '',
        'bounding_box_coords': d.bounding_box_coords or '',
        'stream_threshold': d.delineation.stream_threshold if d.delineation else 5000.0,
        'forcing_dataset': f.dataset or 'ERA5',
        'hydrological_model': m.hydrological_model or 'SUMMA',
        'optimization_algorithm': o.algorithm or 'PSO',
        'optimization_metric': o.metric or 'KGE',
        'iterations': o.iterations if o.iterations else 1000,
        'population_size': o.population_size if o.population_size else 50,
    }


def params_to_config_overrides(params) -> dict:
    """
    Convert BasicConfigParams values into a flat-key override dict
    (uppercase keys) suitable for SymfluenceConfig(**flat_dict).

    Args:
        params: BasicConfigParams instance (or dict of its values)

    Returns:
        dict with SYMFLUENCE flat config keys
    """
    if not isinstance(params, dict):
        params = {name: getattr(params, name) for name in params.param if name != 'name'}

    overrides = {}

    _map = {
        'domain_name': 'DOMAIN_NAME',
        'experiment_id': 'EXPERIMENT_ID',
        'time_start': 'EXPERIMENT_TIME_START',
        'time_end': 'EXPERIMENT_TIME_END',
        'calibration_period': 'CALIBRATION_PERIOD',
        'evaluation_period': 'EVALUATION_PERIOD',
        'definition_method': 'DOMAIN_DEFINITION_METHOD',
        'discretization': 'SUB_GRID_DISCRETIZATION',
        'pour_point_coords': 'POUR_POINT_COORDS',
        'bounding_box_coords': 'BOUNDING_BOX_COORDS',
        'stream_threshold': 'STREAM_THRESHOLD',
        'forcing_dataset': 'FORCING_DATASET',
        'hydrological_model': 'HYDROLOGICAL_MODEL',
        'optimization_algorithm': 'ITERATIVE_OPTIMIZATION_ALGORITHM',
        'optimization_metric': 'OPTIMIZATION_METRIC',
        'iterations': 'NUMBER_OF_ITERATIONS',
        'population_size': 'POPULATION_SIZE',
    }

    for param_name, flat_key in _map.items():
        val = params.get(param_name)
        if val is not None and val != '':
            overrides[flat_key] = val

    return overrides
