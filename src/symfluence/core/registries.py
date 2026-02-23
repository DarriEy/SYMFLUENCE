"""Unified Registries facade for SYMFLUENCE.

All domain registries live as class-level :class:`Registry` instances on
the :class:`Registries` class.  Import the convenience alias ``R`` for
terse usage::

    from symfluence.core.registries import R

    runner_cls = R.runners["SUMMA"]
    R.for_model("SUMMA")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from symfluence.core.registry import Registry


class Registries:
    """Single entry-point for every SYMFLUENCE component registry.

    Each attribute is a :class:`Registry` instance.  Class methods provide
    cross-cutting queries (e.g. *for_model*, *registered_models*).
    """

    # ==================================================================
    # Model execution
    # ==================================================================
    preprocessors: Registry = Registry("preprocessors", doc="Model preprocessor classes")
    runners: Registry = Registry("runners", doc="Model runner classes")
    postprocessors: Registry = Registry("postprocessors", doc="Model postprocessor classes")
    visualizers: Registry = Registry("visualizers", doc="Model visualizer callables")

    # ==================================================================
    # Model configuration
    # ==================================================================
    config_adapters: Registry = Registry("config_adapters", doc="Config adapter classes")
    config_schemas: Registry = Registry("config_schemas", doc="Pydantic config schema classes")
    config_defaults: Registry = Registry("config_defaults", doc="Default config dicts")
    config_transformers: Registry = Registry("config_transformers", doc="Flat-to-nested field mappings")
    config_validators: Registry = Registry("config_validators", doc="Custom validation callables")

    # ==================================================================
    # Result extraction
    # ==================================================================
    result_extractors: Registry = Registry("result_extractors", doc="Result extractor classes")

    # ==================================================================
    # Optimization
    # ==================================================================
    optimizers: Registry = Registry("optimizers", doc="Model-specific optimizer classes")
    workers: Registry = Registry("workers", doc="Model-specific worker classes")
    parameter_managers: Registry = Registry("parameter_managers", doc="Parameter manager classes")
    calibration_targets: Registry = Registry(
        "calibration_targets", doc="Calibration target classes (composite keys: MODEL_TARGET)"
    )
    objectives: Registry = Registry("objectives", doc="Objective function classes")

    # ==================================================================
    # Data
    # ==================================================================
    acquisition_handlers: Registry = Registry(
        "acquisition_handlers", normalize=str.lower, doc="Data acquisition handler classes"
    )
    dataset_handlers: Registry = Registry(
        "dataset_handlers", normalize=str.lower, doc="Dataset preprocessing handler classes"
    )
    observation_handlers: Registry = Registry(
        "observation_handlers", normalize=str.lower, doc="Observation data handler classes"
    )

    # ==================================================================
    # Evaluation & Analysis
    # ==================================================================
    evaluators: Registry = Registry("evaluators", doc="Performance evaluator classes")
    sensitivity_analyzers: Registry = Registry("sensitivity_analyzers", doc="Sensitivity analyzer classes")
    decision_analyzers: Registry = Registry("decision_analyzers", doc="Decision/structure analyzer classes")
    koopman_analyzers: Registry = Registry("koopman_analyzers", doc="Koopman operator analyzer classes")
    metrics: Registry = Registry("metrics", doc="Metric functions and metadata")

    # ==================================================================
    # Visualization
    # ==================================================================
    plotters: Registry = Registry("plotters", doc="Model-specific plotter classes")
    visualization_funcs: Registry = Registry("visualization_funcs", doc="Visualization function callables")

    # ==================================================================
    # Geospatial
    # ==================================================================
    delineation_strategies: Registry = Registry(
        "delineation_strategies", normalize=str.lower, doc="Delineation strategy classes"
    )

    # ==================================================================
    # Coupling
    # ==================================================================
    bmi_adapters: Registry = Registry("bmi_adapters", doc="BMI/dCoupler component adapter classes")

    # ==================================================================
    # Infrastructure
    # ==================================================================
    forcing_adapters: Registry = Registry("forcing_adapters", doc="Forcing adapter classes")
    build_instructions: Registry = Registry(
        "build_instructions", normalize=str.lower, doc="Model build instruction providers"
    )
    presets: Registry = Registry("presets", doc="Initialization preset dicts/loaders")

    # ==================================================================
    # Cross-cutting class methods
    # ==================================================================

    # Registries used by ``for_model`` / ``registered_models`` to determine
    # what constitutes a "model" in the execution sense.
    _MODEL_REGISTRIES: tuple[str, ...] = (
        "preprocessors",
        "runners",
        "postprocessors",
        "config_adapters",
        "result_extractors",
    )

    @classmethod
    def all_registries(cls) -> Dict[str, Registry]:
        """Return ``{name: Registry}`` for every registry on this class."""
        return {
            attr: getattr(cls, attr)
            for attr in sorted(dir(cls))
            if isinstance(getattr(cls, attr, None), Registry)
        }

    @classmethod
    def registered_models(cls) -> List[str]:
        """Return the union of model names across the execution-layer registries."""
        names: set[str] = set()
        for attr_name in cls._MODEL_REGISTRIES:
            reg: Registry = getattr(cls, attr_name)
            names.update(reg.keys())
        return sorted(names)

    @classmethod
    def for_model(cls, name: str) -> Dict[str, Any]:
        """Return everything registered for *name* across all registries.

        Returns a dict mapping ``registry_name -> value`` for every
        registry where *name* has an entry (skips registries where it is
        absent).
        """
        result: Dict[str, Any] = {}
        for reg_name, reg in cls.all_registries().items():
            val = reg.get(name)
            if val is not None:
                result[reg_name] = val
        return result

    @classmethod
    def summary(cls) -> Dict[str, Dict[str, Any]]:
        """Return ``{registry_name: registry.summary()}`` for every registry."""
        return {name: reg.summary() for name, reg in cls.all_registries().items()}

    @classmethod
    def validate_model(cls, name: str) -> Dict[str, Any]:
        """Check registration completeness for *name*.

        Returns a dict with ``valid``, ``registered``, ``missing``, and
        ``optional_missing`` keys.
        """
        required = ("preprocessors", "runners", "postprocessors")
        optional = (
            "visualizers",
            "config_adapters",
            "result_extractors",
            "plotters",
        )

        registered = {}
        missing: list[str] = []
        optional_missing: list[str] = []

        for attr in required:
            reg: Registry = getattr(cls, attr)
            val = reg.get(name)
            if val is not None:
                registered[attr] = val
            else:
                missing.append(attr)

        for attr in optional:
            reg = getattr(cls, attr)
            val = reg.get(name)
            if val is not None:
                registered[attr] = val
            else:
                optional_missing.append(attr)

        return {
            "model_name": name.upper(),
            "valid": len(missing) == 0,
            "registered": registered,
            "missing": missing,
            "optional_missing": optional_missing,
        }

    @classmethod
    def get_calibration_target(
        cls, model: str, target_type: str = "streamflow"
    ) -> Optional[Type]:
        """Convenience helper for composite-key calibration targets."""
        key = f"{model.upper()}_{target_type.upper()}"
        return cls.calibration_targets.get(key)


# Convenience alias
R = Registries
