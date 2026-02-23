"""Regression tests for the SYMFLUENCE exception hierarchy.

Validates structural correctness of the exception hierarchy to prevent
accidental re-parenting, orphaned exceptions, or duplicate definitions.
"""

import pytest

from symfluence.core.exceptions import (
    CodeAnalysisError,
    ConfigurationError,
    ConfigValidationError,
    DataAcquisitionError,
    DiscretizationError,
    EvaluationError,
    FileOperationError,
    GeospatialError,
    ModelExecutionError,
    OptimizationError,
    RasterProcessingError,
    ReportingError,
    RetryExhaustedError,
    ShapefileError,
    SYMFLUENCEError,
    ValidationError,
    WorkerExecutionError,
)


class TestHierarchyRoots:
    """All custom exceptions must be rooted under SYMFLUENCEError."""

    @pytest.mark.parametrize("exc_cls", [
        ConfigurationError,
        ConfigValidationError,
        ModelExecutionError,
        DataAcquisitionError,
        OptimizationError,
        WorkerExecutionError,
        RetryExhaustedError,
        GeospatialError,
        ValidationError,
        FileOperationError,
        DiscretizationError,
        ShapefileError,
        RasterProcessingError,
        CodeAnalysisError,
        EvaluationError,
        ReportingError,
    ])
    def test_all_exceptions_subclass_symfluence_error(self, exc_cls):
        assert issubclass(exc_cls, SYMFLUENCEError), (
            f"{exc_cls.__name__} is not a subclass of SYMFLUENCEError"
        )

    def test_symfluence_error_subclasses_exception(self):
        assert issubclass(SYMFLUENCEError, Exception)


class TestDomainParenting:
    """Validate specific parent-child relationships."""

    def test_config_validation_error_under_configuration(self):
        assert issubclass(ConfigValidationError, ConfigurationError)

    def test_worker_execution_error_under_optimization(self):
        assert issubclass(WorkerExecutionError, OptimizationError)

    def test_retry_exhausted_error_under_optimization(self):
        assert issubclass(RetryExhaustedError, OptimizationError)

    def test_discretization_error_under_geospatial(self):
        assert issubclass(DiscretizationError, GeospatialError)

    def test_shapefile_error_under_geospatial(self):
        assert issubclass(ShapefileError, GeospatialError)

    def test_raster_processing_error_under_geospatial(self):
        assert issubclass(RasterProcessingError, GeospatialError)

    def test_evaluation_error_direct_subclass(self):
        assert EvaluationError.__bases__ == (SYMFLUENCEError,)

    def test_reporting_error_direct_subclass(self):
        assert ReportingError.__bases__ == (SYMFLUENCEError,)


class TestObservationHierarchy:
    """ObservationError and subtypes must be under DataAcquisitionError."""

    def test_observation_error_under_data_acquisition(self):
        from symfluence.data.observation.base import ObservationError
        assert issubclass(ObservationError, DataAcquisitionError)

    def test_observation_acquisition_error_under_observation(self):
        from symfluence.data.observation.base import (
            ObservationAcquisitionError,
            ObservationError,
        )
        assert issubclass(ObservationAcquisitionError, ObservationError)
        assert issubclass(ObservationAcquisitionError, DataAcquisitionError)

    def test_observation_processing_error_under_observation(self):
        from symfluence.data.observation.base import (
            ObservationError,
            ObservationProcessingError,
        )
        assert issubclass(ObservationProcessingError, ObservationError)
        assert issubclass(ObservationProcessingError, DataAcquisitionError)

    def test_observation_validation_error_under_observation(self):
        from symfluence.data.observation.base import (
            ObservationError,
            ObservationValidationError,
        )
        assert issubclass(ObservationValidationError, ObservationError)
        assert issubclass(ObservationValidationError, DataAcquisitionError)


class TestHubEauHierarchy:
    """HubEauAPIError must be under DataAcquisitionError."""

    def test_hubeau_api_error_under_data_acquisition(self):
        from symfluence.data.observation.handlers.hubeau import HubEauAPIError
        assert issubclass(HubEauAPIError, DataAcquisitionError)


class TestDeduplication:
    """Verify there is a single identity for deduplicated classes."""

    def test_shapefile_error_single_identity(self):
        from symfluence.core.exceptions import ShapefileError as core_cls
        from symfluence.geospatial.exceptions import ShapefileError as geo_cls
        assert core_cls is geo_cls

    def test_config_validation_error_single_identity_via_base(self):
        from symfluence.core.exceptions import ConfigValidationError as core_cls
        from symfluence.models.base import ConfigValidationError as base_cls
        assert core_cls is base_cls

    def test_config_validation_error_single_identity_via_config(self):
        from symfluence.core.exceptions import ConfigValidationError as core_cls
        from symfluence.models.base.base_config import ConfigValidationError as config_cls
        assert core_cls is config_cls

    def test_config_validation_error_single_identity_via_extractor(self):
        from symfluence.core.exceptions import ConfigValidationError as core_cls
        from symfluence.models.base.base_extractor import ConfigValidationError as ext_cls
        assert core_cls is ext_cls


class TestCatchability:
    """Verify that typed catches work as expected across the hierarchy."""

    def test_observation_error_caught_by_data_acquisition(self):
        from symfluence.data.observation.base import ObservationError
        with pytest.raises(DataAcquisitionError):
            raise ObservationError("test")

    def test_observation_error_caught_by_symfluence_error(self):
        from symfluence.data.observation.base import ObservationError
        with pytest.raises(SYMFLUENCEError):
            raise ObservationError("test")

    def test_config_validation_caught_by_configuration_error(self):
        with pytest.raises(ConfigurationError):
            raise ConfigValidationError("test")

    def test_worker_execution_caught_by_optimization_error(self):
        with pytest.raises(OptimizationError):
            raise WorkerExecutionError("test")

    def test_shapefile_error_caught_by_geospatial_error(self):
        with pytest.raises(GeospatialError):
            raise ShapefileError("test")
