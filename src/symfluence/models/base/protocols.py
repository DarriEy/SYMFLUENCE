"""Structural Protocols defining the contract for SYMFLUENCE model components.

These are *structural* (duck-typing) Protocols — they document the interface
that model runners, preprocessors, and postprocessors must satisfy without
requiring inheritance from ``BaseModelRunner`` et al.

``@runtime_checkable`` enables ``isinstance()`` checks in registry code if
desired, but the primary purpose is static type-checking and documentation.
"""

from typing import Optional, Protocol, runtime_checkable
from pathlib import Path


@runtime_checkable
class ModelRunner(Protocol):
    """Protocol defining the contract for all SYMFLUENCE model runners.

    Every model runner must expose a ``MODEL_NAME`` class attribute and a
    ``run()`` method that executes the model and returns the path to the
    primary output file (or ``None`` on failure).
    """

    MODEL_NAME: str

    def run(self, **kwargs) -> Optional[Path]: ...


@runtime_checkable
class ModelPreProcessor(Protocol):
    """Protocol defining the contract for all model preprocessors.

    Every preprocessor must expose a ``MODEL_NAME`` class attribute and a
    ``run_preprocessing()`` method that prepares model inputs, returning
    ``True`` on success.
    """

    MODEL_NAME: str

    def run_preprocessing(self) -> bool: ...


@runtime_checkable
class ModelPostProcessor(Protocol):
    """Protocol defining the contract for all model postprocessors.

    Every postprocessor must expose a ``MODEL_NAME`` class attribute and an
    ``extract_streamflow()`` method that produces processed output, returning
    the path to the result file (or ``None`` on failure).
    """

    MODEL_NAME: str

    def extract_streamflow(self) -> Optional[Path]: ...
