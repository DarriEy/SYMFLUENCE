"""
FEWS run_info.xml parser.

Parses the run_info.xml file that FEWS writes into the module working directory
before invoking the General Adapter. Extracts simulation time window, paths,
and properties needed to drive a SYMFLUENCE model run.
"""

import xml.etree.ElementTree as ET  # nosec B405 - parsing trusted FEWS run_info.xml
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .exceptions import RunInfoParseError

# FEWS datetime format: "2023-01-15T00:00:00Z" or "2023-01-15 00:00:00"
_FEWS_DT_FORMATS = [
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
]


def _parse_fews_datetime(text: str) -> datetime:
    """Parse a FEWS datetime string into a timezone-aware UTC datetime."""
    text = text.strip()
    for fmt in _FEWS_DT_FORMATS:
        try:
            dt = datetime.strptime(text, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise RunInfoParseError(f"Cannot parse FEWS datetime: '{text}'")


@dataclass(frozen=True)
class RunInfo:
    """Parsed contents of a FEWS run_info.xml file."""

    work_dir: Path
    input_dir: Path
    output_dir: Path
    state_input_dir: Optional[Path]
    state_output_dir: Optional[Path]
    start_time: datetime
    end_time: datetime
    time_zero: Optional[datetime]
    time_step_seconds: Optional[int]
    properties: Dict[str, str] = field(default_factory=dict)

    def to_config_overrides(self) -> Dict[str, Any]:
        """Convert run info into config overrides for SymfluenceConfig.

        Returns flat-key overrides that can be passed to SymfluenceConfig.from_file().
        """
        overrides: Dict[str, Any] = {
            "EXPERIMENT_TIME_START": self.start_time.strftime("%Y-%m-%d %H:%M"),
            "EXPERIMENT_TIME_END": self.end_time.strftime("%Y-%m-%d %H:%M"),
        }
        if "DOMAIN_NAME" in self.properties:
            overrides["DOMAIN_NAME"] = self.properties["DOMAIN_NAME"]
        if "EXPERIMENT_ID" in self.properties:
            overrides["EXPERIMENT_ID"] = self.properties["EXPERIMENT_ID"]
        if "HYDROLOGICAL_MODEL" in self.properties:
            overrides["HYDROLOGICAL_MODEL"] = self.properties["HYDROLOGICAL_MODEL"]
        return overrides


def parse_run_info(path: Path) -> RunInfo:
    """Parse a FEWS run_info.xml file.

    Args:
        path: Path to run_info.xml

    Returns:
        RunInfo dataclass with all extracted fields

    Raises:
        RunInfoParseError: If the file cannot be read or is malformed
    """
    path = Path(path)
    if not path.is_file():
        raise RunInfoParseError(f"run_info.xml not found: {path}")

    try:
        tree = ET.parse(str(path))  # nosec B314
    except ET.ParseError as exc:
        raise RunInfoParseError(f"Malformed XML in {path}: {exc}") from exc

    root = tree.getroot()

    def _text(tag: str, required: bool = True) -> Optional[str]:
        elem = root.find(tag)
        if elem is None or elem.text is None:
            if required:
                raise RunInfoParseError(f"Missing required element <{tag}> in {path}")
            return None
        return elem.text.strip()

    def _path_elem(tag: str, required: bool = True) -> Optional[Path]:
        text = _text(tag, required=required)
        if text is None:
            return None
        return Path(text)

    work_dir = _path_elem("workDir") or path.parent
    input_dir = _path_elem("inputDir", required=False)
    if input_dir is None:
        input_dir = work_dir / "toModel"
    output_dir = _path_elem("outputDir", required=False)
    if output_dir is None:
        output_dir = work_dir / "toFews"

    start_text = _text("startDateTime", required=True)
    end_text = _text("endDateTime", required=True)

    start_time = _parse_fews_datetime(start_text)  # type: ignore[arg-type]
    end_time = _parse_fews_datetime(end_text)  # type: ignore[arg-type]

    time_zero_text = _text("time0", required=False)
    time_zero = _parse_fews_datetime(time_zero_text) if time_zero_text else None

    ts_text = _text("timeStep", required=False)
    time_step_seconds = int(ts_text) if ts_text else None

    state_input_dir = _path_elem("stateInputDir", required=False)
    state_output_dir = _path_elem("stateOutputDir", required=False)

    # Parse <properties> block
    properties: Dict[str, str] = {}
    props_elem = root.find("properties")
    if props_elem is not None:
        for prop in props_elem.findall("*"):
            tag = prop.tag
            # Handle both <key>value</key> and <property key="..." value="..."/> patterns
            if "key" in prop.attrib and "value" in prop.attrib:
                properties[prop.attrib["key"]] = prop.attrib["value"]
            elif prop.text is not None:
                properties[tag] = prop.text.strip()

    return RunInfo(
        work_dir=work_dir,
        input_dir=input_dir,
        output_dir=output_dir,
        state_input_dir=state_input_dir,
        state_output_dir=state_output_dir,
        start_time=start_time,
        end_time=end_time,
        time_zero=time_zero,
        time_step_seconds=time_step_seconds,
        properties=properties,
    )
