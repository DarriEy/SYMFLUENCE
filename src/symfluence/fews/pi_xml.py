"""
PI-XML timeseries reader and writer.

Implements read/write for the Delft-FEWS PI-XML timeseries format using only
``xml.etree.ElementTree`` (stdlib) and ``xarray``.

PI-XML namespace: ``http://www.wldelft.nl/fews/PI``
"""

import xml.etree.ElementTree as ET  # nosec B405 - parsing trusted FEWS PI-XML files
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import xarray as xr

from .exceptions import PIXMLError

# FEWS PI namespace
_NS = "http://www.wldelft.nl/fews/PI"
_NS_MAP = {"pi": _NS}


def _tag(name: str) -> str:
    """Prepend the PI namespace to a tag name."""
    return f"{{{_NS}}}{name}"


def _strip_ns(tag: str) -> str:
    """Remove namespace prefix from tag."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

def read_pi_xml_timeseries(path: Path, missing_value: float = -999.0) -> xr.Dataset:
    """Read a PI-XML timeseries file into an xarray Dataset.

    Each ``<series>`` becomes a data variable keyed by its parameterId.
    The time axis is the union of all event timestamps.

    Args:
        path: Path to PI-XML file
        missing_value: Value to interpret as NaN

    Returns:
        xr.Dataset with time dimension and one variable per series

    Raises:
        PIXMLError: If the file cannot be parsed
    """
    path = Path(path)
    if not path.is_file():
        raise PIXMLError(f"PI-XML file not found: {path}")

    try:
        tree = ET.parse(str(path))  # nosec B314
    except ET.ParseError as exc:
        raise PIXMLError(f"Malformed PI-XML in {path}: {exc}") from exc

    root = tree.getroot()

    # Detect namespace
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    series_elements = root.findall(f"{ns}series")
    if not series_elements:
        # Fallback: try without namespace
        series_elements = root.findall("series")

    if not series_elements:
        raise PIXMLError(f"No <series> elements found in {path}")

    all_times: List[datetime] = []
    series_data: List[Dict] = []

    for series_elem in series_elements:
        header = series_elem.find(f"{ns}header")
        if header is None:
            header = series_elem.find("header")
        if header is None:
            raise PIXMLError("Missing <header> in <series>")

        param_elem = header.find(f"{ns}parameterId")
        if param_elem is None:
            param_elem = header.find("parameterId")
        param_id = param_elem.text.strip() if param_elem is not None and param_elem.text else "unknown"

        loc_elem = header.find(f"{ns}locationId")
        if loc_elem is None:
            loc_elem = header.find("locationId")
        location_id = loc_elem.text.strip() if loc_elem is not None and loc_elem.text else "default"

        # Read events
        events: Dict[datetime, float] = {}
        for event in series_elem.findall(f"{ns}event"):
            dt_str = event.get("date", "") + "T" + event.get("time", "00:00:00")
            try:
                dt = datetime.fromisoformat(dt_str).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            val_str = event.get("value", "")
            try:
                val = float(val_str)
            except ValueError:
                val = np.nan
            events[dt] = val

        if not events:
            # Try without namespace
            for event in series_elem.findall("event"):
                dt_str = event.get("date", "") + "T" + event.get("time", "00:00:00")
                try:
                    dt = datetime.fromisoformat(dt_str).replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
                val_str = event.get("value", "")
                try:
                    val = float(val_str)
                except ValueError:
                    val = np.nan
                events[dt] = val

        all_times.extend(events.keys())
        series_data.append({
            "param_id": param_id,
            "location_id": location_id,
            "events": events,
        })

    if not all_times:
        raise PIXMLError(f"No timesteps found in {path}")

    # Build unified time axis
    unique_times = sorted(set(all_times))
    time_index = np.array([dt.replace(tzinfo=None) for dt in unique_times], dtype="datetime64[ns]")

    data_vars = {}
    for sd in series_data:
        values = np.full(len(unique_times), np.nan, dtype=np.float64)
        for i, t in enumerate(unique_times):
            if t in sd["events"]:
                val = sd["events"][t]
                if np.isclose(val, missing_value):
                    values[i] = np.nan
                else:
                    values[i] = val
        var_name = sd["param_id"]
        data_vars[var_name] = xr.DataArray(
            values,
            dims=["time"],
            attrs={"location_id": sd["location_id"]},
        )

    ds = xr.Dataset(data_vars, coords={"time": time_index})
    return ds


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def write_pi_xml_timeseries(
    dataset: xr.Dataset,
    path: Path,
    *,
    missing_value: float = -999.0,
    location_id: str = "default",
    time_zone: float = 0.0,
) -> None:
    """Write an xarray Dataset as PI-XML timeseries.

    Each data variable in the dataset becomes a ``<series>``.

    Args:
        dataset: Dataset with a ``time`` coordinate
        path: Output file path
        missing_value: Value to use for NaN in PI-XML
        location_id: Default location ID (overridden by variable attr if present)
        time_zone: Timezone offset in hours (0.0 = UTC)

    Raises:
        PIXMLError: If writing fails
    """
    path = Path(path)

    try:
        root = ET.Element("TimeSeries")
        root.set("xmlns", _NS)
        root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")

        tz_elem = ET.SubElement(root, "timeZone")
        tz_elem.text = str(time_zone)

        times = dataset["time"].values

        for var_name in dataset.data_vars:
            da = dataset[var_name]
            series = ET.SubElement(root, "series")
            header = ET.SubElement(series, "header")

            type_elem = ET.SubElement(header, "type")
            type_elem.text = "instantaneous"

            loc_elem = ET.SubElement(header, "locationId")
            loc_elem.text = da.attrs.get("location_id", location_id)

            param_elem = ET.SubElement(header, "parameterId")
            param_elem.text = str(var_name)

            miss_elem = ET.SubElement(header, "missVal")
            miss_elem.text = str(missing_value)

            unit_elem = ET.SubElement(header, "units")
            unit_elem.text = da.attrs.get("units", "-")

            values = da.values
            for i, t in enumerate(times):
                val = float(values[i])
                ts = np.datetime64(t, "ns")
                dt = ts.astype("datetime64[s]").astype(datetime)

                event = ET.SubElement(series, "event")
                event.set("date", dt.strftime("%Y-%m-%d"))
                event.set("time", dt.strftime("%H:%M:%S"))
                if np.isnan(val):
                    event.set("value", str(missing_value))
                else:
                    event.set("value", f"{val:.6g}")

        path.parent.mkdir(parents=True, exist_ok=True)
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(str(path), xml_declaration=True, encoding="UTF-8")

    except Exception as exc:  # noqa: BLE001 — wrap-and-raise to domain error
        raise PIXMLError(f"Failed to write PI-XML to {path}: {exc}") from exc
