"""
FEWS PI Diagnostics collector and writer.

Implements the Delft-FEWS diagnostics contract: the adapter MUST write a
``diag.xml`` file to the output directory after every run, even on catastrophic
failure. The ``write()`` method therefore catches all internal exceptions.
"""

import logging
import xml.etree.ElementTree as ET  # nosec B405 - writing trusted local diagnostics XML
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal

logger = logging.getLogger(__name__)

DiagLevel = Literal["0", "1", "2", "3", "4"]
# Levels: 0=Debug, 1=Info, 2=Warning, 3=Error, 4=Fatal


@dataclass
class DiagMessage:
    """Single diagnostic message."""
    level: DiagLevel
    text: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class DiagnosticsCollector:
    """Buffers diagnostic messages and writes PI ``<Diag>`` XML.

    Usage::

        diag = DiagnosticsCollector(output_dir / "diag.xml")
        diag.info("Pre-adapter started")
        try:
            ...  # adapter logic
        except Exception as exc:
            diag.fatal(str(exc))
        finally:
            diag.write()  # ALWAYS writes, never raises
    """

    def __init__(self, output_path: Path) -> None:
        self._output_path = Path(output_path)
        self._messages: List[DiagMessage] = []

    # ---- convenience methods ----

    def debug(self, text: str) -> None:
        self._messages.append(DiagMessage(level="0", text=text))

    def info(self, text: str) -> None:
        self._messages.append(DiagMessage(level="1", text=text))

    def warning(self, text: str) -> None:
        self._messages.append(DiagMessage(level="2", text=text))

    def error(self, text: str) -> None:
        self._messages.append(DiagMessage(level="3", text=text))

    def fatal(self, text: str) -> None:
        self._messages.append(DiagMessage(level="4", text=text))

    # ---- query ----

    @property
    def has_fatal(self) -> bool:
        return any(m.level == "4" for m in self._messages)

    @property
    def has_errors(self) -> bool:
        return any(m.level in ("3", "4") for m in self._messages)

    @property
    def messages(self) -> List[DiagMessage]:
        return list(self._messages)

    # ---- writer ----

    def write(self) -> None:
        """Write ``diag.xml``.  **Never raises** — logs errors internally."""
        try:
            root = ET.Element("Diag")
            root.set("xmlns", "http://www.wldelft.nl/fews/PI")
            root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")

            for msg in self._messages:
                line = ET.SubElement(root, "line")
                level_elem = ET.SubElement(line, "level")
                level_elem.text = msg.level
                desc = ET.SubElement(line, "description")
                desc.text = msg.text

            self._output_path.parent.mkdir(parents=True, exist_ok=True)

            tree = ET.ElementTree(root)
            ET.indent(tree, space="  ")
            tree.write(
                str(self._output_path),
                xml_declaration=True,
                encoding="UTF-8",
            )
        except Exception:
            # Contract: write() MUST NOT raise.
            logger.exception("Failed to write FEWS diagnostics to %s", self._output_path)
