"""Tests for FEWS PI Diagnostics collector."""

import xml.etree.ElementTree as ET

import pytest

from symfluence.fews.pi_diagnostics import DiagnosticsCollector


class TestDiagnosticsCollector:
    def test_all_levels(self, tmp_path):
        diag = DiagnosticsCollector(tmp_path / "diag.xml")
        diag.debug("debug msg")
        diag.info("info msg")
        diag.warning("warn msg")
        diag.error("error msg")
        diag.fatal("fatal msg")
        assert len(diag.messages) == 5
        assert diag.messages[0].level == "0"
        assert diag.messages[4].level == "4"

    def test_has_fatal(self, tmp_path):
        diag = DiagnosticsCollector(tmp_path / "diag.xml")
        assert diag.has_fatal is False
        diag.info("ok")
        assert diag.has_fatal is False
        diag.fatal("boom")
        assert diag.has_fatal is True

    def test_has_errors(self, tmp_path):
        diag = DiagnosticsCollector(tmp_path / "diag.xml")
        assert diag.has_errors is False
        diag.error("err")
        assert diag.has_errors is True

    def test_write_creates_valid_xml(self, tmp_path):
        diag = DiagnosticsCollector(tmp_path / "diag.xml")
        diag.info("test message")
        diag.error("error message")
        diag.write()

        # Parse the output
        tree = ET.parse(str(tmp_path / "diag.xml"))
        root = tree.getroot()
        assert "Diag" in root.tag
        ns = {"pi": "http://www.wldelft.nl/fews/PI"}
        lines = root.findall("pi:line", ns)
        assert len(lines) == 2

    def test_write_never_raises(self, tmp_path):
        # Point to a non-writable path
        diag = DiagnosticsCollector(tmp_path / "nonexistent_dir" / "deep" / "nested" / "diag.xml")
        diag.info("test")
        # write() should not raise even if directory creation works
        diag.write()

    def test_write_creates_parent_dirs(self, tmp_path):
        diag = DiagnosticsCollector(tmp_path / "subdir" / "diag.xml")
        diag.info("test")
        diag.write()
        assert (tmp_path / "subdir" / "diag.xml").exists()

    def test_empty_diagnostics(self, tmp_path):
        diag = DiagnosticsCollector(tmp_path / "diag.xml")
        diag.write()
        assert (tmp_path / "diag.xml").exists()
        tree = ET.parse(str(tmp_path / "diag.xml"))
        root = tree.getroot()
        ns = {"pi": "http://www.wldelft.nl/fews/PI"}
        assert len(root.findall("pi:line", ns)) == 0

    def test_message_text_preserved(self, tmp_path):
        diag = DiagnosticsCollector(tmp_path / "diag.xml")
        diag.info("Special chars: <>&\"'")
        diag.write()
        tree = ET.parse(str(tmp_path / "diag.xml"))
        root = tree.getroot()
        ns = {"pi": "http://www.wldelft.nl/fews/PI"}
        desc = root.find("pi:line/pi:description", ns)
        assert desc is not None
        assert "<>&" in desc.text
