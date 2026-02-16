"""Tests for FEWS run_info.xml parser."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from symfluence.fews.exceptions import RunInfoParseError
from symfluence.fews.run_info import RunInfo, parse_run_info


class TestParseRunInfo:
    def test_basic_parse(self, sample_run_info_xml, tmp_work_dir):
        ri = parse_run_info(sample_run_info_xml)
        assert isinstance(ri, RunInfo)
        assert ri.work_dir == tmp_work_dir
        assert ri.input_dir == tmp_work_dir / "toModel"
        assert ri.output_dir == tmp_work_dir / "toFews"

    def test_datetime_parsing(self, sample_run_info_xml):
        ri = parse_run_info(sample_run_info_xml)
        assert ri.start_time == datetime(2023, 1, 1, tzinfo=timezone.utc)
        assert ri.end_time == datetime(2023, 1, 10, tzinfo=timezone.utc)

    def test_time_zero(self, sample_run_info_xml):
        ri = parse_run_info(sample_run_info_xml)
        assert ri.time_zero == datetime(2023, 1, 5, tzinfo=timezone.utc)

    def test_time_step(self, sample_run_info_xml):
        ri = parse_run_info(sample_run_info_xml)
        assert ri.time_step_seconds == 3600

    def test_properties(self, sample_run_info_xml):
        ri = parse_run_info(sample_run_info_xml)
        assert ri.properties["DOMAIN_NAME"] == "test_basin"
        assert ri.properties["EXPERIMENT_ID"] == "fews_run_1"
        assert ri.properties["HYDROLOGICAL_MODEL"] == "GR"

    def test_config_overrides(self, sample_run_info_xml):
        ri = parse_run_info(sample_run_info_xml)
        overrides = ri.to_config_overrides()
        assert overrides["EXPERIMENT_TIME_START"] == "2023-01-01 00:00"
        assert overrides["EXPERIMENT_TIME_END"] == "2023-01-10 00:00"
        assert overrides["DOMAIN_NAME"] == "test_basin"

    def test_file_not_found(self, tmp_path):
        with pytest.raises(RunInfoParseError, match="not found"):
            parse_run_info(tmp_path / "nonexistent.xml")

    def test_malformed_xml(self, tmp_path):
        bad_file = tmp_path / "bad.xml"
        bad_file.write_text("<broken><xml")
        with pytest.raises(RunInfoParseError, match="Malformed XML"):
            parse_run_info(bad_file)

    def test_missing_start_datetime(self, tmp_path):
        xml = f"<Run><workDir>{tmp_path}</workDir><endDateTime>2023-01-10T00:00:00Z</endDateTime></Run>"
        path = tmp_path / "run_info.xml"
        path.write_text(xml)
        with pytest.raises(RunInfoParseError, match="startDateTime"):
            parse_run_info(path)

    def test_state_dirs(self, sample_run_info_xml, tmp_work_dir):
        ri = parse_run_info(sample_run_info_xml)
        assert ri.state_input_dir == tmp_work_dir / "states_in"
        assert ri.state_output_dir == tmp_work_dir / "states_out"

    def test_frozen_dataclass(self, sample_run_info_xml):
        ri = parse_run_info(sample_run_info_xml)
        with pytest.raises(AttributeError):
            ri.start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
