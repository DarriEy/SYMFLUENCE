# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Observation acceptance gate: provider-API parity tier vs dataset-artifact provenance tier (0.5.0)."""
from __future__ import annotations

import pytest

from symfluence.data.backends.contract import (
    ObservationCapability,
    Redistribution,
    SourceKind,
)
from symfluence.data.backends.selection import _observation_decline_reason

pytestmark = [pytest.mark.unit, pytest.mark.quick]


def _cap(**over):
    base = dict(
        provider_id="X",
        kinds=frozenset({"streamflow"}),
        station_id_scheme="scheme",
        temporal=None,
        auth=frozenset(),
        parity_grade=None,
    )
    base.update(over)
    return ObservationCapability(**base)


def _decline(cap, *, name="community", allow_ungated=False):
    return _observation_decline_reason(
        name, cap, kind="streamflow", window=None, allow_ungated=allow_ungated
    )


class TestProviderApiTier:
    def test_ungraded_provider_api_is_refused(self):
        assert _decline(_cap(parity_grade=None)) is not None

    def test_graded_provider_api_is_admitted(self):
        assert _decline(_cap(parity_grade="bit-identical", redistribution=Redistribution.OPEN)) is None

    def test_native_is_exempt_even_when_ungraded(self):
        assert _decline(_cap(parity_grade=None), name="native") is None


class TestDatasetArtifactTier:
    def _artifact(self, **over):
        base = dict(
            source_kind=SourceKind.DATASET_ARTIFACT,
            redistribution=Redistribution.ATTRIBUTION,
            dataset_doi="10.5281/zenodo.5153305",
            dataset_version="v1.6",
            dataset_checksum="sha256:abc123",
            parity_grade=None,  # no native API to be parity-equivalent to
        )
        base.update(over)
        return _cap(**base)

    def test_fully_provenanced_artifact_is_admitted_without_parity(self):
        assert _decline(self._artifact()) is None

    @pytest.mark.parametrize("field", ["dataset_doi", "dataset_version", "dataset_checksum"])
    def test_missing_any_provenance_field_is_refused(self, field):
        reason = _decline(self._artifact(**{field: ""}))
        assert reason is not None and "provenance" in reason

    def test_unknown_license_is_refused(self):
        reason = _decline(self._artifact(redistribution=Redistribution.UNKNOWN))
        assert reason is not None and "license" in reason

    def test_restricted_artifact_is_refused_as_legal_constraint(self):
        # Restricted is refused for every non-native source kind, before the
        # provenance path, and is not waivable by allow_ungated.
        assert _decline(self._artifact(redistribution=Redistribution.RESTRICTED)) is not None
        assert _decline(
            self._artifact(redistribution=Redistribution.RESTRICTED), allow_ungated=True
        ) is not None

    def test_allow_ungated_waives_missing_provenance(self):
        assert _decline(self._artifact(dataset_checksum=""), allow_ungated=True) is None

    def test_native_artifact_is_exempt(self):
        assert _decline(self._artifact(dataset_doi="", dataset_checksum=""), name="native") is None

    def test_open_license_also_accepted(self):
        assert _decline(self._artifact(redistribution=Redistribution.OPEN)) is None

    def test_noncommercial_artifact_is_still_admitted(self):
        # NonCommercial is a use-restriction surfaced to the user (logged on
        # selection), orthogonal to redistribution — it does not refuse here.
        cap = self._artifact(redistribution=Redistribution.ATTRIBUTION, noncommercial=True)
        assert cap.noncommercial is True
        assert _decline(cap) is None
