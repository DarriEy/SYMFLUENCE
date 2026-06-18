# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Attribute acceptance gate: parity tier + license-posture-only tier (0.4.0/0.5.0).

The attribute-flavored sibling of ``test_observation_dataset_artifact_gate``.
There is no native attribute backend (the in-tree processors are the default
path), so every backend reaching the gate is a mirroring/community backend; the
restricted-redistribution refusal and the posture-only admission therefore apply
to all of them. Attribute outputs are *computed* zonal statistics, not published
artifacts, so there is no dataset-artifact provenance tier (unlike observations).
"""
from __future__ import annotations

import pytest

from symfluence.data.backends.contract import (
    AttributeCapability,
    Redistribution,
    SchemaId,
)
from symfluence.data.backends.selection import _attribute_decline_reason

pytestmark = [pytest.mark.unit, pytest.mark.quick]


def _cap(**over):
    base = dict(
        provider_id="CAS",
        attribute_ids=frozenset({"copernicus_dem:elevation"}),
        output_kind="per_hru_stats",
        schema=SchemaId.HRU_STATS_V1,
        auth=frozenset(),
        parity_grade=None,
    )
    base.update(over)
    return AttributeCapability(**base)


def _decline(cap, *, attribute_ids=None, allow_ungated=False):
    return _attribute_decline_reason(
        cap, attribute_ids=attribute_ids, allow_ungated=allow_ungated
    )


class TestClaimAndIds:
    def test_unclaimed_provider_declines(self):
        assert _decline(None) == "does not claim the provider"

    def test_unservable_attribute_ids_decline(self):
        reason = _decline(
            _cap(parity_grade="value-within:1%"),
            attribute_ids=frozenset({"isric_soilgrids:clay_0-5cm"}),
        )
        assert reason is not None and "attribute ids" in reason


class TestParityTier:
    def test_graded_provider_is_admitted(self):
        # Tolerance parity grade admits regardless of posture detail.
        assert _decline(
            _cap(parity_grade="value-within:1%", redistribution=Redistribution.UNKNOWN)
        ) is None


class TestPostureOnlyTier:
    def test_ungraded_unknown_posture_is_refused(self):
        # No parity grade AND posture undeclared -> needs the opt-in.
        assert _decline(
            _cap(parity_grade=None, redistribution=Redistribution.UNKNOWN)
        ) is not None

    @pytest.mark.parametrize("posture", [Redistribution.OPEN, Redistribution.ATTRIBUTION])
    def test_ungraded_open_posture_is_admitted(self, posture):
        # Posture-only gate: zonal stats with no native reference may still serve
        # when the source license is open/attribution.
        assert _decline(_cap(parity_grade=None, redistribution=posture)) is None

    def test_ungraded_unknown_posture_waived_by_allow_ungated(self):
        assert _decline(
            _cap(parity_grade=None, redistribution=Redistribution.UNKNOWN),
            allow_ungated=True,
        ) is None


class TestRestrictedIsNonWaivable:
    def test_restricted_is_refused_even_with_posture_gate(self):
        assert _decline(
            _cap(parity_grade=None, redistribution=Redistribution.RESTRICTED)
        ) is not None

    def test_restricted_is_not_waivable_by_allow_ungated(self):
        # ALLOW_UNGATED_BACKENDS waives the parity grade, never the licence.
        assert _decline(
            _cap(parity_grade=None, redistribution=Redistribution.RESTRICTED),
            allow_ungated=True,
        ) is not None


class TestNonCommercial:
    def test_noncommercial_is_surfaced_not_refused(self):
        # NonCommercial is a use-restriction (logged on selection), orthogonal to
        # redistribution — an attribution+NC provider still serves.
        cap = _cap(
            parity_grade=None,
            redistribution=Redistribution.ATTRIBUTION,
            noncommercial=True,
            data_license="CC-BY-NC-SA-4.0",
        )
        assert cap.noncommercial is True
        assert _decline(cap) is None

    def test_default_capability_is_not_noncommercial(self):
        assert _cap().noncommercial is False
