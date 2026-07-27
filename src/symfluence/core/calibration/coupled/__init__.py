# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Generic multi-model coupled calibration via dCoupler: compose the standalone parameter
managers/workers of whatever models the coupling graph wires (land, snow, groundwater, routing),
delegate parameter subsets to each, run the dCoupler graph, and score one shared objective.

Model-agnostic framework machinery -- no model package is imported here; the participating
models are resolved through the ``R`` registries. Promoted from
``symfluence.models.coupled.calibration``, which remains as a back-compat shim (and is still
the path ``optimization._autodiscover`` imports to fire the ``COUPLED`` registrations).
"""
