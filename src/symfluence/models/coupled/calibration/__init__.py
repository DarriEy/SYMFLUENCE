# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Back-compat shim package: the generic coupled calibration components moved to
:mod:`symfluence.core.calibration.coupled` (they are model-agnostic framework
machinery). The ``optimizer`` / ``parameter_manager`` / ``worker`` modules here
re-export the canonical implementations, so ``optimization._autodiscover`` keeps
firing their ``COUPLED`` registrations from this historical location."""
