"""JAX-based adapters wrapping SYMFLUENCE's differentiable models.

Each adapter wraps a SYMFLUENCE JAX model's step() function as a dCoupler
JAXComponent, enabling gradient flow through the CouplingGraph.
"""

from __future__ import annotations

import logging
from typing import Optional

from dcoupler.core.component import (
    FluxDirection,
    FluxSpec,
    ParameterSpec,
)
from dcoupler.wrappers.jax import JAXComponent

logger = logging.getLogger(__name__)


def _get_snow17_step():
    """Import Snow-17 step function."""
    from symfluence.models.snow17.model import snow17_step
    return snow17_step


def _get_snow17_params():
    """Import Snow-17 parameter bounds."""
    from symfluence.models.snow17.parameters import SNOW17_PARAM_BOUNDS
    return SNOW17_PARAM_BOUNDS


def _get_xaj_step():
    """Import XAJ step function."""
    from symfluence.models.xinanjiang.model import step_jax
    return step_jax


def _get_sacsma_step():
    """Import SAC-SMA step function."""
    from symfluence.models.sacsma.sacsma import sacsma_step
    return sacsma_step


class Snow17JAXComponent(JAXComponent):
    """Wraps Snow-17 as JAXComponent with BMI lifecycle.

    Reuses: symfluence.models.snow17.model.snow17_step (the JAX kernel)
    Reuses: symfluence.models.snow17.bmi.Snow17BMI (BMI pattern)

    10 parameters (all linear transform):
        SCF, PXTEMP, MFMAX, MFMIN, NMF, MBASE, TIPM, UADJ, PLWHC, DAYGM
    """

    SNOW17_PARAMS = [
        ("SCF", 0.7, 1.4),
        ("PXTEMP", -2.0, 2.0),
        ("MFMAX", 0.5, 4.0),
        ("MFMIN", 0.05, 2.0),
        ("NMF", 0.001, 0.5),
        ("MBASE", 0.0, 1.0),
        ("TIPM", 0.01, 1.0),
        ("UADJ", 0.01, 0.4),
        ("PLWHC", 0.01, 0.3),
        ("DAYGM", 0.0, 0.3),
    ]

    def __init__(self, name: str = "snow17", config: Optional[dict] = None):
        try:
            jax_step = _get_snow17_step()
        except ImportError:
            raise ImportError("Snow-17 model not available in SYMFLUENCE") from None

        param_specs = [
            ParameterSpec(pname, lo, hi)
            for pname, lo, hi in self.SNOW17_PARAMS
        ]

        input_flux_specs = [
            FluxSpec("precip", "mm/dt", FluxDirection.INPUT, "hru", 86400, ("time",)),
            FluxSpec("temp", "C", FluxDirection.INPUT, "hru", 86400, ("time",)),
            FluxSpec("doy", "day", FluxDirection.INPUT, "hru", 86400, ("time",),
                     optional=True),
        ]
        output_flux_specs = [
            FluxSpec("rain_plus_melt", "mm/dt", FluxDirection.OUTPUT, "hru", 86400,
                     ("time",), conserved_quantity="water_mass"),
        ]

        def step_wrapper(inputs, state, params, dt):
            import jax.numpy as jnp

            from symfluence.models.snow17.parameters import (
                DEFAULT_ADC,
                Snow17State,
                params_dict_to_namedtuple,
            )
            snow17_params = params_dict_to_namedtuple(params, use_jax=True)
            adc = jnp.array(DEFAULT_ADC)
            # Unpack flat array → Snow17State namedtuple
            snow_state = Snow17State(
                w_i=state[0], w_q=state[1], w_qx=state[2],
                deficit=state[3], ati=state[4], swe=state[5],
            )
            # Get day-of-year from input (controls seasonal melt factor)
            doy = inputs.get("doy", jnp.float32(1))
            # snow17_step returns (new_state: Snow17State, rain_plus_melt)
            new_snow_state, rain_plus_melt = jax_step(
                precip=inputs["precip"],
                temp=inputs["temp"],
                dt=dt,
                state=snow_state,
                params=snow17_params,
                doy=doy,
                adc=adc,
                xp=jnp,
            )
            # Pack Snow17State back to flat array
            new_state = jnp.stack([
                new_snow_state.w_i, new_snow_state.w_q, new_snow_state.w_qx,
                new_snow_state.deficit, new_snow_state.ati, new_snow_state.swe,
            ])
            return rain_plus_melt, new_state

        # Snow-17 state: (w_i, w_q, w_qx, deficit, ati, swe) = 6 vars
        super().__init__(
            name=name,
            jax_step_fn=step_wrapper,
            param_specs=param_specs,
            state_size=6,
            input_flux_specs=input_flux_specs,
            output_flux_specs=output_flux_specs,
        )


class XAJJAXComponent(JAXComponent):
    """Wraps Xinanjiang (XAJ) model as JAXComponent.

    Reuses: symfluence.models.xinanjiang.model.step_jax (per-timestep kernel)

    15 parameters from XAJ parameter registry.
    """

    # Param names match XinanjiangParams namedtuple fields exactly
    XAJ_PARAMS = [
        ("K", 0.0, 1.0),       # PET correction factor
        ("B", 0.1, 0.6),       # Tension water capacity curve exponent
        ("IM", 0.0, 0.1),      # Impervious area fraction
        ("UM", 5.0, 50.0),     # Upper layer tension water capacity (mm)
        ("LM", 10.0, 100.0),   # Lower layer tension water capacity (mm)
        ("DM", 10.0, 100.0),   # Deep layer tension water capacity (mm)
        ("C", 0.05, 0.2),      # Deep layer ET coefficient
        ("SM", 1.0, 100.0),    # Free water capacity (mm)
        ("EX", 0.5, 2.0),      # Free water capacity curve exponent
        ("KI", 0.0, 0.7),      # Interflow outflow coefficient
        ("KG", 0.0, 0.7),      # Groundwater outflow coefficient
        ("CS", 0.0, 1.0),      # Channel recession constant
        ("L", 0.0, 5.0),       # Lag time (timesteps)
        ("CI", 0.0, 1.0),      # Interflow recession constant
        ("CG", 0.0, 1.0),      # Groundwater recession constant
    ]

    def __init__(self, name: str = "xaj", config: Optional[dict] = None):
        try:
            jax_step = _get_xaj_step()
        except ImportError:
            raise ImportError("XAJ model not available in SYMFLUENCE") from None

        param_specs = [
            ParameterSpec(pname, lo, hi)
            for pname, lo, hi in self.XAJ_PARAMS
        ]

        input_flux_specs = [
            FluxSpec("precip", "mm/dt", FluxDirection.INPUT, "hru", 86400, ("time",)),
            FluxSpec("pet", "mm/dt", FluxDirection.INPUT, "hru", 86400, ("time",)),
        ]
        output_flux_specs = [
            FluxSpec("runoff", "mm/dt", FluxDirection.OUTPUT, "hru", 86400,
                     ("time",), conserved_quantity="water_mass"),
        ]

        def step_wrapper(inputs, state, params, dt):
            import jax.numpy as jnp

            from symfluence.models.xinanjiang.parameters import (
                XinanjiangParams,
                XinanjiangState,
            )
            # Unpack flat array → XinanjiangState namedtuple
            xaj_state = XinanjiangState(
                wu=state[0], wl=state[1], wd=state[2],
                s=state[3], fr=state[4], qi=state[5], qg=state[6],
            )
            # Build XinanjiangParams from dict (names match fields)
            xaj_params = XinanjiangParams(**params)
            # step_jax signature: (precip, pet, state, params) -> (new_state, outflow)
            new_xaj_state, outflow = jax_step(
                precip=inputs["precip"],
                pet=inputs["pet"],
                state=xaj_state,
                params=xaj_params,
            )
            # Pack XinanjiangState back to flat array
            new_state = jnp.stack([
                new_xaj_state.wu, new_xaj_state.wl, new_xaj_state.wd,
                new_xaj_state.s, new_xaj_state.fr,
                new_xaj_state.qi, new_xaj_state.qg,
            ])
            return outflow, new_state

        # XAJ state size: WU, WL, WD, S, FR, QI, QG (7 vars)
        super().__init__(
            name=name,
            jax_step_fn=step_wrapper,
            param_specs=param_specs,
            state_size=7,
            input_flux_specs=input_flux_specs,
            output_flux_specs=output_flux_specs,
        )


class SacSmaJAXComponent(JAXComponent):
    """Wraps SAC-SMA as JAXComponent.

    Reuses: symfluence.models.sacsma.sacsma.sacsma_step (per-timestep kernel)

    16 parameters from SAC-SMA parameter registry.
    """

    SACSMA_PARAMS = [
        ("UZTWM", 1.0, 150.0),
        ("UZFWM", 1.0, 150.0),
        ("UZK", 0.1, 0.5),
        ("PCTIM", 0.0, 0.1),
        ("ADIMP", 0.0, 0.4),
        ("RIVA", 0.0, 0.2),
        ("ZPERC", 1.0, 250.0),
        ("REXP", 1.0, 5.0),
        ("LZTWM", 1.0, 500.0),
        ("LZFSM", 1.0, 1000.0),
        ("LZFPM", 1.0, 1000.0),
        ("LZSK", 0.01, 0.25),
        ("LZPK", 0.001, 0.025),
        ("PFREE", 0.0, 0.6),
        ("SIDE", 0.0, 0.5),
        ("RSERV", 0.0, 0.4),
    ]

    def __init__(self, name: str = "sacsma", config: Optional[dict] = None):
        try:
            jax_step = _get_sacsma_step()
        except ImportError:
            raise ImportError("SAC-SMA model not available in SYMFLUENCE") from None

        param_specs = [
            ParameterSpec(pname, lo, hi)
            for pname, lo, hi in self.SACSMA_PARAMS
        ]

        input_flux_specs = [
            FluxSpec("precip", "mm/dt", FluxDirection.INPUT, "hru", 86400, ("time",)),
            FluxSpec("pet", "mm/dt", FluxDirection.INPUT, "hru", 86400, ("time",)),
        ]
        output_flux_specs = [
            FluxSpec("runoff", "mm/dt", FluxDirection.OUTPUT, "hru", 86400,
                     ("time",), conserved_quantity="water_mass"),
        ]

        def step_wrapper(inputs, state, params, dt):
            import jax.numpy as jnp

            from symfluence.models.sacsma.parameters import SacSmaParameters
            from symfluence.models.sacsma.sacsma import SacSmaState
            # Unpack flat array → SacSmaState namedtuple
            sac_state = SacSmaState(
                uztwc=state[0], uzfwc=state[1], lztwc=state[2],
                lzfpc=state[3], lzfsc=state[4], adimc=state[5],
            )
            sac_params = SacSmaParameters(**params)
            # sacsma_step returns (new_state, surface, interflow, baseflow, et)
            new_sac_state, surface, interflow, baseflow, et = jax_step(
                pxv=inputs["precip"],
                pet=inputs["pet"],
                dt=dt,
                state=sac_state,
                params=sac_params,
                xp=jnp,
            )
            runoff = surface + interflow + baseflow
            # Pack SacSmaState back to flat array
            new_state = jnp.stack([
                new_sac_state.uztwc, new_sac_state.uzfwc, new_sac_state.lztwc,
                new_sac_state.lzfpc, new_sac_state.lzfsc, new_sac_state.adimc,
            ])
            return runoff, new_state

        # SAC-SMA state: UZTWC, UZFWC, LZTWC, LZFSC, LZFPC, ADIMC (6 vars)
        super().__init__(
            name=name,
            jax_step_fn=step_wrapper,
            param_specs=param_specs,
            state_size=6,
            input_flux_specs=input_flux_specs,
            output_flux_specs=output_flux_specs,
        )
