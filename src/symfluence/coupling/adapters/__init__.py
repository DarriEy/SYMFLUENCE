"""SYMFLUENCE model adapters for dCoupler components."""

__all__: list = []

try:
    from .process_adapters import (
        CLMProcessComponent,
        MESHProcessComponent,
        MizuRouteProcessComponent,
        MODFLOWProcessComponent,
        ParFlowProcessComponent,
        SUMMAProcessComponent,
        TRouteProcessComponent,
    )
    __all__.extend([
        "SUMMAProcessComponent",
        "MizuRouteProcessComponent",
        "TRouteProcessComponent",
        "ParFlowProcessComponent",
        "MODFLOWProcessComponent",
        "MESHProcessComponent",
        "CLMProcessComponent",
    ])
except ImportError:
    pass

try:
    from .jax_adapters import (
        SacSmaJAXComponent,
        Snow17JAXComponent,
        XAJJAXComponent,
    )
    __all__.extend([
        "Snow17JAXComponent",
        "XAJJAXComponent",
        "SacSmaJAXComponent",
    ])
except ImportError:
    pass
