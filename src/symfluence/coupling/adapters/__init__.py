"""SYMFLUENCE model adapters for dCoupler components."""

__all__: list = []

try:
    from .process_adapters import (
        SUMMAProcessComponent,
        MizuRouteProcessComponent,
        TRouteProcessComponent,
        ParFlowProcessComponent,
        MODFLOWProcessComponent,
        MESHProcessComponent,
        CLMProcessComponent,
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
        Snow17JAXComponent,
        XAJJAXComponent,
        SacSmaJAXComponent,
    )
    __all__.extend([
        "Snow17JAXComponent",
        "XAJJAXComponent",
        "SacSmaJAXComponent",
    ])
except ImportError:
    pass
