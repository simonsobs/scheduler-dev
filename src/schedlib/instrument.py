from . import core
from chex import dataclass

@dataclass(frozen=True)
class ScanBlock(core.NamedBlock):
    az: float     # deg
    alt: float    # deg
    throw: float  # deg
    patch: str

@dataclass(frozen=True)
class IVBlock(core.NamedBlock): pass