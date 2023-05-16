from . import core
from chex import dataclass

@dataclass(frozen=True)
class ScanBlock(core.NamedBlock):
    az: float
    alt: float
    throw: float
    patch: str

@dataclass(frozen=True)
class IVBlock(core.NamedBlock): pass