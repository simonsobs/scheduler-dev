from typing import Tuple
from chex import dataclass
from functools import partial

from . import core
from .source import SourceBlock, source_block_trim_by_alt_range


@dataclass(frozen=True)
class RestrictAltRange(core.Rule):
    alt_range: Tuple[float, float]
    def apply(self, blocks:core.Blocks) -> core.Blocks:
        filt_alt = partial(source_block_trim_by_alt_range, alt_range=self.alt_range)
        return core.seq_map_when(core.block_isa(SourceBlock), filt_alt, blocks)
