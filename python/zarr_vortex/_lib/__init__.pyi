from typing import Awaitable, Callable
import numpy as np
from zarr.core.common import BytesLike

class PyVortexCodec:
    def decode(self, data: bytes) -> np.ndarray:
        ...
    async def decode_partial(self, size: Callable[[], Awaitable[int]], read: Callable[[int, int], Awaitable[np.ndarray | None]], flat_indices: list[int]) -> np.ndarray:
        ...
    def encode(self, dtype: int, view: np.ndarray) -> bytes:
        ...
