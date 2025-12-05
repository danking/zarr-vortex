import sys
import numpy as np
import vortex as vx

from typing import Literal, final, override
from zarr.abc.codec import ArrayBytesCodec, ArrayBytesCodecPartialDecodeMixin
from zarr.abc.store import ByteGetter, ByteRequest, RangeByteRequest
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import Buffer, NDBuffer
from zarr.core.buffer.core import BufferPrototype
from zarr.core.buffer.cpu import Buffer as CpuBuffer, NDBuffer as CpuNDBuffer
from zarr.core.chunk_grids import RegularChunkGrid
from zarr.core.common import JSON, BytesLike, ShapeLike
from zarr.core.indexing import SelectorTuple, get_indexer
from zarr.storage import StorePath

from ._lib import PyVortexCodec

def to_vortex_dtype(dtype: np.dtype) -> vx.DType:
    if dtype == np.bool():
        return vx.bool_(nullable=True)
    if dtype == np.int8():
        return vx.int_(8, nullable=True)
    if dtype == np.int16():
        return vx.int_(16, nullable=True)
    if dtype == np.int32():
        return vx.int_(32, nullable=True)
    if dtype == np.int64():
        return vx.int_(64, nullable=True)
    if dtype == np.uint8():
        return vx.uint(8, nullable=True)
    if dtype == np.uint16():
        return vx.uint(16, nullable=True)
    if dtype == np.uint32():
        return vx.uint(32, nullable=True)
    if dtype == np.uint64():
        return vx.uint(64, nullable=True)
    if dtype == np.uint8():
        return vx.uint(8, nullable=True)
    if dtype == np.uint16():
        return vx.uint(16, nullable=True)
    if dtype == np.uint32():
        return vx.uint(32, nullable=True)
    if dtype == np.uint64():
        return vx.uint(64, nullable=True)
    if dtype == np.float16():
        return vx.float_(16, nullable=True)
    if dtype == np.float32():
        return vx.float_(32, nullable=True)
    if dtype == np.float64():
        return vx.float_(64, nullable=True)
    raise ValueError(f"unsupported dtype: {dtype}")


def jank_it(dtype: vx.DType) -> int:
    if isinstance(dtype, vx.BoolDType):
        return 0
    if isinstance(dtype, vx.PrimitiveDType):
        ptype: vx.PrimitiveDType = dtype
        if ptype.ptype == vx.PType.U8:
            return 1
        if ptype.ptype == vx.PType.U16:
            return 2
        if ptype.ptype == vx.PType.U32:
            return 3
        if ptype.ptype == vx.PType.U64:
            return 4
        if ptype.ptype == vx.PType.I8:
            return 5
        if ptype.ptype == vx.PType.I16:
            return 6
        if ptype.ptype == vx.PType.I32:
            return 7
        if ptype.ptype == vx.PType.I64:
            return 8
        if ptype.ptype == vx.PType.F16:
            return 9
        if ptype.ptype == vx.PType.F32:
            return 10
        if ptype.ptype == vx.PType.F64:
            return 11
    raise ValueError(f"fuck you {dtype}")

def unjank_it(dtype: int) -> vx.DType:
    if dtype == 0:
        return vx.bool_(nullable=True)
    if dtype == 1:
        return vx.uint(8, nullable=True)
    if dtype == 2:
        return vx.uint(16, nullable=True)
    if dtype == 3:
        return vx.uint(32, nullable=True)
    if dtype == 4:
        return vx.uint(64, nullable=True)
    if dtype == 5:
        return vx.int_(8, nullable=True)
    if dtype == 6:
        return vx.int_(16, nullable=True)
    if dtype == 7:
        return vx.int_(32, nullable=True)
    if dtype == 8:
        return vx.int_(64, nullable=True)
    if dtype == 9:
        return vx.float_(16, nullable=True)
    if dtype == 10:
        return vx.float_(32, nullable=True)
    if dtype == 11:
        return vx.float_(64, nullable=True)
    raise ValueError(f"fuck you {dtype}")


@final
class Codec(ArrayBytesCodec, ArrayBytesCodecPartialDecodeMixin):
    def __init__(self, *, name: str, array_shape: ShapeLike | None = None):
        super().__init__()
        assert name == 'zarr_vortex.codec'

        self.array_shape = array_shape
        self._internal: PyVortexCodec = PyVortexCodec()

    @override
    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Codec:
        return Codec(name='zarr_vortex.codec', array_shape=array_spec.shape)

    @override
    def to_dict(self) -> dict[str, JSON]:
        if isinstance(self.array_shape, int | None):
            array_shape = self.array_shape
        else:
            array_shape = list(self.array_shape)
        return {'name': 'zarr_vortex.codec', 'array_shape': array_shape}

    @override
    async def _decode_single(self, chunk_data: Buffer, chunk_spec: ArraySpec) -> NDBuffer:
        return CpuNDBuffer.from_numpy_array(self._internal.decode(chunk_data.to_bytes()).reshape(chunk_spec.shape))

    @override
    async def _decode_partial_single(
        self, byte_getter: ByteGetter, selection: SelectorTuple, chunk_spec: ArraySpec
    ) -> NDBuffer | None:
        assert isinstance(byte_getter, StorePath)
        assert isinstance(selection, tuple)

        slices: list[tuple[int, int]] = []
        for dim in selection:
            assert isinstance(dim, slice)
            assert isinstance(dim.start, int)  # pyright: ignore[reportAny]
            assert isinstance(dim.stop, int)  # pyright: ignore[reportAny]
            slices.append((dim.start, dim.stop))

        async def read(start: int, end: int) -> np.ndarray | None:
            byte_request = RangeByteRequest(start=start, end=end)
            zarr_buffer = await byte_getter.get(BufferPrototype(CpuBuffer, CpuNDBuffer), byte_request)
            if zarr_buffer:
                return zarr_buffer.as_numpy_array()
            return None

        async def getsize() -> int:
            return await byte_getter.store.getsize(byte_getter.path)

        assert self.array_shape is not None

        # FIXME(DK): obviously need the more complicated general thing here
        assert isinstance(selection, tuple)
        a, b = selection
        assert isinstance(a, slice)
        assert a.step == 1
        assert isinstance(b, slice)
        assert b.step == 1

        shape: tuple[int, ...] = (a.stop - a.start, b.start - b.stop)

        indices: list[int] = []
        if chunk_spec.config.order == 'C':
            for r in range(a.start, a.stop):
                for c in range(b.start, b.stop):
                    indices.append(r * chunk_spec.shape[0] + c)
        else:
            assert chunk_spec.config.order == 'F'
            for c in range(b.start, b.stop):
                for r in range(a.start, a.stop):
                    indices.append(c * chunk_spec.shape[1] + r)

        ndarray = await self._internal.decode_partial(getsize, read, indices)

        ndarray = ndarray.reshape(shape)
        return CpuNDBuffer.from_numpy_array(ndarray)

    @override
    async def _encode_single(
        self, chunk_data: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        dtype = jank_it(to_vortex_dtype(chunk_data.dtype))
        viewed = ensure_native_endian(chunk_data.as_numpy_array())
        return CpuBuffer.from_bytes(self._internal.encode(dtype, viewed))


def ensure_native_endian(arr: np.ndarray) -> np.ndarray:
    """
    Verify array endianness matches system native, and convert if needed.
    Returns array with native byte order.
    """
    native_order = sys.byteorder  # 'little' or 'big'

    # '<' = little-endian, '>' = big-endian, '=' = native, '|' = not applicable
    arr_order = arr.dtype.byteorder

    if arr_order in ('=', '|'):
        # already native or irrelevant (u8)
        return arr

    effectively_native = (
        (arr_order == '<' and native_order == 'little') or
        (arr_order == '>' and native_order == 'big')
    )

    if not effectively_native:
        raise ValueError('fucking fuck')


    native_dtype = arr.dtype.newbyteorder('=')
    return np.ndarray(arr.shape, dtype=native_dtype, buffer=arr.data, strides=arr.strides)
