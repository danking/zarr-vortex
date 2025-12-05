import numpy as np
import zarr
import tempfile
import os

def test_zarr_round_trip_float32():
    rng = np.random.default_rng(seed=42)
    data = rng.random((2048, 2048), dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = os.path.join(tmpdir, "test.zarr")

        z = zarr.open_array(  # pyright: ignore[reportUnknownMemberType]
            store_path,
            mode="w",
            shape=data.shape,
            dtype=data.dtype,
            chunks=(512, 512),
            codecs=[{"name": "zarr_vortex.codec"}]
        )
        z[:] = data

        z2 = zarr.open_array(store_path, mode="r")  # pyright: ignore[reportUnknownMemberType]
        loaded = np.array(z2[:], copy=True)

    assert np.array_equal(data, loaded)


def example():
    import numpy as np
    import zarr

    rng = np.random.default_rng(seed=42)
    data = rng.random((2048, 2048), dtype=np.float32)

    z = zarr.open_array(  # pyright: ignore[reportUnknownMemberType]
        '/tmp/foo.zarr',
        mode="w",
        shape=data.shape,
        dtype=data.dtype,
        chunks=(512, 512),
        codecs=[{"name": "zarr_vortex.codec"}]
    )
    z[:] = data
