use std::sync::Arc;

use bytes::Bytes;
use futures::{TryFutureExt, future::BoxFuture};
use numpy::PyArray;
use pyo3::{
    buffer::{Element, PyBuffer},
    exceptions::PyValueError,
    prelude::*,
};
use pyo3_async_runtimes::tokio::into_future;
use tokio::runtime::Builder;
use tracing::info;
use vortex::{
    VortexSessionDefault as _,
    arrays::PrimitiveArray,
    buffer::{Buffer, ByteBuffer},
    compute::min_max,
    dtype::{DType, NativePType, Nullability, PType, half},
    error::{VortexError, VortexResult},
    file::{OpenOptionsSessionExt as _, WriteOptionsSessionExt as _},
    io::{VortexReadAt, session::RuntimeSessionExt as _},
    session::VortexSession,
    stream::ArrayStreamExt as _,
    validity::Validity,
};

#[pyclass]
pub struct PyVortexCodec {
    session: VortexSession,
}

#[pymethods]
impl PyVortexCodec {
    #[new]
    fn new() -> Self {
        Self {
            session: VortexSession::default().with_tokio(),
        }
    }

    fn decode(&self, py: Python, bytes: Vec<u8>) -> PyResult<Py<PyAny>> {
        let opener = self.session.open_options();
        let bytes = ByteBuffer::from(bytes);
        let fut = opener
            .open_buffer(bytes)
            .map_err(|err| PyValueError::new_err(format!("vortex error: {err}")))?
            .scan()
            .map_err(|err| PyValueError::new_err(format!("vortex error: {err}")))?
            .into_array_stream()
            .map_err(|err| PyValueError::new_err(format!("vortex error: {err}")))?
            .read_all();
        let runtime = Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|err| PyValueError::new_err(format!("tokio error: {err}")))?;
        let array = runtime
            .block_on(fut)
            .map_err(|err| PyValueError::new_err(format!("vortex error: {err}")))?;
        let array = array.to_canonical().into_primitive();

        match array.ptype() {
            PType::U8 => {
                let vec = array.buffer::<u8>().to_vec();
                Ok(PyArray::from_vec(py, vec).into())
            }
            PType::U16 => {
                let vec = array.buffer::<u16>().to_vec();
                Ok(PyArray::from_vec(py, vec).into())
            }
            PType::U32 => {
                let vec = array.buffer::<u32>().to_vec();
                Ok(PyArray::from_vec(py, vec).into())
            }
            PType::U64 => {
                let vec = array.buffer::<u64>().to_vec();
                Ok(PyArray::from_vec(py, vec).into())
            }
            PType::I8 => {
                let vec = array.buffer::<i8>().to_vec();
                Ok(PyArray::from_vec(py, vec).into())
            }
            PType::I16 => {
                let vec = array.buffer::<i16>().to_vec();
                Ok(PyArray::from_vec(py, vec).into())
            }
            PType::I32 => {
                let vec = array.buffer::<i32>().to_vec();
                Ok(PyArray::from_vec(py, vec).into())
            }
            PType::I64 => {
                let vec = array.buffer::<i64>().to_vec();
                Ok(PyArray::from_vec(py, vec).into())
            }
            PType::F16 => {
                let vec = array.buffer::<half::f16>().to_vec();
                let vec = unsafe { std::mem::transmute::<Vec<half::f16>, Vec<u16>>(vec) };
                Ok(PyArray::from_vec(py, vec).into())
            }
            PType::F32 => {
                let vec = array.buffer::<f32>().to_vec();
                Ok(PyArray::from_vec(py, vec).into())
            }
            PType::F64 => {
                let vec = array.buffer::<f64>().to_vec();
                Ok(PyArray::from_vec(py, vec).into())
            }
        }
    }

    async fn decode_partial(
        &self,
        size: Py<PyAny>,
        read: Py<PyAny>,
        indices: Vec<u64>,
    ) -> PyResult<Py<PyAny>> {
        let opener = self.session.open_options();
        let read_at = JankVortexReadAt {
            size: Arc::new(size),
            reader: Arc::new(Python::attach(|py| read.clone_ref(py))),
        };

        let indices = Buffer::from(indices);
        let fut = opener
            .open_read_at(read_at)
            .await
            .map_err(|err| PyValueError::new_err(format!("vortex error: {err}")))?
            .scan()
            .map_err(|err| PyValueError::new_err(format!("vortex error: {err}")))?
            .with_row_indices(indices)
            .into_array_stream()
            .map_err(|err| PyValueError::new_err(format!("vortex error: {err}")))?
            .read_all();
        let runtime = Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|err| PyValueError::new_err(format!("tokio error: {err}")))?;
        let array = runtime
            .block_on(fut)
            .map_err(|err| PyValueError::new_err(format!("vortex error: {err}")))?;
        let array = array.to_canonical().into_primitive();

        match array.ptype() {
            PType::U8 => {
                let vec = array.buffer::<u8>().to_vec();
                Python::attach(|py| Ok(PyArray::from_vec(py, vec).into()))
            }
            PType::U16 => {
                let vec = array.buffer::<u16>().to_vec();
                Python::attach(|py| Ok(PyArray::from_vec(py, vec).into()))
            }
            PType::U32 => {
                let vec = array.buffer::<u32>().to_vec();
                Python::attach(|py| Ok(PyArray::from_vec(py, vec).into()))
            }
            PType::U64 => {
                let vec = array.buffer::<u64>().to_vec();
                Python::attach(|py| Ok(PyArray::from_vec(py, vec).into()))
            }
            PType::I8 => {
                let vec = array.buffer::<i8>().to_vec();
                Python::attach(|py| Ok(PyArray::from_vec(py, vec).into()))
            }
            PType::I16 => {
                let vec = array.buffer::<i16>().to_vec();
                Python::attach(|py| Ok(PyArray::from_vec(py, vec).into()))
            }
            PType::I32 => {
                let vec = array.buffer::<i32>().to_vec();
                Python::attach(|py| Ok(PyArray::from_vec(py, vec).into()))
            }
            PType::I64 => {
                let vec = array.buffer::<i64>().to_vec();
                Python::attach(|py| Ok(PyArray::from_vec(py, vec).into()))
            }
            PType::F16 => {
                let vec = array.buffer::<half::f16>().to_vec();
                let vec = unsafe { std::mem::transmute::<Vec<half::f16>, Vec<u16>>(vec) };
                Python::attach(|py| Ok(PyArray::from_vec(py, vec).into()))
            }
            PType::F32 => {
                let vec = array.buffer::<f32>().to_vec();
                Python::attach(|py| Ok(PyArray::from_vec(py, vec).into()))
            }
            PType::F64 => {
                let vec = array.buffer::<f64>().to_vec();
                Python::attach(|py| Ok(PyArray::from_vec(py, vec).into()))
            }
        }
    }

    fn encode(&self, py: Python, dtype: u64, view: Bound<PyAny>) -> PyResult<Vec<u8>> {
        let dtype = decode_vortex_dtype(dtype);

        if let DType::Bool(..) = dtype {
            todo!()
        }

        let DType::Primitive(ptype, ..) = dtype else {
            todo!()
        };

        let runtime = Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|err| PyValueError::new_err(format!("tokio error: {err}")))?;

        match ptype {
            PType::U8 => {
                runtime.block_on(self.encode_typed(py, PyBuffer::<u8>::extract_bound(&view)?))
            }
            PType::U16 => {
                runtime.block_on(self.encode_typed(py, PyBuffer::<u16>::extract_bound(&view)?))
            }
            PType::U32 => {
                runtime.block_on(self.encode_typed(py, PyBuffer::<u32>::extract_bound(&view)?))
            }
            PType::U64 => {
                runtime.block_on(self.encode_typed(py, PyBuffer::<u64>::extract_bound(&view)?))
            }
            PType::I8 => {
                runtime.block_on(self.encode_typed(py, PyBuffer::<i8>::extract_bound(&view)?))
            }
            PType::I16 => {
                runtime.block_on(self.encode_typed(py, PyBuffer::<i16>::extract_bound(&view)?))
            }
            PType::I32 => {
                runtime.block_on(self.encode_typed(py, PyBuffer::<i32>::extract_bound(&view)?))
            }
            PType::I64 => {
                runtime.block_on(self.encode_typed(py, PyBuffer::<i64>::extract_bound(&view)?))
            }
            PType::F16 => {
                todo!()
            }
            PType::F32 => {
                runtime.block_on(self.encode_typed(py, PyBuffer::<f32>::extract_bound(&view)?))
            }
            PType::F64 => {
                runtime.block_on(self.encode_typed(py, PyBuffer::<f64>::extract_bound(&view)?))
            }
        }
    }
}

impl PyVortexCodec {
    async fn encode_typed<T: NativePType + Element>(
        &self,
        py: Python<'_>,
        bytes: PyBuffer<T>,
    ) -> PyResult<Vec<u8>> {
        let slice = bytes.as_slice(py).ok_or_else(|| {
            PyValueError::new_err(format!(
                "Buffer must be c-contiguous and contain values of type {}.",
                T::PTYPE
            ))
        })?;
        // SAFETY: ReadOnlyCell is transparent, UnsafeCell has the same memory
        // layout. If Python mutates this concurrently, we're dead.
        let slice: &[u8] = unsafe { std::mem::transmute(slice) };
        let vortex_bytes = Buffer::<T>::from_byte_buffer(Buffer::from(Bytes::from_owner(slice)));
        // TODO(DK): Can we get invalidity information from Zarr?
        let vortex_array = PrimitiveArray::new(vortex_bytes, Validity::AllValid);

        let x = min_max(&vortex_array.to_array()).unwrap().unwrap();
        info!(min = %x.min, max = %x.max);

        let mut out = Vec::<u8>::new();
        let summary = self
            .session
            .write_options()
            .write(&mut out, vortex_array.to_array_stream())
            .await
            .map_err(|err| PyValueError::new_err(format!("vortex error: {err}")))?;
        info!(layout = %summary.footer().layout().display_tree_verbose(true));
        Ok(out)
    }
}

#[allow(unused)]
fn encode_vortex_dtype(dtype: DType) -> u64 {
    match dtype {
        DType::Bool(..) => 0,
        DType::Primitive(PType::U8, ..) => 1,
        DType::Primitive(PType::U16, ..) => 2,
        DType::Primitive(PType::U32, ..) => 3,
        DType::Primitive(PType::U64, ..) => 4,
        DType::Primitive(PType::I8, ..) => 5,
        DType::Primitive(PType::I16, ..) => 6,
        DType::Primitive(PType::I32, ..) => 7,
        DType::Primitive(PType::I64, ..) => 8,
        DType::Primitive(PType::F16, ..) => 9,
        DType::Primitive(PType::F32, ..) => 10,
        DType::Primitive(PType::F64, ..) => 11,
        _ => todo!(),
    }
}

fn decode_vortex_dtype(dtype: u64) -> DType {
    match dtype {
        0 => DType::Bool(Nullability::Nullable),
        1 => DType::Primitive(PType::U8, Nullability::Nullable),
        2 => DType::Primitive(PType::U16, Nullability::Nullable),
        3 => DType::Primitive(PType::U32, Nullability::Nullable),
        4 => DType::Primitive(PType::U64, Nullability::Nullable),
        5 => DType::Primitive(PType::I8, Nullability::Nullable),
        6 => DType::Primitive(PType::I16, Nullability::Nullable),
        7 => DType::Primitive(PType::I32, Nullability::Nullable),
        8 => DType::Primitive(PType::I64, Nullability::Nullable),
        9 => DType::Primitive(PType::F16, Nullability::Nullable),
        10 => DType::Primitive(PType::F32, Nullability::Nullable),
        11 => DType::Primitive(PType::F64, Nullability::Nullable),
        _ => todo!(),
    }
}

struct JankVortexReadAt {
    size: Arc<Py<PyAny>>,
    reader: Arc<Py<PyAny>>,
}

impl VortexReadAt for JankVortexReadAt {
    fn read_at(
        &self,
        offset: u64,
        length: usize,
        _alignment: vortex::buffer::Alignment,
    ) -> BoxFuture<'static, VortexResult<ByteBuffer>> {
        let reader = self.reader.clone();
        // FIXME(DK): alignment obviously
        Box::pin(
            async move {
                let buf = Python::attach(|py| {
                    let awaitable = reader.call1(py, (offset, offset + length as u64))?;
                    into_future(awaitable.into_bound(py))
                })?
                .await?;

                let vortex_bytes = Python::attach(|py| {
                    let buf = PyBuffer::<u8>::get(buf.bind(py))?;
                    let slice = buf.as_slice(py).ok_or_else(|| {
                        PyValueError::new_err("ByteGetter must return a c-contiguous byte buffer")
                    })?;

                    // SAFETY: ReadOnlyCell is transparent, UnsafeCell has the same memory
                    // layout. If Python mutates this concurrently, we're dead.
                    let slice: &[u8] = unsafe { std::mem::transmute(slice) };

                    Ok::<_, PyErr>(Buffer::from(Bytes::from_owner(slice)))
                })?;

                Ok(vortex_bytes)
            }
            .map_err(|err: PyErr| VortexError::generic(Box::new(err))),
        )
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let size = self.size.clone();
        Box::pin(
            async move {
                let size = Python::attach(|py| {
                    let awaitable = size.call0(py)?;
                    into_future(awaitable.into_bound(py))
                })?
                .await?;
                let size = Python::attach(|py| size.extract::<u64>(py))?;
                Ok::<_, PyErr>(size)
            }
            .map_err(|err: PyErr| VortexError::generic(Box::new(err))),
        )
    }
}
