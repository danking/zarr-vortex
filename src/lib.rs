use std::sync::Arc;

use bytes::Bytes;
use tokio::runtime::Builder;
use tracing::info;
use vortex::{
    VortexSessionDefault as _,
    arrays::PrimitiveArray,
    buffer::{Alignment, Buffer, ByteBuffer},
    dtype::{DType, NativePType, Nullability, PType, match_each_native_ptype},
    file::{OpenOptionsSessionExt as _, WriteOptionsSessionExt},
    io::session::RuntimeSessionExt as _,
    session::VortexSession,
    stream::ArrayStreamExt,
    validity::Validity,
};
use zarrs::{
    array::{
        ArrayBytes, ArrayCodecTraits, BytesRepresentation, ChunkRepresentation, DataType, RawBytes,
        RecommendedConcurrency,
        codec::{
            ArrayToBytesCodecTraits, AsyncArrayPartialDecoderTraits,
            AsyncArrayPartialEncoderTraits, AsyncBytesPartialDecoderTraits,
            AsyncBytesPartialEncoderTraits, Codec, CodecError, CodecMetadataOptions, CodecOptions,
            CodecPlugin, CodecTraits, PartialDecoderCapability, PartialEncoderCapability,
        },
    },
    metadata::{Configuration, v3::MetadataV3},
    plugin::PluginCreateError,
};

mod async_partial_decoder;
mod async_partial_encoder;

pub const VORTEX_IDENTIFIER: &'static str = "vortex";

#[derive(Debug)]
struct VortexCodec {
    session: VortexSession,
}

inventory::submit! {
    CodecPlugin::new(VORTEX_IDENTIFIER, |x| x == VORTEX_IDENTIFIER, vortex_codec)
}

pub fn vortex_codec(_metadata: &MetadataV3) -> Result<Codec, PluginCreateError> {
    Ok(Codec::ArrayToBytes(Arc::from(VortexCodec {
        session: VortexSession::default().with_tokio(),
    })))
}

impl CodecTraits for VortexCodec {
    fn identifier(&self) -> &str {
        VORTEX_IDENTIFIER
    }

    fn configuration_opt(
        &self,
        _name: &str,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        Some(Configuration::default())
    }

    fn partial_decoder_capability(&self) -> PartialDecoderCapability {
        PartialDecoderCapability {
            // FIXME(DK): Implement partial reads.
            partial_read: false,
            partial_decode: true,
        }
    }

    fn partial_encoder_capability(&self) -> PartialEncoderCapability {
        PartialEncoderCapability {
            // FIXME(DK): In principle, some Vortex encodings should be writable in their compressed forms, right?
            partial_encode: false,
        }
    }
}

impl ArrayCodecTraits for VortexCodec {
    fn recommended_concurrency(
        &self,
        _decoded_representation: &ChunkRepresentation,
    ) -> Result<RecommendedConcurrency, CodecError> {
        // FIXME(DK): This is a bit of a weird fit. Vortex should own concurrency and file I/O.
        Ok(RecommendedConcurrency::new(1..1))
    }
}

impl ArrayToBytesCodecTraits for VortexCodec {
    fn into_dyn(self: std::sync::Arc<Self>) -> std::sync::Arc<dyn ArrayToBytesCodecTraits> {
        self
    }

    fn encoded_representation(
        &self,
        _decoded_representation: &ChunkRepresentation,
    ) -> Result<BytesRepresentation, CodecError> {
        // TODO(DK): We ought to be able to at least do BoundedSize but I'm not sure how to
        // calculate the amount of overhead in a Vortex layout.
        Ok(BytesRepresentation::UnboundedSize)
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        decoded_representation: &ChunkRepresentation,
        _options: &CodecOptions,
    ) -> Result<RawBytes<'a>, CodecError> {
        // TODO(DK): we could use decoded_representation to store the array as nested
        // FixedSizeList. This might yield better compression than treating the ndarray as a vector.

        let dtype = zarr_data_type_to_dtype(decoded_representation.data_type());
        let bytes = bytes.into_fixed()?;

        if let DType::Bool(..) = dtype {
            todo!()
        }

        let DType::Primitive(ptype, ..) = dtype else {
            todo!()
        };

        let runtime = Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|err| CodecError::Other(format!("tokio error: {err}")))?;
        match_each_native_ptype!(ptype, |T| {
            runtime.block_on(self.encode_typed::<T>(bytes))
        })
    }

    fn decode<'a>(
        &self,
        bytes: zarrs::array::RawBytes<'a>,
        _decoded_representation: &ChunkRepresentation,
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let opener = self.session.open_options();
        let bytes = ByteBuffer::from(bytes.into_owned());
        let fut = opener
            .open_buffer(bytes)
            .map_err(|err| CodecError::Other(format!("vortex error: {err}")))?
            .scan()
            .map_err(|err| CodecError::Other(format!("vortex error: {err}")))?
            .into_array_stream()
            .map_err(|err| CodecError::Other(format!("vortex error: {err}")))?
            .read_all();
        let runtime = Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|err| CodecError::Other(format!("tokio error: {err}")))?;
        let array = runtime
            .block_on(fut)
            .map_err(|err| CodecError::Other(format!("vortex error: {err}")))?;
        let array = array.to_canonical().into_primitive();

        Ok(ArrayBytes::Fixed(RawBytes::Owned(
            array.into_byte_buffer().as_bytes().to_vec(),
        )))
    }

    async fn async_partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        decoded_representation: &ChunkRepresentation,
        options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
        todo!()
    }

    async fn async_partial_encoder(
        self: Arc<Self>,
        input_output_handle: Arc<dyn AsyncBytesPartialEncoderTraits>,
        decoded_representation: &ChunkRepresentation,
        options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialEncoderTraits>, CodecError> {
        todo!()
    }
}

impl VortexCodec {
    async fn encode_typed<T: NativePType>(
        &self,
        bytes: RawBytes<'_>,
    ) -> Result<RawBytes<'static>, CodecError> {
        let alignment = Alignment::of::<T>();
        let owned = Bytes::from_owner(bytes.into_owned());
        assert_eq!(
            owned.as_ptr().align_offset(alignment.into()),
            0,
            "unaligned: {:?}",
            owned.as_ptr()
        );
        let vortex_bytes = Buffer::<T>::from_bytes_aligned(owned, alignment);
        // TODO(DK): Can we get invalidity information from Zarr?
        let vortex_array = PrimitiveArray::new(vortex_bytes, Validity::AllValid);
        let mut out = Vec::<u8>::new();
        let summary = self
            .session
            .write_options()
            .write(&mut out, vortex_array.to_array_stream())
            .await
            .map_err(|err| CodecError::Other(format!("vortex error: {err}")))?;
        info!(layout = %summary.footer().layout().display_tree());
        Ok(RawBytes::Owned(out))
    }
}

fn zarr_data_type_to_dtype(zarr: &DataType) -> DType {
    match zarr {
        DataType::Bool => DType::Bool(Nullability::Nullable),
        DataType::Int2 => todo!(),
        DataType::Int4 => todo!(),
        DataType::Int8 => DType::Primitive(PType::I8, Nullability::Nullable),
        DataType::Int16 => DType::Primitive(PType::I16, Nullability::Nullable),
        DataType::Int32 => DType::Primitive(PType::I32, Nullability::Nullable),
        DataType::Int64 => DType::Primitive(PType::I64, Nullability::Nullable),
        DataType::UInt2 => todo!(),
        DataType::UInt4 => todo!(),
        DataType::UInt8 => DType::Primitive(PType::U8, Nullability::Nullable),
        DataType::UInt16 => DType::Primitive(PType::U16, Nullability::Nullable),
        DataType::UInt32 => DType::Primitive(PType::U32, Nullability::Nullable),
        DataType::UInt64 => DType::Primitive(PType::U64, Nullability::Nullable),
        DataType::Float4E2M1FN => todo!(),
        DataType::Float6E2M3FN => todo!(),
        DataType::Float6E3M2FN => todo!(),
        DataType::Float8E3M4 => todo!(),
        DataType::Float8E4M3 => todo!(),
        DataType::Float8E4M3B11FNUZ => todo!(),
        DataType::Float8E4M3FNUZ => todo!(),
        DataType::Float8E5M2 => todo!(),
        DataType::Float8E5M2FNUZ => todo!(),
        DataType::Float8E8M0FNU => todo!(),
        DataType::BFloat16 => todo!(),
        DataType::Float16 => DType::Primitive(PType::F16, Nullability::Nullable),
        DataType::Float32 => DType::Primitive(PType::F32, Nullability::Nullable),
        DataType::Float64 => DType::Primitive(PType::F64, Nullability::Nullable),
        DataType::ComplexBFloat16 => todo!(),
        DataType::ComplexFloat16 => todo!(),
        DataType::ComplexFloat32 => todo!(),
        DataType::ComplexFloat64 => todo!(),
        DataType::ComplexFloat4E2M1FN => todo!(),
        DataType::ComplexFloat6E2M3FN => todo!(),
        DataType::ComplexFloat6E3M2FN => todo!(),
        DataType::ComplexFloat8E3M4 => todo!(),
        DataType::ComplexFloat8E4M3 => todo!(),
        DataType::ComplexFloat8E4M3B11FNUZ => todo!(),
        DataType::ComplexFloat8E4M3FNUZ => todo!(),
        DataType::ComplexFloat8E5M2 => todo!(),
        DataType::ComplexFloat8E5M2FNUZ => todo!(),
        DataType::ComplexFloat8E8M0FNU => todo!(),
        DataType::Complex64 => todo!(),
        DataType::Complex128 => todo!(),
        DataType::RawBits(_) => todo!(),
        DataType::String => todo!(),
        DataType::Bytes => todo!(),
        DataType::NumpyDateTime64 {
            unit: _,
            scale_factor: _,
        } => todo!(),
        DataType::NumpyTimeDelta64 {
            unit: _,
            scale_factor: _,
        } => todo!(),
        DataType::Extension(_data_type_extension) => todo!(),
        _ => todo!(),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex::{
        VortexSessionDefault as _, io::session::RuntimeSessionExt as _, session::VortexSession,
    };
    use zarrs::{
        array::{Array, ArrayBuilder, DataType},
        storage::store::MemoryStore,
    };

    use crate::VortexCodec;

    #[test]
    fn test() {
        env_logger::init();

        let store = Arc::new(MemoryStore::default());
        let array = ArrayBuilder::new(
            vec![400, 600], // array shape
            vec![200, 300], // regular chunk shape
            DataType::Float32,
            f32::NAN,
        )
        .array_to_bytes_codec(Arc::new(VortexCodec {
            session: VortexSession::default().with_tokio(),
        }))
        .dimension_names(["y", "x"].into())
        .build(store.clone(), "/array")
        .unwrap();
        array.store_metadata().unwrap();
        let data = (0..(200 * 300))
            .map(|x| (x % 100) as f32)
            .collect::<Vec<f32>>();
        array.store_chunk_elements(&[0, 0], &data).unwrap();

        let array = Array::open(store.clone(), "/array").unwrap();
        let chunk = array.retrieve_chunk_elements::<f32>(&[0, 0]).unwrap();
        assert_eq!(chunk, data);
    }
}
