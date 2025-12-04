use std::sync::Arc;

use futures::{FutureExt as _, future::BoxFuture};
use itertools::Itertools;
use tokio::runtime::Builder;
use tracing::info;
use vortex::{
    buffer::{Alignment, Buffer, ByteBuffer},
    error::{VortexError, VortexResult},
    file::OpenOptionsSessionExt as _,
    io::VortexReadAt,
    session::VortexSession,
    stream::ArrayStreamExt as _,
};
use zarrs::{
    array::{
        ArrayBytes, DataType, RawBytes,
        codec::{ArrayPartialDecoderTraits, BytesPartialDecoderTraits, CodecError, CodecOptions},
    },
    indexer::Indexer,
    storage::{StorageError, byte_range::ByteRange},
};

pub(crate) struct PartialCodec {
    pub(crate) session: VortexSession,
    pub(crate) input_handle: Arc<dyn BytesPartialDecoderTraits>,
    pub(crate) options: CodecOptions,
    pub(crate) chunk_shape: Vec<u64>,
    pub(crate) data_type: DataType,
}

struct BytesPartialDecoderTraitsIsVortexReadAt {
    bytes_source: Arc<dyn BytesPartialDecoderTraits>,
    options: CodecOptions,
}

impl VortexReadAt for BytesPartialDecoderTraitsIsVortexReadAt {
    fn read_at(
        &self,
        offset: u64,
        length: usize,
        _alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<ByteBuffer>> {
        let bytes_source = self.bytes_source.clone();
        let options = self.options.clone();
        async move {
            info!(offset, length);

            let vec = bytes_source
                .partial_decode(
                    ByteRange::from(offset..(offset + (length as u64))),
                    &options,
                )
                .map_err(|err| VortexError::generic(Box::new(err)))?
                .expect("no nones in partial decode")
                .into_owned();
            Ok(ByteBuffer::from(vec))
        }
        .boxed()
    }

    fn size(&self) -> BoxFuture<'static, vortex::error::VortexResult<u64>> {
        let bytes_source = self.bytes_source.clone();
        async move {
            info!(size_held = %bytes_source.size_held());
            // FIXME(DK): This is probably wrong.
            Ok(bytes_source.size_held() as u64)
        }
        .boxed()
    }
}

impl ArrayPartialDecoderTraits for PartialCodec {
    /// Return the data type of the partial decoder.
    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    /// Returns whether the chunk exists.
    ///
    /// # Errors
    /// Returns [`StorageError`] if a storage operation fails.
    fn exists(&self) -> Result<bool, StorageError> {
        self.input_handle.exists()
    }

    /// Returns the size of chunk bytes held by the partial decoder.
    ///
    /// Intended for use by size-constrained partial decoder caches.
    fn size_held(&self) -> usize {
        self.input_handle.size_held()
    }

    /// Partially decode a chunk.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails, array subset is invalid, or the array subset shape does not match array view subset shape.
    fn partial_decode<'a>(
        &'a self,
        indexer: &dyn Indexer,
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let opener = self.session.open_options();

        let runtime = Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|err| CodecError::Other(format!("tokio error: {err}")))?;

        let file = runtime
            .block_on(
                opener.open_read_at(BytesPartialDecoderTraitsIsVortexReadAt {
                    bytes_source: self.input_handle.clone(),
                    options: self.options.clone(),
                }),
            )
            .map_err(|err| CodecError::Other(format!("vortex error: {err}")))?;

        let row_indices = Buffer::from_iter(indexer.iter_linearised_indices(&self.chunk_shape)?);
        let start = row_indices[0];
        let end = row_indices[row_indices.len() - 1];

        info!(start, end, n_indices = %row_indices.len(), indices = %row_indices.iter().join(","));
        let fut = file
            .scan()
            .map_err(|err| CodecError::Other(format!("vortex error: {err}")))?
            // .with_row_range(start..end)
            .with_row_indices(row_indices)
            .into_array_stream()
            .map_err(|err| CodecError::Other(format!("vortex error: {err}")))?
            .read_all();

        let array = runtime
            .block_on(fut)
            .map_err(|err| CodecError::Other(format!("vortex error: {err}")))?;

        let array = array.to_canonical().into_primitive();

        info!(n_elements = %array.len());

        Ok(ArrayBytes::Fixed(RawBytes::Owned(
            array.into_byte_buffer().as_bytes().to_vec(),
        )))
    }

    fn supports_partial_decode(&self) -> bool {
        true
    }
}
