use zarrs::{
    array::{
        ArrayBytes, DataType,
        codec::{AsyncArrayPartialDecoderTraits, CodecError, CodecOptions},
    },
    indexer::Indexer,
    storage::StorageError,
};

pub(crate) struct AsyncPartialCodec;

#[async_trait::async_trait]
impl AsyncArrayPartialDecoderTraits for AsyncPartialCodec {
    /// Return the data type of the partial decoder.
    fn data_type(&self) -> &DataType {
        todo!()
    }

    /// Returns whether the chunk exists.
    ///
    /// # Errors
    /// Returns [`StorageError`] if a storage operation fails.
    async fn exists(&self) -> Result<bool, StorageError> {
        todo!()
    }

    /// Returns the size of chunk bytes held by the partial decoder.
    ///
    /// Intended for use by size-constrained partial decoder caches.
    fn size_held(&self) -> usize {
        todo!()
    }

    /// Partially decode a chunk.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails, array subset is invalid, or the array subset shape does not match array view subset shape.
    async fn partial_decode<'a>(
        &'a self,
        _indexer: &dyn Indexer,
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        todo!()
    }
    fn supports_partial_decode(&self) -> bool {
        todo!()
    }
}
