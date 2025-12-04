use std::sync::Arc;

use super::async_partial_decoder::AsyncPartialCodec;
use zarrs::{
    array::{
        ArrayBytes,
        codec::{
            AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits, CodecError,
            CodecOptions,
        },
    },
    indexer::Indexer,
};

#[async_trait::async_trait]
impl AsyncArrayPartialEncoderTraits for AsyncPartialCodec {
    fn into_dyn_decoder(self: Arc<Self>) -> Arc<dyn AsyncArrayPartialDecoderTraits> {
        self
    }

    async fn erase(&self) -> Result<(), CodecError> {
        todo!()
    }

    async fn partial_encode(
        &self,
        _indexer: &dyn Indexer,
        _bytes: &ArrayBytes<'_>,
        _options: &CodecOptions,
    ) -> Result<(), CodecError> {
        todo!()
    }

    fn supports_partial_encode(&self) -> bool {
        todo!()
    }
}
