#![allow(clippy::unwrap_used)]

use std::{iter::Iterator, sync::Arc};

use divan::Bencher;
use vortex::{
    VortexSessionDefault as _, io::session::RuntimeSessionExt as _, session::VortexSession,
};
use zarr_vortex::VortexCodec;
use zarrs::{
    array::{
        Array, ArrayBuilder, DataType,
        codec::{ShardingCodecBuilder, ZstdCodec},
    },
    array_subset::ArraySubset,
    filesystem::FilesystemStore,
};

fn main() {
    divan::main();
}

const ARGS: &[(u64, u64)] = &[
    (128, 128),
    (1024, 128),
    (2048, 128),
    (16_384, 128),
    //
    (1024, 256),
    (2048, 256),
    (16_384, 256),
    //
    (1024, 512),
    (2048, 512),
    (16_384, 512),
    //
    (1024, 768),
    (2048, 768),
    (16_384, 768),
    //
    (1024, 1024),
    (2048, 1024),
    (16_384, 1024),
    //
    (2048, 1536),
    (16_384, 1536),
    //
    (2048, 1792),
    (16_384, 1792),
    //
    (2048, 2048),
    (16_384, 2048),
    //
    (16_384, 4096),
    //
    (16_384, 8192),
    // TOO SLOW:
    // (16_384, 16_384),
];

#[divan::bench(args = ARGS)]
fn vortex(bencher: Bencher, (write_size, read_size): (u64, u64)) {
    let store = Arc::new(FilesystemStore::new("/tmp/vortex").unwrap()); // MemoryStore::default()
    let array = ArrayBuilder::new(
        vec![64 * 1024, 64 * 1024],   // array shape
        vec![write_size, write_size], // regular chunk shape
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
    let data = (0..(write_size * write_size))
        .map(|x| (x % 10) as f32 / 10.0f32)
        .collect::<Vec<f32>>();
    // for r in 0..(n / write_size) {
    //     for c in 0..(n / write_size) {
    let r = 0;
    let c = 0;
    array.store_chunk_elements(&[r, c], &data).unwrap();
    //     }
    // }

    let array = Array::open(store.clone(), "/array").unwrap();
    let chunk = array.retrieve_chunk_elements::<f32>(&[0, 0]).unwrap();
    assert_eq!(chunk, data);

    let array = Array::open(store.clone(), "/array").unwrap();

    bencher.bench_local(|| {
        // for r in 0..(n / write_size) {
        //     for c in 0..(n / write_size) {
        let r = 0;
        let c = 0;
        divan::black_box(
            array
                .retrieve_array_subset_elements::<f32>(&ArraySubset::new_with_ranges(&[
                    (r * write_size)..(r * write_size + read_size),
                    (c * write_size)..(c * write_size + read_size),
                ]))
                .unwrap(),
        );
        //     }
        // }
    })
}

#[divan::bench(args = ARGS)]
fn zstd(bencher: Bencher, (write_size, read_size): (u64, u64)) {
    let store = Arc::new(FilesystemStore::new("/tmp/zstd").unwrap()); // MemoryStore::default()
    let array = ArrayBuilder::new(
        vec![64 * 1024, 64 * 1024],   // array shape
        vec![write_size, write_size], // regular chunk shape
        DataType::Float32,
        f32::NAN,
    )
    .array_to_bytes_codec(Arc::new(
        // The sharding codec requires the sharding feature
        ShardingCodecBuilder::new(
            [write_size, write_size].try_into().unwrap(), // inner chunk shape
        )
        .bytes_to_bytes_codecs(vec![Arc::new(ZstdCodec::new(3, false))])
        .build(),
    ))
    .dimension_names(["y", "x"].into())
    .build(store.clone(), "/array")
    .unwrap();
    array.store_metadata().unwrap();
    let data = (0..(write_size * write_size))
        .map(|x| (x % 10) as f32 / 10.0f32)
        .collect::<Vec<f32>>();
    // for r in 0..(n / write_size) {
    //     for c in 0..(n / write_size) {
    let r = 0;
    let c = 0;
    array.store_chunk_elements(&[r, c], &data).unwrap();
    //     }
    // }

    let array = Array::open(store.clone(), "/array").unwrap();
    let chunk = array.retrieve_chunk_elements::<f32>(&[0, 0]).unwrap();
    assert_eq!(chunk, data);

    let array = Array::open(store.clone(), "/array").unwrap();

    bencher.bench_local(|| {
        // for r in 0..(n / write_size) {
        //     for c in 0..(n / write_size) {
        let r = 0;
        let c = 0;
        divan::black_box(
            array
                .retrieve_array_subset_elements::<f32>(&ArraySubset::new_with_ranges(&[
                    (r * write_size)..(r * write_size + read_size),
                    (c * write_size)..(c * write_size + read_size),
                ]))
                .unwrap(),
        );
        //     }
        // }
    });
}
