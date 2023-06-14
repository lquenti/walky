#!/bin/bash

# change how you want, its precomputed from 3/50
# If you need more, see ../utils/benchmark_gen.py
MIN_SIZE=3
MAX_SIZE=9
BINARY=$(readlink -f "../target/release/walky")

pushd ..

cargo build --release
# Since hyperfine spawns a shell, we use the abs path everywhere
# Add --show-output for debugging
hyperfine --warmup 10 --runs 100 --parameter-scan N $MIN_SIZE $MAX_SIZE --export-json results.json "$BINARY $(readlink -f ./benchmarks/inputs)/{N}.xml"

popd
mv ../results.json .
