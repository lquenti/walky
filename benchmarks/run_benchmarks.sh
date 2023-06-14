#!/bin/bash

# change how you want, its precomputed from 3/50
# If you need more, see ../utils/benchmark_gen.py
MIN_SIZE=3
MAX_SIZE=5

pushd ..

cargo build --release
hyperfine --warmup 10 --runs 100 --parameter-scan N $MIN_SIZE $MAX_SIZE --export-json results.json "echo {N}"

popd
mv ../results.json .
