#!/bin/bash

BINARY=$(readlink -f "../target/release/walky")

echo "Compiling hyperfine if not existing"
cargo install hyperfine

pushd ..

# For final benchmarks we can probably improve, but for now
# those are the maximal resonable sizes for a benchmark
# max v0: 12
# max v1: 13
# max v2: 11
# max v3: 14
# max v4: 28
# max v5: 42
# max v6: 42

echo "Building walky..."
cargo build --release
# Add --show-output for debugging


# We go from slow to fast, up to 11 all can do with v2
for ((i = 3; i <= 11; i++))
do
  echo "Benchmark $i"
  hyperfine --shell=none --warmup 3 --runs 10 --parameter-list V v0,v1,v2,v3,v4,v5,v6 --export-json results${i}.json "$BINARY exact {V} $(readlink -f ./benchmarks/inputs)/$i.xml"
done

# We can do 12 with v0
echo "Benchmark 12"
hyperfine --shell=none --warmup 3 --runs 10 --parameter-list V v0,v1,v3,v4,v5,v6 --export-json results${i}.json "$BINARY exact {V} $(readlink -f ./benchmarks/inputs)/$i.xml"

# We can do 13 with v1
echo "Benchmark 13"
hyperfine --shell=none --warmup 3 --runs 10 --parameter-list V v1,v3,v4,v5,v6 --export-json results${i}.json "$BINARY exact {V} $(readlink -f ./benchmarks/inputs)/$i.xml"

# We can do 13 with v3
echo "Benchmark 13"
hyperfine --shell=none --warmup 3 --runs 10 --parameter-list V v3,v4,v5,v6 --export-json results${i}.json "$BINARY exact {V} $(readlink -f ./benchmarks/inputs)/$i.xml"

# We can go up to 28 with v4
for ((i = 14; i <= 28; i++))
do
  echo "Benchmark $i"
  hyperfine --shell=none --warmup 3 --runs 10 --parameter-list V v4,v5,v6 --export-json results${i}.json "$BINARY exact {V} $(readlink -f ./benchmarks/inputs)/$i.xml"
done

# We can finally go up to 42 with v6 without waiting too long
for ((i = 14; i <= 28; i++))
do
  echo "Benchmark $i"
  hyperfine --shell=none --warmup 3 --runs 10 --parameter-list V v5,v6 --export-json results${i}.json "$BINARY exact {V} $(readlink -f ./benchmarks/inputs)/$i.xml"
done

popd
rm -rf results_exact
mkdir results_exact
mv ../results*.json ./results/
