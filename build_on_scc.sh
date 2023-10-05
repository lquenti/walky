#!/bin/bash

# The Scientific Compute Cluster (SCC) is one of the clusters from the GWDG
# https://gwdg.de/en/hpc/systems/scc/
# (used for our benchmarking)

# 1. make sure that Rust is installed (rustup is newer than any module)
# (its idempotent)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# 2. Load all modules required
# LLVM is only available through spack
module load openmpi
module load spack-user
source $SPACK_USER_ROOT/share/spack/setup-env.sh
spack load llvm

# 3. build it with mpi and benchmarking
cargo build --release --features benchmarking,mpi
