#!/bin/bash
#SBATCH -p medium
#SBATCH -t 24:00:00
#SBATCH -o run-%J
#SBATCH -c 24

module load openmpi
module load python/3.9.0

# directory in that **this** file is located
#ABSPATH="$(pwd)/$(dirname "$0")"
ABSPATH="/home/uni08/hpctraining75/walky/benchmarks/MST/QUEUE"

$ABSPATH/run_xxl.sh
