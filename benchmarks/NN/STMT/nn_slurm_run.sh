#!/bin/bash
#SBATCH -p medium
#SBATCH -t 5:00:00
#SBATCH -o run-%J
#SBATCH -c 24

module load openmpi
module load python/3.9.0

/home/uni11/gwdg1/GWDG/lars.quentin01/code/playground/walky-stmt/benchmarks/NN/STMT/run.sh
