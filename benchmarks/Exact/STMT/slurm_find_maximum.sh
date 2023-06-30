#!/bin/bash
#SBATCH -p medium
#SBATCH -t 2:30:00
#SBATCH -o find_maximum-%J

module load openmpi
module load python/3.9.0

/home/uni11/gwdg1/GWDG/lars.quentin01/code/playground/walky/benchmarks/Exact/STMT/find_maximum.sh
