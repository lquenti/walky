#!/bin/bash
#SBATCH -N 2
#SBATCH --ntasks-per-node=16
#SBATCH -p medium
#SBATCH -t 24:00:00

# Your milage may vary
NUMBER_OF_NODES=2
NUMBER_OF_TASKS_PER_NODE=16

# This are sane defaults
NUMBER_OF_WARMUP_RUNS=3
NUMBER_OF_BENCHMARK_RUNS=10
GRAPH_SIZE=50

# Computed from your input, do not touch
N=$((NUMBER_OF_NODES * NUMBER_OF_TASKS_PER_NODE))
OUTPUT_FILE="output_${NUMBER_OF_NODES}_nodes_${NUMBER_OF_TASKS_PER_NODE}_taskspernode.txt"

module load openmpi

# warmup
for ((i=0; i<=${NUMBER_OF_WARMUP_RUNS}; i+=1))
do
  mpirun -n $N \
    /home/uni11/gwdg1/GWDG/lars.quentin01/code/walky/target/release/walky \
    exact \
    v0 \
    -p mpi \
    /home/uni11/gwdg1/GWDG/lars.quentin01/code/walky/utils/gen_matrix_fast/results/${GRAPH_SIZE}.xml
done

# benchmark
for ((i=0; i<=${NUMBER_OF_BENCHMARK_RUNS}; i+=1))
do
  echo "run ${i}/${NUMBER_OF_BENCHMARK_RUNS}" >> $OUTPUT_FILE 2>&1
  mpirun -n $N \
    /home/uni11/gwdg1/GWDG/lars.quentin01/code/walky/target/release/walky \
    exact \
    v0 \
    -p mpi \
    /home/uni11/gwdg1/GWDG/lars.quentin01/code/walky/utils/gen_matrix_fast/results/${GRAPH_SIZE}.xml \
    >> $OUTPUT_FILE 2>&1
done
