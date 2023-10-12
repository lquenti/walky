#!/bin/bash

# directory in that **this** file is located
#ABSPATH="$(pwd)/$(dirname "$0")"
ABSPATH="/home/uni08/hpctraining75/walky/benchmarks/MST/STMT"

# root of the walky project
WALKYDIR=$ABSPATH/../../..

# cluster conf
PROGRAM="$WALKYDIR/target/release/walky"
XML_PATH="/scratch/users/hpctraining75"

# local testing
#PROGRAM="/home/lquenti/code/walky/target/release/walky"
#XML_PATH="/home/lquenti/code/walky/utils/gen_matrix_fast/results/"

#rm -rf results
#mkdir results
module load openmpi
module load python/3.9.0

for ((N=4000; N<=10000; N+=1000)); do
  echo "${N}/10000"
  for PARALLELISM in "single-threaded" "multi-threaded"; do
    command="python3 ../../run_often.py \"$PROGRAM lower-bound mst -p $PARALLELISM ${XML_PATH}/$N.xml\" ./results/result_${N}_${PARALLELISM}.json"
    eval "$command"
  done
done

