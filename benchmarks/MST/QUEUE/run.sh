#!/bin/bash

# directory in that **this** file is located
#ABSPATH="$(pwd)/$(dirname "$0")"
ABSPATH="/home/uni08/hpctraining75/walky/benchmarks/MST/QUEUE"

# root of the walky project
WALKYDIR=$ABSPATH/../../..

# cluster conf
PROGRAM="$WALKYDIR/target/release/walky"
XML_PATH="$WALKYDIR/utils/gen_matrix_fast/results"

# local testing
#PROGRAM="/home/lquenti/code/walky/target/release/walky"
#XML_PATH="/home/lquenti/code/walky/utils/gen_matrix_fast/results/"

#rm -rf results
#mkdir results

for ((N=100; N<=3000; N+=100)); do
  echo "${N}/3000"
  command="python3 ../../run_often.py \"$PROGRAM lower-bound mst-queue ${XML_PATH}/$N.xml\" ./results/result_${N}_${PARALLELISM}.json"
  eval "$command"
done

