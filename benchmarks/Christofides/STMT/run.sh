#!/bin/bash

# cluster conf
PROGRAM="/home/uni11/gwdg1/GWDG/lars.quentin01/code/playground/walky-stmt/target/release/walky"
XML_PATH="/home/uni11/gwdg1/GWDG/lars.quentin01/code/playground/walky-stmt/utils/gen_matrix_fast/results"

# local testing
#PROGRAM="/home/lquenti/code/walky/target/release/walky"
#XML_PATH="/home/lquenti/code/walky/utils/gen_matrix_fast/results/"

rm -rf results
mkdir results

for ((N=100; N<=3000; N+=100)); do
  echo "${N}/3000"
  for PARALLELISM in "single-threaded" "multi-threaded"; do
    command="python3 ../../run_often.py \"$PROGRAM approx christofides -p $PARALLELISM ${XML_PATH}/$N.xml\" ./results/result_${N}_${PARALLELISM}.json"
    eval "$command"
  done
done

