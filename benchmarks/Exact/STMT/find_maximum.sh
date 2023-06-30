#!/bin/bash

parameters=("v0" "v1" "v2" "v3" "v4" "v5" "v6")

for param in "${parameters[@]}"
do
  echo $param
  python3 find_maximum.py "$param" "single-threaded"
done

python3 find_maximum.py "v0" "multi-threaded"
