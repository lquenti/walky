# cluster conf
#PROGRAM="/home/uni11/gwdg1/GWDG/lars.quentin01/code/playground/walky/target/release/walky"
#XML_PATH="/home/uni11/gwdg1/GWDG/lars.quentin01/code/playground/walky/utils/gen_matrix_fast/results"
#v0...

# local testing
PROGRAM="/home/lquenti/code/walky/target/release/walky"
XML_PATH="/home/lquenti/code/walky/utils/gen_matrix_fast/results/"
# tested via find_maximum.sh
MAXv0=12
MAXv1=13
MAXv2=11
MAXv3=12
MAXv4=18
MAXv5=46
MAXv6=46
MAXthreaded=15

echo "cleanup"
rm -rf results
mkdir results

#-----
# the actual runs
# v0
hyperfine \
  --shell=none \
  --warmup 2 \
  --runs 10 \
  --export-json results/results_v0.json \
  --parameter-scan N 3 ${MAXv0} \
  "${PROGRAM} exact -p single-threaded v0 ${XML_PATH}/{N}.xml"

# v1
hyperfine \
  --shell=none \
  --warmup 2 \
  --runs 10 \
  --export-json results/results_v1.json \
  --parameter-scan N 3 ${MAXv1} \
  "${PROGRAM} exact -p single-threaded v1 ${XML_PATH}/{N}.xml"

# v2
hyperfine \
  --shell=none \
  --warmup 2 \
  --runs 10 \
  --export-json results/results_v2.json \
  --parameter-scan N 3 ${MAXv2} \
  "${PROGRAM} exact -p single-threaded v2 ${XML_PATH}/{N}.xml"

# v3
hyperfine \
  --shell=none \
  --warmup 2 \
  --runs 10 \
  --export-json results/results_v3.json \
  --parameter-scan N 3 ${MAXv3} \
  "${PROGRAM} exact -p single-threaded v3 ${XML_PATH}/{N}.xml"

# v4
hyperfine \
  --shell=none \
  --warmup 2 \
  --runs 10 \
  --export-json results/results_v4.json \
  --parameter-scan N 3 ${MAXv4} \
  "${PROGRAM} exact -p single-threaded v4 ${XML_PATH}/{N}.xml"

# v5
hyperfine \
  --shell=none \
  --warmup 2 \
  --runs 10 \
  --export-json results/results_v5.json \
  --parameter-scan N 3 ${MAXv5} \
  "${PROGRAM} exact -p single-threaded v5 ${XML_PATH}/{N}.xml"

# v6
hyperfine \
  --shell=none \
  --warmup 2 \
  --runs 10 \
  --export-json results/results_v6.json \
  --parameter-scan N 3 ${MAXv6} \
  "${PROGRAM} exact -p single-threaded v6 ${XML_PATH}/{N}.xml"

# multithreaded
hyperfine \
  --shell=none \
  --warmup 2 \
  --runs 10 \
  --export-json results/results_multithreaded.json \
  --parameter-scan N 3 ${MAXthreaded} \
  "${PROGRAM} exact -p multi-threaded v0 ${XML_PATH}/{N}.xml"

