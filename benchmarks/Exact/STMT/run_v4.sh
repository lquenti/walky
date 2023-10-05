# cluster conf
PROGRAM="/home/uni11/gwdg1/GWDG/lars.quentin01/code/walky/target/release/walky"
XML_PATH="/home/uni11/gwdg1/GWDG/lars.quentin01/code/walky/utils/gen_matrix_fast/results"
#v0...

# local testing
#PROGRAM="/home/lquenti/code/walky/target/release/walky"
#XML_PATH="/home/lquenti/code/walky/utils/gen_matrix_fast/results/"

VX=v4

hyperfine \
  --shell=none \
  --warmup 2 \
  --runs 10 \
  --export-json results/results_${VX}.json \
  --parameter-scan N 3 50 \
  "${PROGRAM} exact -p single-threaded ${VX} ${XML_PATH}/{N}.xml"
