cargo build --release
rm -rf results
mkdir results
for ((i=3; i<=207; i+=7))
do
  echo "$i/207"
  ./target/release/gen_matrix_fast $i > ./results/$i.xml &
  ./target/release/gen_matrix_fast $((i+1)) > ./results/$((i+1)).xml &
  ./target/release/gen_matrix_fast $((i+2)) > ./results/$((i+2)).xml &
  ./target/release/gen_matrix_fast $((i+3)) > ./results/$((i+3)).xml &
  ./target/release/gen_matrix_fast $((i+4)) > ./results/$((i+4)).xml &
  ./target/release/gen_matrix_fast $((i+5)) > ./results/$((i+5)).xml &
  ./target/release/gen_matrix_fast $((i+6)) > ./results/$((i+6)).xml &
  wait
done

