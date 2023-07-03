mkdir results

for ((i=200;i<=3000;i+=100))
do
  echo "$i/3000"
  ./target/release/gen_matrix_fast $i > ./results/$i.xml &
done

