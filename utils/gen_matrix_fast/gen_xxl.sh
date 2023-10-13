mkdir results

for ((i=7000;i<=9000;i+=1000))
do
  echo "$i/10000"
  ./target/release/gen_matrix_fast $i > /scratch/users/$USER/$i.xml &
done

