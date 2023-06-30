rm -rf big
mkdir big

for ((i=200;i<=2500;i+=100))
do
  echo "$i/2500"
  ./target/release/gen_matrix_fast $i > ./big/$i.xml &
done

