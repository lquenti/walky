for file in ./slurm_*.sh; do
  echo "$file"
  sbatch $file
done

