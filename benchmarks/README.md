# All benchmarks

The structure is
```
<folder_of_benchmark_type>/<STMT> -> Single and Multithreaded single node
<folder_of_benchmark_type>/<MPI> -> MPI distributed memory
```

`run_often.py` just runs it and extracts the internal time, which is only visible if compiled with the benchmarking flag

build with
```
cargo build --release --features mpi,benchmarking
```
