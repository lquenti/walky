# Walky Benchmarks

As we are mainly interested in end2end performance and do not necessarily monitor the performance regression of single testing we chose against an integrated microbenchmarking solution such as [`criterion`](https://github.com/bheisler/criterion.rs). Instead, we use the great [`Hyperfine`](https://github.com/sharkdp/hyperfine) for benchmarking. This also makes benchmarking the MPI-algorithms easier.

## Dependencies

```
cargo install hyperfine
```

## How to run

TODO create shell scripts

## Analysis

The `scripts` folder contains several scripts for further analysis provided by the Hyperfine developers licensed under MIT. In order to use those, hyperfine has to be called with the `--export-json` parameter.
