# Walky - A Highly Parallelized TSP Solver (Supports MPI!)

Walky is a highly parallelized solver for the Travelling Salesman Problem (TSP). It has the following features

- Supports Exact Solving, Approximate Solving, and Lower Bound generation
- Compatible with the canonical [TSPLIB-XML format](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/)
- Multiple Approximate Algorithms: Nearest Neighbour, Christofides Algorithm
- Multiple Lower Bound Algorithms: Minimal Spanning Tree (MST), 1-tree
- Support for Multithreading and distributed-memory, multi-node parallelism using MPI
- Well documented, well tested, highly optimized

For a great visual introduction to the topic, the [video essay by reducible](https://www.youtube.com/watch?v=GiDsjIBOVoA) is highly recommended.

## Technical report

Besides the [full docstring coverage on `docs.rs`](https://docs.rs/walky/latest/walky/), the main technical documentation is a very detailed technical report.

See [`./technical-report/final_report.pdf`](./technical-report/final_report.pdf) for a deep dive on
- How the algorithms work, including visualizations
- The structure of the Rust project
- Detailed benchmarks, including MPI analysis for cluster usage
and much more.

## Installation

Either use cargo (add `--features mpi` for MPI)

```
cargo install walky
```

Or build from git:

```
git clone https://github.com/lquenti/walky
cd walky
cargo build --release (--features mpi)
```

For benchmarking, the `benchmarking` feature can be used.

## Usage

```
$ walky --help
A TSP solver written in Rust

Usage: walky <COMMAND>

Commands:
  exact        Find the exact best solution to a given TSP instance
  approx       Find an approximate solution to a given TSP instance
  lower-bound  Compute a lower bound cost of a TSP instance
  help         Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version
```

### Exact Algorithm

Example invocation (Algorithm `v5`, multithreaded, example generated)

```
$ walky exact v5 -p multi-threaded utils/gen_matrix_fast/results/8.xml
Best Cost: 47.85171352981164
Best Permutation: [0, 6, 1, 3, 5, 2, 7, 4]
```

Full usage:

```
$ walky exact --help
Find the exact best solution to a given TSP instance

Usage: walky exact [OPTIONS] <ALGORITHM> <INPUT_FILE>

Arguments:
  <ALGORITHM>
          The Algorithm to use

          Possible values:
          - v0: Testing each possible (n!) solutions
          - v1: Fixating the first Element, so testing ((n-1)!) solutions
          - v2: Recursive Enumeration; Keep the partial sums cached
          - v3: Stop if partial sum is worse than previous best
          - v4: Stop if partial sum + greedy nearest neighbour graph is bigger than current optimum
          - v5: As V5, but use an MST instead of NN-graph as a tighter bound
          - v6: Cache MST distance once computed

  <INPUT_FILE>
          Path to the TSPLIB-XML file

Options:
  -p, --parallelism <PARALLELISM>
          Whether to solve it sequential or parallel

          [default: single-threaded]

          Possible values:
          - single-threaded: Run in a single threaded
          - multi-threaded:  Run in multiple threads on a single node

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
          Print version
```

### Approximate Algorithms

Example invocation (Algorithm `christofides`, multithreaded, example generated)

```
$ walky approx christofides -p multi-threaded utils/gen_matrix_fast/results/8.xml
Christofides solution weight: 47.87647721988842
```

Full usage:

```
$ walky approx --help
Find an approximate solution to a given TSP instance

Usage: walky approx [OPTIONS] <ALGORITHM> <INPUT_FILE>

Arguments:
  <ALGORITHM>
          The Algorithm to use

          Possible values:
          - nearest-neighbour: Starting at each vertex, always visiting the lowest possible next vertex
          - christofides:      The Christofides(-Serdyukov) algorithm, with randomized min-cost perfect matching solver

  <INPUT_FILE>
          Path to the TSPLIB-XML file

Options:
  -p, --parallelism <PARALLELISM>
          Whether to solve it sequential or parallel
          
          [default: single-threaded]

          Possible values:
          - single-threaded: Run in a single threaded
          - multi-threaded:  Run in multiple threads on a single node

  -l, --lower-bound <LOWER_BOUND>
          Whether to also compute a lower_bound. Optional

          Possible values:
          - one-tree:  The one tree lower bound
          - mst:       The MST lower bound
          - mst-queue: The MST lower bound, computed with prims algorithm using a priority queue

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
          Print version
```

### Lower Bound

Example invocation (Algorithm `1-tree`, example generated)

```
$ walky lower-bound one-tree utils/gen_matrix_fast/results/8.xml
1-tree lower bound: 47.13382548327308
```

Full usage:
```
$ walky lower-bound --help
Compute a lower bound cost of a TSP instance

Usage: walky lower-bound [OPTIONS] <ALGORITHM> <INPUT_FILE>

Arguments:
  <ALGORITHM>
          The Algorithm to use

          Possible values:
          - one-tree:  The one tree lower bound
          - mst:       The MST lower bound
          - mst-queue: The MST lower bound, computed with prims algorithm using a priority queue

  <INPUT_FILE>
          Path to the TSPLIB-XML file

Options:
  -p, --parallelism <PARALLELISM>
          Whether to solve it sequential or parallel
          
          [default: single-threaded]

          Possible values:
          - single-threaded: Run in a single threaded
          - multi-threaded:  Run in multiple threads on a single node

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
          Print version
```

### As a library call

```
let points = vec![[0.0, 0.0], [0.0, 1.0], [2.0, 3.0]];
let graph = NAMatrix::from_points(&points);
let solution = christofides::<{ computation_mode::PAR_COMPUTATION }>(&graph);
```

## Algorithms

The algorithms can be found in the technical report (which will be uploaded soon)

## Test File Generation

Test XML files can be generated using `utils/gen_matrix_fast/{gen,gen_big}.sh`.

## Licenses

This project is licensed under the MIT License.

### Third Party Dependencies

This project includes the `priority-queue` crate, which is dual-licensed under LGPLv3 and MPLv2.
You can find the source code of that project here: <https://github.com/garro95/priority-queue>.
We can include the project in our project since the MPLv2 allows that: <https://www.mozilla.org/en-US/MPL/2.0/FAQ/>
