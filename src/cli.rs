//! This module defines the commands and subcommands of the `walky` cli.

use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

/// This struct contains all the arguments captured from the command line.
#[derive(Clone, Debug, Parser)]
#[command(author, version, about, long_about=None)]
#[command(propagate_version = true)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Clone, Debug, Subcommand)]
pub enum Commands {
    /// Find the exact best solution to a given TSP instance
    Exact {
        /// The Algorithm to use
        algorithm: ExactAlgorithm,
        /// Path to the TSPLIB-XML file
        input_file: PathBuf,
        /// Whether to solve it sequential or parallel
        #[arg(short, long, default_value_t=Parallelism::SingleThreaded, value_enum)]
        parallelism: Parallelism,
    },
    /// Find an approximate solution to a given TSP instance
    Approx {
        /// The Algorithm to use
        algorithm: ApproxAlgorithm,
        /// Path to the TSPLIB-XML file
        input_file: PathBuf,
        #[arg(short, long, default_value_t=Parallelism::SingleThreaded, value_enum)]
        /// Whether to solve it sequential or parallel
        parallelism: Parallelism,
        /// Whether to also compute a lower_bound. Optional.
        #[arg(short, long, value_enum)]
        lower_bound: Option<LowerBoundAlgorithm>,
    },
    /// Compute a lower bound cost of a TSP instance
    LowerBound {
        /// The Algorithm to use
        algorithm: LowerBoundAlgorithm,
        /// Path to the TSPLIB-XML file
        input_file: PathBuf,
        /// Whether to solve it sequential or parallel
        #[arg(short, long, default_value_t=Parallelism::SingleThreaded, value_enum)]
        parallelism: Parallelism,
    },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum ExactAlgorithm {
    /// Testing each possible (n!) solutions
    V0,
    /// Fixating the first Element, so testing ((n-1)!) solutions
    V1,
    /// Recursive Enumeration; Keep the partial sums cached
    V2,
    /// Stop if partial sum is worse than previous best
    V3,
    /// Stop if partial sum + greedy nearest neighbour graph is bigger than current optimum
    V4,
    /// As V5, but use an MST instead of NN-graph as a tighter bound
    V5,
    /// Cache MST distance once computed
    V6,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum ApproxAlgorithm {
    /// Starting at each vertex, always visiting the lowest possible next vertex
    NearestNeighbour,
    /// The Christofides(-Serdyukov) algorithm
    Christofides,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum LowerBoundAlgorithm {
    /// The one tree lower bound
    OneTree,
    /// The MST lower bound
    MST,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum Parallelism {
    /// Run in a single threaded
    SingleThreaded,
    /// Run in multiple threads on a single node
    MultiThreaded,
    /// Run on multiple nodes. Requires MPI.
    #[cfg(feature = "mpi")]
    MPI,
}
