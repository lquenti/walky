use crate::{
    datastructures::{NAMatrix, VecMatrix},
    parser::TravellingSalesmanProblemInstance,
};
use std::{error::Error, fs::File, io::Read, path::PathBuf};

use crate::solvers::exact;
use clap::{Parser, Subcommand, ValueEnum};
use one_tree::one_tree_lower_bound;

pub mod computation_mode;
pub mod datastructures;
pub mod mst;
pub mod one_tree;
pub mod parser;
pub mod preconditions;
pub mod solvers;

/// This struct contains all the arguments captured from the command line.
#[derive(Clone, Debug, Parser)]
#[command(author, version, about, long_about=None)]
#[command(propagate_version = true)]
pub struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Clone, Debug, Subcommand)]
enum Commands {
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
        lower_bound: Option<LowerBoundAlgorithm>
    },
    /// Compute the Minimal Spanning Tree of a given TSP instance
    MST {
        /// The Algorithm to use
        algorithm: MSTAlgorithm,
        /// Path to the TSPLIB-XML file
        input_file: PathBuf,
        /// Whether to solve it sequential or parallel
        #[arg(short, long, default_value_t=Parallelism::SingleThreaded, value_enum)]
        parallelism: Parallelism,
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
enum ExactAlgorithm {
    /// Testing each possible (n!) solutions
    V1,
    /// Fixating the first Element, so testing ((n-1)!) solutions
    V2,
    /// Recursive Enumeration; Stop if partial sum is worse than previous best
    V3,
    /// Stop if partial sum + greedy nearest neighbour graph is bigger than current optimum
    V4,
    /// As V4, but use an MST instead of NN-graph as a tighter bound
    V5,
    /// Cache MST distance once computed
    V6,
    /// The Held-Karp Dynamic Programming Algorithm
    HeldKarp
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum ApproxAlgorithm {
    /// Starting at each vertex, always visiting the lowest possible next vertex
    NearestNeighbour,
    /// The Christofides(-Serdyukov) algorithm
    Christofides
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum MSTAlgorithm {
    /// Prim's algorithm for finding the MST
    Prim,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum LowerBoundAlgorithm {
    /// The one tree lower bound
    OneTree,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum Parallelism {
    /// Run in a single threaded
    SingleThreaded,
    /// Run in multiple threads on a single node
    MultiThreaded,
    /// Run on multiple nodes. Requires MPI.
    MPI,
}

/*
#[derive(Debug, Parser)]
pub struct Args {
    /// path to the input file
    input_file: PathBuf,
}
 */

/// This function calls the main logic of our program.
pub fn run(cli: Cli) -> Result<(), Box<dyn Error>> {
    Ok(())
    /*
    let mut file = File::open(args.input_file)?;
    let mut xml = String::new();
    file.read_to_string(&mut xml)?;

    let tsp_instance = TravellingSalesmanProblemInstance::parse_from_xml(&xml[..])?;

    let m: VecMatrix = tsp_instance.graph.clone().into();
    let (best_cost, best_path) = exact::naive_solver(&m);

    let na_matrix: NAMatrix = (&tsp_instance.graph).into();
    let lower_bound = one_tree_lower_bound(&na_matrix);

    println!("Best Path: {:?}", best_path);
    println!("Best Cost: {}", best_cost);
    println!("1-tree lower bound: {}", lower_bound);
    Ok(())
     */
}
