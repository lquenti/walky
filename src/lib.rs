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
    Exact {
        algorithm: ExactAlgorithm,
        input_file: PathBuf,
        #[arg(short, long, default_value_t=Parallelism::SingleThreaded, value_enum)]
        parallelism: Parallelism,
    },
    Approx {
        algorithm: ApproxAlgorithm,
        input_file: PathBuf,
        #[arg(short, long, default_value_t=Parallelism::SingleThreaded, value_enum)]
        parallelism: Parallelism,
        /// TODO make comment that it is optional
        #[arg(short, long, value_enum)]
        lower_bound: Option<LowerBoundAlgorithm>
    },
    MST {
        algorithm: MSTAlgorithm,
        input_file: PathBuf,
        #[arg(short, long, default_value_t=Parallelism::SingleThreaded, value_enum)]
        parallelism: Parallelism,
    },
    LowerBound {
        algorithm: LowerBoundAlgorithm,
        input_file: PathBuf,
        #[arg(short, long, default_value_t=Parallelism::SingleThreaded, value_enum)]
        parallelism: Parallelism,
    },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum ExactAlgorithm {
    TODO,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum ApproxAlgorithm {
    TODO,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum MSTAlgorithm {
    Prim,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum LowerBoundAlgorithm {
    TODO,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum Parallelism {
    SingleThreaded,
    MultiThreaded,
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
