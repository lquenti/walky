use crate::{
    datastructures::{NAMatrix, VecMatrix},
    parser::TravellingSalesmanProblemInstance,
};
use std::{error::Error, fs::File, io::Read, path::PathBuf};

use cli::{Cli, MSTAlgorithm, Parallelism, ExactAlgorithm, ApproxAlgorithm, LowerBoundAlgorithm};
use one_tree::one_tree;
use solvers::exact;

pub mod cli;
pub mod computation_mode;
pub mod datastructures;
pub mod mst;
pub mod one_tree;
pub mod parser;
pub mod preconditions;
pub mod solvers;

/// Extracts the TSP instance from a TSPLIB-XML file.
fn get_tsp_instance(input_file: PathBuf) -> Result<TravellingSalesmanProblemInstance, Box<dyn Error>> {
    let mut file = File::open(input_file)?;
    let mut xml = String::new();
    file.read_to_string(&mut xml)?;

    Ok(TravellingSalesmanProblemInstance::parse_from_xml(&xml[..])?)
}

/// Executes the driver logic for computing an exact solution
fn exact_run(algorithm: ExactAlgorithm, input_file: PathBuf, parallelism: Parallelism) -> Result<(), Box<dyn Error>> {
    let tsp_instance = get_tsp_instance(input_file)?;

    if (parallelism != Parallelism::SingleThreaded) {
        unimplemented!()
    }

    let m: VecMatrix = tsp_instance.graph.into();
    let (best_cost, best_permutation) = match algorithm {
        ExactAlgorithm::V1 => exact::naive_solver(&m),
        ExactAlgorithm::V2 => exact::first_improved_solver(&m),
        ExactAlgorithm::V3 => unimplemented!(),
        ExactAlgorithm::V4 => unimplemented!(),
        ExactAlgorithm::V5 => unimplemented!(),
        ExactAlgorithm::V6 => unimplemented!(),
        ExactAlgorithm::HeldKarp => unimplemented!()
    };
    println!("Best Cost: {}", best_cost);
    println!("Best Permutation: {:?}", best_permutation);
    Ok(())
}

/// Executes the driver logic for computing an approximate solution
fn approx_run(algorithm: ApproxAlgorithm, input_file: PathBuf, parallelism: Parallelism, lower_bound: Option<LowerBoundAlgorithm>) -> Result<(), Box<dyn Error>> {
    let tsp_instance = get_tsp_instance(input_file)?;
    Ok(())
}

/// Executes the driver logic for computing a minimal spanning tree
fn mst_run(algorithm: MSTAlgorithm, input_file: PathBuf, parallelism: Parallelism) -> Result<(), Box<dyn Error>> {
    let tsp_instance = get_tsp_instance(input_file)?;
    Ok(())
}

/// Executes the driver logic for computing a lower bound
fn lower_bound_run(algorithm: LowerBoundAlgorithm, input_file: PathBuf, parallelism: Parallelism) -> Result<(), Box<dyn Error>> {
    let tsp_instance = get_tsp_instance(input_file)?;
    let na_matrix: NAMatrix = (&tsp_instance.graph).into();
    let lower_bound = one_tree::one_tree_lower_bound(&na_matrix);
    println!("1-tree lower bound: {}", lower_bound);
    Ok(())
}

/// This function calls the main logic of our program.
pub fn run(cli: Cli) -> Result<(), Box<dyn Error>> {
    match cli.command {
        cli::Commands::Exact { algorithm, input_file, parallelism } => exact_run(algorithm, input_file, parallelism),
        cli::Commands::Approx { algorithm, input_file, parallelism, lower_bound } => approx_run(algorithm, input_file, parallelism, lower_bound),
        cli::Commands::MST { algorithm, input_file, parallelism } => mst_run(algorithm, input_file, parallelism),
        cli::Commands::LowerBound { algorithm, input_file, parallelism } => lower_bound_run(algorithm, input_file, parallelism)
    }
}
