use crate::{datastructures::VecMatrix, parser::TravellingSalesmanProblemInstance};
use std::{error::Error, fs::File, io::Read, path::PathBuf};

use crate::solvers::exact;
use clap::Parser;
use one_tree::one_tree_lower_bound;

pub mod datastructures;
pub mod mst;
pub mod one_tree;
pub mod parser;
pub mod preconditions;
pub mod solvers;

/// This struct contains all the arguments captured from the command line.
#[derive(Debug, Parser)]
pub struct Args {
    /// path to the input file
    input_file: PathBuf,
}

/// This function calls the main logic of our program.
pub fn run(args: Args) -> Result<(), Box<dyn Error>> {
    let mut file = File::open(args.input_file)?;
    let mut xml = String::new();
    file.read_to_string(&mut xml)?;
    let tsp_instance = TravellingSalesmanProblemInstance::parse_from_xml(&xml[..])?;
    let m: VecMatrix = tsp_instance.graph.clone().into();
    let (best_cost, best_path) = exact::naive_solver(&m);
    let lower_bound = one_tree_lower_bound(&tsp_instance.graph);
    println!("Best Path: {:?}", best_path);
    println!("Best Cost: {}", best_cost);
    println!("1-tree lower bound: {}", lower_bound);
    Ok(())
}
