use crate::datastructures::{Edge, Graph, Vertex};
use crate::{
    datastructures::NAMatrix, mst::prim, parser::TravellingSalesmanProblemInstance,
    solvers::approximate::christofides::christofides,
};
use std::{error::Error, fs::File, io::Read, path::PathBuf};

use cli::{ApproxAlgorithm, Cli, ExactAlgorithm, LowerBoundAlgorithm, MSTAlgorithm, Parallelism};
use one_tree::one_tree_lower_bound;
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
fn get_tsp_instance(
    input_file: PathBuf,
) -> Result<TravellingSalesmanProblemInstance, Box<dyn Error>> {
    let mut file = File::open(input_file)?;
    let mut xml = String::new();
    file.read_to_string(&mut xml)?;

    Ok(TravellingSalesmanProblemInstance::parse_from_xml(&xml[..])?)
}

/// Executes the driver logic for computing an exact solution
fn exact_run(
    algorithm: ExactAlgorithm,
    input_file: PathBuf,
    parallelism: Parallelism,
) -> Result<(), Box<dyn Error>> {
    let tsp_instance = get_tsp_instance(input_file)?;

    if parallelism != Parallelism::SingleThreaded {
        unimplemented!()
    }

    // TODO replace me with nalgebra
    let m: NAMatrix = (&tsp_instance.graph).into();
    let (best_cost, best_permutation) = match algorithm {
        ExactAlgorithm::V0 => exact::naive_solver(&m),
        ExactAlgorithm::V1 => exact::first_improved_solver(&m),
        ExactAlgorithm::V2 => exact::second_improved_solver(&m),
        ExactAlgorithm::V3 => exact::third_improved_solver(&m),
        ExactAlgorithm::V4 => exact::fourth_improved_solver(&m),
        ExactAlgorithm::V5 => exact::fifth_improved_solver(&m),
        ExactAlgorithm::V6 => exact::sixth_improved_solver(&m),
        ExactAlgorithm::HeldKarp => unimplemented!(),
    };
    println!("Best Cost: {}", best_cost);
    println!("Best Permutation: {:?}", best_permutation);
    Ok(())
}

/// Executes the driver logic for computing an approximate solution
fn approx_run(
    algorithm: ApproxAlgorithm,
    input_file: PathBuf,
    parallelism: Parallelism,
    lower_bound: Option<LowerBoundAlgorithm>,
) -> Result<(), Box<dyn Error>> {
    let tsp_instance = get_tsp_instance(input_file.clone())?;

    match algorithm {
        ApproxAlgorithm::Christofides => {
            let solution = match parallelism {
                Parallelism::SingleThreaded => {
                    christofides::<{ computation_mode::SEQ_COMPUTATION }>(&tsp_instance.graph)
                }
                Parallelism::MultiThreaded => {
                    christofides::<{ computation_mode::PAR_COMPUTATION }>(&tsp_instance.graph)
                }
                Parallelism::MPI => {
                    christofides::<{ computation_mode::MPI_COMPUTATION }>(&tsp_instance.graph)
                }
            };
            println!("Christofides solution weight: {}", solution.0);
        }
        ApproxAlgorithm::NearestNeighbour => todo!(),
    };

    if let Some(lower_bound_algo) = lower_bound {
        lower_bound_run(lower_bound_algo, input_file, parallelism)?
    }
    Ok(())
}

/// Executes the driver logic for computing a minimal spanning tree
fn mst_run(
    algorithm: MSTAlgorithm,
    input_file: PathBuf,
    parallelism: Parallelism,
) -> Result<(), Box<dyn Error>> {
    let tsp_instance = get_tsp_instance(input_file)?;
    let na_matrix: NAMatrix = (&tsp_instance.graph).into();

    let mst = match algorithm {
        MSTAlgorithm::Prim => match parallelism {
            Parallelism::SingleThreaded => {
                prim::<{ computation_mode::SEQ_COMPUTATION }>(&na_matrix)
            }
            Parallelism::MultiThreaded => prim::<{ computation_mode::PAR_COMPUTATION }>(&na_matrix),
            Parallelism::MPI => prim::<{ computation_mode::MPI_COMPUTATION }>(&na_matrix),
        },
    };
    println!("MST weight: {}", mst.undirected_edge_weight());
    Ok(())
}

/// Executes the driver logic for computing a lower bound
fn lower_bound_run(
    algorithm: LowerBoundAlgorithm,
    input_file: PathBuf,
    parallelism: Parallelism,
) -> Result<(), Box<dyn Error>> {
    let tsp_instance = get_tsp_instance(input_file)?;
    let na_matrix: NAMatrix = (&tsp_instance.graph).into();

    match algorithm {
        LowerBoundAlgorithm::OneTree => {
            let lower_bound = match parallelism {
                Parallelism::SingleThreaded => {
                    one_tree_lower_bound::<{ computation_mode::SEQ_COMPUTATION }>(&na_matrix)
                }
                Parallelism::MultiThreaded => {
                    one_tree_lower_bound::<{ computation_mode::PAR_COMPUTATION }>(&na_matrix)
                }
                Parallelism::MPI => {
                    one_tree_lower_bound::<{ computation_mode::MPI_COMPUTATION }>(&na_matrix)
                }
            };
            println!("1-tree lower bound: {}", lower_bound);
        }
    }
    Ok(())
}

fn debug() {
    let xs: Graph = Graph {
        vertices: vec![
            Vertex {
                edges: vec![
                    Edge { to: 1, cost: 5.0 },
                    Edge { to: 2, cost: 4.0 },
                    Edge { to: 3, cost: 10.0 },
                ],
            },
            Vertex {
                edges: vec![
                    Edge { to: 0, cost: 5.0 },
                    Edge { to: 2, cost: 8.0 },
                    Edge { to: 3, cost: 5.0 },
                ],
            },
            Vertex {
                edges: vec![
                    Edge { to: 0, cost: 4.0 },
                    Edge { to: 1, cost: 8.0 },
                    Edge { to: 3, cost: 3.0 },
                ],
            },
            Vertex {
                edges: vec![
                    Edge { to: 0, cost: 10.0 },
                    Edge { to: 1, cost: 5.0 },
                    Edge { to: 2, cost: 3.0 },
                ],
            },
        ],
    };
    let xs: NAMatrix = (&xs).into();
    let _res2 = exact::fifth_improved_solver(&xs);
}

/// This function calls the main logic of our program.
pub fn run(cli: Cli) -> Result<(), Box<dyn Error>> {
    match cli.command {
        cli::Commands::Exact {
            algorithm,
            input_file,
            parallelism,
        } => exact_run(algorithm, input_file, parallelism),
        cli::Commands::Approx {
            algorithm,
            input_file,
            parallelism,
            lower_bound,
        } => approx_run(algorithm, input_file, parallelism, lower_bound),
        cli::Commands::MST {
            algorithm,
            input_file,
            parallelism,
        } => mst_run(algorithm, input_file, parallelism),
        cli::Commands::LowerBound {
            algorithm,
            input_file,
            parallelism,
        } => lower_bound_run(algorithm, input_file, parallelism),
        cli::Commands::Debug => {
            debug();
            Ok(())
        }
    }
}
