use crate::mst::prim_with_excluded_node_priority_queue;
use crate::{
    datastructures::NAMatrix, mst::prim, parser::TravellingSalesmanProblemInstance,
    solvers::approximate::christofides::christofides,
};
use std::{error::Error, fs::File, io::Read, path::PathBuf};

use cli::{ApproxAlgorithm, Cli, ExactAlgorithm, LowerBoundAlgorithm, Parallelism};
use one_tree::one_tree_lower_bound;
use solvers::{approximate::nearest_neighbour::nearest_neighbour, exact};

pub mod cli;
pub mod computation_mode;
pub mod datastructures;
pub mod mst;
pub mod one_tree;
pub mod parser;
pub mod preconditions;
pub mod solvers;

/// rank 0 is the main node
#[cfg(feature = "mpi")]
const ROOT_RANK: mpi::Rank = 0;

#[cfg(feature = "mpi")]
use crate::datastructures::AdjacencyMatrix;
#[cfg(feature = "mpi")]
use crate::solvers::approximate::matching::{bootstrap_mpi_matching_calc, mpi_improve_matching};
#[cfg(feature = "mpi")]
use mpi::topology::*;

#[cfg(feature = "benchmarking")]
use std::time::Instant;

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
    let m: NAMatrix = (&tsp_instance.graph).into();
    #[cfg(feature = "benchmarking")]
    let now = Instant::now();

    let (best_cost, best_permutation) = match parallelism {
        Parallelism::SingleThreaded => match algorithm {
            ExactAlgorithm::V0 => exact::naive_solver(&m),
            ExactAlgorithm::V1 => exact::first_improved_solver(&m),
            ExactAlgorithm::V2 => exact::second_improved_solver(&m),
            ExactAlgorithm::V3 => exact::third_improved_solver(&m),
            ExactAlgorithm::V4 => exact::fourth_improved_solver(&m),
            ExactAlgorithm::V5 => exact::fifth_improved_solver(&m),
            ExactAlgorithm::V6 => exact::sixth_improved_solver(&m),
        },
        Parallelism::MultiThreaded => exact::threaded_solver(&m),
        #[cfg(feature = "mpi")]
        Parallelism::MPI => match algorithm {
            ExactAlgorithm::V0 => exact::static_mpi_solver(&m),
            ExactAlgorithm::V1 => exact::dynamic_mpi_solver(&m),
            _ => unimplemented!(),
        },
    };

    println!("Best Cost: {}", best_cost);
    println!("Best Permutation: {:?}", best_permutation);
    #[cfg(feature = "benchmarking")]
    println!("elapsed seconds: {}", now.elapsed().as_secs_f64());
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
            let graph = (&tsp_instance.graph).into();
            #[cfg(feature = "benchmarking")]
            let now = Instant::now();

            let solution = match parallelism {
                Parallelism::SingleThreaded => {
                    christofides::<{ computation_mode::SEQ_COMPUTATION }>(&graph)
                }
                Parallelism::MultiThreaded => {
                    christofides::<{ computation_mode::PAR_COMPUTATION }>(&graph)
                }
                #[cfg(feature = "mpi")]
                Parallelism::MPI => {
                    let universe = mpi::initialize().unwrap();
                    let world = universe.world();
                    let rank = world.rank();
                    let root_process = world.process_at_rank(0);

                    if rank == ROOT_RANK {
                        christofides::<{ computation_mode::MPI_COMPUTATION }>(&graph)
                    } else {
                        let mut tries = 0;
                        let graph = NAMatrix::from_dim(1);
                        let (mut matching, graph) = bootstrap_mpi_matching_calc(
                            &root_process,
                            &mut [],
                            rank,
                            &mut tries,
                            &graph,
                        );
                        mpi_improve_matching(&graph, matching.as_mut_slice(), tries, world);
                        // non-root process is done here
                        return Ok(());
                    }
                }
            };
            println!("Christofides solution weight: {}", solution.0);
            #[cfg(feature = "benchmarking")]
            println!("elapsed seconds: {}", now.elapsed().as_secs_f64());
        }
        ApproxAlgorithm::NearestNeighbour => {
            let graph = (&tsp_instance.graph).into();
            #[cfg(feature = "benchmarking")]
            let now = Instant::now();

            let solution = match parallelism {
                Parallelism::SingleThreaded => {
                    nearest_neighbour::<{ computation_mode::SEQ_COMPUTATION }>(&graph)
                }
                Parallelism::MultiThreaded => {
                    nearest_neighbour::<{ computation_mode::PAR_COMPUTATION }>(&graph)
                }
                #[cfg(feature = "mpi")]
                Parallelism::MPI => {
                    nearest_neighbour::<{ computation_mode::MPI_COMPUTATION }>(&graph)
                }
            };
            println!("Nearest Neighbour solution weight: {}", solution.0);
            println!("Nearest Neighbour solution: {:?}", solution.1);
            #[cfg(feature = "benchmarking")]
            println!("elapsed seconds: {}", now.elapsed().as_secs_f64());
        }
    };

    if let Some(lower_bound_algo) = lower_bound {
        lower_bound_run(lower_bound_algo, input_file, parallelism)?
    }
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
    #[cfg(feature = "benchmarking")]
    let now = Instant::now();

    match algorithm {
        LowerBoundAlgorithm::OneTree => {
            let lower_bound = match parallelism {
                Parallelism::SingleThreaded => {
                    one_tree_lower_bound::<{ computation_mode::SEQ_COMPUTATION }>(&na_matrix)
                }
                Parallelism::MultiThreaded => {
                    one_tree_lower_bound::<{ computation_mode::PAR_COMPUTATION }>(&na_matrix)
                }
                #[cfg(feature = "mpi")]
                Parallelism::MPI => {
                    let universe = mpi::initialize().unwrap();
                    let world = universe.world();
                    let rank = world.rank();
                    let result =
                        one_tree_lower_bound::<{ computation_mode::MPI_COMPUTATION }>(&na_matrix);
                    // only the root process shall write to stdout
                    if rank != ROOT_RANK {
                        return Ok(());
                    }
                    result
                }
            };
            println!("1-tree lower bound: {}", lower_bound);
        }
        LowerBoundAlgorithm::MST => {
            let mst = match parallelism {
                Parallelism::SingleThreaded => {
                    prim::<{ computation_mode::SEQ_COMPUTATION }>(&na_matrix)
                }
                Parallelism::MultiThreaded => {
                    prim::<{ computation_mode::PAR_COMPUTATION }>(&na_matrix)
                }
                #[cfg(feature = "mpi")]
                Parallelism::MPI => prim::<{ computation_mode::MPI_COMPUTATION }>(&na_matrix),
            };
            println!("MST lower bound: {}", mst.undirected_edge_weight());
        }
        LowerBoundAlgorithm::MSTQueue => {
            if parallelism != Parallelism::SingleThreaded {
                eprintln!("Only sequential computation is supported for this algorithm.");
            }
            let mst = prim_with_excluded_node_priority_queue(&na_matrix, &[]);
            println!("MST lower bound: {}", mst.undirected_edge_weight());
        }
    }
    #[cfg(feature = "benchmarking")]
    println!("elapsed seconds: {}", now.elapsed().as_secs_f64());
    Ok(())
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
        cli::Commands::LowerBound {
            algorithm,
            input_file,
            parallelism,
        } => lower_bound_run(algorithm, input_file, parallelism),
    }
}
