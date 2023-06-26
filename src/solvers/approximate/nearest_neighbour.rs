//! Finding an approximate solution to the MST

use ordered_float::OrderedFloat;
use rand::seq::SliceRandom;
use rayon::iter::ParallelIterator;
use rayon::prelude::IntoParallelIterator;

#[cfg(feature="mpi")]
use mpi::{
    collective::UserOperation, datatype::UserDatatype, internal::memoffset::offset_of, traits::*,
    Address,
};

use crate::{
    computation_mode::{panic_on_invaid_mode, PAR_COMPUTATION, SEQ_COMPUTATION},
    datastructures::{AdjacencyMatrix, NAMatrix, Solution},
};

#[cfg(feature="mpi")]
use crate::computation_mode::MPI_COMPUTATION;



#[derive(Default, Clone, Copy)]
#[cfg(feature="mpi")]
struct MPICostRank(f64, i32);

#[cfg(feature="mpi")]
unsafe impl Equivalence for MPICostRank {
    type Out = UserDatatype;
    fn equivalent_datatype() -> Self::Out {
        UserDatatype::structured(
            &[1, 1],
            &[
                offset_of!(MPICostRank, 0) as Address,
                offset_of!(MPICostRank, 1) as Address,
            ],
            &[f64::equivalent_datatype(), i32::equivalent_datatype()],
        )
    }
}

/// Using the nearest neighbour algorithm for a single starting node.
///
/// The algorithm works greedily; Starting at a specific node, it first goes to the nearest node.
/// From there on, it visits the nearest previously unvisited node. Once all nodes are visited,
/// it returns to the beginning. Thats our tour.
pub fn single_nearest_neighbour(graph_matrix: &NAMatrix, index: usize) -> Solution {
    assert!(index < graph_matrix.dim());
    let num_nodes = graph_matrix.ncols();
    let mut path = Vec::with_capacity(num_nodes);
    let mut visited = vec![false; num_nodes];
    let mut current_node = index;
    let mut distance = 0.0;

    path.push(current_node);
    visited[current_node] = true;

    while path.len() < num_nodes {
        let mut nearest_neighbour = None;
        let mut min_distance = f64::INFINITY;

        for node in 0..num_nodes {
            if !visited[node] && graph_matrix[(current_node, node)] < min_distance {
                min_distance = graph_matrix[(current_node, node)];
                nearest_neighbour = Some(node);
            }
        }

        if let Some(neighbour) = nearest_neighbour {
            current_node = neighbour;
            path.push(current_node);
            distance += min_distance;
            visited[current_node] = true;
        } else {
            // This should never happen; it should be catched by the while loop
            panic!()
        }
    }

    // Add the loop closure
    distance += graph_matrix[(current_node, index)];
    (distance, path)
}

/// Generate n unique random numbers between `[0..n)`
fn n_random_numbers(min: usize, max: usize, n: usize) -> Vec<usize> {
    let mut xs: Vec<usize> = (min..max).collect();
    let mut rng = rand::thread_rng();
    xs.shuffle(&mut rng);
    xs.truncate(n);
    xs
}

/// Call [`single_nearest_neighbour`] n times, randomly.
/// Since [`single_nearest_neighbour`] is deterministic, we use n different starting nodes.
///
/// `MODE`: constant parameter, choose one of the values from [`crate::computation_mode`]
pub fn n_nearest_neighbour<const MODE: usize>(graph_matrix: &NAMatrix, n: usize) -> Solution {
    assert!(n != 0);
    assert!(n <= graph_matrix.dim());
    match MODE {
        SEQ_COMPUTATION => n_random_numbers(0, graph_matrix.dim(), n)
            .into_iter()
            .map(|k| single_nearest_neighbour(graph_matrix, k))
            .min_by_key(|&(distance, _)| OrderedFloat(distance))
            .unwrap(),
        PAR_COMPUTATION => n_random_numbers(0, graph_matrix.dim(), n)
            .into_par_iter()
            .map(|k| single_nearest_neighbour(graph_matrix, k))
            .min_by_key(|&(distance, _)| OrderedFloat(distance))
            .unwrap(),
        #[cfg(feature = "mpi")]
        MPI_COMPUTATION => todo!(), // We just support n=dim for now
        _ => panic_on_invaid_mode::<MODE>(),
    }
}

/// Call [`single_nearest_neighbour`] for every starting node.
pub fn nearest_neighbour<const MODE: usize>(graph_matrix: &NAMatrix) -> Solution {
    #[cfg(feature = "mpi")]
    if MODE == MPI_COMPUTATION {
        return nearest_neighbour_mpi(graph_matrix);
    }
    n_nearest_neighbour::<MODE>(graph_matrix, graph_matrix.dim())
}

// TODO refactor me into the structure
#[cfg(feature="mpi")]
pub fn nearest_neighbour_mpi(graph_matrix: &NAMatrix) -> Solution {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size();
    let rank = world.rank();

    let n: i32 = graph_matrix.dim().try_into().unwrap();

    // Divide the nodes into chunks
    let chunk_size = n / size;
    let start_node = chunk_size * rank;
    let mut end_node = start_node + chunk_size;

    // fix if it isnt divisible
    if (rank == size - 1) && (n % size != 0) {
        end_node = n;
    }

    // Do the solving for our local chunk
    let local_solution = (start_node..end_node)
        .map(|k| single_nearest_neighbour(graph_matrix, k.try_into().unwrap()))
        .min_by_key(|&(distance, _)| OrderedFloat(distance))
        .unwrap();

    // ALLREDUCE all solutions
    // If two solutions have same cost, we choose lower rank
    // This makes it commutative, thus easier to reduce
    let sendbuf = MPICostRank(local_solution.0, rank);
    let mut recvbuf = MPICostRank(f64::INFINITY, -1);
    world.all_reduce_into(
        &sendbuf,
        &mut recvbuf,
        &UserOperation::commutative(|x, y| {
            println!("{:?} {:?}", x, y);
            println!("{}")
            let x: &[MPICostRank] = x.downcast().unwrap();
            let y: &mut [MPICostRank] = y.downcast().unwrap();
            println!("this was successful once!");
            // If y.cost < x.cost, we do nothing as the acc is better
            // If y.cost > x.cost, we set the y to x
            if y[0].0 > x[0].0 {
                y[0] = x[0];
            }
            // Else, we are equal, then we decide on the lower rank
            // We only do this so it is commutative, since we do an
            // ALL_REDUCE and want every note to agree which node
            // won.
            if y[0].1 > x[0].1 {
                y[0] = x[0];
            }
        }),
    );

    let mut best_path = local_solution.1.to_owned();

    let winner_process = world.process_at_rank(recvbuf.1);
    // The winner tells everybody who won
    winner_process.broadcast_into(&mut best_path[..]);

    (recvbuf.0, best_path)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        computation_mode,
        datastructures::{Edge, Graph, NAMatrix, Vertex},
        solvers::exact::first_improved_solver,
    };

    #[test]
    fn integer_graph_5x5() {
        let graph: Graph = Graph {
            vertices: vec![
                Vertex {
                    edges: vec![
                        Edge { to: 1, cost: 6.0 },
                        Edge { to: 2, cost: 8.0 },
                        Edge { to: 3, cost: 5.0 },
                        Edge { to: 4, cost: 3.0 },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge { to: 0, cost: 6.0 },
                        Edge { to: 2, cost: 9.0 },
                        Edge { to: 3, cost: 5.0 },
                        Edge { to: 4, cost: 4.0 },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge { to: 0, cost: 8.0 },
                        Edge { to: 1, cost: 9.0 },
                        Edge { to: 3, cost: 8.0 },
                        Edge { to: 4, cost: 5.0 },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge { to: 0, cost: 5.0 },
                        Edge { to: 1, cost: 5.0 },
                        Edge { to: 2, cost: 8.0 },
                        Edge { to: 4, cost: 6.0 },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge { to: 0, cost: 3.0 },
                        Edge { to: 1, cost: 4.0 },
                        Edge { to: 2, cost: 5.0 },
                        Edge { to: 3, cost: 6.0 },
                    ],
                },
            ],
        };
        let nm: NAMatrix = (&graph).into();
        // Starting Node 0
        let (distance, path) = single_nearest_neighbour(&nm, 0);
        assert_eq!(distance, 28.0);
        assert_eq!(path, vec![0, 4, 1, 3, 2]);
        // Starting Node 1
        let (distance, path) = single_nearest_neighbour(&nm, 1);
        assert_eq!(distance, 29.0);
        assert_eq!(path, vec![1, 4, 0, 3, 2]);
        // Starting Node 2
        let (distance, path) = single_nearest_neighbour(&nm, 2);
        assert_eq!(distance, 27.0);
        assert_eq!(path, vec![2, 4, 0, 3, 1]);
        // Starting Node 3
        let (distance, path) = single_nearest_neighbour(&nm, 3);
        assert_eq!(distance, 29.0);
        assert_eq!(path, vec![3, 0, 4, 1, 2]);
        // Starting Node 4
        let (distance, path) = single_nearest_neighbour(&nm, 4);
        assert_eq!(distance, 27.0);
        assert_eq!(path, vec![4, 0, 3, 1, 2]);

        // Check that total is 27 and one of the best two
        let (distance, path) = nearest_neighbour::<{ computation_mode::SEQ_COMPUTATION }>(&nm);
        assert_eq!(distance, 27.0);
        assert!(path == vec![2, 4, 0, 3, 1] || path == vec![4, 0, 3, 1, 2]);

        // Are they the same?
        let (distance2, path2) = nearest_neighbour::<{ computation_mode::PAR_COMPUTATION }>(&nm);
        assert_eq!(distance, distance2);
        assert!(path2 == vec![2, 4, 0, 3, 1] || path2 == vec![4, 0, 3, 1, 2]);

        // That that we are below or equal to the perfect solution
        let (perfect_d, _) = first_improved_solver(&nm);
        assert!(perfect_d <= distance);
    }

    #[test]
    fn integer_graph_10x10() {
        let graph = Graph {
            vertices: vec![
                Vertex {
                    edges: vec![
                        Edge { to: 1, cost: 8.0 },
                        Edge { to: 2, cost: 6.0 },
                        Edge { to: 3, cost: 7.0 },
                        Edge { to: 4, cost: 4.0 },
                        Edge { to: 5, cost: 2.0 },
                        Edge { to: 6, cost: 4.0 },
                        Edge { to: 7, cost: 5.0 },
                        Edge { to: 8, cost: 5.0 },
                        Edge { to: 9, cost: 5.0 },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge { to: 0, cost: 8.0 },
                        Edge { to: 2, cost: 2.0 },
                        Edge { to: 3, cost: 6.0 },
                        Edge { to: 4, cost: 8.0 },
                        Edge { to: 5, cost: 3.0 },
                        Edge { to: 6, cost: 2.0 },
                        Edge { to: 7, cost: 7.0 },
                        Edge { to: 8, cost: 1.0 },
                        Edge { to: 9, cost: 6.0 },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge { to: 0, cost: 6.0 },
                        Edge { to: 1, cost: 2.0 },
                        Edge { to: 3, cost: 4.0 },
                        Edge { to: 4, cost: 6.0 },
                        Edge { to: 5, cost: 5.0 },
                        Edge { to: 6, cost: 5.0 },
                        Edge { to: 7, cost: 5.0 },
                        Edge { to: 8, cost: 4.0 },
                        Edge { to: 9, cost: 5.0 },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge { to: 0, cost: 7.0 },
                        Edge { to: 1, cost: 6.0 },
                        Edge { to: 2, cost: 4.0 },
                        Edge { to: 4, cost: 5.0 },
                        Edge { to: 5, cost: 6.0 },
                        Edge { to: 6, cost: 6.0 },
                        Edge { to: 7, cost: 4.0 },
                        Edge { to: 8, cost: 2.0 },
                        Edge { to: 9, cost: 2.0 },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge { to: 0, cost: 4.0 },
                        Edge { to: 1, cost: 8.0 },
                        Edge { to: 2, cost: 6.0 },
                        Edge { to: 3, cost: 5.0 },
                        Edge { to: 5, cost: 9.0 },
                        Edge { to: 6, cost: 6.0 },
                        Edge { to: 7, cost: 3.0 },
                        Edge { to: 8, cost: 6.0 },
                        Edge { to: 9, cost: 5.0 },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge { to: 0, cost: 2.0 },
                        Edge { to: 1, cost: 3.0 },
                        Edge { to: 2, cost: 5.0 },
                        Edge { to: 3, cost: 6.0 },
                        Edge { to: 4, cost: 9.0 },
                        Edge { to: 6, cost: 4.0 },
                        Edge { to: 7, cost: 2.0 },
                        Edge { to: 8, cost: 3.0 },
                        Edge { to: 9, cost: 6.0 },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge { to: 0, cost: 4.0 },
                        Edge { to: 1, cost: 2.0 },
                        Edge { to: 2, cost: 5.0 },
                        Edge { to: 3, cost: 6.0 },
                        Edge { to: 4, cost: 6.0 },
                        Edge { to: 5, cost: 4.0 },
                        Edge { to: 7, cost: 5.0 },
                        Edge { to: 8, cost: 3.0 },
                        Edge { to: 9, cost: 4.0 },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge { to: 0, cost: 5.0 },
                        Edge { to: 1, cost: 7.0 },
                        Edge { to: 2, cost: 5.0 },
                        Edge { to: 3, cost: 4.0 },
                        Edge { to: 4, cost: 3.0 },
                        Edge { to: 5, cost: 2.0 },
                        Edge { to: 6, cost: 5.0 },
                        Edge { to: 8, cost: 5.0 },
                        Edge { to: 9, cost: 7.0 },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge { to: 0, cost: 5.0 },
                        Edge { to: 1, cost: 1.0 },
                        Edge { to: 2, cost: 4.0 },
                        Edge { to: 3, cost: 2.0 },
                        Edge { to: 4, cost: 6.0 },
                        Edge { to: 5, cost: 3.0 },
                        Edge { to: 6, cost: 3.0 },
                        Edge { to: 7, cost: 5.0 },
                        Edge { to: 9, cost: 6.0 },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge { to: 0, cost: 5.0 },
                        Edge { to: 1, cost: 6.0 },
                        Edge { to: 2, cost: 5.0 },
                        Edge { to: 3, cost: 2.0 },
                        Edge { to: 4, cost: 5.0 },
                        Edge { to: 5, cost: 6.0 },
                        Edge { to: 6, cost: 4.0 },
                        Edge { to: 7, cost: 7.0 },
                        Edge { to: 8, cost: 6.0 },
                    ],
                },
            ],
        };
        let nm: NAMatrix = (&graph).into();
        // Starting Node 0
        let (distance, path) = single_nearest_neighbour(&nm, 0);
        assert_eq!(distance, 31.0);
        assert_eq!(path, vec![0, 5, 7, 4, 3, 8, 1, 2, 6, 9]);
        // Starting Node 1
        let (distance, path) = single_nearest_neighbour(&nm, 1);
        assert_eq!(distance, 28.0);
        assert_eq!(path, vec![1, 8, 3, 9, 6, 0, 5, 7, 4, 2]);
        // Starting Node 2
        let (distance, path) = single_nearest_neighbour(&nm, 2);
        assert_eq!(distance, 28.0);
        assert_eq!(path, vec![2, 1, 8, 3, 9, 6, 0, 5, 7, 4]);
        // Starting Node 3
        let (distance, path) = single_nearest_neighbour(&nm, 3);
        assert_eq!(distance, 30.0);
        assert_eq!(path, vec![3, 8, 1, 2, 5, 0, 4, 7, 6, 9]);
        // Starting Node 4
        let (distance, path) = single_nearest_neighbour(&nm, 4);
        assert_eq!(distance, 29.0);
        assert_eq!(path, vec![4, 7, 5, 0, 6, 1, 8, 3, 9, 2]);
        // Starting Node 5
        let (distance, path) = single_nearest_neighbour(&nm, 5);
        assert_eq!(distance, 33.0);
        assert_eq!(path, vec![5, 0, 4, 7, 3, 8, 1, 2, 6, 9]);
        // Starting Node 6
        let (distance, path) = single_nearest_neighbour(&nm, 6);
        assert_eq!(distance, 30.0);
        assert_eq!(path, vec![6, 1, 8, 3, 9, 0, 5, 7, 4, 2]);
        // Starting Node 7
        let (distance, path) = single_nearest_neighbour(&nm, 7);
        assert_eq!(distance, 34.0);
        assert_eq!(path, vec![7, 5, 0, 4, 3, 8, 1, 2, 6, 9]);

        // Check that total is 27 and one of the best two
        let (distance, path) = nearest_neighbour::<{ computation_mode::SEQ_COMPUTATION }>(&nm);
        assert_eq!(distance, 28.0);
        assert!(
            path == vec![1, 8, 3, 9, 6, 0, 5, 7, 4, 2]
                || path == vec![2, 1, 8, 3, 9, 6, 0, 5, 7, 4]
        );

        // Are they the same?
        let (distance2, path2) = nearest_neighbour::<{ computation_mode::PAR_COMPUTATION }>(&nm);
        assert_eq!(distance, distance2);
        assert!(
            path2 == vec![1, 8, 3, 9, 6, 0, 5, 7, 4, 2]
                || path2 == vec![2, 1, 8, 3, 9, 6, 0, 5, 7, 4]
        );

        // That that we are below or equal to the perfect solution
        let (perfect_d, _) = first_improved_solver(&nm);
        assert!(perfect_d <= distance);
    }
}
