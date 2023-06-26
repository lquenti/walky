#[cfg(feature = "mpi")]
use std::{
    mem::transmute,
    slice::{from_raw_parts, from_raw_parts_mut},
};

use rand::{seq::SliceRandom, thread_rng};
use rayon::prelude::*;

#[cfg(feature = "mpi")]
use mpi::collective::Root;
#[cfg(feature = "mpi")]
use mpi::topology::*;
#[cfg(feature = "mpi")]
use nalgebra::{Dyn, Matrix};

use crate::{computation_mode::*, datastructures::NAMatrix};

fn randomized_matching(vertices: &mut [usize]) {
    vertices.shuffle(&mut thread_rng())
}

// constructs a randomly selected matching and
// does `tries` many local-search optimizations
// on randomly selected pairs of edges
pub fn approx_min_cost_matching<const MODE: usize>(
    graph: &NAMatrix,
    mut vertices: Vec<usize>,
    tries: usize,
) -> Vec<[usize; 2]> {
    randomized_matching(&mut vertices);
    let mut matching: Vec<[usize; 2]> = vertices
        .chunks_exact(2)
        .map(|chunk| [chunk[0], chunk[1]])
        .collect();

    match MODE {
        SEQ_COMPUTATION => improve_matching(graph, matching.as_mut_slice(), tries),
        PAR_COMPUTATION => par_improve_matching(graph, matching.as_mut_slice(), tries),
        #[cfg(feature = "mpi")]
        MPI_COMPUTATION => mpi_improve_matching(graph, matching.as_mut_slice(), tries),
        _ => panic_on_invaid_mode::<MODE>(),
    }
    matching
}

// `graph`: a reference to the underlying graph
//
// `matching`: a mutable reference to a matching of vertices in the graph. Needs to satisfy
// `matching.len() >= 2`
//
// `tries`: the number of repetitions of the following procedure:
//
// randomly choose two distinct pairs of edges:
// given the situation:
//
// 0 -- 1
//
// 2 -- 3
//
// the function evaluates this two alternative matchings:
//
// 0    1
// |    |
// 2    3
//
// and
//
// 0     1
//  \   /
//   \ /
//    X
//   / \
//  /   \
// 2     3
fn improve_matching(graph: &NAMatrix, matching: &mut [[usize; 2]], tries: usize) {
    // cannot improve a matching with 0 or 1 edge
    if matching.len() <= 1 {
        return;
    }

    let mut rng = thread_rng();

    for _ in 0..tries {
        matching.shuffle(&mut rng);

        // look at consecutive pairs of edges in the matching
        // (ignore the last single edge in a matching of odd number of edges)
        for chunk in matching.chunks_exact_mut(2) {
            // safety: the pointers point to valid memory in the matching,
            // and are non-overlapping.
            // Here unsafe code is used to create two mutable references
            // to two distinct cells of the same array
            let edg01 = unsafe { chunk.as_mut_ptr().offset(0).as_mut().unwrap() };
            let edg23 = unsafe { chunk.as_mut_ptr().offset(1).as_mut().unwrap() };

            optimize_two_matching(edg01, edg23, graph);
        }
    }
}

// given the situation:
//
// 0 -- 1 (i.e. `edg01`)
//
// 2 -- 3 (i.e. `edg23`)
//
// the function evaluates this two alternative matchings:
//
// 0    1
// |    |
// 2    3
//
// and
//
// 0     1
//  \   /
//   \ /
//    X
//   / \
//  /   \
// 2     3
#[inline]
fn optimize_two_matching(edg01: &mut [usize; 2], edg23: &mut [usize; 2], graph: &NAMatrix) {
    let v0 = edg01[0];
    let v1 = edg01[1];
    let v2 = edg23[0];
    let v3 = edg23[1];

    // base case cost
    let cost_edg01 = graph[(v0, v1)];
    let cost_edg23 = graph[(v2, v3)];
    let cost_0 = cost_edg01 + cost_edg23;

    // case 1
    let cost_edg02 = graph[(v0, v2)];
    let cost_edg13 = graph[(v1, v3)];
    let cost_1 = cost_edg02 + cost_edg13;

    // case 2
    let cost_edg03 = graph[(v0, v3)];
    let cost_edg12 = graph[(v1, v2)];
    let cost_2 = cost_edg03 + cost_edg12;

    // safety: the pointers point to valid memory in the matching,
    // and they point to non-overlapping memory,
    // hence no write-conflicts can occour
    if cost_1 < cost_0 {
        if cost_2 < cost_1 {
            // case 2 is optimal
            *edg01 = [v0, v2];
            *edg23 = [v1, v2];
        } else {
            // case 1 is optimal
            *edg01 = [v0, v2];
            *edg23 = [v1, v3];
        }
    } else {
        // cost_1 >= cost_0
        if cost_2 < cost_0 {
            // case 2 is optimal
            *edg01 = [v0, v3];
            *edg23 = [v1, v2];
        } // else base case is optimal
    }
}

/// parallelized version of [`improve_matching`]
fn par_improve_matching(graph: &NAMatrix, matching: &mut [[usize; 2]], tries: usize) {
    // step 1: generate pairs of edges, by shuffling the matching and taking chunks of size 2 as
    // pairs of edges
    let mut rng = thread_rng();

    for _ in 0..tries {
        matching.shuffle(&mut rng);

        // look at consecutive pairs of edges in the matching
        // (ignore the last single edge in a matching of odd number of edges)
        matching
            .par_chunks_exact_mut(2)
            .with_min_len(1000)
            .for_each(|chunk| {
                // safety: the pointers point to valid memory in the matching,
                // and are non-overlapping.
                // Here unsafe code is used to create two mutable references
                // to two distinct cells of the same array
                let edg01 = unsafe { chunk.as_mut_ptr().offset(0).as_mut().unwrap() };
                let edg23 = unsafe { chunk.as_mut_ptr().offset(1).as_mut().unwrap() };

                optimize_two_matching(edg01, edg23, graph);
            });
    }
}

#[cfg(feature = "mpi")]
pub fn mpi_improve_matching(graph: &NAMatrix, matching: &mut [[usize; 2]], tries: usize) {
    use mpi::traits::*;

    let world = SystemCommunicator::world();
    let size = world.size();
    let rank = world.rank();
    let root_process = world.process_at_rank(crate::ROOT_RANK);
    const COST_TAG: mpi::Tag = 0;
    const MATCHING_TAG: mpi::Tag = 1;

    let mut tries_copy = tries;
    if rank == crate::ROOT_RANK {
        bootstrap_mpi_matching_calc(&root_process, matching, rank, &mut tries_copy, graph);
    }

    // for every node / process:
    // try a different matching and randomized improvement
    //improve_matching(graph, matching, tries);
    improve_matching(graph, matching, tries);
    let cost: f64 = matching.iter().map(|&[i, j]| graph[(i, j)]).sum();
    let mut min_cost = cost;

    let mut best_matching_singletons: Vec<usize> = matching.iter().flat_map(|&edg| edg).collect();

    if rank != crate::ROOT_RANK {
        root_process.send_with_tag(&cost, COST_TAG);
        // create a mutable slice over the data of `matching`.
        // If matching is a slice of length n and type `&mut [[usize;2]]`,
        // then matching_singletons is a slice of length 2n and type `&mut [usize]`
        // safety: the memory is valid
        let matching_singletons =
            unsafe { from_raw_parts(&matching[0][0] as *const usize, matching.len() * 2) };
        root_process.send_with_tag(matching_singletons, MATCHING_TAG);
    } else {
        // if rank == crate::ROOT_RANK

        // the matchings will be sent as a Vec<usize>, because [usize; 2] toes not implement
        // the trait mpi::Equivalence

        // find the best solution across all nodes
        for rk in 1..size {
            let (other_cost, _status_cost) =
                world.process_at_rank(rk).receive_with_tag::<f64>(COST_TAG);
            let (msg, _status) = world
                .process_at_rank(rk)
                .receive_vec_with_tag::<usize>(MATCHING_TAG);
            if other_cost < min_cost {
                min_cost = other_cost;
                best_matching_singletons = msg;
            }
        }
    }

    // broadcast best solution from the root_node to all nodes
    root_process.broadcast_into(&mut min_cost);
    root_process.broadcast_into(&mut best_matching_singletons);

    // copy the matching data from `matching_singletons` back into `matching`.
    //
    // If `matching_singletons` has the following content:
    // `[0, 1, 2, 3, ...]` then the vertices will be grouped into edges like this:
    // `[[0,1], [2,3], ...]`.
    matching.copy_from_slice(unsafe {
        from_raw_parts(
            transmute::<*const usize, *const [usize; 2]>(best_matching_singletons.as_ptr()),
            matching.len(),
        )
    });
}

/// Distribute the initial matching
/// from the root_node to all other processes.
///
/// `root_process`: handle of the root process
///
/// `matching`: If called on the root process: should be the matching.
/// If called from any other process: can be chosen arbitrarily, as it will be overwritten from the
/// root process
///
/// `rank`: rank of the current process
///
/// `tries`: If called on the root process: should contain the number of tries.
/// If called from any other process: can be chosen arbitrarily, as it will be overwritten from the
/// root process
///
/// `graph`: If called on the root process: should reference the graph.
/// If called from any other process: can be chosen arbitrarily.
#[cfg(feature = "mpi")]
pub fn bootstrap_mpi_matching_calc<C: Communicator>(
    root_process: &Process<C>,
    matching: &mut [[usize; 2]],
    rank: Rank,
    tries: &mut usize,
    graph: &NAMatrix,
) -> (Vec<[usize; 2]>, NAMatrix) {
    // broadcast tries from the root node to all processes

    root_process.broadcast_into(tries);

    let mut matching_size = matching.len();
    broaadcast_matching_size(root_process, &mut matching_size);

    // broadcast the NAMatrix dim from the root process to all processes
    use crate::datastructures::AdjacencyMatrix;
    let mut dim = graph.dim();
    broadcast_dim(root_process, &mut dim);

    // create storage for the incoming matching
    let mut matching_vec = vec![[0usize; 2]; matching_size];
    if rank == crate::ROOT_RANK {
        matching_vec.copy_from_slice(matching);
    }
    broadcast_matching(root_process, &mut matching_vec);

    // broadcast the NAMatrix from the root process to all processes
    let mut matrix_vec = vec![0.; dim * dim];
    if rank == crate::ROOT_RANK {
        matrix_vec.copy_from_slice(graph.data.as_vec())
    }
    broadcast_matrix_vec(root_process, &mut matrix_vec);
    let matrix = NAMatrix(Matrix::from_vec_generic(Dyn(dim), Dyn(dim), matrix_vec));

    (matching_vec, matrix)
}

/// broadcast the size of the matching to all processes
#[cfg(feature = "mpi")]
fn broaadcast_matching_size<C: Communicator>(root_process: &Process<C>, matching_size: &mut usize) {
    root_process.broadcast_into(matching_size);
}

#[cfg(feature = "mpi")]
fn broadcast_dim<C: Communicator>(root_process: &Process<C>, dim: &mut usize) {
    root_process.broadcast_into(dim);
}

#[cfg(feature = "mpi")]
fn broadcast_matching<C: Communicator>(root_process: &Process<C>, matching: &mut Vec<[usize; 2]>) {
    // cast the matching slice type from `&mut [[usize;2]]` to `&mut [usize]`
    // to be able to send it via MPI
    //let matching_singletons = if rank == 0 {
    //    unsafe { from_raw_parts_mut(&mut matching[0][0] as *mut usize, matching.len() * 2) }
    //} else {
    //    unsafe { from_raw_parts_mut(&mut matching_vec[0][0] as *mut usize, matching.len() * 2) }
    //};
    let matching_singletons =
        unsafe { from_raw_parts_mut(&mut matching[0][0] as *mut usize, matching.len() * 2) };
    // distribute the matching from the root process to all other processes
    root_process.broadcast_into(matching_singletons);
}

#[cfg(feature = "mpi")]
fn broadcast_matrix_vec<C: Communicator>(root_process: &Process<C>, matrix_vec: &mut Vec<f64>) {
    root_process.broadcast_into(matrix_vec);
}
