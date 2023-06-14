use std::mem::swap;

use rand::{seq::SliceRandom, thread_rng};
use rayon::prelude::*;

use crate::{computation_mode::*, datastructures::NAMatrix};

fn randomized_matching(vertices: &mut [usize]) {
    vertices.shuffle(&mut thread_rng())
}

pub fn naive_approx_min_cost_matching(
    graph: &NAMatrix,
    mut vertices: Vec<usize>,
    tries: usize,
) -> Vec<(usize, usize)> {
    let mut min = f64::INFINITY;
    let mut min_shuffle = vec![0; vertices.len()];

    for _ in 0..tries {
        randomized_matching(&mut vertices);
        let sum_cost: f64 = vertices
            .chunks_exact(2)
            .map(|chunk| graph[(chunk[0], chunk[1])])
            .sum();
        if min > sum_cost {
            min = sum_cost;
            min_shuffle.copy_from_slice(vertices.as_slice())
        }
    }
    min_shuffle
        .chunks_exact(2)
        .map(|chunk| (chunk[0], chunk[1]))
        .collect()
}

// constructs a randomly selected matching and
// does `tries` many local-search optimizations
// on randomly selected pairs of edges
pub fn approx_min_cost_matching<const MODE: usize>(
    graph: &NAMatrix,
    mut vertices: Vec<usize>,
    tries: usize,
) -> Vec<(usize, usize)> {
    randomized_matching(&mut vertices);
    let mut matching: Vec<(usize, usize)> = vertices
        .chunks_exact(2)
        .map(|chunk| (chunk[0], chunk[1]))
        .collect();

    match MODE {
        SEQ_COMPUTATION => improve_matching(graph, matching.as_mut_slice(), tries),
        PAR_COMPUTATION => par_improve_matching(graph, matching.as_mut_slice(), tries),
        MPI_COMPUTATION => todo!(),
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
fn improve_matching(graph: &NAMatrix, matching: &mut [(usize, usize)], tries: usize) {
    debug_assert!(
        matching.len() >= 2,
        "The matching should have at least two edges"
    );
    let mut rng = thread_rng();

    for _ in 0..tries {
        //let mut edg01 = matching.choose(&mut rng).unwrap() as *const _ as *mut (usize, usize);
        //let edg23 = matching.choose(&mut rng).unwrap() as *const _ as *mut (usize, usize);
        //while edg01 == edg23 {
        //    edg01 = matching.choose(&mut rng).unwrap() as *const _ as *mut (usize, usize);
        //}
        //debug_assert_ne!(edg01, edg23, "The pointers should be distinct");
        //
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
fn optimize_two_matching(edg01: &mut (usize, usize), edg23: &mut (usize, usize), graph: &NAMatrix) {
    let v0 = edg01.0;
    let v1 = edg01.1;
    let v2 = edg23.0;
    let v3 = edg23.1;

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
            *edg01 = (v0, v2);
            *edg23 = (v1, v2);
        } else {
            // case 1 is optimal
            *edg01 = (v0, v2);
            *edg23 = (v1, v3);
        }
    } else {
        // cost_1 >= cost_0
        if cost_2 < cost_0 {
            // case 2 is optimal
            *edg01 = (v0, v3);
            *edg23 = (v1, v2);
        } // else base case is optimal
    }
}

/// parallelized version of [`improve_matching`]
fn par_improve_matching(graph: &NAMatrix, matching: &mut [(usize, usize)], tries: usize) {
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
