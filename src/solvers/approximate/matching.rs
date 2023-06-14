use rand::{seq::SliceRandom, thread_rng};

use crate::datastructures::NAMatrix;

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
pub fn approx_min_cost_matching(
    graph: &NAMatrix,
    mut vertices: Vec<usize>,
    tries: usize,
) -> Vec<(usize, usize)> {
    randomized_matching(&mut vertices);
    let mut matching: Vec<(usize, usize)> = vertices
        .chunks_exact(2)
        .map(|chunk| (chunk[0], chunk[1]))
        .collect();

    improve_matching(graph, matching.as_mut_slice(), tries);
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
        let mut edg01 = matching.choose(&mut rng).unwrap() as *const _ as *mut (usize, usize);
        let edg23 = matching.choose(&mut rng).unwrap() as *const _ as *mut (usize, usize);
        while edg01 == edg23 {
            edg01 = matching.choose(&mut rng).unwrap() as *const _ as *mut (usize, usize);
        }
        debug_assert_ne!(edg01, edg23, "The pointers should be distinct");

        // safety: the pointers point to valid memory in the matching
        let v0 = unsafe { (*edg01).0 };
        let v1 = unsafe { (*edg01).1 };
        let v2 = unsafe { (*edg23).0 };
        let v3 = unsafe { (*edg23).1 };

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
                unsafe {
                    *edg01 = (v0, v2);
                    *edg23 = (v1, v2);
                }
            } else {
                // case 1 is optimal
                unsafe {
                    *edg01 = (v0, v2);
                    *edg23 = (v1, v3);
                }
            }
        } else {
            // cost_1 >= cost_0
            if cost_2 < cost_0 {
                // case 2 is optimal
                unsafe {
                    *edg01 = (v0, v3);
                    *edg23 = (v1, v2);
                }
            } // else base case is optimal
        }
    }
}
