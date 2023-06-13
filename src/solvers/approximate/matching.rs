use rand::{seq::SliceRandom, thread_rng};

use crate::datastructures::NAMatrix;

fn randomized_matching(vertices: &mut [usize]) {
    vertices.shuffle(&mut thread_rng())
}

pub fn approx_min_cost_matching(
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
