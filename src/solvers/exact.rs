//! Exact methods to solve the TSP problem.

use std::{collections::hash_map::DefaultHasher, hash::{Hash, Hasher}};

use crate::{
    datastructures::{AdjacencyMatrix, NAMatrix, Path, Solution},
    mst,
};

use rustc_hash::FxHashMap;

/// Simplest possible solution: just go through all the nodes in order.
/// No further optimizations. See [`next_permutation`] on how the permutations are generated.
///
/// Runtime: Theta(n * n!)
pub fn naive_solver<T>(graph_matrix: &T) -> Solution
where
    T: AdjacencyMatrix,
{
    let n = graph_matrix.dim();
    let mut best_permutation: Path = (0..n).collect();
    let mut best_cost = f64::INFINITY;

    let mut current_permutation = best_permutation.clone();
    while next_permutation(&mut current_permutation) {
        let cost = graph_matrix.evaluate_circle(&current_permutation);
        if cost < best_cost {
            best_cost = cost;
            best_permutation = current_permutation.clone();
        }
    }
    (best_cost, best_permutation)
}

/// First improvement of [`naive_solver`]:
/// We fixate the first element to be at the start.
/// Since it is a cycle, we do not care where it starts.
///
/// Runtime: Theta(n * (n-1)!) = Theta(n!)
pub fn first_improved_solver<T>(graph_matrix: &T) -> Solution
where
    T: AdjacencyMatrix,
{
    let n = graph_matrix.dim();
    let mut best_permutation: Path = (0..n).collect();
    let mut best_cost = f64::INFINITY;

    let mut current_permutation = best_permutation.clone();
    while next_permutation(&mut current_permutation[1..]) {
        let cost = graph_matrix.evaluate_circle(&current_permutation);
        if cost < best_cost {
            best_cost = cost;
            best_permutation = current_permutation.clone();
        }
    }
    (best_cost, best_permutation)
}

/// Second improvement of [`naive_solver`]:
/// We carry along a partial sum whenever we change an element.
///
/// Since we want the prefix to stay the same, we also use recusrive enumeration.
/// This means that we unfortunately have to inline the traversal, making the function more
/// convoluted.
///
/// The complexity analysis gets tedious here, so we basically still have
/// Runtime: O(n!)
pub fn second_improved_solver<T>(graph_matrix: &T) -> Solution
where
    T: AdjacencyMatrix,
{
    let mut current_prefix = Vec::new();
    current_prefix.reserve(graph_matrix.dim());
    let mut result = (f64::INFINITY, Vec::new());
    _second_improved_solver_rec(graph_matrix, &mut current_prefix, 0.0, &mut result);
    result
}

/// The recursive function underlying [`second_improved_solver`]
///
/// Think of it as Depth first traversal and we evaluate at the leafs.
fn _second_improved_solver_rec<T>(
    graph_matrix: &T,
    current_prefix: &mut Path,
    current_cost: f64,
    result: &mut Solution,
) where
    T: AdjacencyMatrix,
{
    let n = graph_matrix.dim();
    let mut current_cost = current_cost; // Copy trait anyways

    // Base case: Is this one better?
    if current_prefix.len() == n {
        // Add the last edge, finishing the circle
        current_cost += graph_matrix.get(
            *current_prefix.last().unwrap(),
            *current_prefix.first().unwrap(),
        );

        let best_cost = result.0;
        if current_cost < best_cost {
            result.0 = current_cost;
            result.1 = current_prefix.clone();
        }
        return;
    }
    // Branch down with branching factor n-k, where k is the length of current_prefix
    for i in 0..n {
        // Prune if already in prefix
        if current_prefix.contains(&i) {
            continue;
        }

        current_prefix.push(i);
        // If this is a single element, we do not have an edge yet
        if current_prefix.len() == 1 {
            _second_improved_solver_rec(graph_matrix, current_prefix, current_cost, result);
            current_prefix.pop();
            continue;
        }

        // Calculate the cost of our new edge
        let from = current_prefix.len() - 2;
        let to = from + 1;
        let cost_last_edge = graph_matrix.get(current_prefix[from], current_prefix[to]);
        current_cost += cost_last_edge;

        // rec call
        _second_improved_solver_rec(graph_matrix, current_prefix, current_cost, result);

        // remove
        current_cost -= cost_last_edge;
        current_prefix.pop();
    }
}

/// Third improvement of [`naive_solver`]:
/// Prune a lot whenever the partial sum is already bigger than the previous optimum
pub fn third_improved_solver<T>(graph_matrix: &T) -> Solution
where
    T: AdjacencyMatrix,
{
    let mut current_prefix = Vec::new();
    current_prefix.reserve(graph_matrix.dim());
    let mut result = (f64::INFINITY, Vec::new());
    _third_improved_solver_rec(graph_matrix, &mut current_prefix, 0.0, &mut result);
    result
}

/// The recursive function underlying [`third_improved_solver`]
///
/// For the most part it is the same function as [`_second_improved_solver_rec`]. We could just not
/// parametrize it without a whole lot of dependency injection and indirection.
///
/// The only difference is that we prune once it is better than our previously computed best
/// solution.
fn _third_improved_solver_rec<T>(
    graph_matrix: &T,
    current_prefix: &mut Path,
    current_cost: f64,
    result: &mut Solution,
) where
    T: AdjacencyMatrix,
{
    // See [`_second_improved_solver_rec`] for more comments
    let n = graph_matrix.dim();
    let mut current_cost = current_cost;

    if current_prefix.len() == n {
        current_cost += graph_matrix.get(
            *current_prefix.last().unwrap(),
            *current_prefix.first().unwrap(),
        );

        let best_cost = result.0;
        if current_cost < best_cost {
            result.0 = current_cost;
            result.1 = current_prefix.clone();
        }
        return;
    }
    for i in 0..n {
        if current_prefix.contains(&i) {
            continue;
        }

        current_prefix.push(i);
        if current_prefix.len() == 1 {
            _third_improved_solver_rec(graph_matrix, current_prefix, current_cost, result);
            current_prefix.pop();
            continue;
        }

        let from = current_prefix.len() - 2;
        let to = from + 1;
        let cost_last_edge = graph_matrix.get(current_prefix[from], current_prefix[to]);
        current_cost += cost_last_edge;

        // HERE IS THE BIG DIFFERENCE TO [`_second_improved_solver_rec`]!
        if current_cost <= result.0 {
            _third_improved_solver_rec(graph_matrix, current_prefix, current_cost, result);
        }
        // This was it. This is the pruning

        current_cost -= cost_last_edge;
        current_prefix.pop();
    }
}

/// Fourth improvement of [`naive_solver`]:
/// We prune not only if our current path is bigger than the previously known minimum but already
/// before. Instead, we prune if (current path + a lower bound on the remaining vertices) >
/// previously known minimum.
///
/// This works since (Smol < TSP), thus if `current_path` and this smol Graph is ALREADY bigger than
/// the previously known minimum then `current_path` extended to a full TSP will DEFINITELY be
/// huger.
///
/// In this function, we use a Nearest Neighbour graph as a lower bound. This means, that starting
/// at a
///
/// Note that this "graph" is maybe a not even fully connected forest. If we require it to be fully
/// connected it is not obvious that this bound is lower than the TSP.
///
/// See the report for a more formal proof.
pub fn fourth_improved_solver<T>(graph_matrix: &T) -> Solution
where
    T: AdjacencyMatrix,
{
    let mut current_prefix = Vec::new();
    current_prefix.reserve(graph_matrix.dim());
    let mut result = (f64::INFINITY, Vec::new());
    _fourth_improved_solver_rec(graph_matrix, &mut current_prefix, 0.0, &mut result);
    result
}

/// The recursive function underlying [`fourth_improved_solver`]
///
/// As previous, it works like the version before, but this time it also uses a NN computation
/// before pruning.
fn _fourth_improved_solver_rec<T>(
    graph_matrix: &T,
    current_prefix: &mut Path,
    current_cost: f64,
    result: &mut Solution,
) where
    T: AdjacencyMatrix,
{
    let n = graph_matrix.dim();
    let mut current_cost = current_cost;

    // Base case: Is this one better?
    if current_prefix.len() == n {
        // Add the last edge, finishing the circle
        current_cost += graph_matrix.get(
            *current_prefix.last().unwrap(),
            *current_prefix.first().unwrap(),
        );

        let best_cost = result.0;
        if current_cost < best_cost {
            result.0 = current_cost;
            result.1 = current_prefix.clone();
        }
        return;
    }

    // Branch down with branching factor n-k, where k is the length of current_prefix
    for i in 0..n {
        // We do not visit twice
        if current_prefix.contains(&i) {
            continue;
        }

        current_prefix.push(i);
        // If this is a single element, we do not have an edge yet
        if current_prefix.len() == 1 {
            _fourth_improved_solver_rec(graph_matrix, current_prefix, current_cost, result);
            current_prefix.pop();
            continue;
        }

        // Calculate the cost of our new edge
        let from = current_prefix.len() - 2;
        let to = from + 1;
        let cost_last_edge = graph_matrix.get(current_prefix[from], current_prefix[to]);
        current_cost += cost_last_edge;

        // If our current sub-tour, together with a lower bound, is already bigger than the whole
        // tour the whole tour will definitely be bigger than our previous best version
        let lower_bound = compute_nn_of_remaining_vertices(graph_matrix, current_prefix, n);
        //println!("{:?}, {}, {}", current_prefix, current_cost, lower_bound);
        if current_cost + lower_bound <= result.0 {
            _fourth_improved_solver_rec(graph_matrix, current_prefix, current_cost, result);
        }

        // Remove the last edge
        current_cost -= cost_last_edge;
        current_prefix.pop();
    }
}

/// Compute the nearest neighbour for each not yet connected vertex of a given graph_matrix.
/// Note that this does not necessarily has to result in a traversable path with correct degrees.
/// In fact, there are no gurantees that this results in a fully connected graph as one there could
/// just be two vertices that connect to each other, i.e. `A -> B && B -> A`.
fn compute_nn_of_remaining_vertices<T>(graph_matrix: &T, subtour: &Path, n: usize) -> f64
where
    T: AdjacencyMatrix,
{
    let mut res = 0.0;
    for i in 0..n {
        // If it is part of the subtour, the vertex is already used.
        if subtour.contains(&i) {
            continue;
        }
        let mut shortest_edge_cost = f64::INFINITY;

        // Otherwise, we can start finding its nearest neighbour
        for j in (i + 1)..n {
            // we do not connect to ourself
            // Also we just connect within
            if subtour.contains(&j) {
                continue;
            }

            let cost = graph_matrix.get(i, j);
            // if it is lower we take it
            if cost < shortest_edge_cost {
                shortest_edge_cost = cost;
            }
        }
        // Without this check, a single node would have length infinity
        if shortest_edge_cost != f64::INFINITY {
            res += shortest_edge_cost;
        }
    }
    res
}

/// Fifth improvement of [`naive_solver`]:
/// Instead of using a NN-based graph for pruning as in [`fourth_improved_solver`] we instead opt
/// out to use an Minimal Spanning Tree (MST), which we compute for every step.
pub fn fifth_improved_solver(graph_matrix: &NAMatrix) -> Solution {
    let mut current_prefix = Vec::new();
    current_prefix.reserve(graph_matrix.dim());
    let mut result = (f64::INFINITY, Vec::new());
    _fifth_improved_solver_rec(graph_matrix, &mut current_prefix, 0.0, &mut result);
    result
}

/// The recursive function underlying [`fifth_improved_solver`]
///
/// This time we use a MST instead of NN
fn _fifth_improved_solver_rec(
    graph_matrix: &NAMatrix,
    current_prefix: &mut Path,
    current_cost: f64,
    result: &mut Solution,
) {
    let n = graph_matrix.dim();
    let mut current_cost = current_cost;

    // Base case: Is this one better?
    if current_prefix.len() == n {
        // Add the last edge, finishing the circle
        current_cost += graph_matrix.get(
            *current_prefix.last().unwrap(),
            *current_prefix.first().unwrap(),
        );

        let best_cost = result.0;
        if current_cost < best_cost {
            result.0 = current_cost;
            result.1 = current_prefix.clone();
        }
        return;
    }

    // Branch down with branching factor n-k, where k is the length of current_prefix
    for i in 0..n {
        // We do not visit twice
        if current_prefix.contains(&i) {
            continue;
        }

        current_prefix.push(i);
        // If this is a single element, we do not have an edge yet
        if current_prefix.len() == 1 {
            _fifth_improved_solver_rec(graph_matrix, current_prefix, current_cost, result);
            current_prefix.pop();
            continue;
        }

        // Calculate the cost of our new edge
        let from = current_prefix.len() - 2;
        let to = from + 1;
        let cost_last_edge = graph_matrix.get(current_prefix[from], current_prefix[to]);
        current_cost += cost_last_edge;

        // If our current sub-tour, together with a lower bound, is already bigger than the whole
        // tour the whole tour will definitely be bigger than our previous best version
        let lower_bound_graph =
            mst::prim_with_excluded_node_single_threaded(graph_matrix, current_prefix);
        let lower_bound = lower_bound_graph.acc_edge_cost();
        if current_cost + lower_bound <= result.0 {
            _fifth_improved_solver_rec(graph_matrix, current_prefix, current_cost, result);
        }

        // Remove the last edge
        current_cost -= cost_last_edge;
        current_prefix.pop();
    }
}

/// Sixth improvement of [`naive_solver`]:
/// Cache the aforementioned, computed MSTs in a Hashmap
pub fn sixth_improved_solver(graph_matrix: &NAMatrix) -> Solution {
    let mut current_prefix = Vec::new();
    current_prefix.reserve(graph_matrix.dim());
    let mut result = (f64::INFINITY, Vec::new());
    let mut mst_cache: FxHashMap<u64, f64> = FxHashMap::default();
    _sixth_improved_solver_rec(
        graph_matrix,
        &mut current_prefix,
        0.0,
        &mut result,
        &mut mst_cache,
    );
    result
}

/// The recursive function underlying [`sixth_improved_solver`]
///
/// Like the fifth, but we HashMap the MSTs
/// Note that we do not use the default Hashmap but instead FxHash from the core team
///
/// One Note: We use FxHashMap<u64, f64> instead of FxHashMap<Path, f64> for performance reasons.
/// Otherwise we would have to copy the vector every time we do a lookup.
///
/// Unfortunately, we also can't use FxHashMap<&Path, f64> because we mutate on the original Path
/// each iteration, resulting in one clone a iteration as well.
///
/// Lastly, Rusts default hasher has to be allocated each time we call it.
/// Thus, we use another hasher
fn _sixth_improved_solver_rec(
    graph_matrix: &NAMatrix,
    current_prefix: &mut Path,
    current_cost: f64,
    result: &mut Solution,
    mst_cache: &mut FxHashMap<u64, f64>,
) {
    let n = graph_matrix.dim();
    let mut current_cost = current_cost;

    // Base case: Is this one better?
    if current_prefix.len() == n {
        // Add the last edge, finishing the circle
        current_cost += graph_matrix.get(
            *current_prefix.last().unwrap(),
            *current_prefix.first().unwrap(),
        );

        let best_cost = result.0;
        if current_cost < best_cost {
            result.0 = current_cost;
            result.1 = current_prefix.clone();
        }
        return;
    }

    // Branch down with branching factor n-k, where k is the length of current_prefix
    for i in 0..n {
        // We do not visit twice
        if current_prefix.contains(&i) {
            continue;
        }

        current_prefix.push(i);
        // If this is a single element, we do not have an edge yet
        if current_prefix.len() == 1 {
            _sixth_improved_solver_rec(
                graph_matrix,
                current_prefix,
                current_cost,
                result,
                mst_cache,
            );
            current_prefix.pop();
            continue;
        }

        // Calculate the cost of our new edge
        let from = current_prefix.len() - 2;
        let to = from + 1;
        let cost_last_edge = graph_matrix.get(current_prefix[from], current_prefix[to]);
        current_cost += cost_last_edge;

        // If our current sub-tour, together with a lower bound, is already bigger than the whole
        // tour the whole tour will definitely be bigger than our previous best version
        let hash = {
            let mut hasher = DefaultHasher::new();
            current_prefix.hash(&mut hasher);
            hasher.finish()
        };
        let lower_bound = mst_cache.entry(hash).or_insert(
            mst::prim_with_excluded_node_single_threaded(graph_matrix, current_prefix)
                .acc_edge_cost(),
        );
        if current_cost + *lower_bound <= result.0 {
            _sixth_improved_solver_rec(
                graph_matrix,
                current_prefix,
                current_cost,
                result,
                mst_cache,
            );
        }

        // Remove the last edge
        current_cost -= cost_last_edge;
        current_prefix.pop();
    }
}

/// Finding the next permutation given an array.
/// Based on [Nayuki](https://www.nayuki.io/page/next-lexicographical-permutation-algorithm)
///
/// It ends when the array is only decreasing.
/// Thus, in order to get all permutations of [n], start with (1,2,...,n)
fn next_permutation<T: Ord>(array: &mut [T]) -> bool {
    // Find non-increasing suffix
    if array.is_empty() {
        return false;
    }
    let mut i: usize = array.len() - 1;
    while i > 0 && array[i - 1] >= array[i] {
        i -= 1;
    }
    if i == 0 {
        return false;
    }

    // Find successor to pivot
    let mut j: usize = array.len() - 1;
    while array[j] <= array[i - 1] {
        j -= 1;
    }
    array.swap(i - 1, j);

    // Reverse suffix
    array[i..].reverse();
    true
}

#[cfg(test)]
mod exact_solver {
    use approx::relative_eq;

    use super::*;
    use crate::datastructures::{NAMatrix, VecMatrix};
    use crate::parser::{Edge, Graph, Vertex};

    use lazy_static::lazy_static;

    lazy_static! {
        static ref SMALL_FLOAT_GRAPH: Graph = Graph {
            vertices: vec![
                Vertex {
                    edges: vec![
                        Edge {
                            to: 1,
                            cost: 13.215648444670196,
                        },
                        Edge {
                            to: 2,
                            cost: 9.674413400408712,
                        },
                        Edge {
                            to: 3,
                            cost: 1.0970596862282833,
                        },
                        Edge {
                            to: 4,
                            cost: 16.098684067859647,
                        },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge {
                            to: 0,
                            cost: 13.215648444670196,
                        },
                        Edge {
                            to: 2,
                            cost: 12.221639547131913,
                        },
                        Edge {
                            to: 3,
                            cost: 17.306826463341803,
                        },
                        Edge {
                            to: 4,
                            cost: 8.321138140452149,
                        },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge {
                            to: 0,
                            cost: 9.674413400408712,
                        },
                        Edge {
                            to: 1,
                            cost: 12.221639547131913,
                        },
                        Edge {
                            to: 3,
                            cost: 4.6376150266768885,
                        },
                        Edge {
                            to: 4,
                            cost: 15.838066781407072,
                        },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge {
                            to: 0,
                            cost: 1.0970596862282833,
                        },
                        Edge {
                            to: 1,
                            cost: 17.306826463341803,
                        },
                        Edge {
                            to: 2,
                            cost: 4.6376150266768885,
                        },
                        Edge {
                            to: 4,
                            cost: 6.102211932446107,
                        },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge {
                            to: 0,
                            cost: 16.098684067859647,
                        },
                        Edge {
                            to: 1,
                            cost: 8.321138140452149,
                        },
                        Edge {
                            to: 2,
                            cost: 15.838066781407072,
                        },
                        Edge {
                            to: 3,
                            cost: 6.102211932446107,
                        },
                    ],
                },
            ],
        };
    }

    lazy_static! {
        static ref BIG_FLOAT_GRAPH: Graph = Graph {
            vertices: vec![
                Vertex {
                    edges: vec![
                        Edge {
                            to: 1,
                            cost: 5.357166694956081,
                        },
                        Edge {
                            to: 2,
                            cost: 12.673287166274285,
                        },
                        Edge {
                            to: 3,
                            cost: 15.392922519581575,
                        },
                        Edge {
                            to: 4,
                            cost: 1.8824165228898004,
                        },
                        Edge {
                            to: 5,
                            cost: 1.0673823908781577,
                        },
                        Edge {
                            to: 6,
                            cost: 8.668326879490138,
                        },
                        Edge {
                            to: 7,
                            cost: 18.956348946357103,
                        },
                        Edge {
                            to: 8,
                            cost: 5.399642479870355,
                        },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge {
                            to: 0,
                            cost: 5.357166694956081,
                        },
                        Edge {
                            to: 2,
                            cost: 11.139733539749999,
                        },
                        Edge {
                            to: 3,
                            cost: 1.661032458795486,
                        },
                        Edge {
                            to: 4,
                            cost: 18.702631945210115,
                        },
                        Edge {
                            to: 5,
                            cost: 3.847655828276122,
                        },
                        Edge {
                            to: 6,
                            cost: 15.73510598766653,
                        },
                        Edge {
                            to: 7,
                            cost: 0.24655608854276645,
                        },
                        Edge {
                            to: 8,
                            cost: 4.321598762165737,
                        },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge {
                            to: 0,
                            cost: 12.673287166274285,
                        },
                        Edge {
                            to: 1,
                            cost: 11.139733539749999,
                        },
                        Edge {
                            to: 3,
                            cost: 2.1803729313885345,
                        },
                        Edge {
                            to: 4,
                            cost: 16.313099247004377,
                        },
                        Edge {
                            to: 5,
                            cost: 5.585527987185975,
                        },
                        Edge {
                            to: 6,
                            cost: 8.932741722100753,
                        },
                        Edge {
                            to: 7,
                            cost: 12.6998544424725,
                        },
                        Edge {
                            to: 8,
                            cost: 9.05733402266841,
                        },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge {
                            to: 0,
                            cost: 15.392922519581575,
                        },
                        Edge {
                            to: 1,
                            cost: 1.661032458795486,
                        },
                        Edge {
                            to: 2,
                            cost: 2.1803729313885345,
                        },
                        Edge {
                            to: 4,
                            cost: 3.340513012587236,
                        },
                        Edge {
                            to: 5,
                            cost: 1.46551068868777,
                        },
                        Edge {
                            to: 6,
                            cost: 2.6426709551798355,
                        },
                        Edge {
                            to: 7,
                            cost: 4.492948831722041,
                        },
                        Edge {
                            to: 8,
                            cost: 13.41757522658849,
                        },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge {
                            to: 0,
                            cost: 1.8824165228898004,
                        },
                        Edge {
                            to: 1,
                            cost: 18.702631945210115,
                        },
                        Edge {
                            to: 2,
                            cost: 16.313099247004377,
                        },
                        Edge {
                            to: 3,
                            cost: 3.340513012587236,
                        },
                        Edge {
                            to: 5,
                            cost: 9.568614854660245,
                        },
                        Edge {
                            to: 6,
                            cost: 6.849461885327388,
                        },
                        Edge {
                            to: 7,
                            cost: 7.455992424446736,
                        },
                        Edge {
                            to: 8,
                            cost: 19.61866966591363,
                        },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge {
                            to: 0,
                            cost: 1.0673823908781577,
                        },
                        Edge {
                            to: 1,
                            cost: 3.847655828276122,
                        },
                        Edge {
                            to: 2,
                            cost: 5.585527987185975,
                        },
                        Edge {
                            to: 3,
                            cost: 1.46551068868777,
                        },
                        Edge {
                            to: 4,
                            cost: 9.568614854660245,
                        },
                        Edge {
                            to: 6,
                            cost: 7.516298524772413,
                        },
                        Edge {
                            to: 7,
                            cost: 17.155030102652216,
                        },
                        Edge {
                            to: 8,
                            cost: 17.46182408314527,
                        },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge {
                            to: 0,
                            cost: 8.668326879490138,
                        },
                        Edge {
                            to: 1,
                            cost: 15.73510598766653,
                        },
                        Edge {
                            to: 2,
                            cost: 8.932741722100753,
                        },
                        Edge {
                            to: 3,
                            cost: 2.6426709551798355,
                        },
                        Edge {
                            to: 4,
                            cost: 6.849461885327388,
                        },
                        Edge {
                            to: 5,
                            cost: 7.516298524772413,
                        },
                        Edge {
                            to: 7,
                            cost: 5.959449216135542,
                        },
                        Edge {
                            to: 8,
                            cost: 11.172366336098495,
                        },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge {
                            to: 0,
                            cost: 18.956348946357103,
                        },
                        Edge {
                            to: 1,
                            cost: 0.24655608854276645,
                        },
                        Edge {
                            to: 2,
                            cost: 12.6998544424725,
                        },
                        Edge {
                            to: 3,
                            cost: 4.492948831722041,
                        },
                        Edge {
                            to: 4,
                            cost: 7.455992424446736,
                        },
                        Edge {
                            to: 5,
                            cost: 17.155030102652216,
                        },
                        Edge {
                            to: 6,
                            cost: 5.959449216135542,
                        },
                        Edge {
                            to: 8,
                            cost: 8.168048838216963,
                        },
                    ],
                },
                Vertex {
                    edges: vec![
                        Edge {
                            to: 0,
                            cost: 5.399642479870355,
                        },
                        Edge {
                            to: 1,
                            cost: 4.321598762165737,
                        },
                        Edge {
                            to: 2,
                            cost: 9.05733402266841,
                        },
                        Edge {
                            to: 3,
                            cost: 13.41757522658849,
                        },
                        Edge {
                            to: 4,
                            cost: 19.61866966591363,
                        },
                        Edge {
                            to: 5,
                            cost: 17.46182408314527,
                        },
                        Edge {
                            to: 6,
                            cost: 11.172366336098495,
                        },
                        Edge {
                            to: 7,
                            cost: 8.168048838216963,
                        },
                    ],
                },
            ],
        };
    }

    lazy_static! {
        static ref SMALL_INT_GRAPH: Graph = Graph {
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
    }

    /// Checks whether two paths describe the same undirected circle.
    /// This means that it is agnostic of
    /// - starting Vertex
    /// - direction
    fn is_same_undirected_circle(seq1: &Path, seq2: &Path) -> bool {
        if seq1.len() != seq2.len() {
            return false;
        }

        let n = seq1.len();

        // Generate all possible rotations of seq1 in both directions
        let rotations = (0..n).map(|i| {
            seq1[i..]
                .iter()
                .chain(seq1[..i].iter())
                .copied()
                .collect::<Path>()
        });
        let reversed_rotations = rotations
            .clone()
            .map(|xs| xs.into_iter().rev().collect::<Path>());

        // Check if any rotation matches
        for rotation in rotations.chain(reversed_rotations) {
            if rotation[..] == seq2[..] {
                return true;
            }
        }

        false
    }

    #[test]
    fn test_is_same_undirected_circle() {
        assert!(is_same_undirected_circle(
            &vec![1, 2, 3, 4, 5, 6],
            &vec![4, 3, 2, 1, 6, 5]
        ));
    }

    #[test]
    fn test_not_same_undirected_circle() {
        assert!(!is_same_undirected_circle(
            &vec![1, 2, 3, 4, 5, 6],
            &vec![4, 3, 2, 6, 1, 5]
        ));
    }

    #[test]
    fn test_float_tsp_vecmatrix() {
        // Test each solution
        let gm: VecMatrix = SMALL_FLOAT_GRAPH.clone().into();
        for f in [
            naive_solver,
            first_improved_solver,
            second_improved_solver,
            third_improved_solver,
            fourth_improved_solver,
        ]
        .iter()
        {
            let (best_cost, best_permutation) = f(&gm);
            assert!(relative_eq!(37.41646270666716, best_cost));
            assert!(is_same_undirected_circle(
                &vec![0, 3, 4, 1, 2],
                &best_permutation
            ));
        }
    }

    #[test]
    fn test_float_tsp_namatrix() {
        // Test each solution
        let gm: NAMatrix = <NAMatrix as From<&Graph>>::from(&SMALL_FLOAT_GRAPH);
        for f in [
            naive_solver,
            first_improved_solver,
            second_improved_solver,
            third_improved_solver,
            fourth_improved_solver,
            fifth_improved_solver,
            sixth_improved_solver,
        ]
        .iter()
        {
            let (best_cost, best_permutation) = f(&gm);
            assert!(relative_eq!(37.41646270666716, best_cost));
            assert!(is_same_undirected_circle(
                &vec![0, 3, 4, 1, 2],
                &best_permutation
            ));
        }
    }

    #[test]
    fn test_big_floating_tsp_vecmatrix() {
        let gm: VecMatrix = BIG_FLOAT_GRAPH.clone().into();
        for f in [
            naive_solver,
            first_improved_solver,
            second_improved_solver,
            third_improved_solver,
            fourth_improved_solver,
        ]
        .iter()
        {
            let (best_cost, best_permutation) = f(&gm);
            assert!(relative_eq!(33.03008250868411, best_cost));
            assert!(is_same_undirected_circle(
                &vec![0, 5, 3, 2, 8, 1, 7, 6, 4],
                &best_permutation
            ));
        }
    }

    #[test]
    fn test_big_floating_tsp_namatrix() {
        let gm: NAMatrix = <NAMatrix as From<&Graph>>::from(&BIG_FLOAT_GRAPH);
        for f in [
            naive_solver,
            first_improved_solver,
            second_improved_solver,
            third_improved_solver,
            fourth_improved_solver,
            fifth_improved_solver,
            sixth_improved_solver,
        ]
        .iter()
        {
            let (best_cost, best_permutation) = f(&gm);
            assert!(relative_eq!(33.03008250868411, best_cost));
            assert!(is_same_undirected_circle(
                &vec![0, 5, 3, 2, 8, 1, 7, 6, 4],
                &best_permutation
            ));
        }
    }

    #[test]
    fn test_integer_tsp_vecmatrix() {
        let gm: VecMatrix = SMALL_INT_GRAPH.clone().into();
        for f in [
            naive_solver,
            first_improved_solver,
            second_improved_solver,
            third_improved_solver,
            fourth_improved_solver,
        ]
        .iter()
        {
            let (best_cost, best_permutation) = f(&gm);
            assert!(relative_eq!(best_cost, 17.0));
            assert!(is_same_undirected_circle(
                &best_permutation,
                &vec![0, 1, 3, 2]
            ));
        }
    }

    #[test]
    fn test_integer_tsp_namatrix() {
        let gm: NAMatrix = <NAMatrix as From<&Graph>>::from(&SMALL_INT_GRAPH);
        for f in [
            naive_solver,
            first_improved_solver,
            second_improved_solver,
            third_improved_solver,
            fourth_improved_solver,
            fifth_improved_solver,
            sixth_improved_solver,
        ]
        .iter()
        {
            let (best_cost, best_permutation) = f(&gm);
            assert!(relative_eq!(best_cost, 17.0));
            assert!(is_same_undirected_circle(
                &best_permutation,
                &vec![0, 1, 3, 2]
            ));
        }
    }

    #[test]
    fn test_get_all_permutations() {
        let mut starting_vec = (0..4).collect::<Vec<i32>>();
        let mut results = vec![];
        results.push(starting_vec.clone());
        while next_permutation(&mut starting_vec) {
            results.push(starting_vec.clone());
        }

        let expected = vec![
            vec![0, 1, 2, 3],
            vec![0, 1, 3, 2],
            vec![0, 2, 1, 3],
            vec![0, 2, 3, 1],
            vec![0, 3, 1, 2],
            vec![0, 3, 2, 1],
            vec![1, 0, 2, 3],
            vec![1, 0, 3, 2],
            vec![1, 2, 0, 3],
            vec![1, 2, 3, 0],
            vec![1, 3, 0, 2],
            vec![1, 3, 2, 0],
            vec![2, 0, 1, 3],
            vec![2, 0, 3, 1],
            vec![2, 1, 0, 3],
            vec![2, 1, 3, 0],
            vec![2, 3, 0, 1],
            vec![2, 3, 1, 0],
            vec![3, 0, 1, 2],
            vec![3, 0, 2, 1],
            vec![3, 1, 0, 2],
            vec![3, 1, 2, 0],
            vec![3, 2, 0, 1],
            vec![3, 2, 1, 0],
        ];
        assert_eq!(expected, results);
    }
}
