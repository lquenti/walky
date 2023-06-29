use std::env;
use std::f64;

#[derive(Debug)]
struct Point {
    x: f64,
    y: f64,
}

fn euclidean_distance(p1: &Point, p2: &Point) -> f64 {
    let dx = p2.x - p1.x;
    let dy = p2.y - p1.y;
    f64::sqrt(dx * dx + dy * dy)
}

fn generate_xml_from_matrix(matrix: &[Vec<f64>]) {
    let name = "RandomGraph";
    let source = "gh:lquenti/walky";
    let description = "random example";
    let double_precision = 15;
    let ignored_digits = 0;

    println!("<travellingSalesmanProblemInstance>");
    println!("  <name>{}</name>", name);
    println!("  <source>{}</source>", source);
    println!("  <description>{}</description>", description);
    println!("  <doublePrecision>{}</doublePrecision>", double_precision);
    println!("  <ignoredDigits>{}</ignoredDigits>", ignored_digits);
    println!("  <graph>");

    for (_, row) in matrix.iter().enumerate() {
        println!("    <vertex>");
        for (col_index, &value) in row.iter().enumerate() {
            if value == 0.0 {continue;}
            println!("      <edge cost=\"{}\">{}</edge>", value, col_index);
        }
        println!("    </vertex>");
    }

    println!("  </graph>");
    println!("</travellingSalesmanProblemInstance>");
}


fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        println!("No n");
        return;
    }

    let n: usize = match args[1].parse() {
        Ok(value) => value,
        Err(_) => {
            println!("Invalid n");
            return;
        }
    };
    let points: Vec<Point> = (0..n)
        .map(|_| Point {
            x: rand::random::<f64>() * 20.0 - 10.0,
            y: rand::random::<f64>() * 20.0 - 10.0,
        })
        .collect();

    let mut matrix = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            if i != j {
                let distance = euclidean_distance(&points[i], &points[j]);
                matrix[i][j] = distance;
            }
        }
    }

    generate_xml_from_matrix(&matrix);
}

