use std::path::PathBuf;

use clap::Parser;

pub mod parser;

/// This struct contains all the arguments captured from the command line.
#[derive(Debug, Parser)]
pub struct Args {
    /// path to the input file
    input_file: PathBuf,
}

/// This function calls the main logic of our program.
pub fn run(args: Args) {
    todo!();
}
