use crate::parser::TravellingSalesmanProblemInstance;
use std::{fs::File, io::Read, path::PathBuf};

use clap::Parser;

pub mod parser;

/// This struct contains all the arguments captured from the command line.
#[derive(Debug, Parser)]
pub struct Args {
    /// path to the input file
    input_file: PathBuf,
}

/// This function calls the main logic of our program.
pub fn run(args: Args) -> std::io::Result<()> {
    let mut file = File::open(args.input_file)?;
    let mut xml = String::new();
    file.read_to_string(&mut xml)?;
    let tsp_instance = TravellingSalesmanProblemInstance::parse_from_xml(&xml[..]);
    Ok(())
}
