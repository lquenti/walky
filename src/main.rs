use std::error::Error;

use clap::Parser;
use walky::Args;

/// parse the command line arguments and pass them to `[walky::run]`
fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    walky::run(args)
}
