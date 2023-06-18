use std::error::Error;

use clap::Parser;
use walky::cli::Cli;

/// parse the command line arguments and pass them to `[walky::run]`
fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    walky::run(cli)
}
