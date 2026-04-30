use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "neuraldiff")]
#[command(about = "Visual diff between AI model checkpoints")]
#[command(version = "0.1.0")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    Diff {
        model_a: PathBuf,
        model_b: PathBuf,
        #[arg(long)]
        json: bool,
    },
    Inspect {
        model: PathBuf,
    },
}
