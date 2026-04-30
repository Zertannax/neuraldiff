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
        #[arg(help = "First model path (optional if using scanner)")]
        model_a: Option<PathBuf>,
        #[arg(help = "Second model path (optional if using scanner)")]
        model_b: Option<PathBuf>,
        #[arg(long, help = "Output as JSON")]
        json: bool,
    },
    Inspect {
        model: PathBuf,
    },
}
