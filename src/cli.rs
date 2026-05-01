use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// Visual diff between AI model checkpoints.
///
/// Compares two .safetensors files and shows what changed at the layer
/// and tensor level (L2 distance, cosine similarity, anomaly z-score),
/// either in an interactive TUI or as JSON / CSV.
///
/// Run without any subcommand to scan the system for models and pick
/// two interactively — handy when you don't remember a path.
#[derive(Parser)]
#[command(name = "neuraldiff")]
#[command(version)]
#[command(about, long_about = None)]
#[command(propagate_version = true)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Diff two model checkpoints (interactive TUI by default).
    ///
    /// If no paths are given, the scanner runs first so you can pick
    /// model A and model B from the list of detected .safetensors files.
    #[command(alias = "d")]
    Diff {
        /// Path to the first .safetensors file.
        model_a: Option<PathBuf>,
        /// Path to the second .safetensors file.
        model_b: Option<PathBuf>,
        /// Print the full diff as JSON to stdout instead of opening the TUI.
        #[arg(long)]
        json: bool,
        /// Write the diff to the given file (JSON if extension is .json, CSV if .csv).
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,
        /// Hide tensors whose L2 distance is below this threshold (default 1e-6).
        #[arg(long, value_name = "FLOAT", default_value = "0.000001")]
        threshold: f32,
    },

    /// Print a one-screen text summary of a diff (no TUI).
    ///
    /// Useful for CI, scripts, or when you just want the headline numbers
    /// without entering the interactive view.
    Summary {
        /// Path to the first .safetensors file.
        model_a: PathBuf,
        /// Path to the second .safetensors file.
        model_b: PathBuf,
        /// How many top-changed layers to list.
        #[arg(short = 'n', long, default_value = "10")]
        top: usize,
    },

    /// Inspect a single model: list every tensor with shape, dtype, and parameter count.
    #[command(alias = "i")]
    Inspect {
        /// Path to the .safetensors file.
        model: PathBuf,
        /// Show only the top N tensors by parameter count.
        #[arg(short = 'n', long, value_name = "N")]
        top: Option<usize>,
        /// Print as JSON instead of a table.
        #[arg(long)]
        json: bool,
    },

    /// Scan the system for .safetensors models and print the results.
    ///
    /// On WSL this also walks /mnt/<drive>/Users/<you>/ so models stored
    /// on the Windows side are found.
    #[command(alias = "s")]
    Scan {
        /// Scan only this directory (and its subdirs) instead of the default roots.
        #[arg(short, long, value_name = "DIR")]
        root: Option<PathBuf>,
        /// Maximum recursion depth when --root is used.
        #[arg(long, default_value = "5")]
        depth: usize,
        /// Print as JSON instead of a table.
        #[arg(long)]
        json: bool,
    },
}
