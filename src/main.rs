use anyhow::Result;
use clap::Parser;
use neuraldiff::cli::{Cli, Commands};
use neuraldiff::types::AppState;

#[cfg(feature = "tui")]
use neuraldiff::tui;

fn main() -> Result<()> {
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Diff { model_a, model_b, json } => {
            if json {
                println!("JSON output not yet implemented");
            } else {
                #[cfg(feature = "tui")]
                {
                    let state = AppState::default();
                    tui::run_app(state)?;
                }
                #[cfg(not(feature = "tui"))]
                {
                    println!("TUI feature not enabled");
                }
            }
        }
        Commands::Inspect { model } => {
            println!("Inspect not yet implemented");
        }
    }
    
    Ok(())
}
