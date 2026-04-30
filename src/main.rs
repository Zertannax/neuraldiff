use anyhow::Result;
use clap::Parser;
use neuraldiff::cli::{Cli, Commands};
use neuraldiff::diff::compute_diff;
use neuraldiff::types::AppState;

#[cfg(feature = "tui")]
use neuraldiff::tui;

#[cfg(feature = "tui")]
use neuraldiff::scanner;

fn main() -> Result<()> {
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Diff { model_a, model_b, json } => {
            let (path_a, path_b) = match (model_a, model_b) {
                (Some(a), Some(b)) => (a, b),
                _ => {
                    #[cfg(feature = "tui")]
                    {
                        let (a, b) = scanner::run_model_selection()?;
                        match (a, b) {
                            (Some(pa), Some(pb)) => (pa, pb),
                            _ => {
                                println!("No models selected. Exiting.");
                                return Ok(());
                            }
                        }
                    }
                    #[cfg(not(feature = "tui"))]
                    {
                        println!("Usage: neuraldiff diff <MODEL_A> <MODEL_B>");
                        println!("Or enable TUI feature for model scanning.");
                        return Ok(());
                    }
                }
            };
            
            let result = compute_diff(&path_a, &path_b)?;
            
            if json {
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else {
                #[cfg(feature = "tui")]
                {
                    let mut state = AppState::default();
                    state.diff = Some(result);
                    tui::run_app(state)?;
                }
                #[cfg(not(feature = "tui"))]
                {
                    println!("TUI feature not enabled");
                }
            }
        }
        Commands::Inspect { model: _ } => {
            println!("Inspect not yet implemented");
        }
    }
    
    Ok(())
}
