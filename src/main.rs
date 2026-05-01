use anyhow::Result;
use clap::Parser;
use neuraldiff::cli::{Cli, Commands};
use neuraldiff::diff::compute_diff;
use neuraldiff::loader::load;

#[cfg(feature = "tui")]
use neuraldiff::tui;

#[cfg(feature = "tui")]
use neuraldiff::scanner;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

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
                        return Ok(());
                    }
                }
            };

            if json {
                let result = compute_diff(&path_a, &path_b)?;
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else {
                #[cfg(feature = "tui")]
                tui::run_with_loading(&path_a, &path_b)?;

                #[cfg(not(feature = "tui"))]
                {
                    let result = compute_diff(&path_a, &path_b)?;
                    println!("Diff complete. {} layers.", result.layers.len());
                    println!("Use --json for output or enable the tui feature.");
                }
            }
        }

        Commands::Inspect { model } => {
            let snapshot = load(&model)?;

            println!();
            println!("  Model  : {}", model.display());
            println!("  Params : {}", format_params(snapshot.total_params));
            println!("  Tensors: {}", snapshot.tensors.len());
            println!();
            println!(
                "  {:<50}  {:>20}  {:>6}  {:>12}",
                "Name", "Shape", "DType", "Params"
            );
            println!("  {}", "─".repeat(96));

            let mut tensors: Vec<_> = snapshot.tensors.values().collect();
            tensors.sort_by(|a, b| a.name.cmp(&b.name));

            for t in tensors {
                let shape_str = format!("{:?}", t.shape);
                let numel: usize = t.shape.iter().product();
                println!(
                    "  {:<50}  {:>20}  {:>6?}  {:>12}",
                    truncate(&t.name, 50),
                    shape_str,
                    t.dtype,
                    format_params(numel as u64),
                );
            }
            println!();
        }
    }

    Ok(())
}

fn format_params(n: u64) -> String {
    if n >= 1_000_000_000 { format!("{:.2}B", n as f64 / 1_000_000_000.0) }
    else if n >= 1_000_000 { format!("{:.2}M", n as f64 / 1_000_000.0) }
    else if n >= 1_000 { format!("{:.2}K", n as f64 / 1_000.0) }
    else { n.to_string() }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max { s.to_string() }
    else { format!("{}…", &s[..max - 1]) }
}
