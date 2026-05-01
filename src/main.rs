use anyhow::{Context, Result};
use clap::Parser;
use neuraldiff::cli::{Cli, Commands};
use neuraldiff::diff::compute_diff;
use neuraldiff::loader::load;

#[cfg(feature = "tui")]
use neuraldiff::tui;

use neuraldiff::scanner;
use std::path::Path;

fn main() -> Result<()> {
    // Log to stderr so --json output stays clean on stdout.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_max_level(tracing::Level::WARN)
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    // No subcommand → behave like `diff` without args (interactive scan + pick).
    let command = cli.command.unwrap_or(Commands::Diff {
        model_a: None,
        model_b: None,
        json: false,
        output: None,
        threshold: 0.000_001,
    });

    match command {
        Commands::Diff { model_a, model_b, json, output, threshold: _ } => {
            let (path_a, path_b) = resolve_diff_paths(model_a, model_b)?;

            // File output (preferred — explicit format from extension)
            if let Some(out) = output {
                let result = compute_diff(&path_a, &path_b)?;
                let ext = out.extension().and_then(|e| e.to_str()).unwrap_or("");
                match ext {
                    "csv" => write_csv(&result, &out)?,
                    _ => {
                        let json = serde_json::to_string_pretty(&result)?;
                        std::fs::write(&out, json)
                            .with_context(|| format!("Failed to write {}", out.display()))?;
                    }
                }
                eprintln!("→ wrote {}", out.display());
                return Ok(());
            }

            if json {
                let result = compute_diff(&path_a, &path_b)?;
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else {
                #[cfg(feature = "tui")]
                tui::run_with_loading(&path_a, &path_b)?;

                #[cfg(not(feature = "tui"))]
                {
                    let result = compute_diff(&path_a, &path_b)?;
                    eprintln!("Diff complete. {} layers.", result.layers.len());
                    eprintln!("Use --json for output or enable the tui feature.");
                }
            }
        }

        Commands::Summary { model_a, model_b, top } => {
            let result = compute_diff(&model_a, &model_b)?;
            print_summary(&result, top);
        }

        Commands::Inspect { model, top, json } => {
            let snapshot = load(&model)?;

            if json {
                let payload = serde_json::json!({
                    "model": model.display().to_string(),
                    "total_params": snapshot.total_params,
                    "tensors": snapshot.tensors.values().map(|t| serde_json::json!({
                        "name": t.name,
                        "shape": t.shape,
                        "dtype": format!("{:?}", t.dtype),
                        "params": t.shape.iter().product::<usize>(),
                    })).collect::<Vec<_>>(),
                });
                println!("{}", serde_json::to_string_pretty(&payload)?);
                return Ok(());
            }

            println!();
            println!("  Model    {}", model.display());
            println!("  Params   {}", format_params(snapshot.total_params));
            println!("  Tensors  {}", snapshot.tensors.len());
            println!();
            println!(
                "  {:<50}  {:>20}  {:>6}  {:>12}",
                "Name", "Shape", "DType", "Params"
            );
            println!("  {}", "─".repeat(96));

            // Sort by parameter count (largest first) when --top given,
            // otherwise sort by name like before.
            let mut tensors: Vec<_> = snapshot.tensors.values().collect();
            if top.is_some() {
                tensors.sort_by(|a, b| {
                    let pa: usize = a.shape.iter().product();
                    let pb: usize = b.shape.iter().product();
                    pb.cmp(&pa)
                });
            } else {
                tensors.sort_by(|a, b| a.name.cmp(&b.name));
            }

            let limit = top.unwrap_or(tensors.len());
            for t in tensors.into_iter().take(limit) {
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

        Commands::Scan { root, depth, json } => {
            let models = match root {
                Some(ref r) => scanner::scan_in_root(r, depth)?,
                None => scanner::scan_for_models()?,
            };

            if json {
                let payload: Vec<_> = models.iter().map(|m| serde_json::json!({
                    "name": m.name,
                    "path": m.path.display().to_string(),
                    "size_mb": m.size_mb,
                    "location": m.location,
                })).collect();
                println!("{}", serde_json::to_string_pretty(&payload)?);
                return Ok(());
            }

            if models.is_empty() {
                eprintln!("No .safetensors models found.");
                if root.is_none() {
                    eprintln!("Searched in: home, Downloads, Documents, Desktop, .cache/huggingface");
                    eprintln!("Try `neuraldiff scan --root <DIR>` to scan a specific path.");
                }
            } else {
                println!("Found {} model(s):\n", models.len());
                println!("  {:<50}  {:>10}  {}", "Name", "Size", "Location");
                println!("  {}", "─".repeat(100));
                for m in models {
                    println!(
                        "  {:<50}  {:>10}  {}",
                        truncate(&m.name, 50),
                        scanner::format_size(m.size_mb),
                        m.location
                    );
                }
            }
        }
    }

    Ok(())
}

fn resolve_diff_paths(
    model_a: Option<std::path::PathBuf>,
    model_b: Option<std::path::PathBuf>,
) -> Result<(std::path::PathBuf, std::path::PathBuf)> {
    match (model_a, model_b) {
        (Some(a), Some(b)) => Ok((a, b)),
        _ => {
            #[cfg(feature = "tui")]
            {
                let (a, b) = scanner::run_model_selection()?;
                match (a, b) {
                    (Some(pa), Some(pb)) => Ok((pa, pb)),
                    _ => {
                        eprintln!("No models selected. Exiting.");
                        std::process::exit(0);
                    }
                }
            }
            #[cfg(not(feature = "tui"))]
            {
                anyhow::bail!("Usage: neuraldiff diff <MODEL_A> <MODEL_B>")
            }
        }
    }
}

fn print_summary(result: &neuraldiff::types::DiffResult, top: usize) {
    let s = &result.summary;
    println!();
    println!("  ◆ NEURALDIFF — diff summary");
    println!();
    println!("  A: {}", result.model_a);
    println!("  B: {}", result.model_b);
    println!();
    println!(
        "  Σ {} params   ▣ {} layers   ⚡ {} changed ({:.1}%)   ⚠ {} anomalies",
        format_params(result.total_params),
        s.total_layers,
        s.changed_layers,
        s.change_ratio_percent,
        s.anomalies.len()
    );
    println!("  μ delta {:.4}    max delta {:.4}", s.mean_delta, s.max_delta);

    if !s.missing_tensors.is_empty() {
        println!("  ⚠ {} tensor(s) only in one model", s.missing_tensors.len());
    }
    println!();

    println!("  Top {} changed layers:", top);
    println!(
        "  {:>3}  {:<22}  {:<8}  {:>10}  {:>8}",
        "#", "Layer", "Type", "L2", "Severity"
    );
    println!("  {}", "─".repeat(60));
    for (i, idx) in s.top_changed_indices.iter().enumerate().take(top) {
        if let Some(layer) = result.layers.get(*idx) {
            let severity = match layer.aggregate_l2 {
                v if v < 0.001 => "low",
                v if v < 0.3 => "medium",
                v if v < 0.6 => "high",
                _ => "CRITICAL",
            };
            println!(
                "  {:>3}  {:<22}  {:<8}  {:>10.4}  {:>8}",
                i + 1,
                truncate(&layer.layer_name, 22),
                format!("{}", layer.layer_type),
                layer.aggregate_l2,
                severity
            );
        }
    }

    if !s.anomalies.is_empty() {
        println!();
        println!("  Anomalies (z > 2.0):");
        for a in &s.anomalies {
            println!("    z={:>5.2}   {}", a.z_score, a.layer_name);
        }
    }
    println!();
}

fn write_csv(result: &neuraldiff::types::DiffResult, out: &Path) -> Result<()> {
    use std::fmt::Write as _;
    let mut s = String::new();
    s.push_str("layer_name,tensor_name,shape,l2,cosine,max_delta,mean_delta,std_delta,changed\n");
    for layer in &result.layers {
        for t in &layer.tensors {
            let shape: String = t
                .shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join("x");
            let _ = writeln!(
                s,
                "{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{}",
                csv_escape(&layer.layer_name),
                csv_escape(&t.name),
                shape,
                t.l2_distance,
                t.cosine_similarity,
                t.max_delta,
                t.mean_delta,
                t.std_delta,
                t.changed
            );
        }
    }
    std::fs::write(out, s)
        .with_context(|| format!("Failed to write {}", out.display()))?;
    Ok(())
}

fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') || s.contains('\r') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn format_params(n: u64) -> String {
    if n >= 1_000_000_000 { format!("{:.2}B", n as f64 / 1_000_000_000.0) }
    else if n >= 1_000_000 { format!("{:.2}M", n as f64 / 1_000_000.0) }
    else if n >= 1_000 { format!("{:.2}K", n as f64 / 1_000.0) }
    else { n.to_string() }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max { return s.to_string(); }
    let keep = max.saturating_sub(1);
    let end = (0..=keep).rev()
        .find(|&i| s.is_char_boundary(i))
        .unwrap_or(0);
    format!("{}…", &s[..end])
}
