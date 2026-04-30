use crate::types::{AppState, FilterMode, LayerDiff, Severity, SortMode, ViewMode};
use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::ExecutableCommand;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Cell, Clear, Paragraph, Row, Table};
use ratatui::Frame;
use ratatui::Terminal;
use std::io;
use std::time::{Duration, Instant};

// Theme colors
const BG: Color = Color::Rgb(10, 10, 10);
const SURFACE: Color = Color::Rgb(23, 23, 23);
const TEXT_PRIMARY: Color = Color::Rgb(229, 229, 229);
const TEXT_SECONDARY: Color = Color::Rgb(115, 115, 115);
const ACCENT: Color = Color::Rgb(16, 185, 129);     // emerald
const ACCENT_INFO: Color = Color::Rgb(249, 115, 22); // orange
const GREEN: Color = Color::Rgb(34, 197, 94);
const YELLOW: Color = Color::Rgb(234, 179, 8);
const ORANGE: Color = Color::Rgb(249, 115, 22);
const RED: Color = Color::Rgb(239, 68, 68);
const PINK: Color = Color::Rgb(236, 72, 153);
const BORDER: Color = Color::Rgb(38, 38, 38);

pub fn run_app(mut state: AppState) -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    stdout.execute(EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut last_tick = Instant::now();
    let tick_rate = Duration::from_millis(250);

    let res = run_app_loop(&mut terminal, &mut state, &mut last_tick, tick_rate);

    disable_raw_mode()?;
    terminal.backend_mut().execute(LeaveAlternateScreen)?;

    res
}

fn run_app_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    state: &mut AppState,
    last_tick: &mut Instant,
    tick_rate: Duration,
) -> Result<()> {
    loop {
        terminal.draw(|f| draw_ui(f, state))?;

        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or_else(|| Duration::from_secs(0));

        if crossterm::event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                if handle_key_event(key, state) {
                    return Ok(());
                }
            }
        }

        if last_tick.elapsed() >= tick_rate {
            *last_tick = Instant::now();
        }
    }
}

fn handle_key_event(key: KeyEvent, state: &mut AppState) -> bool {
    if key.modifiers == KeyModifiers::CONTROL && key.code == KeyCode::Char('c') {
        return true;
    }

    if state.show_help {
        if key.code == KeyCode::Char('?') || key.code == KeyCode::Esc {
            state.show_help = false;
        }
        return false;
    }

    match key.code {
        KeyCode::Char('q') => return true,
        KeyCode::Char('?') => state.show_help = true,
        KeyCode::Char('j') | KeyCode::Down => {
            if state.view_mode == ViewMode::Detail {
                let max = get_filtered_layers(state).len().saturating_sub(1);
                state.selected_layer = (state.selected_layer + 1).min(max);
            } else {
                state.view_mode = ViewMode::Detail;
            }
        }
        KeyCode::Char('k') | KeyCode::Up => {
            if state.view_mode == ViewMode::Detail {
                state.selected_layer = state.selected_layer.saturating_sub(1);
            }
        }
        KeyCode::Char('h') | KeyCode::Left => {
            state.selected_tensor = state.selected_tensor.saturating_sub(1);
        }
        KeyCode::Char('l') | KeyCode::Right => {
            if let Some(ref diff) = state.diff {
                let layers = get_filtered_layers(state);
                if let Some(layer) = layers.get(state.selected_layer) {
                    let max = layer.tensors.len().saturating_sub(1);
                    state.selected_tensor = (state.selected_tensor + 1).min(max);
                }
            }
        }
        KeyCode::Enter => {
            if state.view_mode == ViewMode::Summary {
                state.view_mode = ViewMode::Detail;
            } else {
                state.show_heatmap = !state.show_heatmap;
            }
        }
        KeyCode::Char('b') => {
            state.view_mode = ViewMode::Summary;
            state.selected_layer = 0;
            state.selected_tensor = 0;
        }
        KeyCode::Char('s') => {
            state.sort_mode = match state.sort_mode {
                SortMode::L2Desc => SortMode::LayerIndex,
                SortMode::LayerIndex => SortMode::AnomalyScore,
                SortMode::AnomalyScore => SortMode::L2Desc,
            };
        }
        KeyCode::Char('f') => {
            state.filter_mode = match state.filter_mode {
                FilterMode::All => FilterMode::ChangedOnly,
                FilterMode::ChangedOnly => FilterMode::All,
            };
        }
        KeyCode::Char('J') => {
            if let Some(ref diff) = state.diff {
                if let Ok(json) = serde_json::to_string_pretty(diff) {
                    if std::fs::write("diff.json", json).is_ok() {
                        state.status_message = Some("Exported to diff.json".to_string());
                    }
                }
            }
        }
        _ => {}
    }
    false
}

fn get_filtered_layers(state: &AppState) -> Vec<&LayerDiff> {
    let diff = match state.diff {
        Some(ref d) => d,
        None => return vec![],
    };

    let mut layers: Vec<&LayerDiff> = diff.layers.iter().collect();

    if state.filter_mode == FilterMode::ChangedOnly {
        layers.retain(|l| l.aggregate_l2 > 1e-6);
    }

    match state.sort_mode {
        SortMode::L2Desc => {
            layers.sort_by(|a, b| b.aggregate_l2.partial_cmp(&a.aggregate_l2).unwrap());
        }
        SortMode::LayerIndex => {
            layers.sort_by(|a, b| {
                match (a.layer_index, b.layer_index) {
                    (Some(ai), Some(bi)) => ai.cmp(&bi),
                    _ => std::cmp::Ordering::Equal,
                }
            });
        }
        SortMode::AnomalyScore => {
            layers.sort_by(|a, b| b.anomaly_score.partial_cmp(&a.anomaly_score).unwrap());
        }
    }

    layers
}

fn draw_ui(f: &mut Frame, state: &AppState) {
    let area = f.area();
    f.render_widget(
        Block::default().style(Style::default().bg(BG)),
        area,
    );

    if state.show_help {
        draw_help(f, state, area);
        return;
    }

    match state.view_mode {
        ViewMode::Summary => draw_summary(f, state, area),
        ViewMode::Detail => draw_detail(f, state, area),
    }
}

// ============================================
// SUMMARY VIEW - Intuitive overview
// ============================================
fn draw_summary(f: &mut Frame, state: &AppState, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([
            Constraint::Length(4),   // Header with model comparison
            Constraint::Length(6),   // Legend + key metrics
            Constraint::Min(12),     // Top changed layers
            Constraint::Length(3),   // Footer
        ])
        .split(area);

    draw_comparison_header(f, state, chunks[0]);
    draw_legend_and_metrics(f, state, chunks[1]);
    draw_top_changed(f, state, chunks[2]);
    draw_footer(f, state, chunks[3]);
}

fn draw_comparison_header(f: &mut Frame, state: &AppState, area: Rect) {
    let diff = match state.diff {
        Some(ref d) => d,
        None => {
            let block = Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(BORDER));
            f.render_widget(block, area);
            return;
        }
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(ACCENT))
        .style(Style::default().bg(SURFACE));

    let mut lines = vec![];

    // Title
    lines.push(Line::from(vec![
        Span::styled(" MODEL COMPARISON ", Style::default().fg(TEXT_PRIMARY).add_modifier(Modifier::BOLD)),
    ]));

    // Model A vs Model B
    let model_a_name = diff.model_a.as_deref().unwrap_or("Model A");
    let model_b_name = diff.model_b.as_deref().unwrap_or("Model B");

    lines.push(Line::from(vec![
        Span::styled("A: ", Style::default().fg(ACCENT)),
        Span::styled(truncate_path(model_a_name, 30), Style::default().fg(TEXT_PRIMARY)),
        Span::styled("  |  ", Style::default().fg(TEXT_SECONDARY)),
        Span::styled("B: ", Style::default().fg(ACCENT_INFO)),
        Span::styled(truncate_path(model_b_name, 30), Style::default().fg(TEXT_PRIMARY)),
    ]));

    // Params
    lines.push(Line::from(vec![
        Span::styled("Parameters: ", Style::default().fg(TEXT_SECONDARY)),
        Span::styled(format_params(diff.total_params), Style::default().fg(TEXT_PRIMARY).add_modifier(Modifier::BOLD)),
        Span::styled("  |  ", Style::default().fg(TEXT_SECONDARY)),
        Span::styled("Layers: ", Style::default().fg(TEXT_SECONDARY)),
        Span::styled(diff.summary.total_layers.to_string(), Style::default().fg(TEXT_PRIMARY).add_modifier(Modifier::BOLD)),
    ]));

    let paragraph = Paragraph::new(Text::from(lines))
        .style(Style::default().fg(TEXT_PRIMARY))
        .alignment(Alignment::Left)
        .block(block);

    f.render_widget(paragraph, area);
}

fn draw_legend_and_metrics(f: &mut Frame, state: &AppState, area: Rect) {
    let diff = match state.diff {
        Some(ref d) => d,
        None => return,
    };

    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    // Left: Legend explaining the UI
    let legend_text = vec![
        Line::from(vec![
            Span::styled("LEGEND:", Style::default().fg(TEXT_SECONDARY).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled("L2 Distance ", Style::default().fg(TEXT_PRIMARY)),
            Span::styled("- Magnitude of changes (0=identical, >1=drastic)", Style::default().fg(TEXT_SECONDARY)),
        ]),
        Line::from(vec![
            Span::styled("Cosine Sim  ", Style::default().fg(TEXT_PRIMARY)),
            Span::styled("- Direction similarity (-1=opposite, 1=identical)", Style::default().fg(TEXT_SECONDARY)),
        ]),
        Line::from(vec![
            Span::styled("Z-Score     ", Style::default().fg(TEXT_PRIMARY)),
            Span::styled("- How unusual the change is vs other layers", Style::default().fg(TEXT_SECONDARY)),
        ]),
    ];

    let legend_block = Block::default()
        .title(" What These Numbers Mean ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(BORDER))
        .style(Style::default().bg(SURFACE));

    let legend = Paragraph::new(Text::from(legend_text))
        .style(Style::default().fg(TEXT_PRIMARY))
        .block(legend_block);

    f.render_widget(legend, chunks[0]);

    // Right: Color scale
    let scale_text = vec![
        Line::from(vec![
            Span::styled("CHANGE SCALE:", Style::default().fg(TEXT_SECONDARY).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled("No change    ", Style::default().fg(TEXT_SECONDARY)),
            Span::styled("[LOW]  ", Style::default().fg(GREEN)),
            Span::styled("Minor (<0.3)", Style::default().fg(TEXT_SECONDARY)),
        ]),
        Line::from(vec![
            Span::styled("Moderate     ", Style::default().fg(TEXT_SECONDARY)),
            Span::styled("[MED]  ", Style::default().fg(YELLOW)),
            Span::styled("Noticeable (0.3-0.6)", Style::default().fg(TEXT_SECONDARY)),
        ]),
        Line::from(vec![
            Span::styled("Significant  ", Style::default().fg(TEXT_SECONDARY)),
            Span::styled("[HIGH] ", Style::default().fg(ORANGE)),
            Span::styled("Major (0.6-0.8)", Style::default().fg(TEXT_SECONDARY)),
        ]),
        Line::from(vec![
            Span::styled("Drastic      ", Style::default().fg(TEXT_SECONDARY)),
            Span::styled("[CRIT] ", Style::default().fg(RED)),
            Span::styled("Critical (>0.8)", Style::default().fg(TEXT_SECONDARY)),
        ]),
    ];

    let scale_block = Block::default()
        .title(" Severity Levels ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(BORDER))
        .style(Style::default().bg(SURFACE));

    let scale = Paragraph::new(Text::from(scale_text))
        .style(Style::default().fg(TEXT_PRIMARY))
        .block(scale_block);

    f.render_widget(scale, chunks[1]);
}

fn draw_top_changed(f: &mut Frame, state: &AppState, area: Rect) {
    let diff = match state.diff {
        Some(ref d) => d,
        None => return,
    };

    let block = Block::default()
        .title(" Top Changed Layers (Press Enter to explore) ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(BORDER))
        .style(Style::default().bg(SURFACE));

    let mut lines = vec![];

    // Header row
    lines.push(Line::from(vec![
        Span::styled("#  ", Style::default().fg(TEXT_SECONDARY)),
        Span::styled("Layer Name          ", Style::default().fg(TEXT_SECONDARY)),
        Span::styled("Type  ", Style::default().fg(TEXT_SECONDARY)),
        Span::styled("L2 Distance    ", Style::default().fg(TEXT_SECONDARY)),
        Span::styled("Severity", Style::default().fg(TEXT_SECONDARY)),
        Span::styled("  ", Style::default()),
        Span::styled("Change Bar", Style::default().fg(TEXT_SECONDARY)),
    ]));
    lines.push(Line::from(""));

    for (i, idx) in diff.summary.top_changed_indices.iter().enumerate().take(10) {
        if let Some(layer) = diff.layers.get(*idx) {
            let bar = render_l2_bar(layer.aggregate_l2, 20);
            let severity = Severity::from_l2(layer.aggregate_l2);
            let color = l2_color(layer.aggregate_l2);

            let row = Line::from(vec![
                Span::styled(format!("{:>2} ", i + 1), Style::default().fg(TEXT_SECONDARY)),
                Span::styled(format!("{:<18} ", layer.layer_name.clone()), Style::default().fg(TEXT_PRIMARY)),
                Span::styled(format!("{:5} ", layer.layer_type.to_string()), Style::default().fg(TEXT_SECONDARY)),
                Span::styled(format!("{:>8.4}  ", layer.aggregate_l2), Style::default().fg(color)),
                Span::styled(severity.as_str(), Style::default().fg(color).add_modifier(Modifier::BOLD)),
                Span::styled("  ", Style::default()),
                Span::styled(bar, Style::default().fg(color)),
            ]);
            lines.push(row);
        }
    }

    if diff.summary.anomalies.is_empty() {
        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::styled("No anomalies detected - all changes are within normal range", Style::default().fg(GREEN)),
        ]));
    } else {
        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::styled("⚠ ANOMALIES: ", Style::default().fg(PINK).add_modifier(Modifier::BOLD)),
        ]));
        for anomaly in &diff.summary.anomalies {
            lines.push(Line::from(vec![
                Span::styled("  • ", Style::default().fg(PINK)),
                Span::styled(format!("{} ", anomaly.layer_name), Style::default().fg(TEXT_PRIMARY)),
                Span::styled(format!("(z-score: {:.2}) - Unusually large change", anomaly.z_score), Style::default().fg(TEXT_SECONDARY)),
            ]));
        }
    }

    let paragraph = Paragraph::new(Text::from(lines))
        .style(Style::default().fg(TEXT_PRIMARY))
        .block(block);

    f.render_widget(paragraph, area);
}

// ============================================
// DETAIL VIEW - Layer by layer comparison
// ============================================
fn draw_detail(f: &mut Frame, state: &AppState, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([
            Constraint::Length(3),   // Header
            Constraint::Min(10),     // Main content
            Constraint::Length(3),   // Footer
        ])
        .split(area);

    draw_detail_header(f, state, chunks[0]);
    draw_detail_content(f, state, chunks[1]);
    draw_footer(f, state, chunks[2]);
}

fn draw_detail_header(f: &mut Frame, state: &AppState, area: Rect) {
    let diff = match state.diff {
        Some(ref d) => d,
        None => {
            let block = Block::default().borders(Borders::ALL);
            f.render_widget(block, area);
            return;
        }
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(ACCENT))
        .style(Style::default().bg(SURFACE));

    let mut spans = vec![
        Span::styled(" NEURALDIFF ", Style::default().fg(TEXT_PRIMARY).add_modifier(Modifier::BOLD)),
    ];

    if let Some(ref a) = diff.model_a {
        spans.push(Span::styled(truncate_path(a, 20), Style::default().fg(TEXT_SECONDARY)));
    }
    spans.push(Span::styled(" → ", Style::default().fg(ACCENT)));
    if let Some(ref b) = diff.model_b {
        spans.push(Span::styled(truncate_path(b, 20), Style::default().fg(TEXT_SECONDARY)));
    }

    let paragraph = Paragraph::new(Line::from(spans))
        .alignment(Alignment::Center)
        .block(block);

    f.render_widget(paragraph, area);
}

fn draw_detail_content(f: &mut Frame, state: &AppState, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(35), Constraint::Percentage(65)])
        .split(area);

    draw_layer_list(f, state, chunks[0]);
    draw_tensor_comparison(f, state, chunks[1]);
}

fn draw_layer_list(f: &mut Frame, state: &AppState, area: Rect) {
    let layers = get_filtered_layers(state);
    let mut lines = vec![];

    for (i, layer) in layers.iter().enumerate() {
        let is_selected = i == state.selected_layer;
        let bar = render_l2_bar(layer.aggregate_l2, 12);
        let color = l2_color(layer.aggregate_l2);
        let severity = Severity::from_l2(layer.aggregate_l2);

        let style = if is_selected {
            Style::default().bg(ACCENT).fg(BG).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(TEXT_PRIMARY)
        };

        let line = Line::from(vec![
            Span::styled(
                format!("{:>3} ", layer.layer_index.map_or("-".to_string(), |i| i.to_string())),
                style,
            ),
            Span::styled(
                format!("{:4} ", layer.layer_type.to_string()),
                style,
            ),
            Span::styled(
                bar,
                if is_selected { Style::default().fg(BG) } else { Style::default().fg(color) },
            ),
            Span::styled(
                format!(" {:>5.3} ", layer.aggregate_l2),
                style,
            ),
            Span::styled(
                severity.as_str(),
                if is_selected { Style::default().fg(BG) } else { Style::default().fg(color) },
            ),
        ]);
        lines.push(line);
    }

    let title = format!(" Layers [{}] ", match state.sort_mode {
        SortMode::L2Desc => "L2↓",
        SortMode::LayerIndex => "Idx",
        SortMode::AnomalyScore => "Anom",
    });

    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(Style::default().fg(BORDER))
        .style(Style::default().bg(SURFACE));

    let paragraph = Paragraph::new(Text::from(lines))
        .style(Style::default().fg(TEXT_PRIMARY))
        .block(block);

    f.render_widget(paragraph, area);
}

fn draw_tensor_comparison(f: &mut Frame, state: &AppState, area: Rect) {
    let layers = get_filtered_layers(state);
    let layer = match layers.get(state.selected_layer) {
        Some(l) => l,
        None => {
            let block = Block::default()
                .title(" Tensor Comparison ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(BORDER));
            f.render_widget(block, area);
            return;
        }
    };

    // Get model names for title
    let title = if let Some(ref diff) = state.diff {
        let a = diff.model_a.as_deref().unwrap_or("Model A");
        let b = diff.model_b.as_deref().unwrap_or("Model B");
        format!(" {} → {} ", truncate_path(a, 20), truncate_path(b, 20))
    } else {
        " Tensor Comparison ".to_string()
    };

    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(Style::default().fg(BORDER))
        .style(Style::default().bg(SURFACE));

    let mut lines = vec![];

    // Layer info with model context
    lines.push(Line::from(vec![
        Span::styled("Layer: ", Style::default().fg(TEXT_SECONDARY)),
        Span::styled(&layer.layer_name, Style::default().fg(TEXT_PRIMARY).add_modifier(Modifier::BOLD)),
        Span::styled("  |  Type: ", Style::default().fg(TEXT_SECONDARY)),
        Span::styled(layer.layer_type.to_string().to_uppercase(), Style::default().fg(ACCENT_INFO).add_modifier(Modifier::BOLD)),
    ]));

    lines.push(Line::from(vec![
        Span::styled("L2: ", Style::default().fg(TEXT_SECONDARY)),
        Span::styled(format!("{:.6}", layer.aggregate_l2), Style::default().fg(l2_color(layer.aggregate_l2)).add_modifier(Modifier::BOLD)),
        Span::styled("  |  Params: ", Style::default().fg(TEXT_SECONDARY)),
        Span::styled(format_params(layer.param_count), Style::default().fg(TEXT_PRIMARY)),
        Span::styled("  |  Tensors: ", Style::default().fg(TEXT_SECONDARY)),
        Span::styled(layer.tensors.len().to_string(), Style::default().fg(TEXT_PRIMARY)),
        Span::styled("  |  Changed: ", Style::default().fg(TEXT_SECONDARY)),
        Span::styled(
            format!("{}/{}", layer.tensors.iter().filter(|t| t.changed).count(), layer.tensors.len()),
            Style::default().fg(if layer.tensors.iter().any(|t| t.changed) { RED } else { GREEN }),
        ),
    ]));

    lines.push(Line::from(""));

    // Table header
    let header_prefix = "  ";
    lines.push(Line::from(vec![
        Span::styled(header_prefix, Style::default()),
        Span::styled(format!("{:<24}", "Tensor Name"), Style::default().fg(TEXT_SECONDARY).add_modifier(Modifier::BOLD)),
        Span::styled(format!("{:>12}", "Shape"), Style::default().fg(TEXT_SECONDARY).add_modifier(Modifier::BOLD)),
        Span::styled(format!("{:>10}", "L2"), Style::default().fg(TEXT_SECONDARY).add_modifier(Modifier::BOLD)),
        Span::styled(format!("{:>10}", "Cosine"), Style::default().fg(TEXT_SECONDARY).add_modifier(Modifier::BOLD)),
        Span::styled(format!("{:>12}", "Max Delta"), Style::default().fg(TEXT_SECONDARY).add_modifier(Modifier::BOLD)),
        Span::styled(format!("{:>8}", "Status"), Style::default().fg(TEXT_SECONDARY).add_modifier(Modifier::BOLD)),
    ]));
    lines.push(Line::from(vec![
        Span::styled("─".repeat(82), Style::default().fg(BORDER)),
    ]));

    for (i, tensor) in layer.tensors.iter().enumerate() {
        let is_selected = i == state.selected_tensor;
        let color = l2_color(tensor.l2_distance);
        let prefix = if is_selected { "▶" } else { " " };
        
        let shape_str = format!("{:?}", tensor.shape);
        let shape_display = if shape_str.len() > 12 {
            format!("{}..]", &shape_str[..9])
        } else {
            shape_str
        };

        let cosine_color = if tensor.cosine_similarity > 0.9 { GREEN }
            else if tensor.cosine_similarity > 0.5 { YELLOW }
            else { RED };

        let status_text = if tensor.changed { "CHANGED" } else { "SAME" };
        let status_color = if tensor.changed { RED } else { GREEN };


        let bold_mod = if is_selected { Modifier::BOLD } else { Modifier::empty() };

        let spans = vec![
            Span::styled(format!(" {:^1}", prefix), Style::default().fg(ACCENT).add_modifier(Modifier::BOLD | bold_mod)),
            Span::styled(format!("{:<24}", truncate_str(&tensor.name, 24)), Style::default().fg(TEXT_PRIMARY).add_modifier(bold_mod)),
            Span::styled(format!("{:>12}", shape_display), Style::default().fg(TEXT_SECONDARY).add_modifier(bold_mod)),
            Span::styled(format!("{:>10.4}", tensor.l2_distance), Style::default().fg(color).add_modifier(bold_mod)),
            Span::styled(format!("{:>10.4}", tensor.cosine_similarity), Style::default().fg(cosine_color).add_modifier(bold_mod)),
            Span::styled(format!("{:>12.6}", tensor.max_delta), Style::default().fg(color).add_modifier(bold_mod)),
            Span::styled(format!("{:>8}", status_text), Style::default().fg(status_color).add_modifier(bold_mod)),
        ];

        lines.push(Line::from(spans));
    }

    // Distribution section
    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled("─".repeat(82), Style::default().fg(BORDER)),
    ]));
    
    let low_count = layer.tensors.iter().filter(|t| t.l2_distance < 0.1).count();
    let med_count = layer.tensors.iter().filter(|t| t.l2_distance >= 0.1 && t.l2_distance < 0.5).count();
    let high_count = layer.tensors.iter().filter(|t| t.l2_distance >= 0.5).count();
    let total = layer.tensors.len();

    if total > 0 {
        let low_bar = "█".repeat((low_count * 20 / total).max(1));
        let med_bar = "█".repeat((med_count * 20 / total).max(1));
        let high_bar = "█".repeat((high_count * 20 / total).max(1));

        lines.push(Line::from(vec![
            Span::styled("Distribution: ", Style::default().fg(TEXT_SECONDARY)),
            Span::styled(format!("Low({})", low_count), Style::default().fg(GREEN)),
            Span::styled(low_bar, Style::default().fg(GREEN)),
            Span::styled("  ", Style::default()),
            Span::styled(format!("Med({})", med_count), Style::default().fg(YELLOW)),
            Span::styled(med_bar, Style::default().fg(YELLOW)),
            Span::styled("  ", Style::default()),
            Span::styled(format!("High({})", high_count), Style::default().fg(RED)),
            Span::styled(high_bar, Style::default().fg(RED)),
        ]));
    }

    let paragraph = Paragraph::new(Text::from(lines))
        .style(Style::default().fg(TEXT_PRIMARY))
        .block(block);

    f.render_widget(paragraph, area);
}

fn draw_footer(f: &mut Frame, state: &AppState, area: Rect) {
    let mut spans = vec![
        Span::styled("[↑↓/jk] Navigate  ", Style::default().fg(TEXT_SECONDARY)),
        Span::styled("[←→/hl] Tensor  ", Style::default().fg(TEXT_SECONDARY)),
        Span::styled("[Enter] Heatmap  ", Style::default().fg(TEXT_SECONDARY)),
        Span::styled("[b] Back  ", Style::default().fg(TEXT_SECONDARY)),
        Span::styled("[s] Sort  ", Style::default().fg(TEXT_SECONDARY)),
        Span::styled("[f] Filter  ", Style::default().fg(TEXT_SECONDARY)),
        Span::styled("[J] JSON  ", Style::default().fg(TEXT_SECONDARY)),
        Span::styled("[?] Help  ", Style::default().fg(TEXT_SECONDARY)),
        Span::styled("[q] Quit", Style::default().fg(TEXT_SECONDARY)),
    ];

    if let Some(ref msg) = state.status_message {
        spans.push(Span::styled(format!("  |  {}", msg), Style::default().fg(ACCENT)));
    }

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(BORDER))
        .style(Style::default().bg(SURFACE));

    let paragraph = Paragraph::new(Line::from(spans))
        .alignment(Alignment::Center)
        .block(block);

    f.render_widget(paragraph, area);
}

fn draw_help(f: &mut Frame, _state: &AppState, area: Rect) {
    let help_text = vec![
        Line::from(vec![
            Span::styled("Keyboard Shortcuts", Style::default().fg(TEXT_PRIMARY).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from("Navigation:"),
        Line::from("  ↑/k    Move up in layer list"),
        Line::from("  ↓/j    Move down in layer list"),
        Line::from("  ←/h    Previous tensor"),
        Line::from("  →/l    Next tensor"),
        Line::from("  Enter  Toggle heatmap / Enter detail view"),
        Line::from(""),
        Line::from("Commands:"),
        Line::from("  s      Cycle sort mode (L2 → Index → Anomaly)"),
        Line::from("  f      Toggle filter (All → Changed only)"),
        Line::from("  J      Export JSON to diff.json"),
        Line::from("  ?      Toggle this help"),
        Line::from("  q      Quit"),
        Line::from(""),
        Line::from("Understanding the metrics:"),
        Line::from("  L2 Distance  - How much the weights changed (0 = identical)"),
        Line::from("  Cosine Sim   - Whether changes point in same direction (1 = same)"),
        Line::from("  Max Delta    - Largest single weight change"),
        Line::from("  Z-Score      - How unusual vs other layers (>2 = anomaly)"),
    ];

    let block = Block::default()
        .title(" Help ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(ACCENT))
        .style(Style::default().bg(SURFACE));

    let paragraph = Paragraph::new(Text::from(help_text))
        .style(Style::default().fg(TEXT_PRIMARY))
        .block(block);

    let area = centered_rect(70, 80, area);
    f.render_widget(Clear, area);
    f.render_widget(paragraph, area);
}

// ============================================
// Helpers
// ============================================
fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}

fn render_l2_bar(l2: f32, width: usize) -> String {
    let filled = (l2 * width as f32).ceil() as usize;
    let filled = filled.min(width);
    let empty = width - filled;
    format!("{}{}", "█".repeat(filled), "░".repeat(empty))
}

fn l2_color(l2: f32) -> Color {
    if l2 < 0.001 {
        TEXT_SECONDARY
    } else if l2 < 0.3 {
        GREEN
    } else if l2 < 0.6 {
        YELLOW
    } else if l2 < 0.8 {
        ORANGE
    } else {
        RED
    }
}

fn format_params(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.2}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.2}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

fn truncate_path(path: &str, max_len: usize) -> String {
    if path.len() <= max_len {
        path.to_string()
    } else {
        format!("...{}", &path[path.len() - max_len + 3..])
    }
}

fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}
