use crate::types::{AppState, DiffResult, FilterMode, LayerDiff, Severity, SortMode, ViewMode};
use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::ExecutableCommand;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Margin, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols::border;
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Cell, Clear, Gauge, Paragraph, Row, Table, Wrap};
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
            if let Some(ref diff) = state.diff {
                let layers = get_filtered_layers(state);
                if let Some(layer) = layers.get(state.selected_layer) {
                    state.selected_tensor = state.selected_tensor.saturating_sub(1);
                }
            }
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

fn draw_summary(f: &mut Frame, state: &AppState, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([
            Constraint::Length(3),  // Header
            Constraint::Length(7),  // Metric cards
            Constraint::Min(10),    // Top changed + anomalies
            Constraint::Length(3),  // Footer
        ])
        .split(area);

    draw_header(f, state, chunks[0]);
    draw_metrics(f, state, chunks[1]);
    draw_top_changed(f, state, chunks[2]);
    draw_footer(f, state, chunks[3]);
}

fn draw_header(f: &mut Frame, state: &AppState, area: Rect) {
    let title = if let Some(ref diff) = state.diff {
        format!(
            " NEURALDIFF v0.1.0    {} -> {}    {} params",
            diff.model_a.as_deref().unwrap_or("?"),
            diff.model_b.as_deref().unwrap_or("?"),
            format_params(diff.total_params)
        )
    } else {
        " NEURALDIFF v0.1.0 ".to_string()
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(BORDER))
        .style(Style::default().bg(SURFACE));

    let paragraph = Paragraph::new(title)
        .style(Style::default().fg(TEXT_PRIMARY).add_modifier(Modifier::BOLD))
        .alignment(Alignment::Center)
        .block(block);

    f.render_widget(paragraph, area);
}

fn draw_metrics(f: &mut Frame, state: &AppState, area: Rect) {
    let diff = match state.diff {
        Some(ref d) => d,
        None => return,
    };

    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
        ])
        .split(area);

    draw_metric_card(f, "Parameters", &format_params(diff.total_params), chunks[0]);
    draw_metric_card(f, "Layers", &diff.summary.total_layers.to_string(), chunks[1]);
    draw_metric_card(f, "Changed", &diff.summary.changed_layers.to_string(), chunks[2]);
    draw_metric_card(f, "Unchanged", &diff.summary.unchanged_layers.to_string(), chunks[3]);
}

fn draw_metric_card(f: &mut Frame, label: &str, value: &str, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(BORDER))
        .style(Style::default().bg(SURFACE));

    let text = format!("{}\n{}", value, label);
    let paragraph = Paragraph::new(text)
        .style(Style::default().fg(TEXT_PRIMARY))
        .alignment(Alignment::Center)
        .block(block);

    f.render_widget(paragraph, area);
}

fn draw_top_changed(f: &mut Frame, state: &AppState, area: Rect) {
    let diff = match state.diff {
        Some(ref d) => d,
        None => return,
    };

    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(area);

    let mut rows = vec![];
    for (i, idx) in diff.summary.top_changed_indices.iter().enumerate() {
        if let Some(layer) = diff.layers.get(*idx) {
            let bar = render_l2_bar(layer.aggregate_l2, 30);
            let severity = Severity::from_l2(layer.aggregate_l2);
            let color = l2_color(layer.aggregate_l2);
            
            let line = Line::from(vec![
                Span::styled(
                    format!("#{} ", i + 1),
                    Style::default().fg(TEXT_SECONDARY),
                ),
                Span::styled(
                    format!("{:20} ", layer.layer_name),
                    Style::default().fg(TEXT_PRIMARY),
                ),
                Span::styled(bar, Style::default().fg(color)),
                Span::styled(
                    format!(" {:>6.3} ", layer.aggregate_l2),
                    Style::default().fg(color),
                ),
                Span::styled(severity.as_str(), Style::default().fg(color)),
            ]);
            rows.push(line);
        }
    }

    let block = Block::default()
        .title(" Top Changed Layers ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(BORDER))
        .style(Style::default().bg(SURFACE));

    let paragraph = Paragraph::new(Text::from(rows))
        .style(Style::default().fg(TEXT_PRIMARY))
        .block(block);

    f.render_widget(paragraph, chunks[0]);

    let mut anomaly_lines = vec![];
    if diff.summary.anomalies.is_empty() {
        anomaly_lines.push(Line::from("No anomalies detected"));
    } else {
        anomaly_lines.push(Line::from(vec![
            Span::styled("[R] ", Style::default().fg(PINK)),
            Span::styled("Anomalies Detected", Style::default().fg(TEXT_PRIMARY).add_modifier(Modifier::BOLD)),
        ]));
        for anomaly in &diff.summary.anomalies {
            anomaly_lines.push(Line::from(format!(
                "  {} (z-score: {:.2})",
                anomaly.layer_name, anomaly.z_score
            )));
        }
    }

    let block = Block::default()
        .title(" Anomalies ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(BORDER))
        .style(Style::default().bg(SURFACE));

    let paragraph = Paragraph::new(Text::from(anomaly_lines))
        .style(Style::default().fg(TEXT_PRIMARY))
        .block(block);

    f.render_widget(paragraph, chunks[1]);
}

fn draw_detail(f: &mut Frame, state: &AppState, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([
            Constraint::Length(3),  // Header
            Constraint::Min(10),    // Main content
            Constraint::Length(3),  // Footer
        ])
        .split(area);

    draw_header(f, state, chunks[0]);
    draw_detail_content(f, state, chunks[1]);
    draw_footer(f, state, chunks[2]);
}

fn draw_detail_content(f: &mut Frame, state: &AppState, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
        .split(area);

    draw_layer_list(f, state, chunks[0]);
    draw_layer_detail(f, state, chunks[1]);
}

fn draw_layer_list(f: &mut Frame, state: &AppState, area: Rect) {
    let layers = get_filtered_layers(state);
    let mut rows = vec![];

    for (i, layer) in layers.iter().enumerate() {
        let is_selected = i == state.selected_layer;
        let bar = render_l2_bar(layer.aggregate_l2, 15);
        let color = l2_color(layer.aggregate_l2);
        
        let style = if is_selected {
            Style::default().bg(ACCENT).fg(BG).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(TEXT_PRIMARY)
        };

        let line = Line::from(vec![
            Span::styled(format!("{:>3} ", layer.layer_index.map_or("-".to_string(), |i| i.to_string())), style),
            Span::styled(format!("{:4} ", layer.layer_type.to_string()), style),
            Span::styled(bar, if is_selected { Style::default().fg(BG) } else { Style::default().fg(color) }),
            Span::styled(format!(" {:>5.3}", layer.aggregate_l2), style),
        ]);
        rows.push(line);
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

    let paragraph = Paragraph::new(Text::from(rows))
        .style(Style::default().fg(TEXT_PRIMARY))
        .block(block);

    f.render_widget(paragraph, area);
}

fn draw_layer_detail(f: &mut Frame, state: &AppState, area: Rect) {
    let layers = get_filtered_layers(state);
    let layer = match layers.get(state.selected_layer) {
        Some(l) => l,
        None => {
            let block = Block::default()
                .title(" Detail ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(BORDER));
            f.render_widget(block, area);
            return;
        }
    };

    let mut lines = vec![];
    lines.push(Line::from(vec![
        Span::styled("Layer: ", Style::default().fg(TEXT_SECONDARY)),
        Span::styled(&layer.layer_name, Style::default().fg(TEXT_PRIMARY).add_modifier(Modifier::BOLD)),
    ]));
    lines.push(Line::from(vec![
        Span::styled("Type:  ", Style::default().fg(TEXT_SECONDARY)),
        Span::styled(layer.layer_type.to_string(), Style::default().fg(ACCENT_INFO)),
    ]));
    lines.push(Line::from(vec![
        Span::styled("L2:    ", Style::default().fg(TEXT_SECONDARY)),
        Span::styled(format!("{:.6}", layer.aggregate_l2), Style::default().fg(l2_color(layer.aggregate_l2))),
    ]));
    lines.push(Line::from(vec![
        Span::styled("Params:", Style::default().fg(TEXT_SECONDARY)),
        Span::styled(format_params(layer.param_count), Style::default().fg(TEXT_PRIMARY)),
    ]));
    lines.push(Line::from(""));

    lines.push(Line::from(vec![
        Span::styled("Tensors:", Style::default().fg(TEXT_SECONDARY).add_modifier(Modifier::BOLD)),
    ]));

    for (i, tensor) in layer.tensors.iter().enumerate() {
        let is_selected = i == state.selected_tensor;
        let color = l2_color(tensor.l2_distance);
        let prefix = if is_selected { "> " } else { "  " };
        
        lines.push(Line::from(vec![
            Span::styled(prefix, if is_selected { Style::default().fg(ACCENT) } else { Style::default() }),
            Span::styled(
                format!("{:30} ", tensor.name),
                if is_selected { Style::default().fg(TEXT_PRIMARY).add_modifier(Modifier::BOLD) } else { Style::default().fg(TEXT_PRIMARY) },
            ),
            Span::styled(
                format!("{:?} ", tensor.shape),
                Style::default().fg(TEXT_SECONDARY),
            ),
            Span::styled(
                format!("L2={:.3}", tensor.l2_distance),
                Style::default().fg(color),
            ),
        ]));
    }

    let block = Block::default()
        .title(" Detail ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(BORDER))
        .style(Style::default().bg(SURFACE));

    let paragraph = Paragraph::new(Text::from(lines))
        .style(Style::default().fg(TEXT_PRIMARY))
        .block(block);

    f.render_widget(paragraph, area);
}

fn draw_footer(f: &mut Frame, state: &AppState, area: Rect) {
    let mut spans = vec![
        Span::styled("[↑↓/jk] navigate  ", Style::default().fg(TEXT_SECONDARY)),
        Span::styled("[Enter] select  ", Style::default().fg(TEXT_SECONDARY)),
        Span::styled("[s] sort  ", Style::default().fg(TEXT_SECONDARY)),
        Span::styled("[f] filter  ", Style::default().fg(TEXT_SECONDARY)),
        Span::styled("[J] JSON  ", Style::default().fg(TEXT_SECONDARY)),
        Span::styled("[?] help  ", Style::default().fg(TEXT_SECONDARY)),
        Span::styled("[q] quit", Style::default().fg(TEXT_SECONDARY)),
    ];

    if let Some(ref msg) = state.status_message {
        spans.push(Span::styled(format!("  |  {}", msg), Style::default().fg(ACCENT)));
    }

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(BORDER))
        .style(Style::default().bg(SURFACE));

    let paragraph = Paragraph::new(Line::from(spans))
        .style(Style::default().fg(TEXT_PRIMARY))
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
        Line::from("  s      Cycle sort mode (L2 -> Index -> Anomaly)"),
        Line::from("  f      Toggle filter (All -> Changed only)"),
        Line::from("  J      Export JSON to diff.json"),
        Line::from("  ?      Toggle this help"),
        Line::from("  q      Quit"),
    ];

    let block = Block::default()
        .title(" Help ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(ACCENT))
        .style(Style::default().bg(SURFACE));

    let paragraph = Paragraph::new(Text::from(help_text))
        .style(Style::default().fg(TEXT_PRIMARY))
        .block(block);

    let area = centered_rect(60, 60, area);
    f.render_widget(Clear, area);
    f.render_widget(paragraph, area);
}

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
