use crate::types::{AppState, FilterMode, HeatmapData, LayerDiff, Severity, SortMode, ViewMode};
use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::ExecutableCommand;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use ratatui::Frame;
use ratatui::Terminal;
use std::io;
use std::path::Path;
use std::sync::mpsc;
use std::time::{Duration, Instant};

// Theme
const BG: Color = Color::Rgb(10, 10, 10);
const SURFACE: Color = Color::Rgb(23, 23, 23);
const TEXT_PRIMARY: Color = Color::Rgb(229, 229, 229);
const TEXT_SECONDARY: Color = Color::Rgb(115, 115, 115);
const ACCENT: Color = Color::Rgb(16, 185, 129);
const ACCENT_INFO: Color = Color::Rgb(249, 115, 22);
const GREEN: Color = Color::Rgb(34, 197, 94);
const YELLOW: Color = Color::Rgb(234, 179, 8);
const ORANGE: Color = Color::Rgb(249, 115, 22);
const RED: Color = Color::Rgb(239, 68, 68);
const PINK: Color = Color::Rgb(236, 72, 153);
const BORDER: Color = Color::Rgb(38, 38, 38);

const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

// ============================================
// Public entry points
// ============================================

/// Run the TUI with a pre-computed diff result.
pub fn run_app(mut state: AppState) -> Result<()> {
    let mut terminal = setup_terminal()?;
    let res = run_main_loop(&mut terminal, &mut state);
    restore_terminal(&mut terminal)?;
    res
}

/// Show a loading screen while computing the diff in a background thread,
/// then transition directly into the main TUI.
pub fn run_with_loading(path_a: &Path, path_b: &Path) -> Result<()> {
    let mut terminal = setup_terminal()?;

    let path_a = path_a.to_path_buf();
    let path_b = path_b.to_path_buf();
    let (tx, rx) = mpsc::channel();

    std::thread::spawn(move || {
        let result = crate::diff::compute_diff(&path_a, &path_b);
        tx.send(result).ok();
    });

    let mut frame_idx: usize = 0;
    let tick = Duration::from_millis(80);
    let start = Instant::now();
    // Store paths for the loading screen display
    let display_a = path_a.display().to_string();
    let display_b = path_b.display().to_string();

    // Loading loop
    let diff_result = loop {
        terminal.draw(|f| {
            draw_loading(f, &display_a, &display_b, SPINNER[frame_idx % 10], start.elapsed().as_secs_f64())
        })?;
        frame_idx += 1;

        match rx.try_recv() {
            Ok(result) => break result,
            Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => {
                restore_terminal(&mut terminal)?;
                anyhow::bail!("Diff computation thread panicked");
            }
        }

        if crossterm::event::poll(tick)? {
            if let Event::Key(key) = event::read()? {
                let quit = key.code == KeyCode::Char('q')
                    || (key.modifiers == KeyModifiers::CONTROL
                        && key.code == KeyCode::Char('c'));
                if quit {
                    restore_terminal(&mut terminal)?;
                    return Ok(());
                }
            }
        }
    };

    let diff = diff_result?;
    let mut state = AppState::default();
    state.diff = Some(diff);

    let res = run_main_loop(&mut terminal, &mut state);
    restore_terminal(&mut terminal)?;
    res
}

fn setup_terminal() -> Result<Terminal<CrosstermBackend<io::Stdout>>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    stdout.execute(EnterAlternateScreen)?;
    Ok(Terminal::new(CrosstermBackend::new(stdout))?)
}

fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> Result<()> {
    disable_raw_mode()?;
    terminal.backend_mut().execute(LeaveAlternateScreen)?;
    Ok(())
}

fn run_main_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    state: &mut AppState,
) -> Result<()> {
    let mut last_tick = Instant::now();
    let tick_rate = Duration::from_millis(250);

    loop {
        terminal.draw(|f| draw_ui(f, state))?;

        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or(Duration::ZERO);

        if crossterm::event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                if handle_key_event(key, state) {
                    return Ok(());
                }
            }
        }

        if last_tick.elapsed() >= tick_rate {
            last_tick = Instant::now();
        }
    }
}

// ============================================
// Key handling
// ============================================

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
                state.selected_tensor = 0;
                state.show_heatmap = false;
                state.heatmap_data = None;
            } else {
                state.view_mode = ViewMode::Detail;
            }
        }
        KeyCode::Char('k') | KeyCode::Up => {
            if state.view_mode == ViewMode::Detail {
                state.selected_layer = state.selected_layer.saturating_sub(1);
                state.selected_tensor = 0;
                state.show_heatmap = false;
                state.heatmap_data = None;
            }
        }
        KeyCode::Char('h') | KeyCode::Left => {
            state.selected_tensor = state.selected_tensor.saturating_sub(1);
            state.show_heatmap = false;
            state.heatmap_data = None;
        }
        KeyCode::Char('l') | KeyCode::Right => {
            if state.diff.is_some() {
                let max = get_filtered_layers(state)
                    .get(state.selected_layer)
                    .map(|l| l.tensors.len().saturating_sub(1))
                    .unwrap_or(0);
                state.selected_tensor = (state.selected_tensor + 1).min(max);
                state.show_heatmap = false;
                state.heatmap_data = None;
            }
        }
        KeyCode::Enter => {
            if state.view_mode == ViewMode::Summary {
                state.view_mode = ViewMode::Detail;
            } else if state.show_heatmap {
                state.show_heatmap = false;
                state.heatmap_data = None;
            } else {
                match compute_heatmap(state) {
                    Some(data) => {
                        state.heatmap_data = Some(data);
                        state.show_heatmap = true;
                        state.status_message = None;
                    }
                    None => {
                        state.status_message = Some("Heatmap unavailable for this tensor".to_string());
                    }
                }
            }
        }
        KeyCode::Char('b') => {
            if state.show_heatmap {
                state.show_heatmap = false;
                state.heatmap_data = None;
            } else {
                state.view_mode = ViewMode::Summary;
                state.selected_layer = 0;
                state.selected_tensor = 0;
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

// ============================================
// Heatmap computation
// ============================================

fn compute_heatmap(state: &AppState) -> Option<HeatmapData> {
    let diff = state.diff.as_ref()?;
    let layers = get_filtered_layers(state);
    let layer = layers.get(state.selected_layer)?;
    let tensor = layer.tensors.get(state.selected_tensor)?;

    let tensor_name = tensor.name.clone();
    let shape = tensor.shape.clone();
    let path_a = diff.model_a.clone();
    let path_b = diff.model_b.clone();

    let snap_a = crate::loader::load(Path::new(&path_a)).ok()?;
    let snap_b = crate::loader::load(Path::new(&path_b)).ok()?;
    let data_a = crate::loader::load_tensor_data(&snap_a, &tensor_name).ok()?;
    let data_b = crate::loader::load_tensor_data(&snap_b, &tensor_name).ok()?;

    let deltas: Vec<f32> = data_a.iter().zip(data_b.iter()).map(|(a, b)| (b - a).abs()).collect();

    // Determine 2D display dimensions (max 64 cols × 20 rows)
    let (src_rows, src_cols) = tensor_2d_shape(&shape);
    let max_cols: usize = 64;
    let max_rows: usize = 20;
    let dst_cols = src_cols.min(max_cols);
    let dst_rows = src_rows.min(max_rows);

    let grid = downsample_grid(&deltas, src_rows, src_cols, dst_rows, dst_cols);

    let min = grid.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = grid.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    Some(HeatmapData { grid, rows: dst_rows, cols: dst_cols, min, max, tensor_name })
}

fn tensor_2d_shape(shape: &[usize]) -> (usize, usize) {
    match shape.len() {
        0 => (1, 1),
        1 => {
            let n = shape[0];
            let cols = (n as f64).sqrt().ceil() as usize;
            let rows = n.div_ceil(cols);
            (rows, cols)
        }
        _ => {
            let rows = shape[0];
            let cols: usize = shape[1..].iter().product();
            (rows, cols)
        }
    }
}

fn downsample_grid(
    data: &[f32],
    src_rows: usize,
    src_cols: usize,
    dst_rows: usize,
    dst_cols: usize,
) -> Vec<f32> {
    let mut grid = vec![0.0f32; dst_rows * dst_cols];

    for dr in 0..dst_rows {
        for dc in 0..dst_cols {
            let sr_start = dr * src_rows / dst_rows;
            let sr_end = ((dr + 1) * src_rows / dst_rows).max(sr_start + 1);
            let sc_start = dc * src_cols / dst_cols;
            let sc_end = ((dc + 1) * src_cols / dst_cols).max(sc_start + 1);

            let mut sum = 0.0f32;
            let mut count = 0usize;
            for sr in sr_start..sr_end.min(src_rows) {
                for sc in sc_start..sc_end.min(src_cols) {
                    let idx = sr * src_cols + sc;
                    if let Some(&v) = data.get(idx) {
                        sum += v;
                        count += 1;
                    }
                }
            }
            grid[dr * dst_cols + dc] = if count > 0 { sum / count as f32 } else { 0.0 };
        }
    }

    grid
}

// ============================================
// Layer helpers
// ============================================

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
            layers.sort_by(|a, b| {
                b.aggregate_l2.partial_cmp(&a.aggregate_l2).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        SortMode::LayerIndex => {
            layers.sort_by(|a, b| match (a.layer_index, b.layer_index) {
                (Some(ai), Some(bi)) => ai.cmp(&bi),
                _ => std::cmp::Ordering::Equal,
            });
        }
        SortMode::AnomalyScore => {
            layers.sort_by(|a, b| {
                b.anomaly_score.partial_cmp(&a.anomaly_score).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
    }

    layers
}

// ============================================
// Top-level draw dispatch
// ============================================

fn draw_ui(f: &mut Frame, state: &AppState) {
    let area = f.area();
    f.render_widget(Block::default().style(Style::default().bg(BG)), area);

    if state.show_help {
        draw_help(f, area);
        return;
    }

    match state.view_mode {
        ViewMode::Summary => draw_summary(f, state, area),
        ViewMode::Detail => draw_detail(f, state, area),
    }
}

// ============================================
// Loading screen
// ============================================

fn draw_loading(f: &mut Frame, path_a: &str, path_b: &str, spinner: &str, elapsed: f64) {
    let area = f.area();
    f.render_widget(Block::default().style(Style::default().bg(BG)), area);

    let popup = centered_rect(70, 50, area);
    f.render_widget(ratatui::widgets::Clear, popup);

    let lines = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled(format!("  {}  ", spinner), Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)),
            Span::styled("NEURALDIFF", Style::default().fg(TEXT_PRIMARY).add_modifier(Modifier::BOLD)),
            Span::styled("  Computing diff...", Style::default().fg(TEXT_SECONDARY)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("  A → ", Style::default().fg(ACCENT)),
            Span::styled(truncate_path(path_a, 50), Style::default().fg(TEXT_PRIMARY)),
        ]),
        Line::from(vec![
            Span::styled("  B → ", Style::default().fg(ACCENT_INFO)),
            Span::styled(truncate_path(path_b, 50), Style::default().fg(TEXT_PRIMARY)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled(format!("  Elapsed: {:.1}s", elapsed), Style::default().fg(TEXT_SECONDARY)),
            Span::styled("                    ", Style::default()),
            Span::styled("[q] Cancel", Style::default().fg(TEXT_SECONDARY)),
        ]),
        Line::from(""),
    ];

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(ACCENT))
        .style(Style::default().bg(SURFACE));

    let paragraph = Paragraph::new(Text::from(lines))
        .block(block)
        .alignment(Alignment::Left);

    f.render_widget(paragraph, popup);
}

// ============================================
// Summary view
// ============================================

fn draw_summary(f: &mut Frame, state: &AppState, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([
            Constraint::Length(4),
            Constraint::Length(6),
            Constraint::Min(12),
            Constraint::Length(3),
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
            f.render_widget(Block::default().borders(Borders::ALL).border_style(Style::default().fg(BORDER)), area);
            return;
        }
    };

    let lines = vec![
        Line::from(vec![Span::styled(
            " MODEL COMPARISON ",
            Style::default().fg(TEXT_PRIMARY).add_modifier(Modifier::BOLD),
        )]),
        Line::from(vec![
            Span::styled("A: ", Style::default().fg(ACCENT)),
            Span::styled(truncate_path(&diff.model_a, 40), Style::default().fg(TEXT_PRIMARY)),
            Span::styled("  |  ", Style::default().fg(TEXT_SECONDARY)),
            Span::styled("B: ", Style::default().fg(ACCENT_INFO)),
            Span::styled(truncate_path(&diff.model_b, 40), Style::default().fg(TEXT_PRIMARY)),
        ]),
        Line::from(vec![
            Span::styled("Parameters: ", Style::default().fg(TEXT_SECONDARY)),
            Span::styled(format_params(diff.total_params), Style::default().fg(TEXT_PRIMARY).add_modifier(Modifier::BOLD)),
            Span::styled("  |  Layers: ", Style::default().fg(TEXT_SECONDARY)),
            Span::styled(diff.summary.total_layers.to_string(), Style::default().fg(TEXT_PRIMARY).add_modifier(Modifier::BOLD)),
            Span::styled("  |  Changed: ", Style::default().fg(TEXT_SECONDARY)),
            Span::styled(
                format!("{}  ({:.1}%)", diff.summary.changed_layers, diff.summary.change_ratio_percent),
                Style::default().fg(if diff.summary.changed_layers > 0 { ORANGE } else { GREEN }).add_modifier(Modifier::BOLD),
            ),
        ]),
    ];

    let paragraph = Paragraph::new(Text::from(lines))
        .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(ACCENT)).style(Style::default().bg(SURFACE)));

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

    let legend_text = vec![
        Line::from(vec![Span::styled("LEGEND:", Style::default().fg(TEXT_SECONDARY).add_modifier(Modifier::BOLD))]),
        Line::from(vec![
            Span::styled("L2 Distance ", Style::default().fg(TEXT_PRIMARY)),
            Span::styled("— Magnitude of change (0=identical, >1=drastic)", Style::default().fg(TEXT_SECONDARY)),
        ]),
        Line::from(vec![
            Span::styled("Cosine Sim  ", Style::default().fg(TEXT_PRIMARY)),
            Span::styled("— Direction similarity (-1=opposite, 1=identical)", Style::default().fg(TEXT_SECONDARY)),
        ]),
        Line::from(vec![
            Span::styled("Z-Score     ", Style::default().fg(TEXT_PRIMARY)),
            Span::styled("— Unusualness vs other layers (>2.0 = anomaly)", Style::default().fg(TEXT_SECONDARY)),
        ]),
    ];

    f.render_widget(
        Paragraph::new(Text::from(legend_text))
            .block(Block::default().title(" What These Numbers Mean ").borders(Borders::ALL).border_style(Style::default().fg(BORDER)).style(Style::default().bg(SURFACE))),
        chunks[0],
    );

    let missing_note = if !diff.summary.missing_tensors.is_empty() {
        format!("  ⚠ {} tensor(s) only in one model", diff.summary.missing_tensors.len())
    } else {
        "  All tensors matched".to_string()
    };

    let scale_text = vec![
        Line::from(vec![Span::styled("SEVERITY SCALE:", Style::default().fg(TEXT_SECONDARY).add_modifier(Modifier::BOLD))]),
        Line::from(vec![Span::styled("[LOW]  ", Style::default().fg(GREEN)), Span::styled("< 0.001  — unchanged", Style::default().fg(TEXT_SECONDARY))]),
        Line::from(vec![Span::styled("[MED]  ", Style::default().fg(YELLOW)), Span::styled("0.001–0.3 — minor", Style::default().fg(TEXT_SECONDARY))]),
        Line::from(vec![Span::styled("[HIGH] ", Style::default().fg(ORANGE)), Span::styled("0.3–0.6  — significant", Style::default().fg(TEXT_SECONDARY))]),
        Line::from(vec![Span::styled("[CRIT] ", Style::default().fg(RED)), Span::styled(format!("> 0.6    — drastic{}", missing_note), Style::default().fg(TEXT_SECONDARY))]),
    ];

    f.render_widget(
        Paragraph::new(Text::from(scale_text))
            .block(Block::default().title(" Severity Levels ").borders(Borders::ALL).border_style(Style::default().fg(BORDER)).style(Style::default().bg(SURFACE))),
        chunks[1],
    );
}

fn draw_top_changed(f: &mut Frame, state: &AppState, area: Rect) {
    let diff = match state.diff {
        Some(ref d) => d,
        None => return,
    };

    let mut lines = vec![
        Line::from(vec![
            Span::styled("#  ", Style::default().fg(TEXT_SECONDARY)),
            Span::styled(format!("{:<18} ", "Layer Name"), Style::default().fg(TEXT_SECONDARY).add_modifier(Modifier::BOLD)),
            Span::styled(format!("{:5} ", "Type"), Style::default().fg(TEXT_SECONDARY).add_modifier(Modifier::BOLD)),
            Span::styled(format!("{:>10}  ", "L2"), Style::default().fg(TEXT_SECONDARY).add_modifier(Modifier::BOLD)),
            Span::styled(format!("{:<8}", "Severity"), Style::default().fg(TEXT_SECONDARY).add_modifier(Modifier::BOLD)),
            Span::styled("  Change Bar", Style::default().fg(TEXT_SECONDARY).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
    ];

    for (i, idx) in diff.summary.top_changed_indices.iter().enumerate().take(10) {
        if let Some(layer) = diff.layers.get(*idx) {
            let bar = render_l2_bar(layer.aggregate_l2, 20);
            let severity = Severity::from_l2(layer.aggregate_l2);
            let color = l2_color(layer.aggregate_l2);

            lines.push(Line::from(vec![
                Span::styled(format!("{:>2} ", i + 1), Style::default().fg(TEXT_SECONDARY)),
                Span::styled(format!("{:<18} ", layer.layer_name), Style::default().fg(TEXT_PRIMARY)),
                Span::styled(format!("{:5} ", layer.layer_type), Style::default().fg(TEXT_SECONDARY)),
                Span::styled(format!("{:>10.4}  ", layer.aggregate_l2), Style::default().fg(color)),
                Span::styled(format!("{:<8}", severity.as_str()), Style::default().fg(color).add_modifier(Modifier::BOLD)),
                Span::styled(format!("  {}", bar), Style::default().fg(color)),
            ]));
        }
    }

    if diff.summary.anomalies.is_empty() {
        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::styled("No anomalies detected — all changes within normal range", Style::default().fg(GREEN)),
        ]));
    } else {
        lines.push(Line::from(""));
        lines.push(Line::from(vec![Span::styled("⚠ ANOMALIES: ", Style::default().fg(PINK).add_modifier(Modifier::BOLD))]));
        for anomaly in &diff.summary.anomalies {
            lines.push(Line::from(vec![
                Span::styled("  • ", Style::default().fg(PINK)),
                Span::styled(format!("{} ", anomaly.layer_name), Style::default().fg(TEXT_PRIMARY)),
                Span::styled(format!("(z={:.2})", anomaly.z_score), Style::default().fg(TEXT_SECONDARY)),
            ]));
        }
    }

    f.render_widget(
        Paragraph::new(Text::from(lines))
            .block(Block::default().title(" Top Changed Layers (Enter to explore) ").borders(Borders::ALL).border_style(Style::default().fg(BORDER)).style(Style::default().bg(SURFACE))),
        area,
    );
}

// ============================================
// Detail view
// ============================================

fn draw_detail(f: &mut Frame, state: &AppState, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([Constraint::Length(3), Constraint::Min(10), Constraint::Length(3)])
        .split(area);

    draw_detail_header(f, state, chunks[0]);

    if state.show_heatmap {
        draw_heatmap(f, state, chunks[1]);
    } else {
        draw_detail_content(f, state, chunks[1]);
    }

    draw_footer(f, state, chunks[2]);
}

fn draw_detail_header(f: &mut Frame, state: &AppState, area: Rect) {
    let diff = match state.diff {
        Some(ref d) => d,
        None => {
            f.render_widget(Block::default().borders(Borders::ALL), area);
            return;
        }
    };

    let mut spans = vec![
        Span::styled(" NEURALDIFF ", Style::default().fg(TEXT_PRIMARY).add_modifier(Modifier::BOLD)),
    ];
    spans.push(Span::styled(truncate_path(&diff.model_a, 20), Style::default().fg(TEXT_SECONDARY)));
    spans.push(Span::styled(" → ", Style::default().fg(ACCENT)));
    spans.push(Span::styled(truncate_path(&diff.model_b, 20), Style::default().fg(TEXT_SECONDARY)));

    f.render_widget(
        Paragraph::new(Line::from(spans))
            .alignment(Alignment::Center)
            .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(ACCENT)).style(Style::default().bg(SURFACE))),
        area,
    );
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
        let color_style = if is_selected { Style::default().fg(BG) } else { Style::default().fg(color) };

        lines.push(Line::from(vec![
            Span::styled(format!("{:>3} ", layer.layer_index.map_or("-".to_string(), |i| i.to_string())), style),
            Span::styled(format!("{:4} ", layer.layer_type), style),
            Span::styled(bar, color_style),
            Span::styled(format!(" {:>5.3} ", layer.aggregate_l2), style),
            Span::styled(severity.as_str(), color_style),
        ]));
    }

    let title = format!(" Layers [{}] ", match state.sort_mode {
        SortMode::L2Desc => "L2↓",
        SortMode::LayerIndex => "Idx",
        SortMode::AnomalyScore => "Anom",
    });

    f.render_widget(
        Paragraph::new(Text::from(lines))
            .block(Block::default().title(title).borders(Borders::ALL).border_style(Style::default().fg(BORDER)).style(Style::default().bg(SURFACE))),
        area,
    );
}

fn draw_tensor_comparison(f: &mut Frame, state: &AppState, area: Rect) {
    let layers = get_filtered_layers(state);
    let layer = match layers.get(state.selected_layer) {
        Some(l) => l,
        None => {
            f.render_widget(
                Block::default().title(" Tensor Comparison ").borders(Borders::ALL).border_style(Style::default().fg(BORDER)),
                area,
            );
            return;
        }
    };

    let title = if let Some(ref diff) = state.diff {
        format!(" {} → {} ", truncate_path(&diff.model_a, 20), truncate_path(&diff.model_b, 20))
    } else {
        " Tensor Comparison ".to_string()
    };

    let mut lines = vec![];

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
        Span::styled("  |  ", Style::default().fg(TEXT_SECONDARY)),
        Span::styled(
            format!("{}/{} changed", layer.tensors.iter().filter(|t| t.changed).count(), layer.tensors.len()),
            Style::default().fg(if layer.tensors.iter().any(|t| t.changed) { RED } else { GREEN }),
        ),
    ]));

    lines.push(Line::from(""));

    lines.push(Line::from(vec![
        Span::styled(format!("{:<26}", "  Tensor Name"), Style::default().fg(TEXT_SECONDARY).add_modifier(Modifier::BOLD)),
        Span::styled(format!("{:>12}", "Shape"), Style::default().fg(TEXT_SECONDARY).add_modifier(Modifier::BOLD)),
        Span::styled(format!("{:>10}", "L2"), Style::default().fg(TEXT_SECONDARY).add_modifier(Modifier::BOLD)),
        Span::styled(format!("{:>10}", "Cosine"), Style::default().fg(TEXT_SECONDARY).add_modifier(Modifier::BOLD)),
        Span::styled(format!("{:>12}", "Max Delta"), Style::default().fg(TEXT_SECONDARY).add_modifier(Modifier::BOLD)),
        Span::styled(format!("{:>8}", "Status"), Style::default().fg(TEXT_SECONDARY).add_modifier(Modifier::BOLD)),
    ]));
    lines.push(Line::from(vec![Span::styled("─".repeat(80), Style::default().fg(BORDER))]));

    for (i, tensor) in layer.tensors.iter().enumerate() {
        let is_selected = i == state.selected_tensor;
        let color = l2_color(tensor.l2_distance);
        let prefix = if is_selected { "▶" } else { " " };
        let bold = if is_selected { Modifier::BOLD } else { Modifier::empty() };

        let shape_str = format!("{:?}", tensor.shape);
        let shape_display = if shape_str.len() > 12 { format!("{}..]", &shape_str[..9]) } else { shape_str };

        let cosine_color = if tensor.cosine_similarity > 0.9 { GREEN } else if tensor.cosine_similarity > 0.5 { YELLOW } else { RED };

        lines.push(Line::from(vec![
            Span::styled(format!(" {:^1}", prefix), Style::default().fg(ACCENT).add_modifier(Modifier::BOLD | bold)),
            Span::styled(format!("{:<24}", truncate_str(&tensor.name, 24)), Style::default().fg(TEXT_PRIMARY).add_modifier(bold)),
            Span::styled(format!("{:>12}", shape_display), Style::default().fg(TEXT_SECONDARY).add_modifier(bold)),
            Span::styled(format!("{:>10.4}", tensor.l2_distance), Style::default().fg(color).add_modifier(bold)),
            Span::styled(format!("{:>10.4}", tensor.cosine_similarity), Style::default().fg(cosine_color).add_modifier(bold)),
            Span::styled(format!("{:>12.6}", tensor.max_delta), Style::default().fg(color).add_modifier(bold)),
            Span::styled(format!("{:>8}", if tensor.changed { "CHANGED" } else { "SAME" }), Style::default().fg(if tensor.changed { RED } else { GREEN }).add_modifier(bold)),
        ]));
    }

    lines.push(Line::from(""));
    lines.push(Line::from(vec![Span::styled("─".repeat(80), Style::default().fg(BORDER))]));

    let total = layer.tensors.len();
    if total > 0 {
        let low = layer.tensors.iter().filter(|t| t.l2_distance < 0.1).count();
        let med = layer.tensors.iter().filter(|t| (0.1..0.5).contains(&t.l2_distance)).count();
        let high = layer.tensors.iter().filter(|t| t.l2_distance >= 0.5).count();

        lines.push(Line::from(vec![
            Span::styled("Distribution: ", Style::default().fg(TEXT_SECONDARY)),
            Span::styled(format!("Low({}) ", low), Style::default().fg(GREEN)),
            Span::styled("█".repeat((low * 16 / total).max(if low > 0 { 1 } else { 0 })), Style::default().fg(GREEN)),
            Span::styled(format!("  Med({}) ", med), Style::default().fg(YELLOW)),
            Span::styled("█".repeat((med * 16 / total).max(if med > 0 { 1 } else { 0 })), Style::default().fg(YELLOW)),
            Span::styled(format!("  High({}) ", high), Style::default().fg(RED)),
            Span::styled("█".repeat((high * 16 / total).max(if high > 0 { 1 } else { 0 })), Style::default().fg(RED)),
        ]));
    }

    lines.push(Line::from(vec![
        Span::styled("  [Enter] Show heatmap for selected tensor", Style::default().fg(TEXT_SECONDARY)),
    ]));

    f.render_widget(
        Paragraph::new(Text::from(lines))
            .block(Block::default().title(title).borders(Borders::ALL).border_style(Style::default().fg(BORDER)).style(Style::default().bg(SURFACE))),
        area,
    );
}

// ============================================
// Heatmap rendering
// ============================================

fn draw_heatmap(f: &mut Frame, state: &AppState, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(35), Constraint::Percentage(65)])
        .split(area);

    draw_layer_list(f, state, chunks[0]);

    let heatmap = match &state.heatmap_data {
        Some(h) => h,
        None => {
            f.render_widget(
                Paragraph::new("No heatmap data")
                    .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(BORDER))),
                chunks[1],
            );
            return;
        }
    };

    let range = heatmap.max - heatmap.min;
    let mut lines = vec![];

    lines.push(Line::from(vec![
        Span::styled("Tensor: ", Style::default().fg(TEXT_SECONDARY)),
        Span::styled(&heatmap.tensor_name, Style::default().fg(TEXT_PRIMARY).add_modifier(Modifier::BOLD)),
    ]));
    lines.push(Line::from(vec![
        Span::styled(format!("Grid: {}×{}  ", heatmap.rows, heatmap.cols), Style::default().fg(TEXT_SECONDARY)),
        Span::styled(format!("min={:.4}  max={:.4}", heatmap.min, heatmap.max), Style::default().fg(TEXT_SECONDARY)),
        Span::styled("  [Enter/b] Back", Style::default().fg(TEXT_SECONDARY)),
    ]));
    lines.push(Line::from(""));

    // Color scale legend
    lines.push(Line::from(vec![
        Span::styled("Scale: ", Style::default().fg(TEXT_SECONDARY)),
        Span::styled("░ low ", Style::default().fg(GREEN)),
        Span::styled("▒ med ", Style::default().fg(YELLOW)),
        Span::styled("▓ high ", Style::default().fg(ORANGE)),
        Span::styled("█ max", Style::default().fg(RED)),
    ]));
    lines.push(Line::from(""));

    for row in 0..heatmap.rows {
        let mut spans = vec![];
        for col in 0..heatmap.cols {
            let val = heatmap.grid[row * heatmap.cols + col];
            let norm = if range > 1e-9 { (val - heatmap.min) / range } else { 0.0 };
            let (ch, color) = heatmap_cell(norm);
            spans.push(Span::styled(ch, Style::default().fg(color)));
        }
        lines.push(Line::from(spans));
    }

    f.render_widget(
        Paragraph::new(Text::from(lines))
            .block(Block::default().title(" Δ Heatmap (absolute delta) ").borders(Borders::ALL).border_style(Style::default().fg(ACCENT)).style(Style::default().bg(SURFACE))),
        chunks[1],
    );
}

fn heatmap_cell(norm: f32) -> (&'static str, Color) {
    match norm {
        n if n < 0.15 => ("░", GREEN),
        n if n < 0.35 => ("░", YELLOW),
        n if n < 0.55 => ("▒", YELLOW),
        n if n < 0.75 => ("▓", ORANGE),
        _ => ("█", RED),
    }
}

// ============================================
// Footer & Help
// ============================================

fn draw_footer(f: &mut Frame, state: &AppState, area: Rect) {
    let heatmap_hint = if state.view_mode == ViewMode::Detail && state.show_heatmap {
        "[Enter/b] Exit heatmap  "
    } else if state.view_mode == ViewMode::Detail {
        "[Enter] Heatmap  "
    } else {
        "[Enter] Explore  "
    };

    let mut spans = vec![
        Span::styled("[↑↓/jk] Navigate  ", Style::default().fg(TEXT_SECONDARY)),
        Span::styled("[←→/hl] Tensor  ", Style::default().fg(TEXT_SECONDARY)),
        Span::styled(heatmap_hint, Style::default().fg(TEXT_SECONDARY)),
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

    f.render_widget(
        Paragraph::new(Line::from(spans))
            .alignment(Alignment::Center)
            .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(BORDER)).style(Style::default().bg(SURFACE))),
        area,
    );
}

fn draw_help(f: &mut Frame, area: Rect) {
    let help_text = vec![
        Line::from(vec![Span::styled("Keyboard Shortcuts", Style::default().fg(TEXT_PRIMARY).add_modifier(Modifier::BOLD))]),
        Line::from(""),
        Line::from("Navigation:"),
        Line::from("  ↑/k    Move up in layer list"),
        Line::from("  ↓/j    Move down in layer list"),
        Line::from("  ←/h    Previous tensor"),
        Line::from("  →/l    Next tensor"),
        Line::from("  Enter  Heatmap for selected tensor / Enter detail view"),
        Line::from("  b      Back (exit heatmap → exit detail → summary)"),
        Line::from(""),
        Line::from("Commands:"),
        Line::from("  s      Cycle sort (L2↓ → Index → Anomaly)"),
        Line::from("  f      Toggle filter (All ↔ Changed only)"),
        Line::from("  J      Export full diff to diff.json"),
        Line::from("  ?      Toggle this help"),
        Line::from("  q      Quit"),
        Line::from(""),
        Line::from("Metrics:"),
        Line::from("  L2 Distance  — Weight change magnitude (0=identical)"),
        Line::from("  Cosine Sim   — Change direction (1=same direction)"),
        Line::from("  Max Delta    — Largest single weight change"),
        Line::from("  Z-Score      — Unusualness vs other layers (>2=anomaly)"),
    ];

    let popup = centered_rect(70, 80, area);
    f.render_widget(Clear, popup);
    f.render_widget(
        Paragraph::new(Text::from(help_text))
            .block(Block::default().title(" Help ").borders(Borders::ALL).border_style(Style::default().fg(ACCENT)).style(Style::default().bg(SURFACE))),
        popup,
    );
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
    format!("{}{}", "█".repeat(filled), "░".repeat(width - filled))
}

fn l2_color(l2: f32) -> Color {
    if l2 < 0.001 { TEXT_SECONDARY }
    else if l2 < 0.3 { GREEN }
    else if l2 < 0.6 { YELLOW }
    else if l2 < 0.8 { ORANGE }
    else { RED }
}

fn format_params(n: u64) -> String {
    if n >= 1_000_000_000 { format!("{:.2}B", n as f64 / 1_000_000_000.0) }
    else if n >= 1_000_000 { format!("{:.2}M", n as f64 / 1_000_000.0) }
    else if n >= 1_000 { format!("{:.2}K", n as f64 / 1_000.0) }
    else { n.to_string() }
}

fn truncate_path(path: &str, max_len: usize) -> String {
    if path.len() <= max_len { path.to_string() }
    else { format!("...{}", &path[path.len().saturating_sub(max_len - 3)..]) }
}

fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len { s.to_string() }
    else { format!("{}…", &s[..max_len.saturating_sub(1)]) }
}
