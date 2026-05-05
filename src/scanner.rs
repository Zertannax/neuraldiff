use anyhow::Result;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub path: PathBuf,
    pub name: String,
    pub size_mb: f64,
    pub location: String,
}

pub fn scan_for_models() -> Result<Vec<ModelInfo>> {
    let mut models = Vec::new();
    let mut seen = HashSet::new();

    for (dir, max_depth) in get_scan_roots() {
        scan_dir_recursive(&dir, 0, max_depth, &mut seen, &mut models);
    }

    models.sort_by(|a, b| b.size_mb.partial_cmp(&a.size_mb).unwrap_or(std::cmp::Ordering::Equal));
    Ok(models)
}

/// Scan a single directory tree for .safetensors files. Used by `scan --root`.
pub fn scan_in_root(root: &Path, max_depth: usize) -> Result<Vec<ModelInfo>> {
    let mut models = Vec::new();
    let mut seen = HashSet::new();
    scan_dir_recursive(root, 0, max_depth, &mut seen, &mut models);
    models.sort_by(|a, b| b.size_mb.partial_cmp(&a.size_mb).unwrap_or(std::cmp::Ordering::Equal));
    Ok(models)
}

fn scan_dir_recursive(
    dir: &Path,
    depth: usize,
    max_depth: usize,
    seen: &mut HashSet<PathBuf>,
    models: &mut Vec<ModelInfo>,
) {
    if depth > max_depth {
        return;
    }

    // If this dir is itself a sharded model, surface it as one entry and don't recurse.
    let index_file = dir.join("model.safetensors.index.json");
    if index_file.is_file() {
        if seen.insert(dir.to_path_buf()) {
            let total_size_mb = total_safetensors_size_mb(dir);
            let name = dir
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string();
            let location = format_location(dir.parent().unwrap_or(dir));
            models.push(ModelInfo {
                path: dir.to_path_buf(),
                name,
                size_mb: total_size_mb,
                location,
            });
        }
        return;
    }

    let Ok(entries) = std::fs::read_dir(dir) else { return };

    for entry in entries.flatten() {
        let path = entry.path();

        if path.is_dir() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name.starts_with('.') && name != ".cache" {
                continue;
            }
            if is_skip_dir(name) {
                continue;
            }
            scan_dir_recursive(&path, depth + 1, max_depth, seen, models);
        } else if path.extension().is_some_and(|ext| ext == "safetensors") {
            if seen.insert(path.clone()) {
                if let Ok(meta) = path.metadata() {
                    let size_mb = meta.len() as f64 / (1024.0 * 1024.0);
                    let name = path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("unknown")
                        .to_string();
                    let location =
                        format_location(path.parent().unwrap_or(dir));
                    models.push(ModelInfo { path, name, size_mb, location });
                }
            }
        }
    }
}

/// Sum the size in MB of every .safetensors file at the top level of `dir`.
/// Used to display a meaningful size for sharded directories.
fn total_safetensors_size_mb(dir: &Path) -> f64 {
    let Ok(entries) = std::fs::read_dir(dir) else { return 0.0 };
    let mut total: u64 = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() && path.extension().is_some_and(|e| e == "safetensors") {
            if let Ok(meta) = path.metadata() {
                total += meta.len();
            }
        }
    }
    total as f64 / (1024.0 * 1024.0)
}

/// Returns (root_dir, max_depth) pairs.
/// HuggingFace hub gets a deeper limit because its path structure is 5+ levels deep.
fn get_scan_roots() -> Vec<(PathBuf, usize)> {
    let mut roots = Vec::new();

    if let Ok(current) = std::env::current_dir() {
        roots.push((current, 5));
    }

    if let Some(home) = dirs::home_dir() {
        // Scan home directory itself (where git clone often puts repos)
        roots.push((home.clone(), 2));

        // HuggingFace hub: hub/models--x--y/snapshots/hash/model.safetensors — needs depth 6
        roots.push((home.join(".cache/huggingface"), 6));
        roots.push((home.join(".cache/transformers"), 5));

        // Common user-created model dirs
        for sub in &[
            "Downloads", "Documents", "Desktop", "models", "AI", "ml",
            "checkpoints", "weights", "huggingface", "transformers",
        ] {
            roots.push((home.join(sub), 5));
        }
    }

    #[cfg(target_os = "windows")]
    {
        // Scan common drives on Windows
        for drive in &["C:\\", "D:\\", "E:\\"] {
            let path = PathBuf::from(drive);
            if path.exists() {
                roots.push((path, 4));
            }
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        roots.push((PathBuf::from("/tmp"), 3));
        roots.push((PathBuf::from("/models"), 4));
        roots.push((PathBuf::from("/opt"), 4));
        roots.push((PathBuf::from("/usr/share"), 3));

        // WSL: also scan the Windows user home and any /mnt/<drive>/Users/<user>/
        // dirs that exist, so models stored on the Windows side are picked up.
        if is_wsl() {
            let user = std::env::var("USER").unwrap_or_default();
            for drive_letter in &["c", "d", "e", "f"] {
                let win_home = PathBuf::from(format!("/mnt/{}/Users/{}", drive_letter, user));
                if win_home.exists() {
                    roots.push((win_home.clone(), 3));
                    for sub in &["Downloads", "Documents", "Desktop", "models", "AI", "ml"] {
                        let p = win_home.join(sub);
                        if p.exists() {
                            roots.push((p, 5));
                        }
                    }
                }
            }
        }
    }

    roots
}

fn is_wsl() -> bool {
    std::fs::read_to_string("/proc/version")
        .map(|s| s.to_lowercase().contains("microsoft") || s.to_lowercase().contains("wsl"))
        .unwrap_or(false)
}

fn is_skip_dir(name: &str) -> bool {
    matches!(
        name,
        "AppData"
            | "Windows"
            | "Program Files"
            | "Program Files (x86)"
            | "ProgramData"
            | "$Recycle.Bin"
            | "$RECYCLE.BIN"
            | "System Volume Information"
            | "node_modules"
            | "target"           // Rust build artifacts
            | "venv"
            | ".venv"
            | "__pycache__"
            | "snap"
            | "miniforge3"
            | "anaconda3"
            | "Anaconda3"
            | "Library"          // macOS, but harmless on Linux
    )
}

fn format_location(path: &Path) -> String {
    if let Some(home) = dirs::home_dir()
        && let Ok(stripped) = path.strip_prefix(&home)
    {
        return format!("~/{}", stripped.display());
    }
    // Friendly display of WSL Windows mounts: /mnt/c/Users/foo -> C:\Users\foo
    let s = path.display().to_string();
    if let Some(rest) = s.strip_prefix("/mnt/") {
        if let Some((drive, tail)) = rest.split_once('/') {
            if drive.len() == 1 {
                return format!("{}:\\{}", drive.to_uppercase(), tail.replace('/', "\\"));
            }
        }
    }
    s
}

pub fn format_size(size_mb: f64) -> String {
    if size_mb >= 1024.0 {
        format!("{:.1} GB", size_mb / 1024.0)
    } else if size_mb >= 1.0 {
        format!("{:.1} MB", size_mb)
    } else {
        format!("{:.0} KB", size_mb * 1024.0)
    }
}

// Model Selection UI
use crate::terminal::TerminalGuard;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Frame;
use ratatui::Terminal;
use std::io;
use std::time::{Duration, Instant};

const BG: Color = Color::Rgb(10, 10, 10);
const SURFACE: Color = Color::Rgb(23, 23, 23);
const TEXT_PRIMARY: Color = Color::Rgb(229, 229, 229);
const TEXT_SECONDARY: Color = Color::Rgb(115, 115, 115);
const ACCENT: Color = Color::Rgb(16, 185, 129);
const ACCENT_INFO: Color = Color::Rgb(249, 115, 22);
const BORDER: Color = Color::Rgb(38, 38, 38);
const RED: Color = Color::Rgb(239, 68, 68);

pub struct ModelSelectionState {
    pub models: Vec<ModelInfo>,
    pub selected_a: Option<usize>,
    pub selected_b: Option<usize>,
    pub current_selection: SelectionMode,
    pub status_message: Option<String>,
}

#[derive(Clone, Copy)]
pub enum SelectionMode {
    SelectA,
    SelectB,
}

impl ModelSelectionState {
    pub fn new() -> Result<Self> {
        let models = scan_for_models()?;
        Ok(Self::from_models(models))
    }

    pub fn from_models(models: Vec<ModelInfo>) -> Self {
        Self {
            models,
            selected_a: None,
            selected_b: None,
            current_selection: SelectionMode::SelectA,
            status_message: None,
        }
    }
}

/// Standalone entry: creates its own terminal, runs the selection UI,
/// and tears it down. Use this when invoked outside the unified flow.
pub fn run_model_selection() -> Result<(Option<PathBuf>, Option<PathBuf>)> {
    let models = scan_for_models()?;
    let mut terminal = TerminalGuard::new()?;
    run_model_selection_with(&mut terminal, models)
}

/// Runs the model-selection UI on a *caller-provided* terminal so the
/// same TerminalGuard can be reused across scanner → loading → detail
/// without ever leaving raw mode. Pre-scanned models are passed in so
/// the caller can run the (slow) filesystem walk on a worker thread
/// while displaying its own progress UI.
pub fn run_model_selection_with(
    terminal: &mut TerminalGuard,
    models: Vec<ModelInfo>,
) -> Result<(Option<PathBuf>, Option<PathBuf>)> {
    let mut state = ModelSelectionState::from_models(models);
    let result = run_selection_loop(terminal, &mut state);

    match result {
        Ok(true) => Ok((
            state.selected_a.map(|i| state.models[i].path.clone()),
            state.selected_b.map(|i| state.models[i].path.clone()),
        )),
        _ => Ok((None, None)),
    }
}

fn run_selection_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    state: &mut ModelSelectionState,
) -> Result<bool> {
    let mut last_tick = Instant::now();
    let tick_rate = Duration::from_millis(250);

    loop {
        terminal.draw(|f| draw_selection_ui(f, state))?;

        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or_else(|| Duration::from_secs(0));

        if crossterm::event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                match handle_selection_key(key, state) {
                    SelectionAction::Confirm => return Ok(true),
                    SelectionAction::Cancel => return Ok(false),
                    SelectionAction::Continue => {}
                }
            }
        }

        if last_tick.elapsed() >= tick_rate {
            last_tick = Instant::now();
        }
    }
}

enum SelectionAction {
    Confirm,
    Cancel,
    Continue,
}

fn handle_selection_key(key: KeyEvent, state: &mut ModelSelectionState) -> SelectionAction {
    // Ignore key repeats and releases to avoid double-triggering on a single press
    if key.kind != KeyEventKind::Press {
        return SelectionAction::Continue;
    }

    if key.modifiers == KeyModifiers::CONTROL && key.code == KeyCode::Char('c') {
        return SelectionAction::Cancel;
    }

    match key.code {
        KeyCode::Char('q') => return SelectionAction::Cancel,
        KeyCode::Char('j') | KeyCode::Down => {
            let max = state.models.len().saturating_sub(1);
            match state.current_selection {
                SelectionMode::SelectA => {
                    state.selected_a = Some(state.selected_a.map_or(0, |i| (i + 1).min(max)));
                }
                SelectionMode::SelectB => {
                    state.selected_b = Some(state.selected_b.map_or(0, |i| (i + 1).min(max)));
                }
            }
        }
        KeyCode::Char('k') | KeyCode::Up => {
            match state.current_selection {
                SelectionMode::SelectA => {
                    state.selected_a =
                        Some(state.selected_a.map_or(0, |i| i.saturating_sub(1)));
                }
                SelectionMode::SelectB => {
                    state.selected_b =
                        Some(state.selected_b.map_or(0, |i| i.saturating_sub(1)));
                }
            }
        }
        KeyCode::Tab => {
            state.current_selection = match state.current_selection {
                SelectionMode::SelectA => SelectionMode::SelectB,
                SelectionMode::SelectB => SelectionMode::SelectA,
            };
        }
        KeyCode::Enter => {
            if state.selected_a.is_some() && state.selected_b.is_some() {
                if state.selected_a != state.selected_b {
                    return SelectionAction::Confirm;
                } else {
                    state.status_message = Some("Cannot compare model with itself".to_string());
                }
            } else {
                state.status_message = Some("Please select both models".to_string());
            }
        }
        _ => {}
    }
    SelectionAction::Continue
}

fn draw_selection_ui(f: &mut Frame, state: &ModelSelectionState) {
    let area = f.area();

    f.render_widget(Block::default().style(Style::default().bg(BG)), area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(2)
        .constraints([
            Constraint::Length(5),
            Constraint::Min(10),
            Constraint::Length(3),
            Constraint::Length(3),
        ])
        .split(area);

    draw_selection_header(f, state, chunks[0]);
    draw_model_list(f, state, chunks[1]);
    draw_selection_status(f, state, chunks[2]);
    draw_selection_footer(f, chunks[3]);
}

fn draw_selection_header(f: &mut Frame, state: &ModelSelectionState, area: Rect) {
    let mut lines = vec![
        Line::from(vec![Span::styled(
            " NEURALDIFF - Model Selection ",
            Style::default().fg(TEXT_PRIMARY).add_modifier(Modifier::BOLD),
        )]),
        Line::from(""),
    ];

    let mode_text = match state.current_selection {
        SelectionMode::SelectA => {
            let a_name = state
                .selected_a
                .and_then(|i| state.models.get(i))
                .map(|m| m.name.as_str())
                .unwrap_or("Not selected");
            format!("[Model A]: {}  |  Press Tab to select Model B", a_name)
        }
        SelectionMode::SelectB => {
            let b_name = state
                .selected_b
                .and_then(|i| state.models.get(i))
                .map(|m| m.name.as_str())
                .unwrap_or("Not selected");
            format!("Press Tab to select Model A  |  [Model B]: {}", b_name)
        }
    };

    lines.push(Line::from(vec![
        Span::styled("-> ", Style::default().fg(ACCENT)),
        Span::styled(mode_text, Style::default().fg(TEXT_SECONDARY)),
    ]));

    let paragraph = Paragraph::new(Text::from(lines))
        .style(Style::default().fg(TEXT_PRIMARY))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(BORDER))
                .style(Style::default().bg(SURFACE)),
        );

    f.render_widget(paragraph, area);
}

fn draw_model_list(f: &mut Frame, state: &ModelSelectionState, area: Rect) {
    let mut lines = vec![];

    if state.models.is_empty() {
        lines.push(Line::from(vec![Span::styled(
            "No .safetensors models found on this system.",
            Style::default().fg(TEXT_SECONDARY),
        )]));
        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::styled("Searched in: ", Style::default().fg(TEXT_SECONDARY)),
            Span::styled(
                "Home, Downloads, Documents, .cache/huggingface, current directory",
                Style::default().fg(ACCENT_INFO),
            ),
        ]));
    } else {
        lines.push(Line::from(vec![Span::styled(
            format!("Found {} model(s):", state.models.len()),
            Style::default().fg(TEXT_SECONDARY).add_modifier(Modifier::BOLD),
        )]));
        lines.push(Line::from(""));

        for (i, model) in state.models.iter().enumerate() {
            let is_a = state.selected_a == Some(i);
            let is_b = state.selected_b == Some(i);

            let mut spans = vec![];

            if is_a && is_b {
                spans.push(Span::styled("[A+B] ", Style::default().fg(RED).add_modifier(Modifier::BOLD)));
            } else if is_a {
                spans.push(Span::styled("[A]   ", Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)));
            } else if is_b {
                spans.push(Span::styled("[B]   ", Style::default().fg(ACCENT_INFO).add_modifier(Modifier::BOLD)));
            } else {
                spans.push(Span::styled("      ", Style::default()));
            }

            spans.push(Span::styled(
                format!("{:25} ", model.name),
                if is_a || is_b {
                    Style::default().fg(TEXT_PRIMARY).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(TEXT_PRIMARY)
                },
            ));
            spans.push(Span::styled(
                format!("{:8} ", format_size(model.size_mb)),
                Style::default().fg(TEXT_SECONDARY),
            ));
            spans.push(Span::styled(
                format!("({})", model.location),
                Style::default().fg(TEXT_SECONDARY),
            ));

            lines.push(Line::from(spans));
        }
    }

    let paragraph = Paragraph::new(Text::from(lines))
        .style(Style::default().fg(TEXT_PRIMARY))
        .block(
            Block::default()
                .title(" Available Models ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(BORDER))
                .style(Style::default().bg(SURFACE)),
        );

    f.render_widget(paragraph, area);
}

fn draw_selection_status(f: &mut Frame, state: &ModelSelectionState, area: Rect) {
    let spans = if let Some(ref msg) = state.status_message {
        vec![Span::styled(msg.as_str(), Style::default().fg(RED))]
    } else if state.selected_a.is_some() && state.selected_b.is_some() {
        vec![Span::styled("Ready to compare! Press Enter to continue", Style::default().fg(ACCENT))]
    } else {
        vec![Span::styled("Select two different models to compare", Style::default().fg(TEXT_SECONDARY))]
    };

    let paragraph = Paragraph::new(Line::from(spans))
        .alignment(Alignment::Center)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(BORDER))
                .style(Style::default().bg(SURFACE)),
        );

    f.render_widget(paragraph, area);
}

fn draw_selection_footer(f: &mut Frame, area: Rect) {
    let spans = vec![
        Span::styled("[up/down] Navigate  ", Style::default().fg(TEXT_SECONDARY)),
        Span::styled("[Tab] Switch A/B  ", Style::default().fg(TEXT_SECONDARY)),
        Span::styled("[Enter] Confirm  ", Style::default().fg(TEXT_SECONDARY)),
        Span::styled("[q] Cancel", Style::default().fg(TEXT_SECONDARY)),
    ];

    let paragraph = Paragraph::new(Line::from(spans))
        .alignment(Alignment::Center)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(BORDER))
                .style(Style::default().bg(SURFACE)),
        );

    f.render_widget(paragraph, area);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_scan_recursive() {
        let dir = tempdir().unwrap();
        let subdir = dir.path().join("sub").join("deep");
        fs::create_dir_all(&subdir).unwrap();
        fs::write(subdir.join("model.safetensors"), b"fake").unwrap();

        let mut seen = HashSet::new();
        let mut models = Vec::new();
        scan_dir_recursive(dir.path(), 0, 5, &mut seen, &mut models);

        assert_eq!(models.len(), 1);
        assert!(models[0].path.ends_with("model.safetensors"));
    }

    #[test]
    fn test_scan_respects_max_depth() {
        let dir = tempdir().unwrap();
        let subdir = dir.path().join("sub").join("deep");
        fs::create_dir_all(&subdir).unwrap();
        fs::write(subdir.join("model.safetensors"), b"fake").unwrap();

        let mut seen = HashSet::new();
        let mut models = Vec::new();
        scan_dir_recursive(dir.path(), 0, 1, &mut seen, &mut models);

        assert!(models.is_empty(), "Should not find models beyond max_depth");
    }

    #[test]
    fn test_scan_skips_hidden_dirs() {
        let dir = tempdir().unwrap();
        let hidden = dir.path().join(".hidden").join("deep");
        fs::create_dir_all(&hidden).unwrap();
        fs::write(hidden.join("model.safetensors"), b"fake").unwrap();

        let mut seen = HashSet::new();
        let mut models = Vec::new();
        scan_dir_recursive(dir.path(), 0, 5, &mut seen, &mut models);

        assert!(models.is_empty(), "Should skip hidden directories");
    }

    #[test]
    fn test_scan_allows_cache_dir() {
        let dir = tempdir().unwrap();
        let cache = dir.path().join(".cache").join("huggingface");
        fs::create_dir_all(&cache).unwrap();
        fs::write(cache.join("model.safetensors"), b"fake").unwrap();

        let mut seen = HashSet::new();
        let mut models = Vec::new();
        scan_dir_recursive(dir.path(), 0, 5, &mut seen, &mut models);

        assert_eq!(models.len(), 1, "Should allow .cache directory");
    }
}
