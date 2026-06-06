//! Live full-screen dashboard (opt-in via `--tui`), built on `ratatui`.
//!
//! Shows a per-card table (Sol/s, power, efficiency, shares), a totals row, and
//! a pane of recent log lines (logs are captured into a ring buffer so they
//! don't scribble over the UI). Quit with `q`, `Esc`, or `Ctrl-C`.

use std::collections::VecDeque;
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use log::{Level, LevelFilter, Metadata, Record};

use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Cell, Paragraph, Row, Sparkline, Table, Wrap};

use crate::controls::Controls;
use crate::cpu_groups::{CpuGroups, CpuMining};
use crate::miner::{CardSnapshot, Stats};
use crate::stratum::StratumClient;

/// Step sizes for the live hardware controls (per key press).
const CORE_STEP_MHZ: i32 = 15;
const MEM_STEP_MHZ: i32 = 50;
const TDP_STEP_W: i32 = 10;

const LOG_CAP: usize = 500;
/// How many Sol/s samples (≈ seconds) of history each graph keeps.
const HIST_CAP: usize = 600;
/// How often the CPU-group Sol/s figures are refreshed (seconds). Longer than
/// the per-card 1 s window because a CPU-group solve takes several seconds.
const CPU_POLL_SECS: u64 = 10;

fn log_buf() -> &'static Mutex<VecDeque<String>> {
    static B: OnceLock<Mutex<VecDeque<String>>> = OnceLock::new();
    B.get_or_init(|| Mutex::new(VecDeque::new()))
}

/// Separate ring buffer for pool "new job" notifications (shown in their own
/// pane, kept out of the general log).
fn job_buf() -> &'static Mutex<VecDeque<String>> {
    static B: OnceLock<Mutex<VecDeque<String>>> = OnceLock::new();
    B.get_or_init(|| Mutex::new(VecDeque::new()))
}

/// A `log` sink that captures recent lines into a ring buffer for the TUI pane.
struct CaptureLogger {
    level: LevelFilter,
}

impl log::Log for CaptureLogger {
    fn enabled(&self, m: &Metadata) -> bool {
        m.level() <= self.level
    }
    fn log(&self, record: &Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let lvl = match record.level() {
            Level::Error => "ERR",
            Level::Warn => "WRN",
            Level::Info => "inf",
            Level::Debug => "dbg",
            Level::Trace => "trc",
        };
        let msg = record.args().to_string();
        // Pool "new job <id> (epoch N)" notices go to their own pane (compact,
        // without the level tag); everything else to the general log.
        let (buf, line) = match msg.strip_prefix("new job ") {
            Some(rest) => (job_buf(), format!("{} {rest}", utc_hms())),
            None => (log_buf(), format!("{} {lvl} {msg}", utc_hms())),
        };
        let mut b = buf.lock().unwrap();
        b.push_back(line);
        while b.len() > LOG_CAP {
            b.pop_front();
        }
    }
    fn flush(&self) {}
}

/// Route logging into the in-memory ring buffer instead of the terminal. Call
/// instead of initialising `env_logger` when the TUI is active.
pub fn install_logger() {
    let _ = log::set_boxed_logger(Box::new(CaptureLogger { level: LevelFilter::Info }));
    log::set_max_level(LevelFilter::Info);
}

/// Run the dashboard until the user quits or `running` is cleared. Blocks.
///
/// If the terminal can't be taken over (rare on a real TTY), falls back to
/// periodic text stats on stderr so output is never lost.
#[allow(clippy::too_many_arguments)]
pub fn run(stats: &Stats, pool: &str, running: &AtomicBool, start: Instant, client: &StratumClient, controls: &Controls, cpu_mining: &CpuMining) {
    match ratatui::try_init() {
        Ok(mut terminal) => {
            let _ = render_loop(&mut terminal, stats, pool, running, start, client, controls, cpu_mining);
            ratatui::restore();
        }
        Err(e) => {
            eprintln!("dashboard unavailable ({e}); falling back to text stats");
            text_fallback(stats, running);
        }
    }
}

/// Degraded mode: print per-card stats + captured log lines to stderr every 10 s.
/// Used only if `ratatui` init fails after we'd already chosen the capture logger.
fn text_fallback(stats: &Stats, running: &AtomicBool) {
    let mut prev: Vec<u64> = stats.cards().iter().map(|c| c.solutions).collect();
    let mut t = Instant::now();
    while running.load(Ordering::Relaxed) {
        std::thread::sleep(Duration::from_millis(500));
        if t.elapsed() < Duration::from_secs(10) {
            continue;
        }
        let dt = t.elapsed().as_secs_f64();
        t = Instant::now();
        // Surface captured log + job lines, then per-card stats.
        for l in log_buf().lock().unwrap().drain(..) {
            eprintln!("{l}");
        }
        for l in job_buf().lock().unwrap().drain(..) {
            eprintln!("new job {l}");
        }
        let cards = stats.cards();
        prev.resize(cards.len(), 0);
        for (i, c) in cards.iter().enumerate() {
            let rate = c.solutions.saturating_sub(prev[i]) as f64 / dt;
            prev[i] = c.solutions;
            let w = c.power_mw as f64 / 1000.0;
            if w > 0.0 {
                eprintln!("{}: {rate:.1} Sol/s ({} shares, {w:.0} W, {:.2} Sol/W)", c.label, c.shares, rate / w);
            } else {
                eprintln!("{}: {rate:.1} Sol/s ({} shares)", c.label, c.shares);
            }
        }
    }
    for l in log_buf().lock().unwrap().drain(..) {
        eprintln!("{l}");
    }
}

#[allow(clippy::too_many_arguments)]
fn render_loop(
    terminal: &mut ratatui::DefaultTerminal,
    stats: &Stats,
    pool: &str,
    running: &AtomicBool,
    start: Instant,
    client: &StratumClient,
    controls: &Controls,
    cpu_mining: &CpuMining,
) -> io::Result<()> {
    // Sol/s is computed over ~1 s windows; redraw is faster (for the clock/logs).
    let mut prev: Vec<u64> = stats.cards().iter().map(|c| c.solutions).collect();
    let mut rates: Vec<f64> = vec![0.0; prev.len()];
    let mut hist: Vec<VecDeque<u64>> = vec![VecDeque::new(); prev.len()];
    // Current CPU grouping snapshot; refreshed when the group size changes (the
    // generation bumps). `controls` tracks the row count for Tab selection.
    let mut groups = cpu_mining.groups();
    let mut last_gen = cpu_mining.generation();
    controls.set_group_rows(groups.len());
    // Per-CPU-group Sol/s, refreshed every CPU_POLL_SECS. A slow CPU core finds
    // only a few solutions per refresh, so a short-window delta (integer / ~10 s)
    // quantises to ~0.1 steps and the second decimal is always zero. Instead we
    // show a cumulative average since the core was enabled: the elapsed window
    // grows over time, giving genuine two-decimal precision and a steady figure.
    let mut rates_g: Vec<f64> = vec![0.0; groups.len()];
    // Start instant + solution count at the beginning of each group's current
    // enabled streak (`None` while disabled), used to compute the average.
    let mut cpu_since: Vec<Option<Instant>> = vec![None; groups.len()];
    let mut cpu_base: Vec<u64> = vec![0; groups.len()];
    let mut prev_t = Instant::now();
    let mut prev_tg = Instant::now();

    while running.load(Ordering::Relaxed) {
        // Pick up a regroup (group-size change): refresh the snapshot, the row
        // count, and reset the per-group stat tracking to the new grouping.
        let gen = cpu_mining.generation();
        if gen != last_gen {
            groups = cpu_mining.groups();
            controls.set_group_rows(groups.len());
            rates_g = vec![0.0; groups.len()];
            cpu_since = vec![None; groups.len()];
            cpu_base = vec![0; groups.len()];
            last_gen = gen;
        }

        if prev_t.elapsed() >= Duration::from_secs(1) {
            let dt = prev_t.elapsed().as_secs_f64();
            let cards = stats.cards();
            prev.resize(cards.len(), 0);
            rates.resize(cards.len(), 0.0);
            while hist.len() < cards.len() {
                hist.push(VecDeque::new());
            }
            for (i, c) in cards.iter().enumerate() {
                rates[i] = c.solutions.saturating_sub(prev[i]) as f64 / dt;
                prev[i] = c.solutions;
                push_hist(&mut hist[i], rates[i]);
            }
            prev_t = Instant::now();
        }

        if prev_tg.elapsed() >= Duration::from_secs(CPU_POLL_SECS) {
            let now = Instant::now();
            for (i, g) in groups.iter().enumerate() {
                if !g.enabled() {
                    cpu_since[i] = None;
                    rates_g[i] = 0.0;
                    continue;
                }
                match cpu_since[i] {
                    // First refresh of an enabled streak: start averaging here.
                    None => {
                        cpu_since[i] = Some(now);
                        cpu_base[i] = g.solutions();
                        rates_g[i] = 0.0;
                    }
                    // Cumulative average over the (growing) enabled window.
                    Some(t0) => {
                        let elapsed = now.duration_since(t0).as_secs_f64();
                        if elapsed >= 1.0 {
                            rates_g[i] = g.solutions().saturating_sub(cpu_base[i]) as f64 / elapsed;
                        }
                    }
                }
            }
            prev_tg = now;
        }

        let cards = stats.cards();
        let mode = client.mining_mode();
        let group_size = cpu_mining.group_size();
        terminal.draw(|frame| draw(frame, pool, mode, start, &cards, &rates, &hist, controls, &groups, group_size, &rates_g))?;

        if event::poll(Duration::from_millis(200))? {
            if let Event::Key(k) = event::read()? {
                if k.kind == KeyEventKind::Press {
                    let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
                    match k.code {
                        KeyCode::Char('q') | KeyCode::Esc => {
                            running.store(false, Ordering::Relaxed);
                            break;
                        }
                        KeyCode::Char('c') if ctrl => {
                            running.store(false, Ordering::Relaxed);
                            break;
                        }
                        // Tab cycles the selection over the GPUs then the CPU groups.
                        KeyCode::Tab => controls.select_next(),
                        // 'g' cycles the CPU group size (cores per mining row); the
                        // supervisor regroups the workers to match.
                        KeyCode::Char('g') => {
                            cpu_mining.cycle_group_size();
                            log::info!("CPU group size -> {} core(s) per row", cpu_mining.group_size());
                        }
                        // Backspace toggles mining on the selected row — a GPU
                        // device or a CPU group.
                        KeyCode::Backspace => {
                            if let Some(d) = controls.selected_device() {
                                let dev = controls.device(d);
                                dev.toggle_enabled();
                                let label = cards.get(d).map(|c| short_label(&c.label).to_string()).unwrap_or_else(|| format!("GPU {d}"));
                                log::info!("{label}: mining {}", if dev.enabled() { "enabled" } else { "disabled" });
                            } else if let Some(gi) = controls.selected_group() {
                                if gi < groups.len() {
                                    cpu_mining.toggle_group(gi);
                                    let g = groups.group(gi);
                                    log::info!("{}: CPU mining {}", g.label(), if g.enabled() { "enabled" } else { "disabled" });
                                }
                            }
                        }
                        // Live hardware controls for the selected GPU — only when
                        // a GPU is selected and unlocked via --unlock-controls.
                        KeyCode::Char('z') if controls.unlocked() => {
                            if let Some(d) = controls.selected_device() {
                                controls.device(d).adjust_core(-CORE_STEP_MHZ);
                            }
                        }
                        KeyCode::Char('x') if controls.unlocked() => {
                            if let Some(d) = controls.selected_device() {
                                controls.device(d).adjust_core(CORE_STEP_MHZ);
                            }
                        }
                        KeyCode::Char('c') if controls.unlocked() => {
                            if let Some(d) = controls.selected_device() {
                                controls.device(d).adjust_mem(-MEM_STEP_MHZ);
                            }
                        }
                        KeyCode::Char('v') if controls.unlocked() => {
                            if let Some(d) = controls.selected_device() {
                                controls.device(d).adjust_mem(MEM_STEP_MHZ);
                            }
                        }
                        KeyCode::Char('b') if controls.unlocked() => {
                            if let Some(d) = controls.selected_device() {
                                controls.device(d).adjust_power(-TDP_STEP_W);
                            }
                        }
                        KeyCode::Char('n') if controls.unlocked() => {
                            if let Some(d) = controls.selected_device() {
                                controls.device(d).adjust_power(TDP_STEP_W);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn draw(
    frame: &mut Frame,
    pool: &str,
    mode: &str,
    start: Instant,
    cards: &[CardSnapshot],
    rates: &[f64],
    hist: &[VecDeque<u64>],
    controls: &Controls,
    cpu_groups: &CpuGroups,
    group_size: usize,
    group_rates: &[f64],
) {
    // Tall enough for the controls panel's lines (state + core/mem/tdp + clock
    // readout, in a border), plus a row per CPU core and a CPU TOTAL row below
    // the GPU totals.
    let cpu_total_rows = if cpu_groups.is_empty() { 0 } else { 1 };
    let dev_h = (cards.len() as u16 + cpu_groups.len() as u16 + 4 + cpu_total_rows).max(7);
    let chunks = Layout::vertical([
        Constraint::Length(1),                                 // title
        Constraint::Length(dev_h),                             // device table (+ auto-tune + controls)
        Constraint::Length(9),                                 // network Sol/s text + per-GPU graphs
        Constraint::Min(3),                                    // log pane
        Constraint::Length(1),                                 // footer
    ])
    .split(frame.area());

    // Title bar: name badge, then the pool payout mode, then pool + uptime.
    let title = Line::from(vec![
        Span::styled(" jackpotminer ", Style::new().fg(Color::Black).bg(Color::Green).bold()),
        Span::raw(" "),
        Span::styled(format!(" {mode} "), Style::new().fg(Color::Black).bg(Color::Cyan).bold()),
        Span::raw(format!("  pool {pool}    up {}", fmt_dur(start.elapsed()))),
    ]);
    frame.render_widget(Paragraph::new(title), chunks[0]);

    // Device table + totals.
    let header = Row::new(["DEVICE", "Sol/s", "Power", "Temp", "Sol/W", "Shares"])
        .style(Style::new().fg(Color::Cyan).bold());

    let (mut tot_rate, mut tot_w, mut tot_sh, mut max_temp) = (0.0_f64, 0.0_f64, 0u64, 0u32);
    let mut rows: Vec<Row> = Vec::with_capacity(cards.len() + 1);
    for (i, c) in cards.iter().enumerate() {
        let enabled = controls.device(i).enabled();
        let rate = rates.get(i).copied().unwrap_or(0.0);
        let w = c.power_mw as f64 / 1000.0;
        let (power, eff) = if w > 0.0 {
            tot_w += w;
            (format!("{w:.0} W"), format!("{:.2}", rate / w))
        } else {
            ("—".to_string(), "—".to_string())
        };
        let temp = match c.temp_c {
            0 => "—".to_string(),
            t => {
                max_temp = max_temp.max(t);
                format!("{t}°C")
            }
        };
        // Mark the device the controls panel currently targets, and flag a device
        // disabled from the dashboard (Backspace).
        let sel = controls.selected_device() == Some(i);
        let marker = if sel { "▸ " } else { "  " };
        let state = if enabled { "" } else { " [off]" };
        let device = match &c.name {
            Some(n) => format!("{marker}{}  {n}{state}", c.label),
            None => format!("{marker}{}{state}", c.label),
        };
        let rate_str = if enabled { format!("{rate:.1}") } else { "—".to_string() };
        let mut row = Row::new([
            trunc(&device, 53),
            rate_str,
            power,
            temp,
            eff,
            c.shares.to_string(),
        ]);
        row = if sel {
            row.style(Style::new().fg(Color::Yellow).bold())
        } else if !enabled {
            row.style(Style::new().fg(Color::DarkGray))
        } else {
            row
        };
        rows.push(row);
        if enabled {
            tot_rate += rate;
        }
        tot_sh += c.shares;
    }
    let (tpower, teff) = if tot_w > 0.0 {
        (format!("{tot_w:.0} W"), format!("{:.2}", tot_rate / tot_w))
    } else {
        ("—".to_string(), "—".to_string())
    };
    let ttemp = if max_temp > 0 { format!("{max_temp}°C") } else { "—".to_string() };
    rows.push(
        Row::new([
            Cell::from("  TOTAL"),
            Cell::from(format!("{tot_rate:.1}")),
            Cell::from(tpower),
            Cell::from(ttemp),
            Cell::from(teff),
            Cell::from(tot_sh.to_string()),
        ])
        .style(Style::new().bold()),
    );

    // CPU core rows below the totals: each mines when enabled (Backspace toggles
    // the selected one). Disabled cores are dimmed and show no Sol/s. A CPU TOTAL
    // row (summing enabled cores' Sol/s and all cores' shares) follows.
    let (mut cpu_rate, mut cpu_sh) = (0.0_f64, 0u64);
    for (i, g) in cpu_groups.iter().enumerate() {
        let sel = controls.selected_group() == Some(i);
        let on = g.enabled();
        let r = group_rates.get(i).copied().unwrap_or(0.0);
        let marker = if sel { "▸ " } else { "  " };
        let state = if on { "[on] " } else { "[off]" };
        let device = format!("{marker}{state} {}", g.label());
        let rate = if on { format!("{r:.2}") } else { "—".to_string() };
        let mut row = Row::new([
            trunc(&device, 53),
            rate,
            "—".to_string(),
            "—".to_string(),
            "—".to_string(),
            g.shares().to_string(),
        ]);
        row = if sel {
            row.style(Style::new().fg(Color::Yellow).bold())
        } else if on {
            row.style(Style::new().fg(Color::Green))
        } else {
            row.style(Style::new().fg(Color::DarkGray))
        };
        rows.push(row);
        if on {
            cpu_rate += r;
        }
        cpu_sh += g.shares();
    }
    if !cpu_groups.is_empty() {
        rows.push(
            Row::new([
                Cell::from("  CPU TOTAL"),
                Cell::from(format!("{cpu_rate:.2}")),
                Cell::from("—"),
                Cell::from("—"),
                Cell::from("—"),
                Cell::from(cpu_sh.to_string()),
            ])
            .style(Style::new().bold()),
        );
    }

    let widths = [
        Constraint::Length(53),
        Constraint::Length(9),
        Constraint::Length(8),
        Constraint::Length(6),
        Constraint::Length(7),
        Constraint::Length(7),
    ];
    // Device table on the left; live hardware-control panel on the right.
    let table = Table::new(rows, widths)
        .header(header)
        .block(Block::bordered().title(" devices "));
    let dev_cols = Layout::horizontal([Constraint::Min(0), Constraint::Length(26)]).split(chunks[1]);
    frame.render_widget(table, dev_cols[0]);
    draw_controls(frame, dev_cols[1], controls, cards, cpu_groups);

    // One Sol/s graph per GPU, side by side, across the full width.
    if !cards.is_empty() {
        let cols = Layout::horizontal(vec![Constraint::Ratio(1, cards.len() as u32); cards.len()])
            .split(chunks[2]);
        for (i, c) in cards.iter().enumerate() {
            let rate = rates.get(i).copied().unwrap_or(0.0);
            let title = format!(" {}  {rate:.0} Sol/s ", short_label(&c.label));
            let data = hist.get(i).map(|h| h.iter().copied().collect::<Vec<u64>>()).unwrap_or_default();
            frame.render_widget(
                Sparkline::default()
                    .block(Block::bordered().title(title))
                    .data(data)
                    .style(Style::new().fg(Color::Cyan)),
                cols[i],
            );
        }
    }

    // Bottom row: recent log on the left, pool job notices on the right. Each
    // shows the most recent lines that fill its pane (down to the bottom).
    let log_cols = Layout::horizontal([Constraint::Min(0), Constraint::Length(34)]).split(chunks[3]);
    let visible = (chunks[3].height.saturating_sub(2) as usize).max(1); // minus the borders
    let tail = |buf: &Mutex<VecDeque<String>>| -> Vec<Line> {
        let b = buf.lock().unwrap();
        let n = b.len();
        b.iter()
            .skip(n.saturating_sub(visible))
            .map(|l| Line::raw(l.clone()))
            .collect()
    };
    // Recent log: most-recent lines, coloured by severity and wrapped to the
    // pane width so long messages don't overflow off the edge.
    let log_w = log_cols[0].width.saturating_sub(2).max(1) as usize; // inside borders
    frame.render_widget(
        Paragraph::new(Text::from(recent_log_lines(log_buf(), visible, log_w)))
            .wrap(Wrap { trim: false })
            .block(Block::bordered().title(" recent ")),
        log_cols[0],
    );
    frame.render_widget(
        Paragraph::new(Text::from(tail(job_buf()))).block(Block::bordered().title(" jobs ")),
        log_cols[1],
    );

    // Footer.
    frame.render_widget(
        Paragraph::new(
            Line::from(if controls.unlocked() {
                format!("  q/Esc quit · Tab select · Backspace enable/disable · g cpu group size ({group_size}) · z/x core · c/v mem · b/n tdp")
            } else {
                format!("  q/Esc quit · Tab select · Backspace enable/disable · g cpu group size ({group_size}) · controls locked")
            })
            .style(Style::new().fg(Color::DarkGray)),
        ),
        chunks[4],
    );
}

/// Build the recent-log lines: the newest entries that fit `visible` rows once
/// wrapped to `width` columns, coloured by severity. Selecting wrap-aware (from
/// newest backward, stopping before the wrapped height exceeds the pane) keeps
/// the latest line on screen instead of letting wrapping push it off the bottom.
fn recent_log_lines(buf: &Mutex<VecDeque<String>>, visible: usize, width: usize) -> Vec<Line<'static>> {
    let b = buf.lock().unwrap();
    let width = width.max(1);
    let mut chosen: Vec<&String> = Vec::new();
    let mut rows = 0usize;
    for l in b.iter().rev() {
        // Rows this line occupies once wrapped (ceil of width, at least one).
        let h = (l.chars().count() + width - 1) / width;
        let h = h.max(1);
        if rows + h > visible && !chosen.is_empty() {
            break;
        }
        rows += h;
        chosen.push(l);
        if rows >= visible {
            break;
        }
    }
    chosen.iter().rev().map(|l| style_log_line(l)).collect()
}

/// Colour a captured log line: red for an invalid-address share rejection,
/// green for an accepted share, yellow for any other warning/error, default for
/// everything else (including share submissions). Lines are "HH:MM:SS LVL
/// message", so the level tag sits at byte offset 9..12.
fn style_log_line(l: &str) -> Line<'static> {
    let lower = l.to_lowercase();
    if lower.contains("invalid address") {
        Line::styled(l.to_string(), Style::new().fg(Color::Red).bold())
    } else if lower.contains("share accepted") {
        // The pool accepted a share — good news, show it green.
        Line::styled(l.to_string(), Style::new().fg(Color::Green))
    } else if matches!(l.get(9..12), Some("WRN") | Some("ERR")) {
        Line::styled(l.to_string(), Style::new().fg(Color::Yellow))
    } else {
        Line::raw(l.to_string())
    }
}

/// Push a Sol/s sample (rounded) into a capped history buffer.
fn push_hist(h: &mut VecDeque<u64>, rate: f64) {
    h.push_back(rate.max(0.0).round() as u64);
    while h.len() > HIST_CAP {
        h.pop_front();
    }
}

/// Short graph label: the part before " (" (e.g. "GPU 0 (CUDA)" -> "GPU 0").
fn short_label(label: &str) -> &str {
    label.split(" (").next().unwrap_or(label)
}

fn trunc(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        s.chars().take(max.saturating_sub(1)).collect::<String>() + "…"
    }
}

/// The live hardware-control panel for the selected GPU: per-row label,
/// decrease/increase keys, and the current target value. Adjusted by the key
/// handler in `render_loop`; Tab selects which GPU it targets.
fn draw_controls(frame: &mut Frame, area: Rect, controls: &Controls, cards: &[CardSnapshot], cpu_groups: &CpuGroups) {
    // When a CPU group is selected, show its panel (with the Backspace hint)
    // instead of the GPU clock/power controls.
    if let Some(gi) = controls.selected_group() {
        let g = cpu_groups.group(gi);
        let lbl = Style::new().fg(Color::Cyan);
        let val = Style::new().fg(Color::Green);
        let key = Style::new().fg(Color::Yellow).bold();
        let (state, state_style) = if g.enabled() {
            ("mining", Style::new().fg(Color::Green).bold())
        } else {
            ("idle", Style::new().fg(Color::DarkGray).bold())
        };
        let lines = vec![
            Line::from(vec![Span::styled("cores  ", lbl), Span::styled(format!("{}", g.ncores()), val)]),
            Line::from(vec![Span::styled("state  ", lbl), Span::styled(state, state_style)]),
            Line::from(vec![Span::styled("shares ", lbl), Span::styled(format!("{}", g.shares()), val)]),
            Line::from(vec![Span::styled("Backspace", key), Span::raw(" toggle")]),
        ];
        let title = format!(" {} ", g.label());
        frame.render_widget(Paragraph::new(lines).block(Block::bordered().title(title)), area);
        return;
    }

    let sel = controls.selected_device().unwrap_or(0);
    let dev = controls.device(sel);
    let off = |v: i32| format!("{v:+} MHz");
    let pw = match dev.power_w() {
        0 => "default".to_string(),
        w => format!("{w} W"),
    };
    let key = Style::new().fg(Color::Yellow).bold();
    let lbl = Style::new().fg(Color::Cyan);
    let val = Style::new().fg(Color::Green);
    // label(5) + "  " + dec-key + "- " + inc-key + "+  " + value
    let row = |name: &str, dn: char, up: char, value: String| {
        Line::from(vec![
            Span::styled(format!("{name:<5}"), lbl),
            Span::styled(dn.to_string(), key),
            Span::raw("- "),
            Span::styled(up.to_string(), key),
            Span::raw("+  "),
            Span::styled(value, val),
        ])
    };
    let (state, state_style) = if dev.enabled() {
        ("mining", Style::new().fg(Color::Green).bold())
    } else {
        ("idle", Style::new().fg(Color::DarkGray).bold())
    };
    let lines = vec![
        // Mining state + the Backspace enable/disable hint.
        Line::from(vec![
            Span::styled("⌫ ", key),
            Span::styled(state, state_style),
        ]),
        row("core", 'z', 'x', off(dev.core_off())),
        row("mem", 'c', 'v', off(dev.mem_off())),
        row("tdp", 'b', 'n', pw),
        // Live actual clocks (sampled from the card), not the offsets above.
        {
            let c = cards.get(sel);
            let clk = |mhz: u32| if mhz > 0 { format!("{mhz}") } else { "—".to_string() };
            Line::from(vec![
                Span::styled("clock ", lbl),
                Span::styled(
                    format!(
                        "{} / {} MHz",
                        clk(c.map(|c| c.core_clock_mhz).unwrap_or(0)),
                        clk(c.map(|c| c.mem_clock_mhz).unwrap_or(0)),
                    ),
                    val,
                ),
            ])
        },
    ];
    // Title names the targeted card (its short label, e.g. "GPU 0") and flags the
    // locked state (the footer says how to unlock).
    let name = cards.get(sel).map(|c| short_label(&c.label)).unwrap_or("GPU");
    let title = if controls.unlocked() {
        format!(" controls · {name} ")
    } else {
        format!(" {name} · LOCKED ")
    };
    frame.render_widget(Paragraph::new(lines).block(Block::bordered().title(title)), area);
}

fn fmt_dur(d: Duration) -> String {
    let s = d.as_secs();
    let (days, rem) = (s / 86_400, s % 86_400);
    format!("{days}d {:02}:{:02}:{:02}", rem / 3600, (rem % 3600) / 60, rem % 60)
}

/// Current wall-clock time of day as `HH:MM:SS` (UTC; no timezone dep).
fn utc_hms() -> String {
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    format!("{:02}:{:02}:{:02}", (secs / 3600) % 24, (secs / 60) % 60, secs % 60)
}
