use std::io::stdout;
use std::sync::Arc;

use crossterm::event::{Event, EventStream, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::execute;
use futures::StreamExt;
use ratatui::Terminal;

use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Alignment, Constraint, Layout};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;
use tokio::sync::{mpsc, Mutex};
use tokio::time::{Duration, Instant};

use crate::config::Config;
use crate::history::Role;
use crate::llm::{LlmClient, StreamEvent};
use crate::tools::ToolCall;
use crate::Agent;

// events from the background streaming task to the UI loop
enum UiEvent {
    Delta(String),
    RoundDone,
    ToolResult { name: String, preview: String },
    Status(String),
    Error(String),
    TurnDone,
}

#[derive(Clone)]
struct TuiMessage {
    role: Role,
    content: String,
}

pub struct TuiAgent {
    agent: Arc<Mutex<Agent>>,
    messages: Vec<TuiMessage>,
    streaming: String,
    input: String,
    status: String,
    busy: bool,
    ui_rx: Option<mpsc::Receiver<UiEvent>>,
}

impl TuiAgent {
    pub fn new(agent: Agent) -> Self {
        Self {
            agent: Arc::new(Mutex::new(agent)),
            messages: Vec::new(),
            streaming: String::new(),
            input: String::new(),
            status: "ready".to_string(),
            busy: false,
            ui_rx: None,
        }
    }

    pub async fn run(mut self) -> anyhow::Result<()> {
        let mut terminal = setup_terminal()?;
        let mut events = EventStream::new();
        self.status = format!(
            "model={} cwd={} — type a message, enter to send, ctrl-c to quit",
            {
                let a = self.agent.lock().await;
                a.config.model.clone()
            },
            {
                let a = self.agent.lock().await;
                a.config.repo_root.display().to_string()
            }
        );

        // tick for spinner animation
        let mut ticker = tokio::time::interval(Duration::from_millis(120));
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            terminal.draw(|f| self.draw(f))?;
            tokio::select! {
                maybe_evt = events.next() => {
                    let Some(Ok(evt)) = maybe_evt else { break; };
                    if !self.handle_event(evt).await? { break; }
                }
                Some(ui_evt) = async {
                    match &mut self.ui_rx {
                        Some(rx) => rx.recv().await,
                        None => None,
                    }
                } => {
                    if !self.handle_ui_event(ui_evt).await? { break; }
                }
                _ = ticker.tick() => {
                    // just trigger a redraw on next loop iteration (no-op)
                }
            }
        }
        restore_terminal()?;
        Ok(())
    }

    fn draw(&self, f: &mut Frame) {
        let chunks = Layout::vertical([
            Constraint::Min(3),
            Constraint::Length(3),
            Constraint::Length(1),
        ]).split(f.area());

        let mut lines: Vec<Line> = Vec::new();

        for m in &self.messages {
            render_bubble(&mut lines, m);
        }

        // live streaming bubble
        if self.busy && !self.streaming.is_empty() {
            render_bubble(&mut lines, &TuiMessage {
                role: Role::Assistant,
                content: self.streaming.clone(),
            });
        }

        // thinking indicator
        if self.busy && self.streaming.is_empty() {
            let spinner = thinking_spinner();
            lines.push(Line::from(""));
            lines.push(Line::styled(
                format!("  {} thinking…", spinner),
                Style::default().fg(Color::DarkGray).bg(Color::Black),
            ));
        }

        let chat = Paragraph::new(lines)
            .style(Style::default().fg(Color::White).bg(Color::Black))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("twobobs")
                    .style(Style::default().fg(Color::White).bg(Color::Black),
            ))
            .wrap(Wrap { trim: false })
            .scroll((0, u16::MAX));
        f.render_widget(chat, chunks[0]);

        // input box
        let input_style = Style::default().fg(Color::Black).bg(Color::White);
        let input = Paragraph::new(self.input.as_str())
            .style(input_style)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("input")
                    .style(input_style),
            );
        f.render_widget(input, chunks[1]);

        // status bar
        let status_text = if self.busy {
            format!("⟳ {} (streaming…)", self.status)
        } else {
            self.status.clone()
        };
        let status_style = Style::default().fg(Color::Yellow).bg(Color::Black);
        let status = Paragraph::new(status_text).style(status_style);
        f.render_widget(status, chunks[2]);
    }

    async fn handle_event(&mut self, evt: Event) -> anyhow::Result<bool> {
        if let Event::Key(k) = evt {
            if k.kind != KeyEventKind::Press {
                return Ok(true);
            }
            match k.code {
                KeyCode::Enter if !self.busy && !self.input.is_empty() => {
                    let prompt = std::mem::take(&mut self.input);
                    self.messages.push(TuiMessage { role: Role::User, content: prompt.clone() });
                    self.busy = true;
                    self.streaming.clear();
                    self.status = "thinking".to_string();
                    self.start_turn(prompt).await;
                }
                KeyCode::Char('c') if k.modifiers.contains(KeyModifiers::CONTROL) => return Ok(false),
                KeyCode::Char('d') if k.modifiers.contains(KeyModifiers::CONTROL) => return Ok(false),
                KeyCode::Char(c) if !self.busy && !k.modifiers.contains(KeyModifiers::CONTROL) => {
                    self.input.push(c);
                }
                KeyCode::Backspace if !self.busy => {
                    self.input.pop();
                }
                _ => {}
            }
        }
        Ok(true)
    }

    // spawn background task that drives one full turn (possibly multiple tool rounds)
    async fn start_turn(&mut self, prompt: String) {
        let (tx, rx) = mpsc::channel::<UiEvent>(64);
        self.ui_rx = Some(rx);
        let agent = self.agent.clone();

        tokio::spawn(async move {
            run_turn_streamed(agent, prompt, tx).await;
        });
    }

    async fn handle_ui_event(&mut self, evt: UiEvent) -> anyhow::Result<bool> {
        match evt {
            UiEvent::Delta(delta) => {
                self.streaming.push_str(&delta);
            }
            UiEvent::RoundDone => {
                if !self.streaming.is_empty() {
                    self.messages.push(TuiMessage {
                        role: Role::Assistant,
                        content: std::mem::take(&mut self.streaming),
                    });
                }
            }
            UiEvent::ToolResult { name, preview } => {
                self.messages.push(TuiMessage {
                    role: Role::Tool,
                    content: format!("{name} → {preview}"),
                });
            }
            UiEvent::Status(s) => {
                self.status = s;
            }
            UiEvent::Error(e) => {
                self.messages.push(TuiMessage {
                    role: Role::Assistant,
                    content: format!("error: {e}"),
                });
                self.streaming.clear();
                self.busy = false;
                self.ui_rx = None;
            }
            UiEvent::TurnDone => {
                self.busy = false;
                self.streaming.clear();
                self.ui_rx = None;
            }
        }
        Ok(true)
    }
}

// background streaming driver — shares Agent via Arc<Mutex<Agent>>
// lock is held only during req build, history append, tool dispatch — never across .recv().await
async fn run_turn_streamed(
    agent: Arc<Mutex<Agent>>,
    prompt: String,
    tx: mpsc::Sender<UiEvent>,
) {
    if let Err(e) = run_turn_inner(&agent, prompt, &tx).await {
        let _ = tx.send(UiEvent::Error(e.to_string())).await;
    }
    let _ = tx.send(UiEvent::TurnDone).await;
}

async fn run_turn_inner(
    agent: &Arc<Mutex<Agent>>,
    prompt: String,
    tx: &mpsc::Sender<UiEvent>,
) -> anyhow::Result<()> {
    {
        let mut a = agent.lock().await;
        a.history.append_user(prompt);
    }

    loop {
        // build request under lock, then release
        let req = {
            let a = agent.lock().await;
            a.history.to_request(&a.config.model, &a.tools.schemas())
        };
        // start stream under lock (needs &self), then release lock during .recv().await
        let mut rx = {
            let a = agent.lock().await;
            a.llm.complete_stream(req).await?
        };

        let mut content = String::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        let mut last_cost: Option<crate::llm::CallCost> = None;

        while let Some(evt) = rx.recv().await {
            match evt {
                StreamEvent::Content(delta) => {
                    content.push_str(&delta);
                    let _ = tx.send(UiEvent::Delta(delta)).await;
                }
                StreamEvent::ToolCallStart(tc) => {
                    tool_calls.push(tc);
                }
                StreamEvent::ToolCallDelta(_) => {}
                StreamEvent::Cost(c) => {
                    last_cost = Some(c);
                }
                StreamEvent::Done => break,
                StreamEvent::Error(e) => {
                    anyhow::bail!("stream error: {e}");
                }
            }
        }

        let _ = tx.send(UiEvent::RoundDone).await;

        if let Some(c) = &last_cost {
            let _ = tx.send(UiEvent::Status(format!(
                "{} cost ${:.6} ({}+{} tok)",
                {
                    let a = agent.lock().await;
                    a.config.model.clone()
                },
                c.total_cost, c.input_tokens, c.output_tokens
            ))).await;
        }

        // append assistant + dispatch tools under lock
        if tool_calls.is_empty() {
            let mut a = agent.lock().await;
            a.history.append_assistant(content);
            return Ok(());
        }
        {
            let mut a = agent.lock().await;
            a.history.append_assistant_with_tools(content, tool_calls.clone());
        }
        for call in &tool_calls {
            let result = {
                let a = agent.lock().await;
                a.tools.dispatch(call).await
            };
            let preview: String = result.chars().take(200).collect();
            let _ = tx.send(UiEvent::ToolResult {
                name: call.name.clone(),
                preview,
            }).await;
            let mut a = agent.lock().await;
            a.history.append_tool_result(call.id.clone(), result);
        }
    }
}

// render an iMessage-style bubble into the lines vec
fn render_bubble(lines: &mut Vec<Line>, m: &TuiMessage) {
    let (fg, bg) = match m.role {
        Role::User => (Color::Black, Color::Gray),
        Role::Assistant => (Color::White, Color::Blue),
        Role::Tool => (Color::Black, Color::Yellow),
        Role::System => (Color::White, Color::DarkGray),
    };

    let style = Style::default().fg(fg).bg(bg);
    let label = match m.role {
        Role::User => "you",
        Role::Assistant => "bob",
        Role::Tool => "tool",
        Role::System => "sys",
    };

    lines.push(Line::from(""));

    let mut first = Line::from(Span::styled(format!("[{label}] "), style));
    let content_lines: Vec<&str> = m.content.lines().collect();
    if content_lines.is_empty() {
        lines.push(first);
        return;
    }
    first.spans.push(Span::styled(content_lines[0].to_string(), style));
    lines.push(first);
    for l in &content_lines[1..] {
        lines.push(Line::from(Span::styled(l.to_string(), style)));
    }
}

fn thinking_spinner() -> char {
    // animate based on current time so the ticker redraws different frames
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    let frames = ['◜', '◝', '◞', '◟'];
    let idx = ((nanos / 250_000_000) as usize) % frames.len();
    frames[idx]
}

fn setup_terminal() -> anyhow::Result<Terminal<CrosstermBackend<std::io::Stdout>>> {
    enable_raw_mode()?;
    execute!(stdout(), EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout());
    Ok(Terminal::new(backend)?)
}

fn restore_terminal() -> anyhow::Result<()> {
    disable_raw_mode()?;
    execute!(stdout(), LeaveAlternateScreen)?;
    Ok(())
}

pub async fn run(config: Config, llm: Box<dyn LlmClient>) -> anyhow::Result<()> {
    let agent = Agent::new(config, llm);
    TuiAgent::new(agent).run().await
}
