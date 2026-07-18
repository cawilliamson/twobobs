use std::io::stdout;
use std::time::Instant;

use crossterm::event::{Event, EventStream, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::execute;
use futures::StreamExt;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Terminal;

use crate::config::Config;
use crate::history::Role;
use crate::llm::{LlmClient, StreamEvent};
use crate::Agent;

pub struct TuiAgent {
    agent: Agent,
    messages: Vec<TuiMessage>,
    input: String,
    status: String,
    busy: bool,
}

struct TuiMessage {
    role: Role,
    content: String,
}

impl TuiAgent {
    pub fn new(agent: Agent) -> Self {
        Self {
            agent,
            messages: Vec::new(),
            input: String::new(),
            status: "ready".to_string(),
            busy: false,
        }
    }

    pub async fn run(mut self) -> anyhow::Result<()> {
        let mut terminal = setup_terminal()?;
        let mut events = EventStream::new();
        self.status = format!("model={} cwd={} — type a message, enter to send, ctrl-c to quit",
            self.agent.config.model,
            self.agent.config.repo_root.display());

        loop {
            terminal.draw(|f| self.draw(f))?;
            tokio::select! {
                maybe_evt = events.next() => {
                    let Some(Ok(evt)) = maybe_evt else { break; };
                    if !self.handle_event(evt).await? { break; }
                }
            }
        }
        restore_terminal()?;
        Ok(())
    }

    fn draw(&self, f: &mut ratatui::Frame) {
        let chunks = Layout::vertical([
            Constraint::Min(3),
            Constraint::Length(3),
            Constraint::Length(1),
        ]).split(f.area());

        // chat scrollback
        let chat_style = Style::default().fg(Color::White).bg(Color::Black);
        let lines: Vec<Line> = self.messages.iter().flat_map(|m| {
            let prefix = match m.role {
                Role::User => Span::styled("you> ", Style::default().fg(Color::Cyan).bg(Color::Black)),
                Role::Assistant => Span::styled("bob> ", Style::default().fg(Color::Green).bg(Color::Black)),
                Role::Tool => Span::styled("tool> ", Style::default().fg(Color::Yellow).bg(Color::Black)),
                Role::System => Span::styled("sys> ", Style::default().fg(Color::DarkGray).bg(Color::Black)),
            };
            m.content.lines().enumerate().map(move |(i, l)| {
                let mut line = Line::default();
                if i == 0 { line.spans.push(prefix.clone()); }
                line.spans.push(Span::styled(l.to_string(), chat_style));
                line
            }).collect::<Vec<_>>()
        }).collect();

        let chat = Paragraph::new(lines)
            .style(chat_style)
            .block(Block::default().borders(Borders::ALL).title("twobobs").style(chat_style))
            .wrap(Wrap { trim: false })
            .scroll((0, u16::MAX));

        // input box
        let input_style = Style::default().fg(Color::Black).bg(Color::White);
        let input = Paragraph::new(self.input.as_str())
            .style(input_style)
            .block(Block::default().borders(Borders::ALL).title("input").style(input_style));
        f.render_widget(input, chunks[1]);

        // status bar
        let status_text = if self.busy {
            format!("⟳ {} (streaming…)", self.status)
        } else {
            self.status.clone()
        };
        let status_style = Style::default().fg(Color::Yellow).bg(Color::Black);
        let status = Paragraph::new(status_text)
            .style(status_style);
        f.render_widget(status, chunks[2]);
    }

    async fn handle_event(&mut self, evt: Event) -> anyhow::Result<bool> {
        if let Event::Key(k) = evt {
            if k.kind != KeyEventKind::Press { return Ok(true); }
            match k.code {
                KeyCode::Enter if !self.busy && !self.input.is_empty() => {
                    let prompt = std::mem::take(&mut self.input);
                    self.messages.push(TuiMessage { role: Role::User, content: prompt.clone() });
                    self.busy = true;
                    self.status = "thinking".to_string();
                    self.run_turn_streamed(prompt).await?;
                    self.busy = false;
                    self.status = "ready".to_string();
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

    async fn run_turn_streamed(&mut self, prompt: String) -> anyhow::Result<()> {
        self.agent.history.append_user(prompt);
        loop {
            let req = self.agent.history.to_request(&self.agent.config.model, &self.agent.tools.schemas());
            let mut rx = self.agent.llm.complete_stream(req).await?;
            let mut content = String::new();
            let mut tool_calls: Vec<crate::tools::ToolCall> = Vec::new();
            let mut last_cost: Option<crate::llm::CallCost> = None;
            while let Some(evt) = rx.recv().await {
                match evt {
                    StreamEvent::Content(delta) => {
                        content.push_str(&delta);
                        self.messages.push(TuiMessage { role: Role::Assistant, content: delta });
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
                        self.messages.push(TuiMessage { role: Role::Assistant, content: format!("error: {e}") });
                        return Ok(());
                    }
                }
            }

            if let Some(c) = &last_cost {
                self.status = format!("{} cost ${:.6} ({}+{} tok)",
                    self.agent.config.model, c.total_cost, c.input_tokens, c.output_tokens);
            }

            if tool_calls.is_empty() {
                self.agent.history.append_assistant(content);
                return Ok(());
            }
            self.agent.history.append_assistant_with_tools(content, tool_calls.clone());
            for call in &tool_calls {
                let result = self.agent.tools.dispatch(call).await;
                self.messages.push(TuiMessage {
                    role: Role::Tool,
                    content: format!("{} → {}", call.name, result.chars().take(200).collect::<String>()),
                });
                self.agent.history.append_tool_result(call.id.clone(), result);
            }
        }
    }
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