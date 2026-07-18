use std::collections::HashMap;

use async_trait::async_trait;
use futures::StreamExt;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::history::{LlmRequest, LlmResponse};
use crate::tools::ToolCall;

#[async_trait]
pub trait LlmClient: Send + Sync {
    async fn complete(&self, req: LlmRequest) -> anyhow::Result<LlmResponse>;
    async fn complete_stream(&self, req: LlmRequest) -> anyhow::Result<mpsc::Receiver<StreamEvent>> {
        let _ = req;
        // default: fall back to batch, emit a single content event
        let mut resp = self.complete(req).await?;
        let (tx, rx) = mpsc::channel(16);
        let _ = tx.try_send(StreamEvent::Content(std::mem::take(&mut resp.content)));
        for tc in resp.tool_calls.into_iter() {
            let _ = tx.try_send(StreamEvent::ToolCallStart(tc));
        }
        let _ = tx.try_send(StreamEvent::Done);
        Ok(rx)
    }
}

/// events emitted during a streamed llm call.
#[derive(Clone, Debug)]
pub enum StreamEvent {
    /// accumulated content delta
    Content(String),
    /// tool call fully assembled (emitted once per call when complete)
    ToolCallStart(ToolCall),
    /// tool call argument fragment (for live display)
    ToolCallDelta(String),
    /// final usage/cost for this call
    Cost(CallCost),
    /// stream finished, no more events
    Done,
    /// non-fatal stream error
    Error(String),
}

const DEFAULT_BASE_URL: &str = "https://api.llmgateway.io/v1";

#[derive(Clone, Debug, Default)]
pub struct ModelRates {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
}

#[derive(Clone, Debug, Default)]
pub struct CallCost {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_write_tokens: u64,
    pub total_cost: f64,
}

impl CallCost {
    pub(crate) fn compute(rates: &ModelRates, usage: &OpenAiUsage) -> Self {
        let input_tokens = usage.prompt_tokens;
        let output_tokens = usage.completion_tokens;
        let cache_read_tokens = usage.prompt_tokens_details.as_ref()
            .map(|d| d.cached_tokens).unwrap_or(0);
        let cache_write_tokens = usage.prompt_tokens_details.as_ref()
            .map(|d| d.cache_write_tokens).unwrap_or(0);
        let input_cost = rates.input * input_tokens as f64;
        let output_cost = rates.output * output_tokens as f64;
        let cache_read_cost = rates.cache_read * cache_read_tokens as f64;
        let cache_write_cost = rates.cache_write * cache_write_tokens as f64;
        let total_cost = input_cost + output_cost + cache_read_cost + cache_write_cost;
        Self { input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, total_cost }
    }

    pub fn format(&self) -> String {
        format!(
            "tokens in={}|cache_read={}|cache_write={} out={} cost ${:.6}",
            self.input_tokens, self.cache_read_tokens, self.cache_write_tokens,
            self.output_tokens, self.total_cost
        )
    }
}

pub struct HttpLlmClient {
    api_key: String,
    base_url: String,
    pub last_cost: std::sync::Arc<std::sync::Mutex<Option<CallCost>>>,
    client: reqwest::Client,
    rates: HashMap<String, ModelRates>,
}

impl HttpLlmClient {
    pub async fn from_env() -> anyhow::Result<Self> {
        let api_key = std::env::var("LLMGATEWAY_API_KEY")
            .map_err(|_| anyhow::anyhow!("LLMGATEWAY_API_KEY not set"))?;
        let client = reqwest::Client::new();
        let rates = fetch_model_rates(&client, DEFAULT_BASE_URL, &api_key).await
            .unwrap_or_default();
        Ok(Self {
            api_key,
            base_url: DEFAULT_BASE_URL.to_string(),
            client,
            rates,
            last_cost: std::sync::Arc::new(std::sync::Mutex::new(None)),
        })
    }

    pub fn with_base_url(api_key: String, base_url: String) -> Self {
        Self {
            api_key,
            base_url,
            client: reqwest::Client::new(),
            rates: HashMap::new(),
            last_cost: std::sync::Arc::new(std::sync::Mutex::new(None)),
        }
    }

    pub fn last_cost(&self) -> Option<CallCost> {
        self.last_cost.lock().ok()?.as_ref().cloned()
    }

    fn record_cost(&self, model: &str, usage: &OpenAiUsage) -> CallCost {
        let rates = self.rates.get(model).cloned().unwrap_or_default();
        let cost = CallCost::compute(&rates, usage);
        if let Ok(mut g) = self.last_cost.lock() {
            *g = Some(cost.clone());
        }
        cost
    }
}

async fn fetch_model_rates(
    client: &reqwest::Client,
    base_url: &str,
    api_key: &str,
) -> anyhow::Result<HashMap<String, ModelRates>> {
    let url = format!("{}/models?exclude_deprecated=true", base_url.trim_end_matches('/'));
    let resp = client.get(&url)
        .header(AUTHORIZATION, format!("Bearer {api_key}"))
        .send().await?;
    let parsed: GatewayModelsResponse = serde_json::from_str(&resp.text().await?)?;
    let mut map = HashMap::new();
    for m in parsed.data {
        let pricing = m.providers.first()
            .and_then(|p| p.pricing.as_ref());
        let rates = if let Some(p) = pricing {
            ModelRates {
                input: parse_price(&p.prompt),
                output: parse_price(&p.completion),
                cache_read: parse_price(&p.input_cache_read),
                cache_write: parse_price(&p.input_cache_write),
            }
        } else {
            ModelRates::default()
        };
        map.insert(m.id, rates);
    }
    Ok(map)
}

fn parse_price(s: &Option<String>) -> f64 {
    s.as_deref()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .unwrap_or(0.0)
}

#[async_trait]
impl LlmClient for HttpLlmClient {
    async fn complete(&self, req: LlmRequest) -> anyhow::Result<LlmResponse> {
        let mut body = build_openai_body(&req);
        body["stream"] = Value::Bool(false);
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));

        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, HeaderValue::from_str(&format!("Bearer {}", self.api_key))?);
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let resp = self.client.post(&url).headers(headers).json(&body).send().await?;
        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            anyhow::bail!("llm gateway {status}: {text}");
        }

        let parsed: OpenAiResponse = serde_json::from_str(&text)
            .map_err(|e| anyhow::anyhow!("parse response: {e} (body: {text})"))?;

        if let Some(usage) = &parsed.usage {
            let cost = self.record_cost(&req.model, usage);
            eprintln!("[cost] {} model={}", cost.format(), req.model);
        }

        let msg = parsed.choices.first()
            .ok_or_else(|| anyhow::anyhow!("empty choices"))?
            .message.clone();
        let tool_calls = msg.tool_calls.unwrap_or_default()
            .into_iter().map(parse_tool_call).collect::<anyhow::Result<Vec<_>>>()?;
        Ok(LlmResponse { content: msg.content.unwrap_or_default(), tool_calls })
    }

    async fn complete_stream(&self, req: LlmRequest) -> anyhow::Result<mpsc::Receiver<StreamEvent>> {
        let mut body = build_openai_body(&req);
        body["stream"] = Value::Bool(true);
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));

        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, HeaderValue::from_str(&format!("Bearer {}", self.api_key))?);
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert("Accept", HeaderValue::from_static("text/event-stream"));

        let resp = self.client.post(&url).headers(headers).json(&body).send().await?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("llm gateway {status}: {text}");
        }

        let (tx, rx) = mpsc::channel::<StreamEvent>(64);
        let model = req.model.clone();
        let rates = self.rates.clone();
        let last_cost = self.last_cost.clone();

        // spawn a task that consumes the sse stream and emits events.
        tokio::spawn(async move {
            let mut stream = resp.bytes_stream();
            let mut buf = String::new();
            // accumulated tool calls by index: (id, name, arguments)
            let mut tool_acc: HashMap<usize, (String, String, String)> = HashMap::new();
            let mut tool_order: Vec<usize> = Vec::new();

            while let Some(chunk) = stream.next().await {
                let bytes = match chunk {
                    Ok(b) => b,
                    Err(e) => {
                        let _ = tx.send(StreamEvent::Error(e.to_string())).await;
                        break;
                    }
                };
                buf.push_str(&String::from_utf8_lossy(&bytes));

                // sse events are separated by \n\n; process complete ones.
                while let Some(idx) = buf.find("\n\n") {
                    let event_str = buf[..idx].to_string();
                    buf.drain(..idx + 2);
                    if let Some(evt) = parse_sse_event(&event_str) {
                        match evt {
                            SseEvent::DeltaContent(text) => {
                                let _ = tx.send(StreamEvent::Content(text)).await;
                            }
                            SseEvent::DeltaTool { index, id, name, arguments } => {
                                let entry = tool_acc.entry(index).or_insert_with(|| (String::new(), String::new(), String::new()));
                                if let Some(i) = id { entry.0 = i; }
                                if let Some(n) = name { entry.1 = n; }
                                if let Some(a) = arguments {
                                    entry.2.push_str(&a);
                                    let _ = tx.send(StreamEvent::ToolCallDelta(a)).await;
                                }
                                if !tool_order.contains(&index) {
                                    tool_order.push(index);
                                }
                            }
                            SseEvent::Usage(usage) => {
                                let rates = rates.get(&model).cloned().unwrap_or_default();
                                let cost = CallCost::compute(&rates, &usage);
                                if let Ok(mut g) = last_cost.lock() {
                                    *g = Some(cost.clone());
                                }
                                let _ = tx.send(StreamEvent::Cost(cost)).await;
                            }
                            SseEvent::Done => {
                                // emit completed tool calls in arrival order
                                for &i in &tool_order {
                                    if let Some((id, name, args)) = tool_acc.remove(&i) {
                                        let args_val: Value = serde_json::from_str(&args)
                                            .unwrap_or(Value::String(args));
                                        let _ = tx.send(StreamEvent::ToolCallStart(ToolCall {
                                            id,
                                            name,
                                            arguments: args_val,
                                        })).await;
                                    }
                                }
                                let _ = tx.send(StreamEvent::Done).await;
                                return;
                            }
                        }
                    }
                }
            }
            // stream ended without explicit done — emit any pending tool calls then done.
            for &i in &tool_order {
                if let Some((id, name, args)) = tool_acc.remove(&i) {
                    let args_val: Value = serde_json::from_str(&args)
                        .unwrap_or(Value::String(args));
                    let _ = tx.send(StreamEvent::ToolCallStart(ToolCall {
                        id,
                        name,
                        arguments: args_val,
                    })).await;
                }
            }
            let _ = tx.send(StreamEvent::Done).await;
        });
        Ok(rx)
    }
}

/// raw parsed sse event types
#[derive(Debug)]
enum SseEvent {
    DeltaContent(String),
    DeltaTool { index: usize, id: Option<String>, name: Option<String>, arguments: Option<String> },
    Usage(OpenAiUsage),
    Done,
}

/// parse a single `data: ...` sse payload (without the trailing \n\n).
fn parse_sse_event(raw: &str) -> Option<SseEvent> {
    // skip comment lines (': ping') and empty data
    let data: Vec<&str> = raw.lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .collect();
    let payload = data.join("\n");
    if payload.is_empty() { return None; }
    if payload == "[DONE]" { return Some(SseEvent::Done); }

    let v: Value = serde_json::from_str(&payload).ok()?;
    // usage can appear at top level on the final chunk
    if let Some(usage) = v.get("usage") {
        if let Ok(u) = serde_json::from_value::<OpenAiUsage>(usage.clone()) {
            return Some(SseEvent::Usage(u));
        }
    }

    let choices = v.get("choices")?.as_array()?;
    let first = choices.first()?;
    let delta = first.get("delta")?;

    // content delta
    if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
        if !content.is_empty() {
            return Some(SseEvent::DeltaContent(content.to_string()));
        }
    }

    // tool_calls delta
    if let Some(tcs) = delta.get("tool_calls").and_then(|t| t.as_array()) {
        let tc = tcs.first()?;
        let index = tc.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
        let id = tc.get("id").and_then(|i| i.as_str()).map(String::from);
        let function = tc.get("function");
        let name = function.and_then(|f| f.get("name")).and_then(|n| n.as_str()).map(String::from);
        let arguments = function.and_then(|f| f.get("arguments")).and_then(|a| a.as_str()).map(String::from);
        return Some(SseEvent::DeltaTool { index, id, name, arguments });
    }

    // finish_reason with no further data — treat as done sentinel
    if first.get("finish_reason").is_some() {
        return None;
    }
    None
}

fn build_openai_body(req: &LlmRequest) -> Value {
    let messages: Vec<Value> = req.messages.iter().map(|m| {
        let mut obj = json!({ "role": match m.role {
            crate::history::Role::System => "system",
            crate::history::Role::User => "user",
            crate::history::Role::Assistant => "assistant",
            crate::history::Role::Tool => "tool",
        }, "content": m.content });
        if !m.tool_calls.is_empty() {
            let tcs: Vec<Value> = m.tool_calls.iter().map(|tc| json!({
                "id": tc.id,
                "type": "function",
                "function": { "name": tc.name, "arguments": serde_json::to_string(&tc.arguments).unwrap_or_default() }
            })).collect();
            obj["tool_calls"] = Value::Array(tcs);
        }
        if let Some(id) = &m.tool_call_id {
            obj["tool_call_id"] = Value::String(id.clone());
        }
        obj
    }).collect();

    let mut body = json!({ "model": req.model, "messages": messages, "stream": false });
    if !req.tools.is_empty() {
        let tools: Vec<Value> = req.tools.iter().map(|t| {
            let mut func = t.clone();
            if let Some(obj) = func.as_object_mut() {
                if let Some(is) = obj.remove("input_schema") {
                    obj.insert("parameters".to_string(), is);
                }
            }
            json!({ "type": "function", "function": func })
        }).collect();
        body["tools"] = Value::Array(tools);
    }
    body
}

fn parse_tool_call(tc: OpenAiToolCall) -> anyhow::Result<ToolCall> {
    let args: Value = serde_json::from_str(&tc.function.arguments)
        .unwrap_or(Value::String(tc.function.arguments.clone()));
    Ok(ToolCall { id: tc.id, name: tc.function.name, arguments: args })
}

// ── response types ───────────────────────────────────────────────────────

#[derive(Deserialize)]
struct OpenAiResponse {
    choices: Vec<OpenAiChoice>,
    usage: Option<OpenAiUsage>,
}

#[derive(Deserialize)]
struct OpenAiChoice {
    message: OpenAiMessage,
}

#[derive(Deserialize, Clone)]
struct OpenAiMessage {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAiToolCall>>,
}

#[derive(Deserialize, Clone)]
struct OpenAiToolCall {
    id: String,
    function: OpenAiToolFunction,
}

#[derive(Deserialize, Clone)]
struct OpenAiToolFunction {
    name: String,
    arguments: String,
}

#[derive(Deserialize, Clone, Debug)]
pub(crate) struct OpenAiUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
    #[serde(default)]
    prompt_tokens_details: Option<OpenAiPromptTokenDetails>,
}

#[derive(Deserialize, Clone, Debug)]
struct OpenAiPromptTokenDetails {
    #[serde(default)]
    cached_tokens: u64,
    #[serde(default)]
    cache_write_tokens: u64,
}

#[derive(Deserialize)]
struct GatewayModelsResponse {
    data: Vec<GatewayModel>,
}

#[derive(Deserialize)]
struct GatewayModel {
    id: String,
    #[serde(default)]
    providers: Vec<GatewayProvider>,
}

#[derive(Deserialize, Default)]
struct GatewayProvider {
    #[serde(default)]
    pricing: Option<GatewayPricing>,
}

#[derive(Deserialize, Default)]
struct GatewayPricing {
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    completion: Option<String>,
    #[serde(default)]
    input_cache_read: Option<String>,
    #[serde(default)]
    input_cache_write: Option<String>,
}

// ── replay client ───────────────────────────────────────────────────────

#[derive(Clone, Debug, Default)]
pub struct ReplayLlmClient {
    pub responses: Vec<LlmResponse>,
    pub index: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl ReplayLlmClient {
    pub fn new(responses: Vec<LlmResponse>) -> Self {
        Self { responses, index: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)) }
    }
}

#[async_trait]
impl LlmClient for ReplayLlmClient {
    async fn complete(&self, _req: LlmRequest) -> anyhow::Result<LlmResponse> {
        let idx = self.index.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.responses.get(idx).cloned()
            .ok_or_else(|| anyhow::anyhow!("replay exhausted at index {idx}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cost_compute_gpt4o_mini_rates() {
        let rates = ModelRates { input: 0.15e-6, output: 0.6e-6, cache_read: 0.075e-6, cache_write: 0.0 };
        let usage = OpenAiUsage {
            prompt_tokens: 174,
            completion_tokens: 10,
            prompt_tokens_details: Some(OpenAiPromptTokenDetails {
                cached_tokens: 0,
                cache_write_tokens: 0,
            }),
        };
        let cost = CallCost::compute(&rates, &usage);
        let expected = 174.0 * 0.15e-6 + 10.0 * 0.6e-6;
        assert!((cost.total_cost - expected).abs() < 1e-9);
        assert_eq!(cost.input_tokens, 174);
        assert_eq!(cost.output_tokens, 10);
    }

    #[test]
    fn sse_parse_done_event() {
        assert!(matches!(parse_sse_event("data: [DONE]"), Some(SseEvent::Done)));
    }

    #[test]
    fn sse_parse_content_delta() {
        let raw = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"index\":0}]}";
        match parse_sse_event(raw) {
            Some(SseEvent::DeltaContent(s)) => assert_eq!(s, "hi"),
            other => panic!("expected DeltaContent, got {other:?}"),
        }
    }

    #[test]
    fn sse_parse_tool_call_delta() {
        let raw = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"read\",\"arguments\":\"{\\\"path\\\":\\\"a.txt\\\"}\"}}]},\"index\":0}]}";
        match parse_sse_event(raw) {
            Some(SseEvent::DeltaTool { index, id, name, arguments }) => {
                assert_eq!(index, 0);
                assert_eq!(id.as_deref(), Some("call_1"));
                assert_eq!(name.as_deref(), Some("read"));
                assert!(arguments.unwrap().contains("a.txt"));
            }
            other => panic!("expected DeltaTool, got {other:?}"),
        }
    }
}