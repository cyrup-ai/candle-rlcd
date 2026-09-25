//! Local HTTP server with TypeSafe's Jev API (`POST /v1/systemone`, `GET /v1/models`).
//!
//! The request, response and error bodies follow TypeSafe's published OpenAPI schema
//! (`https://api.typesafe.ai/openapi.json`, v0.2.0), so the official Python and JavaScript SDKs
//! work against it by pointing their base URL at this server.
//!
//! # Concurrency
//!
//! A loaded [`Laya`] is immutable: every weight is a reference-counted Candle tensor and a
//! forward pass only reads them, so one copy is shared by every thread (`Laya: Send + Sync`,
//! checked at compile time below) and more workers cost no extra weight memory. The server runs:
//!
//! - HTTP on a small Tokio runtime. Handlers validate and tokenize (cheap) and enqueue a job.
//! - `workers` inference threads over one bounded queue, each with its own Rayon pools, which
//!   Candle's CPU kernels run on.
//! - Adaptive width: a worker splits the cores between the passes running and the requests
//!   waiting. A lone request gets every core (lowest latency); a full queue gets one core per
//!   worker (highest throughput: Candle's CPU ops parallelize poorly inside one pass, so
//!   independent passes beat splitting each op).
//! - Optional dynamic batching (`max_batch_rows` > 1): a worker drains what is already queued
//!   (optionally waiting `batch_wait` for more) and runs it as one forward pass
//!   ([`Laya::forward_many`]). It pays off on GPUs; on CPU it did not beat one worker per core.
//! - A full queue answers `529 Overloaded` with `retry-after`, as Jev does, so the SDKs back off.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use crossbeam_channel::{Receiver, Sender, TrySendError};
use serde_json::{json, Map, Value};
use tokio::sync::oneshot;

use crate::agent::{round4, RowOutput};
use crate::pipeline::{Budget, Plan, Truncation};
use crate::sequence::{Criteria, Encoded, QType, Question};
use crate::Laya;

const _: () = {
    const fn shareable<T: Send + Sync>() {}
    shareable::<Laya>();
};

/// Jev's documented limits: at most 255 options per Choice and 10 levels per Score.
pub const MAX_CHOICE_OPTIONS: usize = 255;
pub const MAX_SCORE_LEVELS: usize = 10;

/// Model names accepted as aliases for the served model, as in Jev.
pub const ALIASES: [&str; 2] = ["jev-latest", "jev-preview"];

// ---------------------------------------------------------------------------------------------
// Request validation (FastAPI / pydantic-style 422 bodies)
// ---------------------------------------------------------------------------------------------

/// One entry of a 422 body's `detail` list (`ValidationError` in the OpenAPI schema).
#[derive(Debug, Clone, PartialEq)]
pub struct FieldError {
    pub loc: Vec<Value>,
    pub msg: String,
    pub kind: &'static str,
    pub input: Option<Value>,
    pub ctx: Option<Value>,
}

impl FieldError {
    fn new(loc: Vec<Value>, kind: &'static str, msg: impl Into<String>) -> Self {
        Self {
            loc,
            msg: msg.into(),
            kind,
            input: None,
            ctx: None,
        }
    }

    fn input(mut self, v: &Value) -> Self {
        self.input = Some(v.clone());
        self
    }

    fn ctx(mut self, v: Value) -> Self {
        self.ctx = Some(v);
        self
    }

    fn to_json(&self) -> Value {
        let mut o = Map::new();
        o.insert("type".into(), json!(self.kind));
        o.insert("loc".into(), Value::Array(self.loc.clone()));
        o.insert("msg".into(), json!(self.msg));
        if let Some(i) = &self.input {
            o.insert("input".into(), i.clone());
        }
        if let Some(c) = &self.ctx {
            o.insert("ctx".into(), c.clone());
        }
        Value::Object(o)
    }
}

/// `{"detail": [...]}` with status 422.
pub fn validation_body(errors: &[FieldError]) -> Value {
    json!({"detail": errors.iter().map(FieldError::to_json).collect::<Vec<_>>()})
}

/// A validated `POST /v1/systemone` body.
#[derive(Debug, Clone)]
pub struct SystemOneRequest {
    pub state: Value,
    pub model: String,
    /// Questions in request order, keyed by the caller's ids.
    pub questions: Vec<(String, Question)>,
}

fn loc(path: &[&str]) -> Vec<Value> {
    path.iter().map(|s| json!(s)).collect()
}

fn is_entry(v: &Value) -> bool {
    // `string | object | array` (and `null` where the schema allows it, checked by the caller).
    matches!(v, Value::String(_) | Value::Object(_) | Value::Array(_))
}

fn entry_error(loc: Vec<Value>, v: &Value) -> FieldError {
    FieldError::new(loc, "string_type", "Input should be a valid string").input(v)
}

/// Parses and validates a request body against Jev's schema.
pub fn parse_system_one(body: &[u8]) -> std::result::Result<SystemOneRequest, Vec<FieldError>> {
    let v: Value = serde_json::from_slice(body).map_err(|e| {
        vec![FieldError::new(
            vec![json!("body"), json!(e.column())],
            "json_invalid",
            "JSON decode error",
        )
        .input(&json!({}))
        .ctx(json!({"error": e.to_string()}))]
    })?;
    let Some(o) = v.as_object() else {
        return Err(vec![FieldError::new(
            loc(&["body"]),
            "model_attributes_type",
            "Input should be a valid dictionary or object to extract fields from",
        )
        .input(&v)]);
    };
    let mut errs = Vec::new();
    let state = match o.get("state") {
        None => {
            errs.push(
                FieldError::new(loc(&["body", "state"]), "missing", "Field required").input(&v),
            );
            Value::Null
        }
        Some(s) if is_entry(s) => s.clone(),
        Some(s) => {
            errs.push(entry_error(loc(&["body", "state"]), s));
            Value::Null
        }
    };
    let model = match o.get("model") {
        None => {
            errs.push(
                FieldError::new(loc(&["body", "model"]), "missing", "Field required").input(&v),
            );
            String::new()
        }
        Some(Value::String(m)) => m.clone(),
        Some(m) => {
            errs.push(
                FieldError::new(
                    loc(&["body", "model"]),
                    "string_type",
                    "Input should be a valid string",
                )
                .input(m),
            );
            String::new()
        }
    };
    let mut questions = Vec::new();
    match o.get("questions") {
        None => errs.push(
            FieldError::new(loc(&["body", "questions"]), "missing", "Field required").input(&v),
        ),
        Some(Value::Object(qs)) if qs.is_empty() => errs.push(
            FieldError::new(
                loc(&["body", "questions"]),
                "too_short",
                "Dictionary should have at least 1 item after validation, not 0",
            )
            .input(&json!({}))
            .ctx(json!({"field_type": "Dictionary", "min_length": 1, "actual_length": 0})),
        ),
        Some(Value::Object(qs)) => {
            for (id, q) in qs {
                match parse_question(id, q) {
                    Ok(q) => questions.push((id.clone(), q)),
                    Err(e) => errs.extend(e),
                }
            }
        }
        Some(q) => errs.push(
            FieldError::new(
                loc(&["body", "questions"]),
                "dict_type",
                "Input should be a valid dictionary",
            )
            .input(q),
        ),
    }
    if errs.is_empty() {
        Ok(SystemOneRequest {
            state,
            model,
            questions,
        })
    } else {
        Err(errs)
    }
}

fn parse_question(id: &str, q: &Value) -> std::result::Result<Question, Vec<FieldError>> {
    let base = || vec![json!("body"), json!("questions"), json!(id)];
    let Some(o) = q.as_object() else {
        return Err(vec![FieldError::new(
            base(),
            "model_attributes_type",
            "Input should be a valid dictionary or object to extract fields from",
        )
        .input(q)]);
    };
    let tags = "'noul', 'choice', 'score'";
    let t = match o.get("type") {
        None => {
            return Err(vec![FieldError::new(
                base(),
                "union_tag_not_found",
                "Unable to extract tag using discriminator 'type'",
            )
            .input(q)
            .ctx(json!({"discriminator": "'type'"}))])
        }
        Some(Value::String(s)) if s == "choice" => QType::Choice,
        Some(Value::String(s)) if s == "score" => QType::Score,
        Some(Value::String(s)) if s == "noul" => QType::Noul,
        Some(tag) => {
            let tag = tag
                .as_str()
                .map(str::to_string)
                .unwrap_or_else(|| tag.to_string());
            return Err(vec![FieldError::new(
                base(),
                "union_tag_invalid",
                format!("Input tag '{tag}' found using 'type' does not match any of the expected tags: {tags}"),
            )
            .input(q)
            .ctx(json!({"discriminator": "'type'", "tag": tag, "expected_tags": tags}))]);
        }
    };
    // pydantic puts the union tag in the error path: body.questions.<id>.<type>.<field>.
    let at = |field: &str| {
        let mut l = base();
        l.push(json!(t.name()));
        l.push(json!(field));
        l
    };
    let mut errs = Vec::new();
    let ins = match o.get("instructions") {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(v) if is_entry(v) => crate::sequence::render_criterion(v),
        Some(v) => {
            errs.push(entry_error(at("instructions"), v));
            String::new()
        }
    };
    let crit = o.get("criteria");
    let crit = match t {
        QType::Choice => match crit {
            None => {
                errs.push(FieldError::new(at("criteria"), "missing", "Field required").input(q));
                None
            }
            Some(Value::Object(m)) => {
                if m.is_empty() {
                    errs.push(
                        FieldError::new(at("criteria"), "too_short", "A choice needs at least one option")
                            .input(&json!({}))
                            .ctx(json!({"field_type": "Dictionary", "min_length": 1, "actual_length": 0})),
                    );
                } else if m.len() > MAX_CHOICE_OPTIONS {
                    errs.push(
                        FieldError::new(
                            at("criteria"),
                            "too_long",
                            format!("A choice can have at most {MAX_CHOICE_OPTIONS} options, not {}", m.len()),
                        )
                        .ctx(json!({"field_type": "Dictionary", "max_length": MAX_CHOICE_OPTIONS, "actual_length": m.len()})),
                    );
                }
                for (k, v) in m {
                    if !(v.is_null() || is_entry(v)) {
                        let mut l = at("criteria");
                        l.push(json!(k));
                        errs.push(entry_error(l, v));
                    }
                }
                Some(Criteria::Choice(
                    m.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
                ))
            }
            Some(v) => {
                errs.push(
                    FieldError::new(
                        at("criteria"),
                        "dict_type",
                        "Input should be a valid dictionary",
                    )
                    .input(v),
                );
                None
            }
        },
        QType::Score => match crit {
            None => {
                errs.push(FieldError::new(at("criteria"), "missing", "Field required").input(q));
                None
            }
            Some(Value::Array(a)) => {
                if a.is_empty() {
                    errs.push(
                        FieldError::new(
                            at("criteria"),
                            "too_short",
                            "List should have at least 1 item after validation, not 0",
                        )
                        .input(&json!([]))
                        .ctx(json!({"field_type": "List", "min_length": 1, "actual_length": 0})),
                    );
                } else if a.len() > MAX_SCORE_LEVELS {
                    errs.push(
                        FieldError::new(
                            at("criteria"),
                            "too_long",
                            format!("A score can have at most {MAX_SCORE_LEVELS} levels, not {}", a.len()),
                        )
                        .ctx(json!({"field_type": "List", "max_length": MAX_SCORE_LEVELS, "actual_length": a.len()})),
                    );
                }
                for (i, v) in a.iter().enumerate() {
                    if !is_entry(v) {
                        let mut l = at("criteria");
                        l.push(json!(i));
                        errs.push(entry_error(l, v));
                    }
                }
                Some(Criteria::Score(a.clone()))
            }
            Some(v) => {
                errs.push(
                    FieldError::new(at("criteria"), "list_type", "Input should be a valid list")
                        .input(v),
                );
                None
            }
        },
        QType::Noul => match crit {
            None | Some(Value::Null) => Some(Criteria::Noul {
                false_crit: None,
                true_crit: None,
            }),
            Some(Value::Object(m)) => {
                for k in ["true", "false"] {
                    if let Some(v) = m.get(k) {
                        if !(v.is_null() || is_entry(v)) {
                            let mut l = at("criteria");
                            l.push(json!(k));
                            errs.push(entry_error(l, v));
                        }
                    }
                }
                Some(Criteria::Noul {
                    false_crit: m.get("false").cloned(),
                    true_crit: m.get("true").cloned(),
                })
            }
            Some(v) => {
                errs.push(
                    FieldError::new(
                        at("criteria"),
                        "model_attributes_type",
                        "Input should be a valid dictionary or object to extract fields from",
                    )
                    .input(v),
                );
                None
            }
        },
    };
    match crit {
        Some(crit) if errs.is_empty() => Ok(Question {
            t,
            ins,
            crit,
            labels: None,
        }),
        _ => Err(errs),
    }
}

// ---------------------------------------------------------------------------------------------
// Answers
// ---------------------------------------------------------------------------------------------

/// Jev's confidence for a distribution over `k` options: `(k * max p - 1) / (k - 1)`, 0 for a
/// uniform distribution and 1 for a certain one. This is the formula in TypeSafe's confidence
/// docs for Choice; it also reproduces their 3-level Score examples, so it is used for both.
pub fn jev_confidence(p: &[f32]) -> f64 {
    let k = p.len();
    if k < 2 {
        return 1.0;
    }
    let peak = p.iter().cloned().fold(0f32, f32::max) as f64;
    ((k as f64 * peak - 1.0) / (k as f64 - 1.0)).clamp(0.0, 1.0)
}

/// One Jev answer object. `extended` adds laya's extra fields (`act_probability`, and the
/// entropy-free `answer_confidence` = max p), which Jev clients ignore.
pub fn jev_answer(laya: &Laya, q: &Question, out: &RowOutput, extended: bool) -> Value {
    let p = laya.probabilities(q, out);
    let argmax = p
        .iter()
        .enumerate()
        .fold(0, |best, (i, &v)| if v > p[best] { i } else { best });
    let keys = q.option_keys();
    let probs = || -> Map<String, Value> {
        keys.iter()
            .zip(&p)
            .map(|(k, &v)| (k.clone(), json!(round4(v as f64))))
            .collect()
    };
    let conf = round4(jev_confidence(&p));
    let mut a = match &q.crit {
        Criteria::Choice(_) => json!({
            "type": "choice", "choice": keys[argmax], "probabilities": probs(), "confidence": conf,
        }),
        Criteria::Score(levels) => {
            let score: f64 = p
                .iter()
                .enumerate()
                .map(|(i, &v)| i as f64 * v as f64)
                .sum();
            let legend: Map<String, Value> = levels
                .iter()
                .enumerate()
                .map(|(i, v)| (i.to_string(), v.clone()))
                .collect();
            json!({
                "type": "score", "score": round4(score), "legend": legend,
                "probabilities": probs(), "confidence": conf,
            })
        }
        Criteria::Noul { .. } => json!({"type": "noul", "noul": round4(p[1] as f64)}),
    };
    if extended {
        let o = a.as_object_mut().expect("answer is an object");
        o.insert("answer_confidence".into(), json!(round4(p[argmax] as f64)));
        o.insert(
            "act_probability".into(),
            json!(round4(out.act_probs[0] as f64)),
        );
    }
    a
}

/// Tokens the model reads for a request: the state once plus each question's own tokens in the
/// prefix layout, or every full row in laya's layout.
pub fn input_tokens(rows: &[Encoded]) -> usize {
    match rows.first() {
        Some(r) if r.prefix_len > 0 => {
            r.prefix_len
                + rows
                    .iter()
                    .map(|r| r.ids.len() - r.prefix_len)
                    .sum::<usize>()
        }
        _ => rows.iter().map(|r| r.ids.len()).sum(),
    }
}

// ---------------------------------------------------------------------------------------------
// Inference engine: shared model, worker threads, dynamic batching
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Inference threads, each running whole forward passes.
    pub workers: usize,
    /// Rayon threads per worker for Candle's CPU kernels; 0 shares one global pool (sized by
    /// `RAYON_NUM_THREADS`, default all cores) between all workers.
    pub threads_per_worker: usize,
    /// Most question rows one forward pass takes (a larger single request still runs alone).
    pub max_batch_rows: usize,
    /// How long a worker waits for more requests before running a partial batch. 0 runs
    /// whatever is queued at once.
    pub batch_wait: Duration,
    /// Queued requests beyond this get `529 Overloaded`.
    pub max_queue: usize,
    /// Widen a worker's pool (up to every core) when fewer passes are running or queued than
    /// there are cores; `threads_per_worker` is then the narrowest width.
    pub adaptive: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
        Self {
            workers: 1,
            threads_per_worker: cores,
            max_batch_rows: 64,
            batch_wait: Duration::ZERO,
            max_queue: 1024,
            adaptive: true,
        }
    }
}

type Reply = oneshot::Sender<Result<Vec<RowOutput>, String>>;

struct Job {
    rows: Vec<Encoded>,
    reply: Reply,
}

/// Counters for `/health` and benchmarks.
#[derive(Default)]
pub struct Stats {
    pub requests: AtomicU64,
    pub batches: AtomicU64,
    pub rows: AtomicU64,
    /// Forward passes that ran on every core.
    pub burst: AtomicU64,
}

/// Runs forward passes for many concurrent callers over one shared model.
pub struct Engine {
    pub laya: Arc<Laya>,
    pub config: EngineConfig,
    pub stats: Arc<Stats>,
    tx: Sender<Job>,
}

#[derive(Debug)]
pub enum EngineError {
    Overloaded,
    Failed(String),
}

impl Engine {
    pub fn new(laya: Arc<Laya>, config: EngineConfig) -> Result<Self> {
        anyhow::ensure!(config.workers >= 1, "need at least one worker");
        let (tx, rx) = crossbeam_channel::bounded::<Job>(config.max_queue.max(1));
        let stats = Arc::new(Stats::default());
        let busy = Arc::new(AtomicUsize::new(0));
        for w in 0..config.workers {
            // 0 threads per worker: every worker uses the one global pool.
            let pools = Pools::new(w, &config, busy.clone())?;
            let (laya, rx, cfg, stats) = (laya.clone(), rx.clone(), config.clone(), stats.clone());
            std::thread::Builder::new()
                .name(format!("rlcd-worker-{w}"))
                .spawn(move || worker(&laya, &rx, &cfg, &pools, &stats))?;
        }
        Ok(Self {
            laya,
            config,
            stats,
            tx,
        })
    }

    /// Queues one request's rows and waits for its outputs.
    pub async fn run(
        &self,
        rows: Vec<Encoded>,
    ) -> std::result::Result<Vec<RowOutput>, EngineError> {
        let (reply, rx) = oneshot::channel();
        match self.tx.try_send(Job { rows, reply }) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => return Err(EngineError::Overloaded),
            Err(TrySendError::Disconnected(_)) => {
                return Err(EngineError::Failed("inference workers stopped".into()))
            }
        }
        match rx.await {
            Ok(Ok(out)) => Ok(out),
            Ok(Err(e)) => Err(EngineError::Failed(e)),
            Err(_) => Err(EngineError::Failed(
                "inference worker dropped the request".into(),
            )),
        }
    }

    /// Blocking variant of [`Self::run`] for non-async callers.
    pub fn run_blocking(
        &self,
        rows: Vec<Encoded>,
    ) -> std::result::Result<Vec<RowOutput>, EngineError> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(Job { rows, reply })
            .map_err(|_| EngineError::Failed("inference workers stopped".into()))?;
        match rx.blocking_recv() {
            Ok(Ok(out)) => Ok(out),
            Ok(Err(e)) => Err(EngineError::Failed(e)),
            Err(_) => Err(EngineError::Failed(
                "inference worker dropped the request".into(),
            )),
        }
    }
}

/// Where a worker runs its forward passes.
struct Pools {
    /// The worker's own Rayon pools, narrowest first (`None`: the global pool). With
    /// `adapt`, there is one per power-of-two width up to the core count.
    own: Vec<(usize, rayon::ThreadPool)>,
    /// Workers currently running a forward pass, across the engine.
    busy: Arc<AtomicUsize>,
    cores: usize,
}

impl Pools {
    fn new(worker: usize, cfg: &EngineConfig, busy: Arc<AtomicUsize>) -> Result<Self> {
        let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
        let base = cfg.threads_per_worker;
        let adapt = cfg.adaptive && base > 0 && base < cores;
        let mut widths = vec![];
        if base > 0 {
            widths.push(base);
            if adapt {
                let mut w = base * 2;
                while w < cores {
                    widths.push(w);
                    w *= 2;
                }
                widths.push(cores);
            }
        }
        let own = widths
            .into_iter()
            .map(|n| {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .thread_name(move |i| format!("rlcd-w{worker}x{n}-{i}"))
                    .build()
                    .context("building worker thread pool")?;
                Ok((n, pool))
            })
            .collect::<Result<_>>()?;
        Ok(Self { own, busy, cores })
    }
}

fn worker(laya: &Laya, rx: &Receiver<Job>, cfg: &EngineConfig, pools: &Pools, stats: &Stats) {
    let forward = |groups: &[&[Encoded]]| {
        // Adaptive width: split the cores between the passes running now and the requests
        // waiting, so a lone request gets every core and a full queue gets one worker per
        // core (independent passes parallelize better than splitting each op).
        let running = pools.busy.fetch_add(1, Ordering::SeqCst) + 1;
        let out = match pools.own.as_slice() {
            [] => laya.forward_many(groups),
            [(_, p)] => p.install(|| laya.forward_many(groups)),
            all => {
                let share = pools.cores / (running + rx.len()).max(1);
                let (w, p) = all
                    .iter()
                    .rev()
                    .find(|(w, _)| *w <= share)
                    .unwrap_or(&all[0]);
                if *w == pools.cores {
                    stats.burst.fetch_add(1, Ordering::Relaxed);
                }
                p.install(|| laya.forward_many(groups))
            }
        };
        pools.busy.fetch_sub(1, Ordering::SeqCst);
        out
    };
    while let Ok(first) = rx.recv() {
        let mut rows = first.rows.len();
        let mut jobs = vec![first];
        let deadline = Instant::now() + cfg.batch_wait;
        while rows < cfg.max_batch_rows {
            let next = if cfg.batch_wait.is_zero() {
                rx.try_recv().ok()
            } else {
                rx.recv_deadline(deadline).ok()
            };
            let Some(job) = next else { break };
            rows += job.rows.len();
            jobs.push(job);
        }
        stats.batches.fetch_add(1, Ordering::Relaxed);
        stats.rows.fetch_add(rows as u64, Ordering::Relaxed);
        stats
            .requests
            .fetch_add(jobs.len() as u64, Ordering::Relaxed);
        let groups: Vec<&[Encoded]> = jobs.iter().map(|j| j.rows.as_slice()).collect();
        match forward(&groups) {
            Ok(outs) => {
                for (job, out) in jobs.into_iter().zip(outs) {
                    let _ = job.reply.send(Ok(out));
                }
            }
            Err(e) if jobs.len() > 1 => {
                // Isolate the failure: rerun each request alone.
                eprintln!("batched forward failed ({e:#}); retrying requests one by one");
                for job in jobs {
                    let out = forward(&[job.rows.as_slice()]);
                    let _ = job
                        .reply
                        .send(out.map(|mut o| o.remove(0)).map_err(|e| format!("{e:#}")));
                }
            }
            Err(e) => {
                let _ = jobs.remove(0).reply.send(Err(format!("{e:#}")));
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// HTTP
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ServeConfig {
    pub addr: SocketAddr,
    /// Name reported in responses and listed by `/v1/models`.
    pub model_name: String,
    pub description: String,
    /// `YYYY-MM-DD`.
    pub release_date: String,
    /// When set, requests need `Authorization: Bearer <key>`.
    pub api_key: Option<String>,
    /// Add laya's extra answer fields.
    pub extended: bool,
    /// What to do with a question too long for the model's input.
    pub truncation: TruncationPolicy,
}

/// What the server does when a question can't fit the model's input whole (the state never
/// needs cutting: it runs in chunks).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TruncationPolicy {
    /// Refuse with a 422 naming the question, instead of answering a question the model
    /// only saw part of.
    #[default]
    Strict,
    /// Cut it, answer, and count it in the `x-truncated-questions` header (and per answer
    /// with `extended`).
    Report,
}

struct AppState {
    engine: Engine,
    cfg: ServeConfig,
    budget: Budget,
    next_id: AtomicU64,
    started: Instant,
}

/// Builds the router (exposed for tests).
pub fn router(engine: Engine, cfg: ServeConfig) -> Router {
    let state = Arc::new(AppState {
        budget: Budget::for_model(&engine.laya),
        engine,
        cfg,
        next_id: AtomicU64::new(1),
        started: Instant::now(),
    });
    Router::new()
        .route("/v1/systemone", post(system_one))
        .route("/v1/models", get(models))
        .route("/health", get(health))
        .fallback(not_found)
        .method_not_allowed_fallback(method_not_allowed)
        .layer(axum::extract::DefaultBodyLimit::max(16 << 20))
        .with_state(state)
}

/// Binds `cfg.addr` and serves until Ctrl-C.
pub async fn serve(engine: Engine, cfg: ServeConfig) -> Result<()> {
    let addr = cfg.addr;
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("binding {addr}"))?;
    eprintln!(
        "serving {} on http://{} (POST /v1/systemone, GET /v1/models)",
        cfg.model_name,
        listener.local_addr()?
    );
    axum::serve(listener, router(engine, cfg))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

fn json_response(status: StatusCode, body: Value, request_id: &str) -> Response {
    let mut r = (status, axum::Json(body)).into_response();
    if let Ok(v) = HeaderValue::from_str(request_id) {
        r.headers_mut().insert("x-typesafe-request-id", v);
    }
    r
}

impl AppState {
    fn request_id(&self) -> String {
        let n = self.next_id.fetch_add(1, Ordering::Relaxed);
        format!(
            "req_{:x}{:08x}",
            self.started.elapsed().as_micros() as u64 & 0xffff_ffff,
            n
        )
    }

    fn authorize(&self, headers: &HeaderMap, rid: &str) -> Option<Response> {
        let key = self.cfg.api_key.as_deref()?;
        let ok = headers
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .is_some_and(|k| k.trim() == key);
        if ok {
            return None;
        }
        let mut r = json_response(
            StatusCode::UNAUTHORIZED,
            json!({"detail": "Not authenticated"}),
            rid,
        );
        r.headers_mut()
            .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
        Some(r)
    }
}

async fn system_one(State(app): State<Arc<AppState>>, headers: HeaderMap, body: Bytes) -> Response {
    let rid = app.request_id();
    if let Some(r) = app.authorize(&headers, &rid) {
        return r;
    }
    let req = match parse_system_one(&body) {
        Ok(r) => r,
        Err(errs) => {
            return json_response(
                StatusCode::UNPROCESSABLE_ENTITY,
                validation_body(&errs),
                &rid,
            )
        }
    };
    let laya = &app.engine.laya;
    let questions: Vec<Question> = req.questions.iter().map(|(_, q)| q.clone()).collect();
    let mut plan = match Plan::new(laya, &app.budget, &req.state, questions) {
        Ok(p) => p,
        Err(e) => {
            let err = FieldError::new(loc(&["body"]), "value_error", format!("{e:#}"));
            return json_response(
                StatusCode::UNPROCESSABLE_ENTITY,
                validation_body(&[err]),
                &rid,
            );
        }
    };
    let truncated: Vec<(usize, Truncation)> = (0..req.questions.len())
        .map(|i| (i, plan.truncation(i)))
        .filter(|(_, t)| t.any())
        .collect();
    if app.cfg.truncation == TruncationPolicy::Strict && !truncated.is_empty() {
        let errs: Vec<FieldError> = truncated
            .iter()
            .map(|&(i, t)| truncation_error(&req.questions[i], t))
            .collect();
        return json_response(
            StatusCode::UNPROCESSABLE_ENTITY,
            validation_body(&errs),
            &rid,
        );
    }
    // Round one asks every question; a Choice whose options run in batches needs more.
    loop {
        let rows = plan.rows();
        if rows.is_empty() {
            break;
        }
        let outs = match app.engine.run(rows).await {
            Ok(o) => o,
            Err(EngineError::Overloaded) => {
                let mut r = json_response(
                    StatusCode::from_u16(529).expect("valid status"),
                    json!({"detail": "Overloaded: too many queued requests"}),
                    &rid,
                );
                r.headers_mut()
                    .insert(header::RETRY_AFTER, HeaderValue::from_static("1"));
                return r;
            }
            Err(EngineError::Failed(e)) => {
                return json_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    json!({"detail": e}),
                    &rid,
                )
            }
        };
        if let Err(e) = plan.feed(outs) {
            return json_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                json!({"detail": format!("{e:#}")}),
                &rid,
            );
        }
    }
    let tokens = plan.input_tokens();
    let (chunks, state_tokens) = (plan.state_chunks(), plan.state_tokens());
    let rounds: Vec<Vec<usize>> = (0..req.questions.len()).map(|i| plan.rounds(i)).collect();
    let mut answers = Map::new();
    for (i, ((id, _), (q, out))) in req.questions.iter().zip(plan.finish()).enumerate() {
        let mut a = jev_answer(laya, &q, &out, app.cfg.extended);
        if app.cfg.extended {
            let o = a.as_object_mut().expect("answer is an object");
            o.insert("state_chunks".into(), json!(chunks));
            if rounds[i].len() > 1 {
                o.insert("option_batches".into(), json!(rounds[i]));
            }
            if let Some((_, t)) = truncated.iter().find(|(j, _)| *j == i) {
                o.insert(
                    "truncated".into(),
                    json!({"instructions_tokens": t.instructions, "option_tokens": t.options}),
                );
            }
        }
        answers.insert(id.clone(), a);
    }
    let mut r = json_response(
        StatusCode::OK,
        json!({
            "model": app.cfg.model_name,
            "answers": answers,
            "usage": {"input_tokens": tokens, "output_tokens": 0},
        }),
        &rid,
    );
    let h = r.headers_mut();
    h.insert("x-state-tokens", HeaderValue::from(state_tokens));
    h.insert("x-state-chunks", HeaderValue::from(chunks));
    h.insert("x-truncated-questions", HeaderValue::from(truncated.len()));
    r
}

/// Strict mode's 422 for a question too long for the model's input.
fn truncation_error((id, q): &(String, Question), t: Truncation) -> FieldError {
    let field = if t.instructions > 0 {
        "instructions"
    } else {
        "criteria"
    };
    FieldError::new(
        vec![
            json!("body"),
            json!("questions"),
            json!(id),
            json!(q.t.name()),
            json!(field),
        ],
        "too_long",
        format!(
            "Instructions and options take {} tokens, but at most {} fit in the model's input \
             next to the state; {} would be cut. Shorten them, or start the server with \
             --truncation report to answer anyway.",
            t.question_tokens,
            t.limit,
            t.instructions + t.options
        ),
    )
    .ctx(json!({"max_length": t.limit, "actual_length": t.question_tokens}))
}

async fn models(State(app): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    let rid = app.request_id();
    if let Some(r) = app.authorize(&headers, &rid) {
        return r;
    }
    let c = &app.cfg;
    let mut names = vec![c.model_name.clone()];
    names.extend(
        ALIASES
            .iter()
            .map(|a| a.to_string())
            .filter(|a| *a != c.model_name),
    );
    let models: Vec<Value> = names
        .into_iter()
        .map(|n| json!({"name": n, "description": c.description, "release_date": c.release_date}))
        .collect();
    json_response(StatusCode::OK, json!({"models": models}), &rid)
}

async fn health(State(app): State<Arc<AppState>>) -> Response {
    let s = &app.engine.stats;
    let e = &app.engine.config;
    let rid = app.request_id();
    json_response(
        StatusCode::OK,
        json!({
            "status": "ok",
            "model": app.cfg.model_name,
            "layout": format!("{:?}", app.engine.laya.cfg.layout).to_lowercase(),
            "workers": e.workers,
            "threads_per_worker": e.threads_per_worker,
            "adaptive": e.adaptive,
            "max_batch_rows": e.max_batch_rows,
            "batch_wait_ms": e.batch_wait.as_secs_f64() * 1e3,
            "requests": s.requests.load(Ordering::Relaxed),
            "batches": s.batches.load(Ordering::Relaxed),
            "rows": s.rows.load(Ordering::Relaxed),
        }),
        &rid,
    )
}

async fn not_found() -> Response {
    (
        StatusCode::NOT_FOUND,
        axum::Json(json!({"detail": "Not Found"})),
    )
        .into_response()
}

async fn method_not_allowed() -> Response {
    (
        StatusCode::METHOD_NOT_ALLOWED,
        axum::Json(json!({"detail": "Method Not Allowed"})),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn errs(body: Value) -> Vec<Value> {
        let e = parse_system_one(body.to_string().as_bytes()).unwrap_err();
        validation_body(&e)["detail"].as_array().unwrap().clone()
    }

    #[test]
    fn parses_the_documented_example() {
        let r = parse_system_one(
            br#"{"state": "Help! My payouts have been failing for 3 days.", "model": "jev-latest",
                 "questions": {
                   "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?",
                                 "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"}},
                   "department": {"type": "choice", "instructions": "Which team should handle this?",
                                  "criteria": {"billing": "Payments", "technical": null}},
                   "frustration": {"type": "score", "instructions": {"question": "How frustrated?", "ctx": [1]},
                                   "criteria": ["Calm", "Frustrated", {"level": "Very angry"}]}}}"#,
        )
        .unwrap();
        assert_eq!(r.model, "jev-latest");
        let ids: Vec<&str> = r.questions.iter().map(|(i, _)| i.as_str()).collect();
        assert_eq!(ids, ["is_urgent", "department", "frustration"]);
        assert_eq!(r.questions[1].1.option_keys(), ["billing", "technical"]);
        assert_eq!(
            r.questions[2].1.ins,
            r#"{"question": "How frustrated?", "ctx": [1]}"#
        );
    }

    #[test]
    fn errors_follow_fastapi_shape() {
        let d = errs(json!({"model": "jev-latest", "questions": {}}));
        assert_eq!(d[0]["loc"], json!(["body", "state"]));
        assert_eq!(d[0]["type"], "missing");
        assert_eq!(d[1]["type"], "too_short");

        let d = errs(json!({"state": "x", "model": "m", "questions": {"q": {"type": "maybe"}}}));
        assert_eq!(d[0]["type"], "union_tag_invalid");
        assert_eq!(d[0]["loc"], json!(["body", "questions", "q"]));

        let d = errs(json!({"state": "x", "model": "m", "questions": {"q": {"type": "score"}}}));
        assert_eq!(
            d[0]["loc"],
            json!(["body", "questions", "q", "score", "criteria"])
        );
        assert_eq!(d[0]["type"], "missing");

        let d = errs(json!({"state": 3, "model": "m", "questions": {"q": {"type": "noul"}}}));
        assert_eq!(d[0]["loc"], json!(["body", "state"]));

        let e = parse_system_one(b"{not json").unwrap_err();
        assert_eq!(e[0].kind, "json_invalid");
    }

    #[test]
    fn confidence_matches_jev_docs() {
        // TypeSafe's examples: [0.57, 0.43, 0] -> 0.35, [0.74, 0.26, 0] -> 0.61, uniform -> 0.
        assert!((jev_confidence(&[0.0, 0.57, 0.43]) - 0.355).abs() < 1e-6);
        assert!((jev_confidence(&[0.0, 0.74, 0.26]) - 0.61).abs() < 1e-6);
        assert_eq!(jev_confidence(&[0.25; 4]), 0.0);
        assert_eq!(jev_confidence(&[0.0, 1.0]), 1.0);
    }
}
