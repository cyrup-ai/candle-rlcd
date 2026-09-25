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
//!
//! # Several models
//!
//! One server can hold several checkpoints ([`router_models`]); a request's `model` picks one by
//! name, alias or pinned `name@commit`. Each model has its own [`Engine`], and the engines share
//! one count of running passes so the adaptive width accounts for all of them.

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
use crate::metrics::{label, Histogram, HttpMetrics};
use crate::sequence::{Criteria, Encoded, QType, Question};
use crate::Laya;

const _: () = {
    const fn shareable<T: Send + Sync>() {}
    shareable::<Laya>();
};

/// Jev's documented limits: at most 255 options per Choice and 10 levels per Score.
pub const MAX_CHOICE_OPTIONS: usize = 255;
pub const MAX_SCORE_LEVELS: usize = 10;

/// Jev's per-request token budget.
pub const JEV_MAX_REQUEST_TOKENS: usize = 65_536;

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
    queued: Instant,
}

/// Counters for `/health` and benchmarks.
#[derive(Default)]
pub struct Stats {
    pub requests: AtomicU64,
    pub batches: AtomicU64,
    pub rows: AtomicU64,
    /// Forward passes that ran on every core.
    pub burst: AtomicU64,
    /// Questions answered and the tokens they read (`usage.input_tokens`).
    pub questions: AtomicU64,
    pub input_tokens: AtomicU64,
    /// Time per forward pass.
    pub forward: Histogram,
    /// Time requests spent queued before a worker picked them up.
    pub queue_wait: Histogram,
}

/// Runs forward passes for many concurrent callers over one shared model.
pub struct Engine {
    pub laya: Arc<Laya>,
    pub config: EngineConfig,
    pub stats: Arc<Stats>,
    tx: Sender<Job>,
    busy: Arc<AtomicUsize>,
}

#[derive(Debug)]
pub enum EngineError {
    Overloaded,
    Failed(String),
}

impl Engine {
    pub fn new(laya: Arc<Laya>, config: EngineConfig) -> Result<Self> {
        Self::with_shared_busy(laya, config, Arc::new(AtomicUsize::new(0)))
    }

    /// Like [`Self::new`], counting running passes in `busy`, which several engines (one per
    /// served model) can share so their adaptive width sees the whole machine's load.
    pub fn with_shared_busy(
        laya: Arc<Laya>,
        config: EngineConfig,
        busy: Arc<AtomicUsize>,
    ) -> Result<Self> {
        anyhow::ensure!(config.workers >= 1, "need at least one worker");
        let (tx, rx) = crossbeam_channel::bounded::<Job>(config.max_queue.max(1));
        let stats = Arc::new(Stats::default());
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
            busy,
        })
    }

    /// Requests waiting for a worker.
    pub fn queue_len(&self) -> usize {
        self.tx.len()
    }

    /// Forward passes running now, across every engine sharing this one's count.
    pub fn running(&self) -> usize {
        self.busy.load(Ordering::Relaxed)
    }

    /// Queues one request's rows and waits for its outputs.
    pub async fn run(
        &self,
        rows: Vec<Encoded>,
    ) -> std::result::Result<Vec<RowOutput>, EngineError> {
        let (reply, rx) = oneshot::channel();
        match self.tx.try_send(Job {
            rows,
            reply,
            queued: Instant::now(),
        }) {
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
            .send(Job {
                rows,
                reply,
                queued: Instant::now(),
            })
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
        let t0 = Instant::now();
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
        stats.forward.observe(t0.elapsed());
        out
    };
    // A caller that gave up (timed out) has dropped its reply: don't spend a pass on it.
    let next_live = |job: Job| (!job.reply.is_closed()).then_some(job);
    while let Ok(first) = rx.recv() {
        let Some(first) = next_live(first) else {
            continue;
        };
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
            let Some(job) = next_live(job) else {
                continue;
            };
            rows += job.rows.len();
            jobs.push(job);
        }
        for j in &jobs {
            stats.queue_wait.observe(j.queued.elapsed());
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
    /// Name reported in responses and listed by `/v1/models` (for [`router`]'s single model).
    pub model_name: String,
    pub description: String,
    /// `YYYY-MM-DD`.
    pub release_date: String,
    /// When set, requests need `Authorization: Bearer <key>`.
    pub api_key: Option<String>,
    /// Add laya's extra answer fields.
    pub extended: bool,
    /// Requests with more questions than this get a 422 (Jev has no such cap; it bounds how
    /// long one request can hold a worker).
    pub max_questions: Option<usize>,
    /// Requests whose `usage.input_tokens` would exceed this get a 422, like Jev's 64k cap.
    pub max_request_tokens: Option<usize>,
    /// Requests not answered within this get a 504. A request still queued when it times out
    /// is dropped without running; one already in a forward pass finishes it.
    pub timeout: Option<Duration>,
}

impl Default for ServeConfig {
    fn default() -> Self {
        Self {
            addr: SocketAddr::from(([127, 0, 0, 1], 8080)),
            model_name: "candle-rlcd".into(),
            description: String::new(),
            release_date: "1970-01-01".into(),
            api_key: None,
            extended: false,
            max_questions: None,
            max_request_tokens: Some(JEV_MAX_REQUEST_TOKENS),
            timeout: None,
        }
    }
}

/// One model a server answers with.
pub struct ServedModel {
    /// The name requests use and responses report.
    pub name: String,
    /// Other names it answers to, such as a pinned `laya@55cf4c4`.
    pub aliases: Vec<String>,
    pub description: String,
    /// `YYYY-MM-DD`.
    pub release_date: String,
    pub engine: Engine,
}

struct AppState {
    /// The first model is the default: it answers `jev-latest`, `jev-preview`, and any name
    /// when it is the only model.
    models: Vec<ServedModel>,
    cfg: ServeConfig,
    next_id: AtomicU64,
    started: Instant,
    http: HttpMetrics,
}

/// Builds the router for one model named by `cfg` (exposed for tests).
pub fn router(engine: Engine, cfg: ServeConfig) -> Router {
    let model = ServedModel {
        name: cfg.model_name.clone(),
        aliases: vec![],
        description: cfg.description.clone(),
        release_date: cfg.release_date.clone(),
        engine,
    };
    router_models(vec![model], cfg)
}

/// Builds the router for several models; the first is the default.
pub fn router_models(served: Vec<ServedModel>, cfg: ServeConfig) -> Router {
    assert!(!served.is_empty(), "serve needs at least one model");
    let state = Arc::new(AppState {
        models: served,
        cfg,
        next_id: AtomicU64::new(1),
        started: Instant::now(),
        http: HttpMetrics::default(),
    });
    Router::new()
        .route("/v1/systemone", post(system_one))
        .route("/v1/models", get(models))
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .fallback(not_found)
        .method_not_allowed_fallback(method_not_allowed)
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            track_http,
        ))
        .layer(axum::extract::DefaultBodyLimit::max(16 << 20))
        .with_state(state)
}

/// Binds `cfg.addr` and serves one model until Ctrl-C.
pub async fn serve(engine: Engine, cfg: ServeConfig) -> Result<()> {
    let model = ServedModel {
        name: cfg.model_name.clone(),
        aliases: vec![],
        description: cfg.description.clone(),
        release_date: cfg.release_date.clone(),
        engine,
    };
    serve_models(vec![model], cfg).await
}

/// Binds `cfg.addr` and serves `models` (the first is the default) until Ctrl-C.
pub async fn serve_models(models: Vec<ServedModel>, cfg: ServeConfig) -> Result<()> {
    let addr = cfg.addr;
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("binding {addr}"))?;
    let names: Vec<&str> = models.iter().map(|m| m.name.as_str()).collect();
    eprintln!(
        "serving {} on http://{} (POST /v1/systemone, GET /v1/models, GET /metrics)",
        names.join(", "),
        listener.local_addr()?
    );
    axum::serve(listener, router_models(models, cfg))
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

    /// The model a request's `model` names: a name or alias, `jev-latest` / `jev-preview`
    /// for the default, or anything at all when only one model is served.
    fn pick(&self, name: &str) -> std::result::Result<&ServedModel, Box<FieldError>> {
        let found = self
            .models
            .iter()
            .find(|m| m.name == name || m.aliases.iter().any(|a| a == name));
        if let Some(m) = found {
            return Ok(m);
        }
        if self.models.len() == 1 || ALIASES.contains(&name) {
            return Ok(&self.models[0]);
        }
        let names: Vec<&str> = self.models.iter().map(|m| m.name.as_str()).collect();
        Err(Box::new(
            FieldError::new(
                loc(&["body", "model"]),
                "value_error",
                format!(
                    "Value error, unknown model '{name}'; this server has: {}",
                    names.join(", ")
                ),
            )
            .input(&json!(name)),
        ))
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
    let model = match app.pick(&req.model) {
        Ok(m) => m,
        Err(e) => {
            return json_response(
                StatusCode::UNPROCESSABLE_ENTITY,
                validation_body(&[*e]),
                &rid,
            )
        }
    };
    if let Some(max) = app.cfg.max_questions {
        let n = req.questions.len();
        if n > max {
            let err = FieldError::new(
                loc(&["body", "questions"]),
                "too_long",
                format!("Dictionary should have at most {max} items after validation, not {n}"),
            )
            .ctx(json!({"field_type": "Dictionary", "max_length": max, "actual_length": n}));
            return json_response(
                StatusCode::UNPROCESSABLE_ENTITY,
                validation_body(&[err]),
                &rid,
            );
        }
    }
    let engine = &model.engine;
    let laya = &engine.laya;
    let questions: Vec<Question> = req.questions.iter().map(|(_, q)| q.clone()).collect();
    let rows = match laya.encode(&req.state, &questions) {
        Ok(r) => r,
        Err(e) => {
            let err = FieldError::new(loc(&["body"]), "value_error", format!("{e:#}"));
            return json_response(
                StatusCode::UNPROCESSABLE_ENTITY,
                validation_body(&[err]),
                &rid,
            );
        }
    };
    let tokens = input_tokens(&rows);
    if let Some(max) = app.cfg.max_request_tokens {
        if tokens > max {
            let err = FieldError::new(
                loc(&["body"]),
                "value_error",
                format!(
                    "Value error, the request is {tokens} tokens, more than the limit of {max}"
                ),
            )
            .ctx(json!({"max_tokens": max, "actual_tokens": tokens}));
            return json_response(
                StatusCode::UNPROCESSABLE_ENTITY,
                validation_body(&[err]),
                &rid,
            );
        }
    }
    let run = engine.run(rows);
    let result = match app.cfg.timeout {
        Some(t) => match tokio::time::timeout(t, run).await {
            Ok(r) => r,
            Err(_) => {
                return json_response(
                    StatusCode::GATEWAY_TIMEOUT,
                    json!({"detail": format!("Timed out after {:.1} s", t.as_secs_f64())}),
                    &rid,
                )
            }
        },
        None => run.await,
    };
    let outs = match result {
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
    let stats = &engine.stats;
    stats
        .questions
        .fetch_add(outs.len() as u64, Ordering::Relaxed);
    stats
        .input_tokens
        .fetch_add(tokens as u64, Ordering::Relaxed);
    let mut answers = Map::new();
    for ((id, q), out) in req.questions.iter().zip(&outs) {
        answers.insert(id.clone(), jev_answer(laya, q, out, app.cfg.extended));
    }
    json_response(
        StatusCode::OK,
        json!({
            "model": model.name,
            "answers": answers,
            "usage": {"input_tokens": tokens, "output_tokens": 0},
        }),
        &rid,
    )
}

async fn models(State(app): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    let rid = app.request_id();
    if let Some(r) = app.authorize(&headers, &rid) {
        return r;
    }
    let mut listed: Vec<Value> = vec![];
    let mut seen = std::collections::HashSet::new();
    let entry = |n: &str, m: &ServedModel| json!({"name": n, "description": m.description, "release_date": m.release_date});
    for m in &app.models {
        for n in std::iter::once(&m.name).chain(&m.aliases) {
            if seen.insert(n.clone()) {
                listed.push(entry(n, m));
            }
        }
    }
    for a in ALIASES {
        if seen.insert(a.to_string()) {
            listed.push(entry(a, &app.models[0]));
        }
    }
    json_response(StatusCode::OK, json!({"models": listed}), &rid)
}

async fn health(State(app): State<Arc<AppState>>) -> Response {
    let default = &app.models[0];
    let s = &default.engine.stats;
    let e = &default.engine.config;
    let rid = app.request_id();
    let models: Vec<Value> = app
        .models
        .iter()
        .map(|m| {
            json!({
                "name": m.name,
                "aliases": m.aliases,
                "layout": format!("{:?}", m.engine.laya.cfg.layout).to_lowercase(),
                "requests": m.engine.stats.requests.load(Ordering::Relaxed),
                "queued": m.engine.queue_len(),
            })
        })
        .collect();
    json_response(
        StatusCode::OK,
        json!({
            "status": "ok",
            "model": default.name,
            "layout": format!("{:?}", default.engine.laya.cfg.layout).to_lowercase(),
            "workers": e.workers,
            "threads_per_worker": e.threads_per_worker,
            "adaptive": e.adaptive,
            "max_batch_rows": e.max_batch_rows,
            "batch_wait_ms": e.batch_wait.as_secs_f64() * 1e3,
            "requests": s.requests.load(Ordering::Relaxed),
            "batches": s.batches.load(Ordering::Relaxed),
            "rows": s.rows.load(Ordering::Relaxed),
            "models": models,
        }),
        &rid,
    )
}

/// Records every response's route, status and latency for `/metrics`.
async fn track_http(
    State(app): State<Arc<AppState>>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    // Known routes only, so stray paths can't grow the label set.
    let route = match req.uri().path() {
        "/v1/systemone" => "/v1/systemone",
        "/v1/models" => "/v1/models",
        "/health" => "/health",
        "/metrics" => "/metrics",
        _ => "other",
    };
    let t0 = Instant::now();
    let r = next.run(req).await;
    app.http.record(route, r.status().as_u16(), t0.elapsed());
    r
}

/// Prometheus text format: HTTP responses and latency, and per-model queue depth, passes,
/// rows, questions, tokens and forward-pass time.
async fn metrics(State(app): State<Arc<AppState>>) -> Response {
    use std::fmt::Write;
    let mut out = String::new();
    app.http.render(&mut out);
    let gauge = |out: &mut String, name: &str, help: &str, kind: &str| {
        let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} {kind}");
    };
    type Read = fn(&ServedModel) -> f64;
    let per_model: [(&str, &str, &str, Read); 7] = [
        (
            "candle_rlcd_queue_depth",
            "Requests waiting for an inference worker.",
            "gauge",
            |m| m.engine.queue_len() as f64,
        ),
        (
            "candle_rlcd_workers",
            "Inference worker threads.",
            "gauge",
            |m| m.engine.config.workers as f64,
        ),
        (
            "candle_rlcd_requests_total",
            "Requests run through the model.",
            "counter",
            |m| m.engine.stats.requests.load(Ordering::Relaxed) as f64,
        ),
        (
            "candle_rlcd_forward_passes_total",
            "Forward passes (a batch of one or more requests).",
            "counter",
            |m| m.engine.stats.batches.load(Ordering::Relaxed) as f64,
        ),
        (
            "candle_rlcd_rows_total",
            "Question rows run through the encoder.",
            "counter",
            |m| m.engine.stats.rows.load(Ordering::Relaxed) as f64,
        ),
        (
            "candle_rlcd_questions_total",
            "Questions answered.",
            "counter",
            |m| m.engine.stats.questions.load(Ordering::Relaxed) as f64,
        ),
        (
            "candle_rlcd_input_tokens_total",
            "Tokens read by the model (usage.input_tokens).",
            "counter",
            |m| m.engine.stats.input_tokens.load(Ordering::Relaxed) as f64,
        ),
    ];
    for (name, help, kind, read) in per_model {
        gauge(&mut out, name, help, kind);
        for m in &app.models {
            let _ = writeln!(out, "{name}{{model=\"{}\"}} {}", label(&m.name), read(m));
        }
    }
    gauge(
        &mut out,
        "candle_rlcd_running_passes",
        "Forward passes running now, across all models.",
        "gauge",
    );
    let _ = writeln!(
        out,
        "candle_rlcd_running_passes {}",
        app.models[0].engine.running()
    );
    type Pick = fn(&Stats) -> &Histogram;
    let hists: [(&str, &str, Pick); 2] = [
        (
            "candle_rlcd_forward_seconds",
            "Time per forward pass.",
            |s| &s.forward,
        ),
        (
            "candle_rlcd_queue_wait_seconds",
            "Time a request waited for a worker.",
            |s| &s.queue_wait,
        ),
    ];
    for (name, help, pick) in hists {
        gauge(&mut out, name, help, "histogram");
        for m in &app.models {
            pick(&m.engine.stats).render(&mut out, name, &format!("model=\"{}\"", label(&m.name)));
        }
    }
    let mut r = out.into_response();
    r.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/plain; version=0.0.4; charset=utf-8"),
    );
    r
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
