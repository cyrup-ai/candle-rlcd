//! Closed-loop load generator: `concurrency` clients each send their next request as soon as
//! the previous one returns, either straight into an in-process [`Engine`] or over HTTP to a
//! running `candle-rlcd serve`.

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;

use crate::sequence::Question;
use crate::serve::{Engine, EngineError};

pub enum Target {
    Engine(Arc<Engine>),
    Http(String),
}

/// Jev request bodies from JSONL records (`state` + `questions`; other keys are dropped).
pub fn load_requests(path: &Path) -> Result<Vec<Value>> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let reqs: Vec<Value> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| -> Result<Value> {
            let r: Value = serde_json::from_str(l)?;
            let questions = r
                .get("questions")
                .and_then(Value::as_object)
                .context("record needs questions")?
                .iter()
                .map(|(id, q)| (id.clone(), to_jev_question(q)))
                .collect::<serde_json::Map<_, _>>();
            Ok(json!({"state": r["state"], "model": "jev-latest", "questions": questions}))
        })
        .collect::<Result<_>>()?;
    if reqs.is_empty() {
        bail!("no requests in {}", path.display());
    }
    Ok(reqs)
}

/// laya-style `{t, ins, crit}` to Jev's `{type, instructions, criteria}`.
fn to_jev_question(q: &Value) -> Value {
    let get = |a: &str, b: &str| q.get(a).or_else(|| q.get(b)).cloned();
    let mut o = serde_json::Map::new();
    if let Some(t) = get("t", "type") {
        o.insert("type".into(), t);
    }
    if let Some(i) = get("ins", "instructions") {
        o.insert("instructions".into(), i);
    }
    if let Some(c) = get("crit", "criteria") {
        o.insert("criteria".into(), c);
    }
    Value::Object(o)
}

pub async fn run(
    target: Target,
    reqs: Vec<Value>,
    concurrency: usize,
    total: usize,
    warmup: usize,
) -> Result<Value> {
    let target = Arc::new(target);
    let reqs = Arc::new(reqs);
    let concurrency = concurrency.max(1);
    // Warm-up: every client sends `warmup` requests first (allocations, page cache, pools).
    let mut clients = Vec::new();
    for c in 0..concurrency {
        let (t, r) = (target.clone(), reqs.clone());
        clients.push(tokio::spawn(async move {
            let mut conn = Conn::new(&t);
            for i in 0..warmup {
                conn.send(&t, &r[(c + i * 7) % r.len()]).await?;
            }
            Ok::<_, anyhow::Error>(conn)
        }));
    }
    let mut conns = Vec::new();
    for c in clients {
        conns.push(c.await??);
    }
    let next = Arc::new(AtomicUsize::new(0));
    let start = Instant::now();
    let mut clients = Vec::new();
    for mut conn in conns {
        let (t, r, next) = (target.clone(), reqs.clone(), next.clone());
        clients.push(tokio::spawn(async move {
            let mut lat = Vec::new();
            let mut questions = 0usize;
            loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= total {
                    break;
                }
                let req = &r[i % r.len()];
                let t0 = Instant::now();
                conn.send(&t, req).await?;
                lat.push(t0.elapsed());
                questions += req["questions"].as_object().map_or(0, |q| q.len());
            }
            Ok::<_, anyhow::Error>((lat, questions))
        }));
    }
    let mut lat: Vec<Duration> = Vec::new();
    let mut questions = 0;
    for c in clients {
        let (l, q) = c.await??;
        lat.extend(l);
        questions += q;
    }
    let wall = start.elapsed().as_secs_f64();
    lat.sort();
    let ms = |q: f64| {
        let i = ((lat.len() as f64 - 1.0) * q).round() as usize;
        (lat[i].as_secs_f64() * 1e4).round() / 10.0
    };
    let mean = lat.iter().map(Duration::as_secs_f64).sum::<f64>() / lat.len() as f64;
    let mut report = json!({
        "concurrency": concurrency,
        "requests": lat.len(),
        "questions": questions,
        "wall_s": (wall * 100.0).round() / 100.0,
        "req_per_s": (lat.len() as f64 / wall * 100.0).round() / 100.0,
        "questions_per_s": (questions as f64 / wall * 100.0).round() / 100.0,
        "latency_ms": {"mean": (mean * 1e4).round() / 10.0, "p50": ms(0.5), "p90": ms(0.9), "p99": ms(0.99), "max": ms(1.0)},
    });
    if let Target::Engine(e) = &*target {
        let s = &e.stats;
        let batches = s.batches.load(Ordering::Relaxed).max(1) as f64;
        report["engine"] = json!({
            "workers": e.config.workers,
            "threads_per_worker": e.config.threads_per_worker,
            "max_batch_rows": e.config.max_batch_rows,
            "batch_wait_ms": e.config.batch_wait.as_secs_f64() * 1e3,
            "adaptive": e.config.adaptive,
            "all_core_share": ((s.burst.load(Ordering::Relaxed) as f64 / batches) * 100.0).round() / 100.0,
            "mean_requests_per_batch": ((s.requests.load(Ordering::Relaxed) as f64 / batches) * 100.0).round() / 100.0,
        });
    }
    Ok(report)
}

/// One client: an HTTP/1.1 keep-alive connection, or nothing for the in-process engine.
struct Conn {
    stream: Option<BufReader<TcpStream>>,
}

impl Conn {
    fn new(_t: &Target) -> Self {
        Self { stream: None }
    }

    async fn send(&mut self, t: &Target, req: &Value) -> Result<()> {
        match t {
            Target::Engine(e) => {
                let questions: Vec<Question> = req["questions"]
                    .as_object()
                    .context("questions")?
                    .values()
                    .map(Question::from_json)
                    .collect::<Result<_>>()?;
                let rows = e.laya.encode(&req["state"], &questions)?;
                match e.run(rows).await {
                    Ok(_) => Ok(()),
                    Err(EngineError::Overloaded) => bail!("engine overloaded"),
                    Err(EngineError::Failed(m)) => bail!("{m}"),
                }
            }
            Target::Http(url) => self.post(url, req).await,
        }
    }

    async fn post(&mut self, url: &str, req: &Value) -> Result<()> {
        let hostport = url
            .strip_prefix("http://")
            .context("only http:// URLs are supported")?
            .trim_end_matches('/');
        if self.stream.is_none() {
            let s = TcpStream::connect(hostport).await?;
            s.set_nodelay(true)?;
            self.stream = Some(BufReader::new(s));
        }
        let body = serde_json::to_vec(req)?;
        let s = self.stream.as_mut().expect("connected");
        let head = format!(
            "POST /v1/systemone HTTP/1.1\r\nHost: {hostport}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            body.len()
        );
        s.get_mut().write_all(head.as_bytes()).await?;
        s.get_mut().write_all(&body).await?;
        let mut line = String::new();
        s.read_line(&mut line).await?;
        let status: u16 = line
            .split_whitespace()
            .nth(1)
            .and_then(|c| c.parse().ok())
            .with_context(|| format!("bad status line {line:?}"))?;
        let mut len = 0usize;
        loop {
            line.clear();
            s.read_line(&mut line).await?;
            if line == "\r\n" || line.is_empty() {
                break;
            }
            if let Some((k, v)) = line.split_once(':') {
                if k.eq_ignore_ascii_case("content-length") {
                    len = v.trim().parse()?;
                }
            }
        }
        let mut buf = vec![0u8; len];
        s.read_exact(&mut buf).await?;
        if status != 200 {
            bail!("HTTP {status}: {}", String::from_utf8_lossy(&buf));
        }
        Ok(())
    }
}
