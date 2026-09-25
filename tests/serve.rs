//! The HTTP server end to end on the tiny fixture: Jev response shapes, validation errors,
//! auth, and concurrent requests batched together getting the same answers as alone.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use candle_core::{DType, Device};
use candle_rlcd::serve::{router, router_models, Engine, EngineConfig, ServeConfig, ServedModel};
use candle_rlcd::Laya;
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// The tiny fixture, or a copy of it switched to the encode-once prefix layout.
fn fixture(prefix: bool) -> PathBuf {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny");
    if !prefix {
        return dir;
    }
    let out = std::env::temp_dir().join(format!("crlcd-serve-prefix-{}", std::process::id()));
    if !out.exists() {
        let tmp = out.with_extension("tmp");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        for f in ["model.safetensors", "encoder", "tokenizer"] {
            std::os::unix::fs::symlink(dir.join(f), tmp.join(f)).unwrap();
        }
        let mut cfg: Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("rl_agent_config.json")).unwrap(),
        )
        .unwrap();
        cfg["layout"] = json!("prefix");
        std::fs::write(tmp.join("rl_agent_config.json"), cfg.to_string()).unwrap();
        let _ = std::fs::rename(&tmp, &out);
    }
    out
}

async fn start(api_key: Option<&str>, engine: EngineConfig) -> String {
    start_with(fixture(false), api_key, engine).await
}

async fn start_with(dir: PathBuf, api_key: Option<&str>, engine: EngineConfig) -> String {
    let laya = Laya::load(&dir, &Device::Cpu, DType::F32).unwrap();
    let engine = Engine::new(Arc::new(laya), engine).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let cfg = ServeConfig {
        addr,
        model_name: "tiny".into(),
        description: "test".into(),
        release_date: "2026-09-25".into(),
        api_key: api_key.map(str::to_string),
        extended: false,
        ..ServeConfig::default()
    };
    tokio::spawn(async move { axum::serve(listener, router(engine, cfg)).await.unwrap() });
    addr.to_string()
}

/// One HTTP/1.1 request on a fresh connection; returns (status, headers, JSON body).
async fn call(
    addr: &str,
    method: &str,
    path: &str,
    body: Option<&str>,
    auth: Option<&str>,
) -> (u16, String, Value) {
    let mut s = TcpStream::connect(addr).await.unwrap();
    let body = body.unwrap_or("");
    let auth = auth
        .map(|k| format!("Authorization: Bearer {k}\r\n"))
        .unwrap_or_default();
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n{auth}Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes()).await.unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).await.unwrap();
    let (head, body) = out.split_once("\r\n\r\n").unwrap();
    let status = head.split_whitespace().nth(1).unwrap().parse().unwrap();
    (
        status,
        head.to_lowercase(),
        serde_json::from_str(body).unwrap_or(Value::Null),
    )
}

fn request(state: &str) -> String {
    json!({
        "state": state,
        "model": "jev-latest",
        "questions": {
            "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?",
                          "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"}},
            "department": {"type": "choice", "instructions": "Which team should handle this?",
                           "criteria": {"billing": "Payments, invoicing, refunds", "technical": "Bugs, outages", "sales": null}},
            "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                            "criteria": ["Calm", "Frustrated", "Very angry"]}
        }
    })
    .to_string()
}

fn close(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            (x.as_f64().unwrap() - y.as_f64().unwrap()).abs() < 2e-4
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| close(v, w)))
        }
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(v, w)| close(v, w))
        }
        _ => a == b,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn systemone_matches_jev_shapes() {
    let addr = start(None, EngineConfig::default()).await;
    let (status, head, r) = call(
        &addr,
        "POST",
        "/v1/systemone",
        Some(&request("Help! My payouts have been failing for 3 days.")),
        None,
    )
    .await;
    assert_eq!(status, 200, "{r}");
    assert!(head.contains("x-typesafe-request-id"));
    assert_eq!(r["model"], "tiny");
    assert!(r["usage"]["input_tokens"].as_u64().unwrap() > 0);
    assert_eq!(r["usage"]["output_tokens"], 0);
    let a = &r["answers"];
    // Answers come back in request order with exactly Jev's fields.
    let ids: Vec<&String> = a.as_object().unwrap().keys().collect();
    assert_eq!(ids, ["is_urgent", "department", "frustration"]);
    let keys = |v: &Value| {
        let mut k: Vec<String> = v.as_object().unwrap().keys().cloned().collect();
        k.sort();
        k
    };
    assert_eq!(keys(&a["is_urgent"]), ["noul", "type"]);
    assert_eq!(
        keys(&a["department"]),
        ["choice", "confidence", "probabilities", "type"]
    );
    assert_eq!(
        keys(&a["frustration"]),
        ["confidence", "legend", "probabilities", "score", "type"]
    );
    assert_eq!(
        a["frustration"]["legend"],
        json!({"0": "Calm", "1": "Frustrated", "2": "Very angry"})
    );
    let p: f64 = a["department"]["probabilities"]
        .as_object()
        .unwrap()
        .values()
        .map(|v| v.as_f64().unwrap())
        .sum();
    assert!((p - 1.0).abs() < 1e-3);
    let choice = a["department"]["choice"].as_str().unwrap();
    let best = a["department"]["probabilities"]
        .as_object()
        .unwrap()
        .iter()
        .max_by(|x, y| x.1.as_f64().partial_cmp(&y.1.as_f64()).unwrap())
        .unwrap()
        .0;
    assert_eq!(choice, best);

    let (status, _, m) = call(&addr, "GET", "/v1/models", None, None).await;
    assert_eq!(status, 200);
    let names: Vec<&str> = m["models"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["tiny", "jev-latest", "jev-preview"]);

    let (status, _, e) = call(&addr, "POST", "/v1/systemone", Some(r#"{"state": "x", "model": "m", "questions": {"q": {"type": "score", "criteria": []}}}"#), None).await;
    assert_eq!(status, 422);
    assert_eq!(
        e["detail"][0]["loc"],
        json!(["body", "questions", "q", "score", "criteria"])
    );
    let (status, _, _) = call(&addr, "GET", "/v1/nothing", None, None).await;
    assert_eq!(status, 404);
    let (status, _, _) = call(&addr, "GET", "/v1/systemone", None, None).await;
    assert_eq!(status, 405);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn api_key_is_enforced_when_set() {
    let addr = start(Some("sekret"), EngineConfig::default()).await;
    let body = request("hi");
    assert_eq!(
        call(&addr, "POST", "/v1/systemone", Some(&body), None)
            .await
            .0,
        401
    );
    assert_eq!(
        call(&addr, "POST", "/v1/systemone", Some(&body), Some("wrong"))
            .await
            .0,
        401
    );
    assert_eq!(
        call(&addr, "POST", "/v1/systemone", Some(&body), Some("sekret"))
            .await
            .0,
        200
    );
    assert_eq!(call(&addr, "GET", "/v1/models", None, None).await.0, 401);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_requests_batch_without_changing_answers() {
    batch_check(false).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_prefix_requests_batch_without_changing_answers() {
    batch_check(true).await;
}

async fn batch_check(prefix: bool) {
    let states = [
        "Help! My payouts have been failing for 3 days.",
        "Thanks, all good now.",
        "I was charged twice and nobody answers my emails. This is the third time this month and I am furious. Cancel my account unless someone calls me today.",
        "Can I upgrade to the team plan?",
    ];
    // Reference answers one at a time, no batching.
    let solo = start_with(
        fixture(prefix),
        None,
        EngineConfig {
            max_batch_rows: 1,
            ..EngineConfig::default()
        },
    )
    .await;
    let mut expected = Vec::new();
    for s in states {
        expected.push(
            call(&solo, "POST", "/v1/systemone", Some(&request(s)), None)
                .await
                .2,
        );
    }
    // One worker that waits to fill batches, hit by all requests at once, several times.
    let batched = start_with(
        fixture(prefix),
        None,
        EngineConfig {
            max_batch_rows: 64,
            batch_wait: Duration::from_millis(50),
            ..EngineConfig::default()
        },
    )
    .await;
    let mut tasks = Vec::new();
    for round in 0..3 {
        for (i, s) in states.iter().enumerate() {
            let (addr, body) = (batched.clone(), request(s));
            tasks.push(tokio::spawn(async move {
                (
                    round,
                    i,
                    call(&addr, "POST", "/v1/systemone", Some(&body), None).await,
                )
            }));
        }
    }
    for t in tasks {
        let (_, i, (status, _, r)) = t.await.unwrap();
        assert_eq!(status, 200);
        assert!(
            close(&r["answers"], &expected[i]["answers"]),
            "{}\nvs\n{}",
            r["answers"],
            expected[i]["answers"]
        );
    }
}

/// Two models on one server, the /metrics page, and --max-questions.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn several_models_metrics_and_question_cap() {
    let busy = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let model = |dir: PathBuf, name: &str, aliases: &[&str]| ServedModel {
        name: name.into(),
        aliases: aliases.iter().map(|a| a.to_string()).collect(),
        description: format!("{name} model"),
        release_date: "2026-09-25".into(),
        engine: Engine::with_shared_busy(
            Arc::new(Laya::load(&dir, &Device::Cpu, DType::F32).unwrap()),
            EngineConfig::default(),
            busy.clone(),
        )
        .unwrap(),
    };
    let models = vec![
        model(fixture(false), "joint", &["joint@abc1234"]),
        model(fixture(true), "prefix", &[]),
    ];
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let cfg = ServeConfig {
        addr,
        max_questions: Some(2),
        ..ServeConfig::default()
    };
    tokio::spawn(async move {
        axum::serve(listener, router_models(models, cfg))
            .await
            .unwrap()
    });
    let addr = addr.to_string();
    let body = |model: &str, n: usize| {
        let qs: serde_json::Map<String, Value> = (0..n)
            .map(|i| {
                (
                    format!("q{i}"),
                    json!({"type": "noul", "instructions": "Urgent?"}),
                )
            })
            .collect();
        json!({"state": "Payouts failing", "model": model, "questions": qs}).to_string()
    };

    // Each name reaches its own model; jev-latest is the first; unknown names are refused.
    for (name, served) in [
        ("joint", "joint"),
        ("joint@abc1234", "joint"),
        ("prefix", "prefix"),
        ("jev-latest", "joint"),
    ] {
        let (status, _, r) = call(&addr, "POST", "/v1/systemone", Some(&body(name, 1)), None).await;
        assert_eq!(status, 200, "{name}: {r}");
        assert_eq!(r["model"], served);
    }
    let (status, _, e) = call(&addr, "POST", "/v1/systemone", Some(&body("nope", 1)), None).await;
    assert_eq!(status, 422);
    assert_eq!(e["detail"][0]["loc"], json!(["body", "model"]));

    let (status, _, e) = call(
        &addr,
        "POST",
        "/v1/systemone",
        Some(&body("joint", 3)),
        None,
    )
    .await;
    assert_eq!(status, 422, "{e}");
    assert_eq!(e["detail"][0]["type"], "too_long");
    assert_eq!(e["detail"][0]["ctx"]["max_length"], 2);

    let (_, _, m) = call(&addr, "GET", "/v1/models", None, None).await;
    let names: Vec<&str> = m["models"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["name"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        [
            "joint",
            "joint@abc1234",
            "prefix",
            "jev-latest",
            "jev-preview"
        ]
    );

    // Prometheus text.
    let mut s = TcpStream::connect(&addr).await.unwrap();
    s.write_all(b"GET /metrics HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
        .await
        .unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).await.unwrap();
    assert!(out.contains("text/plain; version=0.0.4"), "{out}");
    for line in [
        "candle_rlcd_requests_total{model=\"joint\"} 3",
        "candle_rlcd_requests_total{model=\"prefix\"} 1",
        "candle_rlcd_questions_total{model=\"joint\"} 3",
        "candle_rlcd_http_requests_total{route=\"/v1/systemone\",status=\"200\"} 4",
        "candle_rlcd_http_requests_total{route=\"/v1/systemone\",status=\"422\"} 2",
        "candle_rlcd_forward_seconds_count{model=\"prefix\"} 1",
        "candle_rlcd_queue_depth{model=\"joint\"} 0",
    ] {
        assert!(out.contains(line), "missing {line:?} in\n{out}");
    }
}
