//! The serving pipeline on the tiny fixture (`max_len` 128, question budget 48, 512
//! positions): nothing a request sends is dropped silently, short requests are unchanged from
//! laya's own encoding, and the batched and chunked paths agree with running each piece alone.

use std::path::{Path, PathBuf};

use candle_core::{DType, Device};
use candle_rlcd::agent::RowOutput;
use candle_rlcd::model::Layout;
use candle_rlcd::pipeline::{combine_chunks, Budget, Plan};
use candle_rlcd::sequence::Question;
use candle_rlcd::Laya;
use serde_json::{json, Value};

fn load(layout: Layout) -> Laya {
    let dir: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny");
    let mut laya = Laya::load(&dir, &Device::Cpu, DType::F32).unwrap();
    laya.cfg.layout = layout;
    laya.model.layout = layout;
    laya
}

fn q(v: Value) -> Question {
    Question::from_json(&v).unwrap()
}

fn questions() -> Vec<Question> {
    vec![
        q(
            json!({"t": "choice", "ins": "Which team handles this?", "crit": {"billing": "payments", "shipping": "delivery", "other": null}}),
        ),
        q(json!({"t": "score", "ins": "How urgent?", "crit": ["low", "medium", "high"]})),
        q(json!({"t": "noul", "ins": "Does the customer want a refund?"})),
    ]
}

fn words(n: usize) -> String {
    (0..n)
        .map(|i| format!("word{} ", i % 97))
        .collect::<String>()
}

fn run(laya: &Laya, budget: &Budget, state: &Value, qs: Vec<Question>) -> Vec<RowOutput> {
    let plan = Plan::new(laya, budget, state, qs).unwrap();
    let (outs, _) = plan.run(|rows| laya.forward(&rows)).unwrap();
    outs.into_iter().map(|(_, o)| o).collect()
}

#[test]
fn short_requests_encode_exactly_as_laya_does() {
    for layout in [Layout::Laya, Layout::Prefix] {
        let laya = load(layout);
        let state = json!("The package arrived damaged and I want a refund.");
        let mut plan = Plan::new(&laya, &Budget::for_model(&laya), &state, questions()).unwrap();
        let rows = plan.rows();
        let old = laya.encode(&state, &questions()).unwrap();
        assert_eq!(rows.len(), old.len());
        for (a, b) in rows.iter().zip(&old) {
            assert_eq!(a.ids, b.ids, "{layout:?}");
            assert_eq!(a.markers, b.markers);
        }
    }
}

#[test]
fn many_options_are_answered_in_batches() {
    let laya = load(Layout::Laya);
    let crit: serde_json::Map<String, Value> = (0..200)
        .map(|i| (format!("opt{i}"), json!(format!("category number {i}"))))
        .collect();
    let big = q(json!({"t": "choice", "ins": "Which category?", "crit": crit}));
    // laya's own encoding refuses this many options.
    assert!(laya
        .encode(&json!("hello"), std::slice::from_ref(&big))
        .is_err());
    let budget = Budget::for_model(&laya);
    let plan = Plan::new(&laya, &budget, &json!("hello"), vec![big.clone()]).unwrap();
    let rounds = plan.rounds(0);
    assert!(rounds[0] > 1, "{rounds:?}");
    let (outs, _) = plan
        .run(|rows| {
            for r in &rows {
                assert!(r.ids.len() <= laya.cfg.max_len, "{}", r.ids.len());
            }
            laya.forward(&rows)
        })
        .unwrap();
    let p = laya.probabilities(&big, &outs[0].1);
    assert_eq!(p.len(), 200);
    assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-4);
    assert!(p.iter().all(|v| v.is_finite() && *v > 0.0));
}

#[test]
fn options_are_no_longer_cut_at_48_tokens() {
    let laya = load(Layout::Laya);
    let long = words(60);
    let question = q(json!({"t": "choice", "ins": "Which?", "crit": {"a": long, "b": "short"}}));
    let mut plan = Plan::new(
        &laya,
        &Budget::for_model(&laya),
        &json!("hi"),
        vec![question],
    )
    .unwrap();
    let rows = plan.rows();
    // Option `a`'s tokens run from its marker to option `b`'s marker.
    let m = &rows[0].markers;
    assert!(m[1] - m[0] > 49, "option a kept {} tokens", m[1] - m[0]);
    assert!(!plan.truncation(0).any());
}

#[test]
fn long_instructions_are_kept_whole() {
    for layout in [Layout::Laya, Layout::Prefix] {
        let laya = load(layout);
        let ins = format!("{} Is the last sentence about refunds?", words(40));
        let question = q(json!({"t": "noul", "ins": ins}));
        let budget = Budget::for_model(&laya);
        let state = json!(words(300));
        let mut plan = Plan::new(&laya, &budget, &state, vec![question.clone()]).unwrap();
        assert!(!plan.truncation(0).any());
        let rows = plan.rows();
        let head = laya
            .tokenizer
            .encode(format!("noul question: {ins}"), false)
            .unwrap();
        let head = head.get_ids();
        for r in &rows {
            // The whole head is in the row, and the state keeps `min_state` tokens beside it.
            assert!(r.ids.windows(head.len()).any(|w| w == head), "{layout:?}");
            assert!(r.ids.len() > laya.cfg.max_len);
            assert!(r.ids.len() >= head.len() + budget.min_state);
        }
        assert!(plan.state_chunks() > 1);
    }
}

#[test]
fn questions_past_the_position_limit_report_what_was_cut() {
    let laya = load(Layout::Laya);
    let ins = format!("{} Is it about refunds?", words(700));
    let question = q(json!({"t": "noul", "ins": ins}));
    let budget = Budget::for_model(&laya);
    let mut plan = Plan::new(&laya, &budget, &json!("hi"), vec![question]).unwrap();
    let t = plan.truncation(0);
    assert!(t.instructions > 0 && t.options == 0, "{t:?}");
    assert!(t.question_tokens > t.limit);
    let rows = plan.rows();
    assert!(rows[0].ids.len() <= budget.max_window);
    // The end of the instructions (the actual question) survives the cut.
    let tail = laya.tokenizer.encode(" refunds?", false).unwrap();
    let tail = tail.get_ids();
    assert!(rows[0].ids.windows(tail.len()).any(|w| w == tail));
}

#[test]
fn a_long_state_runs_in_chunks_that_cover_it() {
    for layout in [Layout::Laya, Layout::Prefix] {
        let laya = load(layout);
        let budget = Budget::for_model(&laya);
        let state = json!(words(600));
        let plan = Plan::new(&laya, &budget, &state, questions()).unwrap();
        let n = plan.state_chunks();
        assert!(n > 3, "{n}");
        let mut plan = plan;
        let rows = plan.rows();
        assert_eq!(rows.len(), n * 3);
        for r in &rows {
            assert!(
                r.ids.len() <= laya.cfg.max_len,
                "{layout:?} {}",
                r.ids.len()
            );
        }
        // Chunked answers equal each chunk's rows run alone, then combined per type. In the
        // prefix layout this checks rows with different state prefixes in one forward.
        let together = laya.forward(&rows).unwrap();
        let alone: Vec<RowOutput> = rows
            .iter()
            .map(|r| laya.forward(std::slice::from_ref(r)).unwrap().remove(0))
            .collect();
        for (a, b) in together.iter().zip(&alone) {
            for (x, y) in a.logits.iter().zip(&b.logits) {
                assert!((x - y).abs() < 1e-4, "{layout:?}: {x} vs {y}");
            }
        }
        let outs = run(&laya, &budget, &state, questions());
        for (qi, question) in questions().iter().enumerate() {
            let per_chunk: Vec<&RowOutput> = (0..n).map(|c| &alone[c * 3 + qi]).collect();
            let (want, _) = combine_chunks(question.t, &per_chunk);
            for (x, y) in outs[qi].logits.iter().zip(&want) {
                assert!((x - y).abs() < 1e-4, "{layout:?} q{qi}: {x} vs {y}");
            }
        }
    }
}
