//! Training-side checks on the tiny fixture: every parameter gets a gradient (including the
//! Q/K rows of `Wqkv`, which upstream Candle's RoPE silently cut off), gradients match finite
//! differences, the encode-once prefix path equals the joint forward it is trained with, and a
//! short run learns, checkpoints, and resumes.

use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use candle_rlcd::agent::{build_model, load_tokenizer};
use candle_rlcd::config::{AgentConfig, EncoderConfig};
use candle_rlcd::loss::{proper_scoring_loss, LossConfig, Targets};
use candle_rlcd::model::Layout;
use candle_rlcd::sequence::Question;
use candle_rlcd::Laya;
use serde_json::{json, Value};

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny")
}

/// The fixture model as trainable variables, in the given layout.
fn trainable(layout: Layout) -> (Laya, VarMap) {
    let dir = fixture();
    let mut cfg: AgentConfig =
        serde_json::from_str(&std::fs::read_to_string(dir.join("rl_agent_config.json")).unwrap())
            .unwrap();
    cfg.layout = layout;
    let enc = EncoderConfig::from_json(
        &std::fs::read_to_string(dir.join("encoder/config.json")).unwrap(),
    )
    .unwrap();
    let dev = Device::Cpu;
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let model = build_model(vb, &cfg, &enc).unwrap();
    varmap.load(dir.join("model.safetensors")).unwrap();
    let (tok, sp) = load_tokenizer(&dir).unwrap();
    (Laya::from_parts(model, cfg, enc, tok, sp, &dev), varmap)
}

fn questions() -> Vec<Question> {
    [
        json!({"t": "choice", "ins": "Which team handles this?", "crit": {"billing": "payments", "shipping": "delivery", "other": null}}),
        json!({"t": "score", "ins": "How urgent?", "crit": ["low", "medium", "high", "critical"]}),
        json!({"t": "noul", "ins": "Does the customer want a refund?"}),
    ]
    .iter()
    .map(|q| Question::from_json(q).unwrap())
    .collect()
}

fn state() -> Value {
    json!("Customer says the package arrived damaged and wants a refund. The invoice total is 1,240 dollars and it is overdue by 30 days.")
}

fn loss(agent: &Laya) -> Tensor {
    let rows = agent.encode(&state(), &questions()).unwrap();
    let b = agent
        .model
        .batch(&rows, agent.specials.pad, agent.device())
        .unwrap();
    let logits = agent.model.forward(&b, false).unwrap().logits;
    let t = Targets::new(
        &[
            (vec![0.1, 0.8, 0.1], false),
            (vec![0.0, 0.2, 0.5, 0.3], true),
            (vec![0.3, 0.7], false),
        ],
        logits.dim(1).unwrap(),
        agent.device(),
    )
    .unwrap();
    proper_scoring_loss(&logits, &t, &LossConfig::default(), 0.0).unwrap()
}

#[test]
fn every_scoring_path_parameter_gets_a_gradient() {
    for layout in [Layout::Laya, Layout::Prefix] {
        let (agent, varmap) = trainable(layout);
        let grads = loss(&agent).backward().unwrap();
        let data = varmap.data().lock().unwrap();
        for (name, var) in data.iter() {
            // The act head and the unused temperature buffer are not on the loss path, and the
            // scorer's output bias shifts every logit equally, which softmax ignores.
            if name.starts_with("act_head") || name == "temperature" || name == "scorer.3.bias" {
                continue;
            }
            let g = grads
                .get(var.as_tensor())
                .unwrap_or_else(|| panic!("{layout:?}: no gradient for {name}"));
            let norm = g
                .sqr()
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(norm > 0.0, "{layout:?}: zero gradient for {name}");
            if name.ends_with("attn.Wqkv.weight") {
                // Rows [0, d) are Q, [d, 2d) are K: both must train.
                let d = g.dim(1).unwrap();
                for (part, off) in [("Q", 0), ("K", d)] {
                    let n = g
                        .narrow(0, off, d)
                        .unwrap()
                        .abs()
                        .unwrap()
                        .sum_all()
                        .unwrap()
                        .to_scalar::<f32>()
                        .unwrap();
                    assert!(n > 0.0, "{layout:?}: {name} {part} rows get no gradient");
                }
            }
        }
    }
}

#[test]
fn gradients_match_finite_differences() {
    let (agent, varmap) = trainable(Layout::Prefix);
    let grads = loss(&agent).backward().unwrap();
    let data = varmap.data().lock().unwrap();
    // A K-row weight of a local (sliding-window) and a global layer, plus the head.
    for (name, row) in [
        ("encoder.layers.1.attn.Wqkv.weight", 128 + 5),
        ("encoder.layers.0.attn.Wqkv.weight", 17),
        ("head.layers.0.self_attn.in_proj_weight", 140),
    ] {
        let var = &data[name];
        let g = grads
            .get(var.as_tensor())
            .unwrap()
            .get(row)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let col = (0..g.len())
            .max_by(|&a, &b| g[a].abs().total_cmp(&g[b].abs()))
            .unwrap();
        let w0 = var.as_tensor().copy().unwrap();
        let eps = 1e-2f32;
        let bump = |delta: f32| {
            let mut w = w0.to_vec2::<f32>().unwrap();
            w[row][col] += delta;
            var.set(&Tensor::new(w, &Device::Cpu).unwrap()).unwrap();
            loss(&agent).to_scalar::<f32>().unwrap() as f64
        };
        let fd = (bump(eps) - bump(-eps)) / (2.0 * eps as f64);
        var.set(&w0).unwrap();
        let an = g[col] as f64;
        assert!(
            (fd - an).abs() < 0.02 * an.abs().max(1e-3),
            "{name}[{row},{col}]: backprop {an} vs finite difference {fd}"
        );
    }
}

#[test]
fn encode_once_matches_joint_forward() {
    let (agent, _vm) = trainable(Layout::Prefix);
    let rows = agent.encode(&state(), &questions()).unwrap();
    let b = agent
        .model
        .batch(&rows, agent.specials.pad, agent.device())
        .unwrap();
    let joint = agent.model.forward(&b, false).unwrap();
    let cached = agent
        .model
        .forward_prefix(&rows, agent.specials.pad, agent.device())
        .unwrap();
    let diff = |a: &Tensor, b: &Tensor| {
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    };
    assert!(diff(&joint.logits, &cached.logits) < 1e-4, "logits differ");
    assert!(
        diff(&joint.act_logits, &cached.act_logits) < 1e-4,
        "act logits differ"
    );
    // And one question alone gives the same answer as in a batch of three.
    let one = agent
        .model
        .forward_prefix(&rows[1..2], agent.specials.pad, agent.device())
        .unwrap();
    let k = rows[1].markers.len();
    assert!(
        diff(
            &one.logits.narrow(1, 0, k).unwrap(),
            &cached
                .logits
                .get(1)
                .unwrap()
                .narrow(0, 0, k)
                .unwrap()
                .unsqueeze(0)
                .unwrap()
        ) < 1e-4
    );
}

#[test]
fn prefix_state_is_blind_to_the_question() {
    // Changing a question must not change the state's hidden states in the prefix layout.
    let (agent, _vm) = trainable(Layout::Prefix);
    let qs = questions();
    let r1 = agent.encode(&state(), &qs[0..1]).unwrap();
    let r2 = agent.encode(&state(), &qs[1..2]).unwrap();
    let p = r1[0].prefix_len;
    assert_eq!(r1[0].ids[..p], r2[0].ids[..p]);
    let h = |rows: &[candle_rlcd::sequence::Encoded]| {
        let b = agent
            .model
            .batch(rows, agent.specials.pad, agent.device())
            .unwrap();
        agent
            .model
            .encoder
            .forward_masked(&b.ids, &b.masks)
            .unwrap()
            .narrow(1, 0, p)
            .unwrap()
    };
    let d = (h(&r1) - h(&r2))
        .unwrap()
        .abs()
        .unwrap()
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(d < 1e-5, "state hidden states depend on the question ({d})");
}

#[test]
fn train_checkpoint_resume_and_load() {
    use candle_rlcd::train::{LayoutArg, TrainConfig, Trainer};
    let dir = std::env::temp_dir().join(format!("crlcd-train-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let data = dir.join("syn.jsonl");
    candle_rlcd::data::synthetic(24, 3, &mut std::fs::File::create(&data).unwrap()).unwrap();
    let cfg = TrainConfig {
        train: data.clone(),
        eval: Some(data.clone()),
        out: dir.join("run"),
        init: Some(fixture()),
        init_encoder: None,
        encoder_config: None,
        tokenizer: None,
        resume: None,
        layout: LayoutArg::Prefix,
        max_len: 128,
        head_max_len: 48,
        head_layers: 2,
        epochs: 1,
        max_steps: Some(4),
        batch_size: 8,
        grad_accum: 2,
        lr_encoder: 1e-3,
        lr_head: 1e-3,
        min_lr_ratio: 0.1,
        warmup: 0.25,
        weight_decay: 0.01,
        clip: 1.0,
        loss: Default::default(),
        augment: Default::default(),
        seed: 1,
        log_every: 1,
        eval_every: 0,
        save_every: 2,
        keep: 2,
        eval_max: 0,
    };
    let dev = Device::Cpu;
    let summary = Trainer::new(cfg.clone(), &dev).unwrap().run().unwrap();
    assert_eq!(summary["step"], 4);
    assert!(summary["eval_calibrated"]["nll"]
        .as_f64()
        .unwrap()
        .is_finite());
    let step2 = dir.join("run/step-2");
    assert!(step2.join("optimizer.safetensors").exists());

    // Resume from step 2 into a new directory and finish the same 4 steps.
    let mut again = cfg.clone();
    again.resume = Some(step2);
    again.init = None;
    again.out = dir.join("resumed");
    let summary = Trainer::new(again, &dev).unwrap().run().unwrap();
    assert_eq!(summary["step"], 4);

    // The final checkpoint is a normal model directory with fitted temperatures.
    let model = Laya::load(dir.join("resumed/final"), &dev, DType::F32).unwrap();
    assert_eq!(model.cfg.layout, Layout::Prefix);
    assert!(!model.cfg.temperature_by_options.is_empty());
    let qs = json!({"team": {"t": "choice", "ins": "Which team?", "crit": {"billing": "charges", "shipping": "parcels"}}});
    let out = model.system_one(&state(), qs.as_object().unwrap()).unwrap();
    assert!(out["team"]["choice"].is_string());
    std::fs::remove_dir_all(&dir).ok();
}
