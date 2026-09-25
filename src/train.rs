//! Training loop: data, forward/backward, optimizer, eval, calibration and checkpoints.
//!
//! A checkpoint directory is a normal model directory (`model.safetensors`,
//! `rl_agent_config.json`, `encoder/config.json`, `tokenizer/`) that `Laya::load` and the CLI
//! run as-is, plus `optimizer.safetensors` and `trainer_state.json` for exact resumption.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, ensure, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::agent::{build_model, load_tokenizer_from, softmax, temperature_key, Laya};
use crate::config::{AgentConfig, EncoderConfig};
use crate::data::{augment, load_jsonl, AugmentConfig, Example, Rng};
use crate::loss::{proper_scoring_loss, LossConfig, Targets};
use crate::model::Layout;
use crate::optim::{clip_grad_norm, AdamW, Group, Param, Schedule};
use crate::sequence::{Encoded, QType};

#[derive(Debug, Clone, Serialize, Deserialize, clap::Args)]
pub struct TrainConfig {
    /// Training records (JSONL, see `data`).
    #[arg(long)]
    pub train: PathBuf,
    /// Held-out records: half fits temperatures, half is reported.
    #[arg(long)]
    pub eval: Option<PathBuf>,
    /// Output directory for checkpoints and `metrics.jsonl`.
    #[arg(long)]
    pub out: PathBuf,

    /// Start from a model directory (a laya snapshot or one of our checkpoints); every tensor
    /// present there is loaded, the rest keep their fresh init.
    #[arg(long)]
    pub init: Option<PathBuf>,
    /// Start the encoder from a Hugging Face ModernBERT directory (`config.json`,
    /// `model.safetensors`, `tokenizer.json`), e.g. `answerdotai/ModernBERT-base`.
    #[arg(long)]
    pub init_encoder: Option<PathBuf>,
    /// Encoder config for a model trained from scratch.
    #[arg(long)]
    pub encoder_config: Option<PathBuf>,
    /// Directory with `tokenizer.json`, when not taken from `--init`/`--init-encoder`.
    #[arg(long)]
    pub tokenizer: Option<PathBuf>,
    /// Continue from a checkpoint written by this trainer (model, optimizer and data position).
    #[arg(long)]
    pub resume: Option<PathBuf>,

    /// `prefix` (state encoded once per request) or `laya` (laya's joint layout).
    #[arg(long, value_enum, default_value = "prefix")]
    pub layout: LayoutArg,
    #[arg(long, default_value_t = 512)]
    pub max_len: usize,
    #[arg(long, default_value_t = 192)]
    pub head_max_len: usize,
    #[arg(long, default_value_t = 2)]
    pub head_layers: usize,

    #[arg(long, default_value_t = 1)]
    pub epochs: usize,
    /// Stop after this many optimizer steps (overrides `epochs` for the schedule).
    #[arg(long)]
    pub max_steps: Option<usize>,
    /// Question rows per micro-batch.
    #[arg(long, default_value_t = 16)]
    pub batch_size: usize,
    #[arg(long, default_value_t = 1)]
    pub grad_accum: usize,
    /// Peak learning rate for the encoder (laya fine-tune: 2.5e-5).
    #[arg(long, default_value_t = 2.5e-5)]
    pub lr_encoder: f64,
    /// Peak learning rate for the decision head (laya fine-tune: 1e-4).
    #[arg(long, default_value_t = 1e-4)]
    pub lr_head: f64,
    /// Final learning rate as a fraction of the peak (laya: cosine to 1e-6).
    #[arg(long, default_value_t = 0.04)]
    pub min_lr_ratio: f64,
    /// Warmup as a fraction of total steps.
    #[arg(long, default_value_t = 0.06)]
    pub warmup: f64,
    #[arg(long, default_value_t = 0.01)]
    pub weight_decay: f64,
    /// Clip gradients to this global L2 norm (0 = off).
    #[arg(long, default_value_t = 1.0)]
    pub clip: f64,

    #[command(flatten)]
    #[serde(default)]
    pub loss: LossConfig,
    #[command(flatten)]
    #[serde(default)]
    pub augment: AugmentConfig,

    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    #[arg(long, default_value_t = 10)]
    pub log_every: usize,
    /// Evaluate every N optimizer steps (0 = only at the end).
    #[arg(long, default_value_t = 0)]
    pub eval_every: usize,
    /// Save a checkpoint every N optimizer steps (0 = only at the end).
    #[arg(long, default_value_t = 0)]
    pub save_every: usize,
    /// Step checkpoints to keep (the final one is always kept).
    #[arg(long, default_value_t = 2)]
    pub keep: usize,
    /// Cap on held-out examples used (0 = all).
    #[arg(long, default_value_t = 0)]
    pub eval_max: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum LayoutArg {
    Prefix,
    Laya,
}

impl From<LayoutArg> for Layout {
    fn from(l: LayoutArg) -> Self {
        match l {
            LayoutArg::Prefix => Layout::Prefix,
            LayoutArg::Laya => Layout::Laya,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TrainerState {
    step: usize,
    epoch: usize,
    /// Next micro-batch index within the epoch.
    batch: usize,
    config: TrainConfig,
}

/// Where the encoder config and tokenizer come from.
struct Sources {
    encoder_json: String,
    tokenizer_dir: PathBuf,
}

fn sources(cfg: &TrainConfig) -> Result<Sources> {
    let read =
        |p: &Path| std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()));
    let (encoder_json, default_tok) = if let Some(p) = &cfg.encoder_config {
        (read(p)?, None)
    } else if let Some(d) = cfg.resume.as_ref().or(cfg.init.as_ref()) {
        (
            read(&d.join("encoder/config.json"))?,
            Some(d.join("tokenizer")),
        )
    } else if let Some(d) = &cfg.init_encoder {
        (read(&d.join("config.json"))?, Some(d.clone()))
    } else {
        bail!("give --init, --init-encoder, --resume or --encoder-config");
    };
    let tokenizer_dir = cfg
        .tokenizer
        .clone()
        .or(default_tok)
        .context("give --tokenizer (a directory with tokenizer.json)")?;
    Ok(Sources {
        encoder_json,
        tokenizer_dir,
    })
}

/// Copies every tensor of `file` that `varmap` has (under `rename(name)`) into the varmap.
fn load_into(
    varmap: &VarMap,
    file: &Path,
    rename: impl Fn(&str) -> Vec<String>,
) -> Result<(usize, Vec<String>)> {
    // SAFETY: read-only mapping of a file we don't modify.
    let st = unsafe { candle_core::safetensors::MmapedSafetensors::new(file)? };
    let names: std::collections::HashSet<String> =
        st.tensors().into_iter().map(|(n, _)| n).collect();
    let data = varmap.data().lock().expect("varmap lock");
    let mut loaded = 0;
    let mut missing = Vec::new();
    for (name, var) in data.iter() {
        let Some(src) = rename(name).into_iter().find(|n| names.contains(n)) else {
            missing.push(name.clone());
            continue;
        };
        let t = st.load(&src, var.device())?.to_dtype(var.dtype())?;
        ensure!(
            t.shape() == var.shape(),
            "{src}: shape {:?} in checkpoint, {:?} in model",
            t.shape(),
            var.shape()
        );
        var.set(&t)?;
        loaded += 1;
    }
    missing.sort();
    Ok((loaded, missing))
}

pub struct Trainer {
    pub cfg: TrainConfig,
    pub agent: Laya,
    varmap: VarMap,
    opt: AdamW,
    encoder_json: String,
    tokenizer_dir: PathBuf,
    epoch: usize,
    batch: usize,
}

impl Trainer {
    pub fn new(mut cfg: TrainConfig, dev: &Device) -> Result<Self> {
        let mut resume_state = None;
        if let Some(r) = &cfg.resume {
            let st: TrainerState =
                serde_json::from_str(&std::fs::read_to_string(r.join("trainer_state.json"))?)?;
            // Keep the run's settings; only the new invocation's paths and step cap apply.
            let (resume, out, max_steps) = (cfg.resume.clone(), cfg.out.clone(), cfg.max_steps);
            cfg = st.config.clone();
            cfg.resume = resume;
            cfg.out = out;
            cfg.max_steps = max_steps.or(cfg.max_steps);
            resume_state = Some(st);
        }
        let src = sources(&cfg)?;
        let encoder_cfg = EncoderConfig::from_json(&src.encoder_json)?;
        let (tokenizer, specials) = load_tokenizer_from(&src.tokenizer_dir)?;
        let agent_cfg: AgentConfig = serde_json::from_value(json!({
            "layout": Layout::from(cfg.layout),
            "head_layers": cfg.head_layers,
            "act_costs": {"act": 0.0},
            "max_len": cfg.max_len,
            "head_max_len": cfg.head_max_len,
        }))?;
        ensure!(
            cfg.max_len <= encoder_cfg.max_position_embeddings,
            "max_len {} exceeds the encoder's {} positions",
            cfg.max_len,
            encoder_cfg.max_position_embeddings
        );

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, dev);
        let model = build_model(vb, &agent_cfg, &encoder_cfg)?;

        if let Some(d) = &cfg.init_encoder {
            let (n, missing) = load_into(&varmap, &d.join("model.safetensors"), |name| {
                name.strip_prefix("encoder.")
                    .map(|r| vec![format!("model.{r}"), r.to_string()])
                    .unwrap_or_default()
            })?;
            let enc_missing: Vec<_> = missing
                .iter()
                .filter(|m| m.starts_with("encoder."))
                .collect();
            ensure!(
                enc_missing.is_empty(),
                "encoder tensors missing from --init-encoder: {enc_missing:?}"
            );
            eprintln!("init-encoder: loaded {n} tensors from {}", d.display());
        }
        if let Some(d) = cfg.resume.as_ref().or(cfg.init.as_ref()) {
            let (n, missing) = load_into(&varmap, &d.join("model.safetensors"), |name| {
                vec![name.to_string()]
            })?;
            if cfg.resume.is_some() {
                ensure!(missing.is_empty(), "checkpoint is missing {missing:?}");
            }
            eprintln!(
                "loaded {n} tensors from {} ({} left at init)",
                d.display(),
                missing.len()
            );
        }

        let mut params = Vec::new();
        {
            let data = varmap.data().lock().expect("varmap lock");
            let mut names: Vec<_> = data.keys().cloned().collect();
            names.sort();
            for name in names {
                if name == "temperature" {
                    continue;
                }
                let var = data[&name].clone();
                params.push(Param {
                    group: if name.starts_with("encoder.") { 0 } else { 1 },
                    decay: var.rank() >= 2,
                    name,
                    var,
                });
            }
        }
        let n_params: usize = params.iter().map(|p| p.var.elem_count()).sum();
        eprintln!(
            "{} tensors, {:.2}M parameters, layout {:?}",
            params.len(),
            n_params as f64 / 1e6,
            cfg.layout
        );
        let mut opt = AdamW::new(
            params,
            vec![
                Group {
                    lr: cfg.lr_encoder,
                    weight_decay: cfg.weight_decay,
                },
                Group {
                    lr: cfg.lr_head,
                    weight_decay: cfg.weight_decay,
                },
            ],
        )?;
        let (mut epoch, mut batch) = (0, 0);
        if let (Some(r), Some(st)) = (&cfg.resume, &resume_state) {
            opt.load(&r.join("optimizer.safetensors"))?;
            ensure!(
                opt.step == st.step,
                "optimizer step {} != trainer step {}",
                opt.step,
                st.step
            );
            (epoch, batch) = (st.epoch, st.batch);
            eprintln!(
                "resuming at step {} (epoch {epoch}, batch {batch})",
                st.step
            );
        }
        let agent = Laya::from_parts(model, agent_cfg, encoder_cfg, tokenizer, specials, dev);
        Ok(Self {
            cfg,
            agent,
            varmap,
            opt,
            encoder_json: src.encoder_json,
            tokenizer_dir: src.tokenizer_dir,
            epoch,
            batch,
        })
    }

    fn encode(&self, ex: &Example) -> Result<Encoded> {
        let mut rows = self
            .agent
            .encode(&ex.state, std::slice::from_ref(&ex.question))?;
        Ok(rows.remove(0))
    }

    /// Logits for a batch of encoded rows, `[n, kmax]`.
    fn logits(&self, rows: &[Encoded], train: bool) -> Result<Tensor> {
        let b = self
            .agent
            .model
            .batch(rows, self.agent.specials.pad, self.agent.device())?;
        Ok(self.agent.model.forward(&b, train)?.logits)
    }

    /// Epoch order: shuffled, then sorted by length inside windows of 50 batches so batches
    /// carry little padding.
    fn epoch_order(&self, lens: &[usize], epoch: usize) -> Vec<usize> {
        let mut rng = Rng(self.cfg.seed ^ (epoch as u64).wrapping_mul(0xA24B_AED4_963E_E407));
        let mut idx: Vec<usize> = (0..lens.len()).collect();
        rng.shuffle(&mut idx);
        let window = self.cfg.batch_size * 50;
        for chunk in idx.chunks_mut(window) {
            chunk.sort_by_key(|&i| lens[i]);
        }
        let mut batches: Vec<Vec<usize>> = idx
            .chunks(self.cfg.batch_size)
            .map(|c| c.to_vec())
            .collect();
        rng.shuffle(&mut batches);
        batches.concat()
    }

    pub fn run(&mut self) -> Result<Value> {
        let dev = self.agent.device().clone();
        let train = load_jsonl(&self.cfg.train)?;
        ensure!(
            !train.is_empty(),
            "no training examples in {}",
            self.cfg.train.display()
        );
        let (calib, report) = match &self.cfg.eval {
            Some(p) => split_eval(load_jsonl(p)?, self.cfg.eval_max, self.cfg.seed),
            None => (vec![], vec![]),
        };
        // Approximate lengths for bucketing: state size in characters.
        let lens: Vec<usize> = train
            .iter()
            .map(|e| crate::sequence::serialize_state(&e.state).len())
            .collect();
        let bs = self.cfg.batch_size;
        let per_epoch = train.len().div_ceil(bs).div_ceil(self.cfg.grad_accum);
        let total = self
            .cfg
            .max_steps
            .unwrap_or(per_epoch * self.cfg.epochs)
            .max(1);
        let sched = Schedule {
            warmup: ((total as f64 * self.cfg.warmup).round() as usize).max(1),
            total,
            min_ratio: self.cfg.min_lr_ratio,
        };
        eprintln!(
            "{} train examples, {} calibration, {} report; {per_epoch} steps/epoch, {total} steps",
            train.len(),
            calib.len(),
            report.len()
        );
        std::fs::create_dir_all(&self.cfg.out)?;
        let mut log = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.cfg.out.join("metrics.jsonl"))?;

        let mut order = self.epoch_order(&lens, self.epoch);
        let n_batches = order.len().div_ceil(bs);
        let (mut loss_acc, mut rows_acc, mut t0) = (0f64, 0usize, Instant::now());
        let n_params = self.opt.params.len();
        while self.opt.step < total {
            let mut grads: Vec<Option<Tensor>> = vec![None; n_params];
            let mut micro_loss = 0f64;
            for _ in 0..self.cfg.grad_accum {
                if self.batch >= n_batches {
                    self.epoch += 1;
                    self.batch = 0;
                    order = self.epoch_order(&lens, self.epoch);
                }
                let ids = &order[self.batch * bs..((self.batch + 1) * bs).min(order.len())];
                let mut rows = Vec::with_capacity(ids.len());
                let mut tgts = Vec::with_capacity(ids.len());
                for &i in ids {
                    let mut rng = Rng(self.cfg.seed
                        ^ ((self.epoch as u64) << 40)
                        ^ (i as u64).wrapping_mul(0x9E37_79B9));
                    let ex = augment(&train[i], &self.cfg.augment, &mut rng);
                    let enc = self.encode(&ex)?;
                    tgts.push((ex.target.clone(), ex.question.t == QType::Score));
                    rows.push(enc);
                }
                self.batch += 1;
                let logits = self.logits(&rows, true)?;
                let targets = Targets::new(&tgts, logits.dim(1)?, &dev)?;
                let sigma = self.cfg.loss.sigma(self.opt.step as f64 / total as f64);
                let loss = proper_scoring_loss(&logits, &targets, &self.cfg.loss, sigma)?;
                let loss = (loss / self.cfg.grad_accum as f64)?;
                micro_loss += loss.to_scalar::<f32>()? as f64;
                let gs = loss.backward()?;
                for (g, p) in grads.iter_mut().zip(&self.opt.params) {
                    if let Some(new) = gs.get(p.var.as_tensor()) {
                        *g = Some(match g.take() {
                            None => new.clone(),
                            Some(old) => (old + new)?,
                        });
                    }
                }
                rows_acc += rows.len();
            }
            ensure!(
                micro_loss.is_finite(),
                "loss is {micro_loss} at step {}",
                self.opt.step
            );
            let norm = clip_grad_norm(&mut grads, self.cfg.clip)?;
            let scale = sched.scale(self.opt.step);
            self.opt.step(&grads, scale)?;
            loss_acc += micro_loss;
            let step = self.opt.step;
            if step.is_multiple_of(self.cfg.log_every) || step == total {
                let n = self.cfg.log_every.min(step) as f64;
                let rec = json!({
                    "step": step, "epoch": self.epoch, "loss": loss_acc / n, "grad_norm": norm,
                    "lr_encoder": self.cfg.lr_encoder * scale, "lr_head": self.cfg.lr_head * scale,
                    "rows_per_s": rows_acc as f64 / t0.elapsed().as_secs_f64(),
                });
                eprintln!("{rec}");
                writeln!(log, "{rec}")?;
                (loss_acc, rows_acc, t0) = (0.0, 0, Instant::now());
            }
            if self.cfg.eval_every > 0
                && step.is_multiple_of(self.cfg.eval_every)
                && step < total
                && !report.is_empty()
            {
                let m = self.evaluate(&report, None)?;
                let rec = json!({"step": step, "eval": m});
                eprintln!("{rec}");
                writeln!(log, "{rec}")?;
            }
            if self.cfg.save_every > 0 && step.is_multiple_of(self.cfg.save_every) && step < total {
                self.save(&self.cfg.out.join(format!("step-{step}")), None)?;
                self.prune()?;
            }
        }

        let mut summary = json!({"step": self.opt.step});
        let mut fitted = None;
        if !report.is_empty() {
            summary["eval_raw"] = self.evaluate(&report, None)?;
            if !calib.is_empty() {
                let temps = self.fit_temperatures(&calib)?;
                summary["temperatures"] = json!(temps);
                summary["eval_calibrated"] = self.evaluate(&report, Some(&temps))?;
                fitted = Some(temps);
            }
        }
        self.save(&self.cfg.out.join("final"), fitted.as_ref())?;
        eprintln!("{}", serde_json::to_string_pretty(&summary)?);
        writeln!(log, "{}", json!({"final": summary}))?;
        Ok(summary)
    }

    /// Raw logits for held-out examples (no augmentation), in batches.
    fn predict(&self, exs: &[Example]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(exs.len());
        for chunk in exs.chunks(self.cfg.batch_size.max(8)) {
            let rows = chunk
                .iter()
                .map(|e| self.encode(e))
                .collect::<Result<Vec<_>>>()?;
            let l = self.logits(&rows, false)?.detach().to_vec2::<f32>()?;
            out.extend(
                l.into_iter()
                    .zip(&rows)
                    .map(|(l, r)| l[..r.markers.len()].to_vec()),
            );
        }
        Ok(out)
    }

    pub fn evaluate(&self, exs: &[Example], temps: Option<&Temperatures>) -> Result<Value> {
        let logits = self.predict(exs)?;
        Ok(metrics(exs, &logits, temps))
    }

    /// Per `(qtype, k-bucket)` temperatures minimising NLL on the calibration split (and a
    /// per-qtype fallback), clamped to laya's [0.5, 5].
    fn fit_temperatures(&self, exs: &[Example]) -> Result<Temperatures> {
        let logits = self.predict(exs)?;
        let mut by_key: std::collections::BTreeMap<String, Vec<usize>> = Default::default();
        let mut by_type: [Vec<usize>; 3] = Default::default();
        for (i, e) in exs.iter().enumerate() {
            by_key
                .entry(temperature_key(e.question.t, e.target.len()))
                .or_default()
                .push(i);
            by_type[e.question.t as usize].push(i);
        }
        let fit = |idx: &[usize]| {
            fit_temperature(idx.iter().map(|&i| (&logits[i][..], &exs[i].target[..])))
        };
        Ok(Temperatures {
            by_options: by_key
                .iter()
                .map(|(k, idx)| (k.clone(), fit(idx)))
                .collect(),
            by_type: by_type
                .iter()
                .map(|idx| if idx.is_empty() { 1.0 } else { fit(idx) })
                .collect(),
        })
    }

    /// Writes a loadable model directory plus optimizer and trainer state.
    pub fn save(&self, dir: &Path, temps: Option<&Temperatures>) -> Result<()> {
        let tmp = dir.with_extension("partial");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("encoder"))?;
        std::fs::create_dir_all(tmp.join("tokenizer"))?;
        self.varmap.save(tmp.join("model.safetensors"))?;
        self.opt.save(&tmp.join("optimizer.safetensors"))?;
        std::fs::write(tmp.join("encoder/config.json"), &self.encoder_json)?;
        for f in [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ] {
            let src = self.tokenizer_dir.join(f);
            if src.exists() {
                std::fs::copy(&src, tmp.join("tokenizer").join(f))?;
            }
        }
        let mut agent = serde_json::to_value(&self.agent.cfg)?;
        if let Some(t) = temps {
            agent["temperature"] = json!(t.by_type);
            agent["temperature_by_options"] = json!(t.by_options);
        }
        agent["training"] = json!({
            "updates": self.opt.step, "epochs_started": self.epoch + 1,
            "loss": self.cfg.loss, "layout": self.cfg.layout,
            "note": "trained by candle-rlcd with the direct proper-scoring loss",
        });
        std::fs::write(
            tmp.join("rl_agent_config.json"),
            serde_json::to_string_pretty(&agent)?,
        )?;
        let st = TrainerState {
            step: self.opt.step,
            epoch: self.epoch,
            batch: self.batch,
            config: self.cfg.clone(),
        };
        std::fs::write(
            tmp.join("trainer_state.json"),
            serde_json::to_string_pretty(&st)?,
        )?;
        let _ = std::fs::remove_dir_all(dir);
        std::fs::rename(&tmp, dir)?;
        eprintln!("saved {}", dir.display());
        Ok(())
    }

    fn prune(&self) -> Result<()> {
        let mut steps: Vec<(usize, PathBuf)> = std::fs::read_dir(&self.cfg.out)?
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let name = e.file_name().into_string().ok()?;
                Some((name.strip_prefix("step-")?.parse().ok()?, e.path()))
            })
            .collect();
        steps.sort();
        while steps.len() > self.cfg.keep {
            std::fs::remove_dir_all(steps.remove(0).1)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Temperatures {
    pub by_options: std::collections::BTreeMap<String, f64>,
    pub by_type: Vec<f64>,
}

impl Temperatures {
    fn get(&self, t: QType, k: usize) -> f64 {
        let v = self.by_options.get(&temperature_key(t, k)).copied();
        crate::config::clamp_temperature(v.or(self.by_type.get(t as usize).copied()).unwrap_or(1.0))
    }
}

/// Deterministic half/half split of held-out examples into calibration and report sets.
fn split_eval(mut exs: Vec<Example>, max: usize, seed: u64) -> (Vec<Example>, Vec<Example>) {
    Rng(seed ^ 0x5EED).shuffle(&mut exs);
    if max > 0 {
        exs.truncate(max);
    }
    let report = exs.split_off(exs.len() / 2);
    (exs, report)
}

fn nll(logits: &[f32], t: &[f32], temp: f64) -> f64 {
    let p = softmax(logits, temp as f32);
    -t.iter()
        .zip(&p)
        .map(|(t, p)| *t as f64 * (*p as f64).max(1e-12).ln())
        .sum::<f64>()
}

/// Golden-section search for the NLL-minimising temperature on a log scale in [0.5, 5].
fn fit_temperature<'a>(items: impl Iterator<Item = (&'a [f32], &'a [f32])> + Clone) -> f64 {
    let f = |lt: f64| items.clone().map(|(l, t)| nll(l, t, lt.exp())).sum::<f64>();
    let (mut a, mut b) = (0.5f64.ln(), 5f64.ln());
    let g = (5f64.sqrt() - 1.0) / 2.0;
    for _ in 0..60 {
        let (c, d) = (b - g * (b - a), a + g * (b - a));
        if f(c) < f(d) {
            b = d;
        } else {
            a = c;
        }
    }
    ((a + b) / 2.0).exp()
}

/// Accuracy, NLL, Brier, 15-bin ECE of max p, and RPS on `score` rows; overall and per type.
pub fn metrics(exs: &[Example], logits: &[Vec<f32>], temps: Option<&Temperatures>) -> Value {
    #[derive(Default)]
    struct Acc {
        n: usize,
        correct: f64,
        nll: f64,
        brier: f64,
        rps: f64,
        n_rps: usize,
        bins: [(f64, f64, usize); 15],
    }
    impl Acc {
        fn add(&mut self, p: &[f32], t: &[f32], score: bool) {
            let arg = |v: &[f32]| {
                v.iter()
                    .enumerate()
                    .fold(0, |b, (i, &x)| if x > v[b] { i } else { b })
            };
            let (ap, at) = (arg(p), arg(t));
            let hit = (ap == at) as u8 as f64;
            self.n += 1;
            self.correct += hit;
            self.nll -= t
                .iter()
                .zip(p)
                .map(|(t, p)| *t as f64 * (*p as f64).max(1e-12).ln())
                .sum::<f64>();
            self.brier += t
                .iter()
                .zip(p)
                .map(|(t, p)| (*p as f64 - *t as f64).powi(2))
                .sum::<f64>();
            if score {
                let (mut cp, mut ct, mut s) = (0f64, 0f64, 0f64);
                for (p, t) in p.iter().zip(t) {
                    cp += *p as f64;
                    ct += *t as f64;
                    s += (cp - ct).powi(2);
                }
                self.rps += s / (p.len().max(2) - 1) as f64;
                self.n_rps += 1;
            }
            let conf = p[ap] as f64;
            let b = ((conf * 15.0) as usize).min(14);
            self.bins[b].0 += conf;
            self.bins[b].1 += hit;
            self.bins[b].2 += 1;
        }
        fn json(&self) -> Value {
            let n = self.n.max(1) as f64;
            let ece: f64 = self
                .bins
                .iter()
                .map(|(c, h, k)| (c - h).abs() * (*k as f64 > 0.0) as u8 as f64)
                .sum::<f64>()
                / n;
            let mut v = json!({
                "n": self.n, "accuracy": self.correct / n, "nll": self.nll / n,
                "brier": self.brier / n, "ece": ece,
            });
            if self.n_rps > 0 {
                v["rps"] = json!(self.rps / self.n_rps as f64);
            }
            v
        }
    }
    let mut all = Acc::default();
    let mut per: [Acc; 3] = Default::default();
    for (e, l) in exs.iter().zip(logits) {
        let temp = temps.map(|t| t.get(e.question.t, l.len())).unwrap_or(1.0);
        let p = softmax(l, temp as f32);
        let score = e.question.t == QType::Score;
        all.add(&p, &e.target, score);
        per[e.question.t as usize].add(&p, &e.target, score);
    }
    let mut v = all.json();
    for (t, a) in [QType::Choice, QType::Score, QType::Noul].iter().zip(&per) {
        if a.n > 0 {
            v[t.name()] = a.json();
        }
    }
    v
}

/// Evaluate a saved model directory on labelled records (uses its fitted temperatures).
pub fn evaluate_dir(model: &Path, data: &Path, dev: &Device) -> Result<Value> {
    let agent = Laya::load(model, dev, DType::F32)?;
    let exs = load_jsonl(data)?;
    let mut logits = Vec::with_capacity(exs.len());
    for e in &exs {
        let rows = agent.encode(&e.state, std::slice::from_ref(&e.question))?;
        logits.push(agent.forward(&rows)?.remove(0).logits);
    }
    let temps = Temperatures {
        by_options: agent.cfg.temperature_by_options.clone(),
        by_type: agent.cfg.temperature.clone(),
    };
    Ok(json!({
        "raw": metrics(&exs, &logits, None),
        "calibrated": metrics(&exs, &logits, Some(&temps)),
    }))
}
