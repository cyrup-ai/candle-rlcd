# candle-rlcd

A Rust/[Candle](https://github.com/huggingface/candle) stack for ModernBERT + RLCD typed-decision
models. It loads [laya](https://huggingface.co/convaiinnovations/laya) checkpoints as-is and
answers `choice` / `score` / `noul` questions about a state in one encoder forward pass, with
no generated tokens.

laya is the reference, not the target. Training uses a direct proper-scoring loss instead of
laya's REINFORCE term, and a new *prefix* layout encodes the state once per request instead of
once per question. Serving comes next.

## Use

```sh
cargo run --release -- run --model /path/to/laya --request request.json          # CPU, f32
cargo run --release --features cuda -- run --model /path/to/laya --dtype bf16 < request.json
```

```json
{"state": "Customer says the package arrived damaged and wants a refund.",
 "questions": {
   "team":    {"t": "choice", "ins": "Which team handles this?", "crit": {"billing": "payments", "shipping": "delivery"}},
   "urgency": {"t": "score",  "ins": "How urgent?", "crit": ["low", "medium", "high"]},
   "refund":  {"t": "noul",   "ins": "Does the customer want a refund?"}}}
```

The output matches laya's `system_one` answers: `choice` / `score` / `noul`, calibrated
`probabilities`, `answer_confidence` (max p, the calibrated number), entropy `confidence`, and
the act head's `act_probability`.

A checkpoint directory holds `rl_agent_config.json`, `model.safetensors`, `encoder/config.json`
and `tokenizer/` (the Hub layout; `huggingface-cli download convaiinnovations/laya` fetches it).

## Layout

| file | what |
|---|---|
| `src/modernbert.rs` | ModernBERT encoder, vendored from candle-transformers and fixed (below) |
| `src/head.rs` | laya's decision head: type embedding, 2 pre-norm transformer layers, marker scorer, act head |
| `src/sequence.rs` | question schema and laya's `[CLS] head [SEP] [MASK] opt ... [SEP] state [SEP]` builder |
| `src/agent.rs` | checkpoint loading, batched forward, temperature calibration, answer decoding |
| `src/config.rs` | encoder config (transformers 4.x and 5.x layouts) and `rl_agent_config.json` |
| `src/model.rs` | encoder + head in laya's joint layout or the encode-once prefix layout |
| `src/autograd.rs` | backward passes for Candle's forward-only fused RoPE and softmax |
| `src/loss.rs` | direct proper-scoring loss (log + spherical + RPS, optional logit noise) |
| `src/optim.rs` | AdamW with saveable state and param groups, clip-by-global-norm, warmup + cosine |
| `src/data.rs` | training records, targets, augmentation, AG News / BoolQ / synthetic data, BPE tokenizer training |
| `src/train.rs` | training loop, eval metrics, temperature fitting, checkpoints and resume |

### Fixes over upstream `candle-transformers` modernbert.rs

- Bias-free norms load (upstream's `layer_norm_no_bias` requires a bias tensor, so `mlp_norm`
  failed and `attn_norm` only loaded through a `.ok()` that also hid real errors).
- F16/BF16 work: masks, softmax and norms run in F32; RoPE tables are computed from integer
  positions in F32 and cast once (BF16 positions were inexact above 256).
- Padded rows stay finite in local layers (upstream's mask add could overflow to `-inf` and
  NaN a fully padded window).
- RoPE thetas come from `rope_parameters` when present (mmBERT uses 160000 for both; transformers
  4.x silently used 10000 for local), and `layer_types` decides which layers are local.
- Weights load from any prefix (laya stores the encoder under `encoder.`), with optional
  attention/MLP/norm biases.

BF16 needs CUDA or Metal: Candle's CPU backend has no BF16 matmul, so on CPU use f32 or f16.

## Parity

`tests/golden.rs` compares against outputs produced by laya's own Python (`DecisionModel`,
`build_sequence`) on a tiny random-weight checkpoint in `tests/fixtures/tiny`, regenerated with
`scripts/make_tiny_fixture.py`:

| dtype | token ids | max logit diff |
|---|---|---|
| f32 (CPU) | identical | 7e-7 |
| f16 (CPU) | identical | 2e-3 |

The fixture covers structured criteria, custom `noul` labels, head-budget overflow, left-truncated
conversations, right-truncated long states, a distinct local RoPE theta and a 16-token sliding
window. Changing the window by one token or swapping the RoPE theta fails the test.

For the published weights, `scripts/golden_from_laya.py` writes the same golden file from
`convaiinnovations/laya`, and `LAYA_DIR=... LAYA_GOLDEN=... cargo test --release real_laya`
checks it.

## Inference notes

- The state is tokenized once per request and every question row goes through one batched
  forward pass; padding is fully masked, so a row's output doesn't depend on its batch.
- In laya's layout each question attends to the state and the state to the question, so the
  state is *encoded* once per question. The prefix layout below removes that.

## Training

```sh
# Data: AG News (choice) from its original CSV release, BoolQ (noul), or synthetic tickets.
candle-rlcd data ag-news --input ag_news_csv/train.csv --output train.jsonl
candle-rlcd data ag-news --input ag_news_csv/test.csv  --output test.jsonl

# Fine-tune from ModernBERT (Hugging Face directory: config.json, model.safetensors, tokenizer.json)
candle-rlcd train --train train.jsonl --eval test.jsonl --out runs/ag \
    --init-encoder ModernBERT-base --epochs 1 --batch-size 16 --grad-accum 4

# ...or from a laya snapshot (every matching tensor is loaded, laya's layout or the prefix one)
candle-rlcd train --train train.jsonl --eval test.jsonl --out runs/ag --init laya

candle-rlcd eval --model runs/ag/final --data test.jsonl
candle-rlcd run  --model runs/ag/final --request request.json
```

Records are JSONL, one state with any number of questions:
`{"state": .., "questions": {id: {"t", "ins", "crit"}}, "targets": {id: target}}`, where a
target is an option key, an index, a boolean or `p(true)` for `noul`, a list of per-option
probabilities, or a `{key: prob}` map (teacher soft labels).

**Default data: AG News.** laya's base data mix isn't published. AG News is small (120k/7.6k),
openly downloadable without Hugging Face (the original CSV release, mirrored on GitHub), and is in
laya's own training mix, so it is comparable to laya's reported 0.950. `data boolq` adds `noul`
questions from BoolQ's JSONL release; `data synthetic` makes multi-question support tickets
(`choice` + `noul` + `score`) for smoke tests without any download.

**Loss.** laya samples noisy logits and estimates the gradient of a proper-scoring reward with
REINFORCE. That reward is a closed-form, differentiable function of the probabilities, so we
backpropagate through it directly: `-[w_log · Σ t log q + w_sph · (t·q)/‖q‖ − w_rps · RPS]`,
with laya's weights (1, 0.75, 1; RPS only on `score`). `--sigma-start/--sigma-end` add laya's
logit noise with reparameterised gradients (off by default). With `--w-sph 0 --w-rps 0` it is
plain soft cross-entropy.

**Augmentation.** Each epoch, `choice` options are shuffled and 30% of questions get opaque
labels (`A`/`option 2`/`x17`, criterion kept), so the model reads options instead of memorising
label words (`--shuffle-options`, `--opaque-labels`).

**Optimiser and schedule.** AdamW with separate encoder/head learning rates (laya: 2.5e-5 /
1e-4), decoupled weight decay on matrices only, warmup then cosine to `--min-lr-ratio`, and
clip-by-global-norm (`--clip 1.0`). Optimiser moments are F32 and saved with the checkpoint.

**Checkpoints.** `--save-every N` writes `out/step-N/` (keeping `--keep`), and the run ends with
`out/final/`. Each is a model directory `run`/`eval`/`Laya::load` open directly, plus
`optimizer.safetensors` and `trainer_state.json`; `--resume out/step-N` continues with the same
data order, schedule and optimiser state.

**Calibration.** Held-out records are split in half: one half fits a temperature per
`(qtype, option-count bucket)` by NLL (clamped to laya's [0.5, 5]) and the other half reports
accuracy, NLL, Brier, ECE and RPS both raw and calibrated. The temperatures are written into
the final `rl_agent_config.json`. Metrics go to `out/metrics.jsonl`.

### Encode-once: the prefix layout

`--layout prefix` (the default for training) puts the state first and lets state tokens attend
only to the state:

```text
[CLS] state [SEP] | [CLS] <t> question: <ins> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP]
   ^ sees state only    ^ sees state + question
```

The state's hidden states and per-layer keys/values then don't depend on the question, so
inference encodes the state once and runs each question as a short suffix against the cache.
It has exactly laya's parameters, so a laya snapshot is a valid starting point (`--init laya`);
training runs the ordinary masked forward, and `tests/training.rs` checks it equals the cached
path. `cargo run --release --example encode_once` times both layouts on a ModernBERT-base-shaped
model (random weights, 512 tokens, 4-core CPU container, f32):

| questions per request | laya layout | prefix layout | speedup |
|---|---|---|---|
| 1 | 1597 ms | 1118 ms | 1.4x |
| 2 | 2960 ms | 1549 ms | 1.9x |
| 4 | 6141 ms | 1852 ms | 3.3x |
| 8 | 12445 ms | 3186 ms | 3.9x |

### Candle fixes for training

- **RoPE had no backward** (`apply_op3_no_bwd`), so Q and K never trained, only V.
  `autograd::rope` keeps the fused kernel and adds the backward (the inverse rotation);
  `autograd::softmax_last_dim` does the same for the fused softmax. Both are no-ops for
  inference. The fused layer norm switches to Candle's differentiable composite when training.
- Fresh weights get ModernBERT's init (norms 1/0, embeddings and linears N(0, σ²) with σ the
  config's `initializer_range`, default 0.02).
- `tests/training.rs` checks every parameter on the scoring path gets a gradient (including
  the Q/K rows of every `Wqkv`) and that gradients match finite differences.

Not done yet: mixed precision/loss scaling, gradient checkpointing and multi-GPU all-reduce. CPU
training works but is slow (Candle's CPU ops are mostly single-threaded at these sizes); use
`--features cuda` for real runs.

Start from pretrained weights. Small models trained from scratch learn `score` questions quickly,
but they stay at chance on option matching like AG News for hundreds of steps. Every option marker
is the same `[MASK]` token, and a fresh model's attention is near uniform, so the markers start
out indistinguishable. A pretrained encoder has already mixed each option's text into its marker.
