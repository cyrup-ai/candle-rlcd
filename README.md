# candle-rlcd

A Rust/[Candle](https://github.com/huggingface/candle) stack for ModernBERT + RLCD typed-decision
models. It loads [laya](https://huggingface.co/convaiinnovations/laya) checkpoints as-is and
answers `choice` / `score` / `noul` questions about a state in one encoder forward pass, with
no generated tokens.

laya is the reference, not the target: training (a direct proper-scoring loss instead of laya's
REINFORCE term) and serving come next.

## Use

```sh
cargo run --release -- --model /path/to/laya --request request.json          # CPU, f32
cargo run --release --features cuda -- --model /path/to/laya --dtype bf16 < request.json
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
checks it. Against `convaiinnovations/laya` (ModernBERT-large, snapshot `55cf4c4`) the token ids
are identical and the worst f32 logit diff is 1.3e-5.

On AG News (a fixed 1,000-row sample of the test set), `candle-rlcd eval` on the published
weights gives accuracy 0.927, Brier 0.115 and ECE 0.036 (raw logits; laya's fitted
temperatures give ECE 0.051 here). It runs at about 0.7 s per question on a 4-core CPU in f32.

## Inference notes

- The state is tokenized once per request and every question row goes through one batched
  forward pass; padding is fully masked, so a row's output doesn't depend on its batch.
- laya's architecture has each question attend to the state, so the state is still *encoded*
  once per question. Encoding it once (shared lower layers, or all questions packed into one
  sequence) and scoring options separately need a model trained that way; they are part of the
  training work, not something the published weights support.
