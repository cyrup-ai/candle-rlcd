# candle-rlcd

A Rust/[Candle](https://github.com/huggingface/candle) stack for ModernBERT + RLCD typed-decision
models. It loads [laya](https://huggingface.co/convaiinnovations/laya) checkpoints as-is and
answers `choice` / `score` / `noul` questions about a state in one encoder forward pass, with
no generated tokens.

laya is the reference, not the target. Training uses a direct proper-scoring loss instead of
laya's REINFORCE term, and a new *prefix* layout encodes the state once per request instead of
once per question. `candle-rlcd serve` exposes it over HTTP with TypeSafe's Jev API.

## Install

- **Prebuilt binaries** for Linux x86_64 and arm64 and for Apple Silicon (built with Metal and
  Accelerate) are attached to each [GitHub release](https://github.com/cyrup-ai/candle-rlcd/releases).
  Pushing a `v*` tag builds them (`.github/workflows/release.yml`).
- **Docker** (CPU): `docker run -p 8080:8080 -v candle-rlcd:/data ghcr.io/cyrup-ai/candle-rlcd`
  serves laya on port 8080. Weights download to the `/data` volume on first start.
- **From source**: `cargo build --release`, adding `--features cuda`, `metal`, `mkl` or
  `accelerate` as available.

```sh
candle-rlcd serve                    # downloads convaiinnovations/laya on first run, then serves it
```

## Use

```sh
cargo run --release -- run --request request.json                     # laya from the Hub, CPU, f32
cargo run --release -- run --model /path/to/checkpoint < request.json
cargo run --release --features cuda -- run --model /path/to/laya --dtype bf16 < request.json
```

`--model` takes a checkpoint directory or a Hub id. It defaults to `convaiinnovations/laya`.

| `--model` | what |
|---|---|
| `runs/x/final` | a local checkpoint directory |
| `convaiinnovations/laya` | a Hub repository's root checkpoint |
| `convaiinnovations/laya/multilingual` | a checkpoint in a sub-folder (laya also has `typed-decisions`) |
| `convaiinnovations/laya@55cf4c4e…` | pinned to a branch, tag or commit |

Hub checkpoints download once into the Hugging Face cache (`HF_HOME`, `HF_HUB_CACHE`) in its
usual layout, so an earlier `huggingface-cli download` is reused. Only the files a checkpoint
needs are fetched: 803 MB for laya's root, not the whole 2.4 GB repository. `HF_HUB_OFFLINE=1`
uses the cache without the network, and `HF_TOKEN` is sent for private repositories.

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
and `tokenizer/`, which is the Hub layout.

## Serve: a local Jev API

```sh
cargo build --release          # add --features cuda / metal / mkl / accelerate as available
./target/release/candle-rlcd serve --port 8080                          # laya from the Hub
./target/release/candle-rlcd serve --model /path/to/model --port 8080
```

The server speaks [TypeSafe's Jev API](https://docs.typesafe.ai/api) with the same request,
response and error types, taken from TypeSafe's published OpenAPI schema
(`https://api.typesafe.ai/openapi.json`, v0.2.0):

| endpoint | what |
|---|---|
| `POST /v1/systemone` | `{"state", "model", "questions": {id: {"type": "choice" \| "score" \| "noul", "instructions", "criteria"}}}` to `{"model", "answers": {id: answer}, "usage"}` |
| `GET /v1/models` | `{"models": [{"name", "description", "release_date"}]}`: the served model plus the `jev-latest` / `jev-preview` aliases |
| `GET /health` | not in Jev: status, engine settings and request/batch counters |
| `GET /metrics` | not in Jev: Prometheus metrics (see below) |

Answers carry exactly Jev's fields. A `choice` has `choice`, `probabilities` and
`confidence`; a `score` has `score` (the expected level), `legend`, `probabilities` and
`confidence`; a `noul` has `noul` (p(yes)). Probabilities use the model's fitted temperatures. Invalid bodies get Jev's FastAPI-style
`422 {"detail": [{"type", "loc", "msg", "input", "ctx"}]}`, a full queue gets `529` with
`retry-after`, and `--api-key` (or `CANDLE_RLCD_API_KEY`) turns on `Authorization: Bearer`
checks (off by default, since it binds to 127.0.0.1). Responses carry `x-typesafe-request-id`.

TypeSafe's own SDKs work unchanged by pointing them at the server:

```python
from typesafe_sdk import TypeSafeClient, Choice, Score, Noul   # pip install typesafe-sdk
with TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8080") as client:
    r = client.system_one(state="I was charged twice, please refund one.",
                          questions={"team": Choice(instructions="Which team?", criteria={"billing": "Payments", "technical": "Bugs"}),
                                     "refund": Noul(instructions="Does the customer want a refund?")})
    print(r.answers["team"].choice, r.answers["team"].confidence, r.answers["refund"].noul)
```

This was checked with `typesafe-sdk` 0.7.1: listing models, all three question types and a 422
error all go through the SDK.

Where it differs from Jev:
- **`model`**: the response reports the served model's name, not a Jev version. With one
  model, any name is accepted. With several (below), `model` picks one.
- **`confidence`**: Jev's docs give `(k * max p - 1) / (k - 1)` for Choice, and that is what
  is returned. The Score formula isn't published. The same formula reproduces TypeSafe's
  3-level Score examples, but not their 4- and 5-level ones, so Score confidence may differ.
- **`usage.output_tokens`** is 0. `input_tokens` counts the tokens the model read (the state
  once plus each question in the prefix layout).
- **Context**: the checkpoint's `max_len` (256 to 1024) applies, not Jev's 32k. Long states are
  truncated the way laya does it, so conversations keep their newest turns.
- **Limits**: at most 255 Choice options and 10 Score levels, per Jev's docs.
- `--extended` adds laya's `answer_confidence` (max p) and `act_probability` to each answer.
- **Request limits.** A request whose `usage.input_tokens` would exceed 65,536 gets a 422,
  like Jev's 64k cap (`--max-request-tokens`, 0 turns it off). In laya's layout the state
  counts once per question, so about 128 questions on a full 512-token state hit the cap.
  - `--timeout-secs` (default 120) answers 504 to a request that hasn't finished in time. A
    request still queued at that point is dropped without running.
  - `--max-questions N` refuses requests with more than N questions with a 422. Jev has no
    such cap.

### Several models

Repeat `--model` to serve several checkpoints from one process, and use `name=` to choose the
names requests use:

```sh
candle-rlcd serve --model laya=convaiinnovations/laya \
                  --model laya-multilingual=convaiinnovations/laya/multilingual \
                  --model support=runs/support/final
```

- A request's `model` picks the checkpoint by name.
- The first model is the default, and it answers `jev-latest` and `jev-preview`.
- An unknown name gets a 422 at `body.model` that lists the served names.
- A Hub model also answers a pinned name such as `laya@55cf4c4`, so a client can hold on to
  one exact version.
- `/v1/models` lists every name.
- Each model gets its own workers. They share one count of running passes, so the adaptive
  width sees the load across all models.

### Metrics

`GET /metrics` serves Prometheus text. It needs no API key, like `/health`. It reports:

- HTTP responses by route and status, and latency histograms by route.
- Per model: queue depth, workers, requests, forward passes, rows, questions answered and input
  tokens.
- Forward-pass time and queue-wait histograms.
- The number of passes running now.

### Calibrating confidence on your data

Confidence-gated routing means acting on an answer when `confidence` is high and escalating
it otherwise. That only works if confidence means what it says on your traffic. laya's shipped
temperatures were fit on its own data. `calibrate` refits them from your labelled requests
without touching the weights:

```sh
candle-rlcd calibrate --data labelled.jsonl --out cal.json   # labelled Jev requests, see below
candle-rlcd serve --calibration cal.json                     # or name=cal.json with several models
```

- It fits on a random half and prints accuracy, NLL, Brier and ECE on the other half, raw,
  with the current temperatures and with the new ones. Then it refits on everything.
- A `(type, option-count)` bucket gets its own temperature only with at least
  `--min-examples` (30) questions. A type with enough data gets a new fallback temperature,
  and its old bucket temperatures are dropped.
- `run`, `eval` and `bench` take `--calibration` too. `--write` stores the result in a local
  checkpoint's `rl_agent_config.json` and keeps the original next to it.

Shipped temperatures never sharpen 11 or more options. laya's root config sets `choice:11+` to
0.1, fit on few examples, and even after the [0.5, 5] clamp that pushed an unanswerable 11-option
question's max p from 0.135 (at 10 options) to 0.213. Unless `calibrate` or `train` refit the
temperatures, 11+ options now use at least 1.0, which gives 0.145.

### Concurrency

The loaded model is immutable. Every weight is an `Arc`-backed Candle tensor, and a forward
pass only reads them, so `Laya` is `Send + Sync` (checked at compile time) and one copy is
shared by all threads. Peak RSS is the same with 1 or 4 workers: 1.3 GB for ModernBERT-base
and 2.5 GB for laya-large. The server runs:

- **HTTP on a 2-thread Tokio runtime.** Handlers validate and tokenize, then queue the request.
- **N inference workers** on one bounded queue, each with its own Rayon thread pools, where
  Candle's CPU kernels run.
- **Adaptive width.** A worker splits the cores between the passes running and the requests
  waiting. A lone request gets every core, and a full queue gets one core per worker.
- **Optional dynamic batching** (`--max-batch-rows N`, `--batch-wait-ms`). A worker drains
  the queued requests into one forward pass. In the encode-once layout the requests' states
  are left-padded to a common length and encoded together. RoPE and the sliding window only
  see relative positions, so each request gets what it would alone (`tests/training.rs`,
  `tests/serve.rs`).

Defaults: on CPU, one adaptive worker per core and no batching. On GPU, one worker batching up
to 64 rows. Everything is a flag: `--workers`, `--threads-per-worker` (0 shares one pool),
`--max-batch-rows`, `--batch-wait-ms`, `--max-queue` and `--no-adaptive`.

`candle-rlcd bench` is a closed-loop load generator. It runs against the engine in process, or
against a running server with `--url`. These numbers are for a 4-core CPU container in f32,
using the ModernBERT-base prefix-layout model:

| AG News, 1 question/request | 1 client | 2 clients | 4 clients | 16 clients |
|---|---|---|---|---|
| serial: 1 worker x 4 threads | 3.3 req/s, 306 ms | - | 3.1 req/s | 3.2 req/s |
| 4 workers x 1 thread, fixed | 1.5 req/s, 686 ms | - | 5.7 req/s | 5.8 req/s |
| 1 worker x 4 threads + batching (64 rows) | 3.4 req/s, 287 ms | - | 3.9 req/s | 3.9 req/s |
| 4 workers x 1 thread + batching | 1.5 req/s | - | 5.3 req/s | 4.0 req/s |
| **default: 4 adaptive workers** | **3.4 req/s, 294 ms** | **4.7 req/s** | **5.4 req/s** | **5.6 req/s** |

| synthetic tickets, 3 questions/request | 1 client | 4 clients | 16 clients |
|---|---|---|---|
| serial | 7.1 q/s, 436 ms | 7.0 q/s | 6.8 q/s |
| 1 worker + batching | 7.1 q/s | 7.0 q/s | 7.1 q/s |
| **default** | **7.0 q/s, 440 ms** | **11.4 q/s** | **11.4 q/s** |

Over HTTP (`bench --url`), the default server gives 3.2 req/s at 311 ms with one client and
5.3 req/s with four, so the HTTP layer adds about 10 ms.

laya-large (joint layout), AG News: 1.6 req/s at 625 ms with one client, and 2.3 req/s with
four clients (serial: 1.7 req/s).

On CPU the model is compute bound. One pass on one thread runs at about 48 GFLOP/s, but four
threads on one pass only reach 2.2x because Candle's elementwise and norm ops are mostly
single-threaded. Four independent passes do reach about 4x. Batching makes the matmuls bigger,
not more efficient, and it pads short states up to the longest, so it doesn't pay on CPU. On a
GPU it's the other way round: one pass leaves most of the device idle and batching fills it.
That's why GPU builds default to one batching worker. The GPU numbers are untested here, since
this container has no GPU.

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
| `src/serve.rs` | Jev-compatible HTTP server: request validation, answers, shared-model worker engine |
| `src/bench.rs` | closed-loop load generator (in process or over HTTP) |
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
checks it. Against `convaiinnovations/laya` (ModernBERT-large, snapshot `55cf4c4`) the token ids
are identical and the worst f32 logit diff is 1.3e-5.

On AG News (a fixed 1,000-row sample of the test set), `candle-rlcd eval` on the published
weights gives accuracy 0.927, Brier 0.115 and ECE 0.036 (raw logits; laya's fitted
temperatures give ECE 0.051 here). It runs at about 0.7 s per question on a 4-core CPU in f32.

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

# ...or from a laya checkpoint, a directory or a Hub id (every matching tensor is loaded, laya's
# layout or the prefix one)
candle-rlcd train --train train.jsonl --eval test.jsonl --out runs/ag --init convaiinnovations/laya

candle-rlcd eval --model runs/ag/final --data test.jsonl
candle-rlcd run  --model runs/ag/final --request request.json
```

Records are JSONL, one state with any number of questions:
`{"state": .., "questions": {id: {"t", "ins", "crit"}}, "targets": {id: target}}`, where a
target is an option key, an index, a boolean or `p(true)` for `noul`, a list of per-option
probabilities, or a `{key: prob}` map (teacher soft labels).

**Your own decisions, in Jev's shape.** A record can also be a `/v1/systemone` request body
(`type` / `instructions` / `criteria`; `model` is ignored) plus `targets`. It can also carry
the `answers` of a response you accepted or corrected, and each Jev answer object is read as
the target:

- `probabilities` when present, as soft labels;
- otherwise the `choice`, the `noul` probability, or the `score` rounded to a level.

So a log of requests and the answers you acted on is training data as it stands, for `train`,
`eval` and `calibrate` alike. [`examples/jev-labelled.jsonl`](examples/jev-labelled.jsonl) shows
all three forms:

```json
{"state": "How do I change the email on my account?", "model": "jev-latest",
 "questions": {"team": {"type": "choice", "instructions": "Which team should handle this ticket?",
                        "criteria": {"billing": "Payments", "technical": "Bugs", "account": "Login and settings"}}},
 "answers": {"team": {"type": "choice", "choice": "account"}}}
```

Instructions and criteria are rendered exactly as the server renders them, so a model trained
on these records sees at serving time what it saw in training. To specialise laya on them,
fine-tune from it and then serve the result:

```sh
candle-rlcd train --train decisions.jsonl --eval held-out.jsonl --out runs/mine --init convaiinnovations/laya
candle-rlcd serve --model mine=runs/mine/final --model laya=convaiinnovations/laya
```

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

### AG News fine-tunes

These runs used a fixed 1,000-row sample of the AG News test set, CPU-only with 4 cores and
16 GB. Temperatures were fitted on 200 held-out train rows. Brier and ECE are for the
calibrated probabilities (laya's row is raw, which is its better ECE here):

| model | layout | rows trained | accuracy | Brier | ECE |
|---|---|---|---|---|---|
| laya as published | laya | - | 0.927 | 0.115 | 0.036 |
| laya weights, untuned | prefix | 0 | 0.781 | - | - |
| `--init laya`, 150 steps | prefix | 2,400 | **0.928** | **0.114** | **0.019** |
| `--init-encoder ModernBERT-base`, 250 steps | prefix | 4,000 | 0.900 | 0.150 | 0.019 |

The encode-once layout loses 15 points on laya's weights untuned. It gets them back after
2,400 rows, then matches laya and is better calibrated. ModernBERT-base with a fresh head
reaches 0.90 on 4,000 rows.

The runs used these settings:
- laya: `--batch-size 2 --grad-accum 8 --max-len 256` and default learning rates. It trains at
  ~0.2 rows/s and peaks around 10 GB RSS. Batch 4 runs out of memory in 16 GB.
- base: `--batch-size 8 --grad-accum 2 --max-len 256 --lr-encoder 5e-5`, at ~0.65 rows/s.

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
