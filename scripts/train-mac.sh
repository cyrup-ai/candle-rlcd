#!/usr/bin/env bash
# One command for the whole training run on an Apple Silicon Mac (Metal), from a fresh clone:
#
#   scripts/train-mac.sh [run-name]
#
# It builds candle-rlcd with Metal, downloads and converts the public training data, trains in
# three stages (general -> near-miss negatives -> task data with 40% general replay), calibrates,
# then evaluates, runs the audit battery and a throughput bench, and writes a report.
#
# Everything lands in runs/<run-name>/ (default: a timestamp):
#   model/              the trained model: serve it with `candle-rlcd serve --model runs/<name>/model`
#   calibration.json    the fitted temperatures (already written into model/rl_agent_config.json)
#   report.md           accuracy / Brier / ECE per test set, battery, throughput, timings
#   report.json         the same numbers, machine readable
#   eval/ battery.* bench-*.json stages/ logs/ timings.tsv run.env
#
# Re-running with the same name picks up where it stopped: finished stages are skipped and an
# interrupted stage resumes from its last checkpoint (saved every SAVE_EVERY steps).
#
# Settings (environment variables):
#   BASE=auto|laya|modernbert-base|modernbert-large   starting weights (auto: laya with >= 32 GB
#                       of memory, else ModernBERT-base)
#   SCALE=1             multiplies the training row counts (0.1 for a ~1/10 run)
#   TEST_SCALE=1        multiplies the dev/test set sizes
#   TASK_DATA=a.jsonl   your own labelled records (Jev requests + targets/answers), added to stage 3
#                       (space separated for several files)
#   MAX_LEN=512 BATCH= ACCUM= LR1= LR2= LR3= SAVE_EVERY=200
#   COMPARE_LAYA=1      also evaluate convaiinnovations/laya as published on the same test sets
#   SKIP_BATTERY=0 SKIP_BENCH=0 SKIP_SMOKE=0
#   CPU=1               train on the CPU (what CI and Linux use); FEATURES= overrides cargo features
#   HF_TOKEN=...        optional, lifts the Hugging Face download rate limit
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$(pwd)
NAME=${1:-${RUN_NAME:-$(date +%Y%m%d-%H%M)}}
RUN="$ROOT/runs/$NAME"
DATA=${DATA_DIR:-$ROOT/data/public}
SCALE=${SCALE:-1}
TEST_SCALE=${TEST_SCALE:-1}
MAX_LEN=${MAX_LEN:-512}
SAVE_EVERY=${SAVE_EVERY:-200}
COMPARE_LAYA=${COMPARE_LAYA:-1}
PORT=${PORT:-18080}
mkdir -p "$RUN/logs" "$RUN/eval" "$RUN/stages"
exec > >(tee -a "$RUN/logs/train-mac.log") 2>&1

say() { printf '\n\033[1m== %s\033[0m  (%s)\n' "$*" "$(date '+%H:%M:%S')"; }
mark() { printf '%s\t%s\n' "$1" "$(date +%s)" >> "$RUN/timings.tsv"; }
die() { echo "error: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------------------------
say "checking the machine"
OS=$(uname -s); ARCH=$(uname -m)
if [ "$OS" = Darwin ]; then
    MEM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))
    MACOS=$(sw_vers -productVersion)
    [ "${MACOS%%.*}" -ge 15 ] || echo "warning: macOS $MACOS; Candle's Metal backend needs macOS 15+, so this will run on the CPU"
    [ "$ARCH" = arm64 ] || echo "warning: $ARCH Mac; Metal training is meant for Apple Silicon"
    DEFAULT_FEATURES=metal,accelerate
else
    MEM_GB=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) / 1048576 ))
    DEFAULT_FEATURES=
fi
FEATURES=${FEATURES-$DEFAULT_FEATURES}
CPUFLAG=
[ "${CPU:-0}" = 1 ] && CPUFLAG=--cpu
echo "$OS $ARCH, ${MEM_GB} GB memory, cargo features: ${FEATURES:-none}${CPUFLAG:+, forced CPU}"
command -v cargo >/dev/null || die "cargo not found; install Rust from https://rustup.rs"
PYTHON=${PYTHON:-python3}
command -v "$PYTHON" >/dev/null || die "python3 not found (xcode-select --install provides one)"

BASE=${BASE:-auto}
if [ "$BASE" = auto ]; then
    if [ "$MEM_GB" -ge 32 ]; then BASE=laya; else BASE=modernbert-base; fi
fi
case "$BASE" in
    laya|modernbert-large) BATCH=${BATCH:-4}; ACCUM=${ACCUM:-8}
        LR1=${LR1:-2.5e-5}; LR2=${LR2:-1.5e-5}; LR3=${LR3:-1e-5} ;;
    modernbert-base) BATCH=${BATCH:-8}; ACCUM=${ACCUM:-4}
        LR1=${LR1:-5e-5}; LR2=${LR2:-3e-5}; LR3=${LR3:-2e-5} ;;
    *) die "BASE must be auto, laya, modernbert-base or modernbert-large (got $BASE)" ;;
esac
cat > "$RUN/run.env" <<EOF
NAME=$NAME
BASE=$BASE
SCALE=$SCALE
MAX_LEN=$MAX_LEN
BATCH=$BATCH
ACCUM=$ACCUM
LR1=$LR1
LR2=$LR2
LR3=$LR3
FEATURES=$FEATURES
CPU=${CPU:-0}
TEST_SCALE=$TEST_SCALE
TASK_DATA=${TASK_DATA:-}
HOST=$OS $ARCH ${MEM_GB}GB
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
EOF
echo "run $NAME: base $BASE, batch $BATCH x $ACCUM, max_len $MAX_LEN, scale $SCALE -> $RUN"
mark start

# ---------------------------------------------------------------------------------------------
say "building candle-rlcd (release${FEATURES:+, $FEATURES})"
cargo build --release --locked --bin candle-rlcd ${FEATURES:+--features "$FEATURES"}
BIN="$ROOT/target/release/candle-rlcd"
mark build

# ---------------------------------------------------------------------------------------------
say "python environment for the data download"
VENV="$ROOT/.venv"
if [ ! -x "$VENV/bin/python" ]; then "$PYTHON" -m venv "$VENV"; fi
"$VENV/bin/python" -c 'import datasets, huggingface_hub' 2>/dev/null ||
    "$VENV/bin/python" -m pip install --quiet --upgrade pip datasets huggingface_hub
PY="$VENV/bin/python"

say "training data (public Hugging Face datasets -> $DATA)"
STAMP="scale=$SCALE test=$TEST_SCALE task=${TASK_DATA:-}"
if [ -f "$DATA/.done" ] && [ "$(cat "$DATA/.done")" = "$STAMP" ]; then
    echo "already prepared ($STAMP); delete $DATA to rebuild"
else
    mkdir -p "$DATA"
    "$BIN" data synthetic --n "$(awk "BEGIN {print int(1500 * $SCALE) + 1}")" --seed 7 --output "$DATA/synthetic_tickets.jsonl"
    EXTRA="--extra $DATA/synthetic_tickets.jsonl"
    for f in ${TASK_DATA:-}; do EXTRA="$EXTRA --extra $f"; done
    # shellcheck disable=SC2086
    "$PY" scripts/fetch_data.py --out "$DATA" --scale "$SCALE" --test-scale "$TEST_SCALE" $EXTRA
    echo "$STAMP" > "$DATA/.done"
fi
cp "$DATA/manifest.json" "$RUN/data-manifest.json"
mark data

# ---------------------------------------------------------------------------------------------
say "starting weights: $BASE"
case "$BASE" in
    laya) INIT=(--init convaiinnovations/laya) ;;   # candle-rlcd downloads it into the HF cache
    modernbert-*)
        SIZE=${BASE#modernbert-}
        DIR="$ROOT/data/models/ModernBERT-$SIZE"
        "$PY" - "$DIR" "answerdotai/ModernBERT-$SIZE" <<'EOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[2], local_dir=sys.argv[1],
                  allow_patterns=["config.json", "model.safetensors", "tokenizer.json",
                                  "tokenizer_config.json", "special_tokens_map.json"])
EOF
        INIT=(--init-encoder "$DIR") ;;
esac
mark weights

# Common training flags. The prefix layout encodes the state once per request.
COMMON=(--layout prefix --max-len "$MAX_LEN" --batch-size "$BATCH" --grad-accum "$ACCUM"
        --save-every "$SAVE_EVERY" --keep 1 --log-every 10 --eval "$DATA/dev.jsonl")

if [ "${SKIP_SMOKE:-0}" != 1 ] && [ ! -f "$RUN/stages/1/metrics.jsonl" ]; then
    say "smoke test: 2 training steps and an eval, to fail fast before the long run"
    head -n 16 "$DATA/stage1_general.jsonl" > "$RUN/logs/smoke-train.jsonl"
    head -n 8 "$DATA/dev.jsonl" > "$RUN/logs/smoke-eval.jsonl"
    rm -rf "$RUN/smoke"
    "$BIN" train $CPUFLAG "${INIT[@]}" --train "$RUN/logs/smoke-train.jsonl" --eval "$RUN/logs/smoke-eval.jsonl" \
        --out "$RUN/smoke" --layout prefix --max-len "$MAX_LEN" --batch-size 2 --grad-accum 1 --max-steps 2 --log-every 1
    "$BIN" eval $CPUFLAG --model "$RUN/smoke/final" --data "$RUN/logs/smoke-eval.jsonl" > /dev/null
    rm -rf "$RUN/smoke"
    echo "smoke test passed"
fi

# train_stage <n> <data> <lr-encoder> <init flags...>
train_stage() {
    local n=$1 data=$2 lr=$3; shift 3
    local dir="$RUN/stages/$n"
    if [ -f "$dir/final/model.safetensors" ]; then
        echo "stage $n already done"
        return
    fi
    local latest
    # The newest complete checkpoint (a save in progress is step-N.partial, without the state file).
    latest=$(ls -d "$dir"/step-* 2>/dev/null | grep -E '/step-[0-9]+$' | awk -F'step-' '{print $NF, $0}' |
        sort -n | cut -d' ' -f2- | while read -r d; do [ -f "$d/trainer_state.json" ] && echo "$d"; done | tail -n 1 || true)
    if [ -n "$latest" ]; then
        echo "resuming stage $n from $latest"
        "$BIN" train $CPUFLAG --train "$data" --out "$dir" --resume "$latest"
    else
        "$BIN" train $CPUFLAG "$@" --train "$data" --out "$dir" --lr-encoder "$lr" "${COMMON[@]}"
    fi
    # Stage checkpoints are only needed to resume; the next stage starts from final/.
    rm -rf "$dir"/step-* "$dir/final/optimizer.safetensors"
}

say "stage 1/3: broad general decisions ($(wc -l < "$DATA/stage1_general.jsonl" | tr -d ' ') records)"
train_stage 1 "$DATA/stage1_general.jsonl" "$LR1" "${INIT[@]}"
mark stage1
say "stage 2/3: near-miss negatives ($(wc -l < "$DATA/stage2_hard.jsonl" | tr -d ' ') records)"
train_stage 2 "$DATA/stage2_hard.jsonl" "$LR2" --init "$RUN/stages/1/final"
mark stage2
say "stage 3/3: task data + 40% general replay ($(wc -l < "$DATA/stage3_task.jsonl" | tr -d ' ') records)"
train_stage 3 "$DATA/stage3_task.jsonl" "$LR3" --init "$RUN/stages/2/final"
mark stage3

# ---------------------------------------------------------------------------------------------
say "packaging the model and calibrating on the dev set"
rm -rf "$RUN/model"
mkdir -p "$RUN/model"
for f in model.safetensors rl_agent_config.json encoder tokenizer; do
    cp -R "$RUN/stages/3/final/$f" "$RUN/model/"
done
"$BIN" calibrate $CPUFLAG --model "$RUN/model" --data "$DATA/dev.jsonl" \
    --out "$RUN/calibration.json" --write > "$RUN/eval/calibrate.json"
mark calibrate

say "evaluating on the held-out test sets"
eval_all() {  # eval_all <model> <out dir>
    mkdir -p "$2"
    for f in "$DATA"/test/*.jsonl; do
        local name
        name=$(basename "$f" .jsonl)
        [ -s "$2/$name.json" ] && continue
        printf '  %-22s' "$name"
        "$BIN" eval $CPUFLAG --model "$1" --data "$f" > "$2/$name.json.tmp" && mv "$2/$name.json.tmp" "$2/$name.json"
        "$PY" -c 'import json,sys; m=json.load(open(sys.argv[1]))["calibrated"]; print("acc %.3f  brier %.3f  ece %.3f  n %d" % (m["accuracy"], m["brier"], m["ece"], m["n"]))' "$2/$name.json"
    done
}
eval_all "$RUN/model" "$RUN/eval/model"
mark eval
if [ "$COMPARE_LAYA" = 1 ]; then
    say "baseline: laya as published, same test sets"
    eval_all convaiinnovations/laya "$RUN/eval/laya"
    mark eval_laya
fi

# ---------------------------------------------------------------------------------------------
if [ "${SKIP_BATTERY:-0}" != 1 ]; then
    say "audit battery (40 hand-written Jev-style cases) against the served model"
    "$BIN" serve $CPUFLAG --model "$RUN/model" --port "$PORT" > "$RUN/logs/serve.log" 2>&1 &
    SERVER=$!
    trap 'kill $SERVER 2>/dev/null || true' EXIT
    for _ in $(seq 1 120); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
        kill -0 $SERVER 2>/dev/null || die "server exited; see $RUN/logs/serve.log"
        sleep 1
    done
    "$PY" scripts/audit_battery.py "http://127.0.0.1:$PORT" --json "$RUN/battery.json" | tee "$RUN/battery.txt"
    kill $SERVER 2>/dev/null || true
    wait $SERVER 2>/dev/null || true
    trap - EXIT
    mark battery
fi

if [ "${SKIP_BENCH:-0}" != 1 ]; then
    say "throughput bench (in process, mixed 1- and multi-question requests)"
    for c in 1 4; do
        "$BIN" bench $CPUFLAG --model "$RUN/model" --data "$DATA/bench.jsonl" --concurrency $c \
            --requests "${BENCH_REQUESTS:-200}" > "$RUN/bench-c$c.json"
        cat "$RUN/bench-c$c.json"
    done
    mark bench
fi

say "report"
"$PY" scripts/make_report.py "$RUN"
mark done
echo
echo "Done. Model: $RUN/model"
echo "Report: $RUN/report.md"
echo "Serve it: $BIN serve --model $NAME=$RUN/model --port 8080"
