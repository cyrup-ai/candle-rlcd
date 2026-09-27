#!/usr/bin/env python3
"""Summarise a scripts/train-mac.sh run: writes <run>/report.md and <run>/report.json.

  python scripts/make_report.py runs/<name>
"""
import json
import os
import sys

run = sys.argv[1]
J = lambda *p: os.path.join(run, *p)


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def kv(path):
    out = {}
    if os.path.exists(path):
        for line in open(path):
            if "=" in line:
                k, v = line.rstrip("\n").split("=", 1)
                out[k] = v
    return out


env = kv(J("run.env"))
report = {"run": os.path.basename(os.path.abspath(run)), "settings": env}

# Wall time per step, from the marks the script writes as it goes.
times = []
if os.path.exists(J("timings.tsv")):
    rows = [line.split("\t") for line in open(J("timings.tsv")) if "\t" in line]
    for (a, ta), (b, tb) in zip(rows, rows[1:]):
        times.append((b, int(tb) - int(ta)))
report["timings_s"] = dict(times)

# Per-stage training summaries (the last line of each stage's metrics.jsonl).
stages = {}
for n in ("1", "2", "3"):
    p = J("stages", n, "metrics.jsonl")
    if not os.path.exists(p):
        continue
    last_log, final = None, None
    for line in open(p):
        r = json.loads(line)
        if "final" in r:
            final = r["final"]
        elif "loss" in r:
            last_log = r
    stages[n] = {"final": final, "last_log": last_log}
report["stages"] = stages


def evals(d):
    out = {}
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.endswith(".json"):
                m = load(os.path.join(d, f))
                if m:
                    out[f[:-5]] = m
    return out


model_ev, laya_ev = evals(J("eval", "model")), evals(J("eval", "laya"))
report["eval"] = {"model": model_ev, "laya": laya_ev}
report["calibrate"] = load(J("eval", "calibrate.json"))
report["battery"] = load(J("battery.json"))
report["bench"] = {f[:-5]: load(J(f)) for f in sorted(os.listdir(run)) if f.startswith("bench-") and f.endswith(".json")}
report["data"] = load(J("data-manifest.json"))
with open(J("report.json"), "w") as f:
    json.dump(report, f, indent=1)

# ---------------------------------------------------------------------------------------------
md = [f"# candle-rlcd training run `{report['run']}`", ""]
md.append(f"Base `{env.get('BASE', '?')}`, prefix layout, max_len {env.get('MAX_LEN', '?')}, "
          f"batch {env.get('BATCH', '?')} x {env.get('ACCUM', '?')}, data scale {env.get('SCALE', '?')}, "
          f"commit {env.get('COMMIT', '?')}, on {env.get('HOST', '?')}"
          f"{' (CPU forced)' if env.get('CPU') == '1' else ''}.")
md += ["", f"Model: `{J('model')}`. Serve it with `candle-rlcd serve --model {report['run']}={J('model')}`.", ""]

md += ["## Held-out test sets", "",
       "Calibrated probabilities (the model's fitted temperatures). Accuracy is argmax against the "
       "label; Brier and ECE measure how trustworthy `confidence` is (lower is better).", ""]
hdr = "| test set | n | accuracy | Brier | ECE |"
sep = "|---|---|---|---|---|"
if laya_ev:
    hdr += " laya accuracy | laya ECE |"
    sep += "---|---|"
md += [hdr, sep]
tot = {"n": 0, "acc": 0.0}
for name, m in model_ev.items():
    c = m["calibrated"]
    row = f"| {name} | {c['n']} | {c['accuracy']:.3f} | {c['brier']:.3f} | {c['ece']:.3f} |"
    if name != "ag_news_test1k":
        tot["n"] += c["n"]
        tot["acc"] += c["accuracy"] * c["n"]
    if laya_ev:
        lc = laya_ev.get(name, {}).get("calibrated")
        row += f" {lc['accuracy']:.3f} | {lc['ece']:.3f} |" if lc else " - | - |"
    md.append(row)
if tot["n"]:
    line = f"| **all sources (question-weighted)** | {tot['n']} | **{tot['acc'] / tot['n']:.3f}** | | |"
    if laya_ev:
        ln = sum(v["calibrated"]["n"] for k, v in laya_ev.items() if k != "ag_news_test1k")
        la = sum(v["calibrated"]["accuracy"] * v["calibrated"]["n"] for k, v in laya_ev.items() if k != "ag_news_test1k")
        line += f" {la / ln:.3f} | |" if ln else " - | |"
    md.append(line)
md += ["", "Earlier reference on ag_news_test1k: laya 0.927 (ECE 0.036 raw); laya init + 2,400 AG News rows "
       "in the prefix layout 0.928 (ECE 0.019); ModernBERT-base + 4,000 rows 0.900 (ECE 0.019).", ""]

b = report["battery"]
if b:
    md += ["## Audit battery", "",
           f"{b['correct']}/{b['total']} correct ("
           + ", ".join(f"{k} {v['correct']}/{v['total']}" for k, v in b["by_type"].items())
           + f"), mean {b['mean_latency_ms']:.0f} ms per request over HTTP. laya as published scored 33/40 "
           "(choice 17/19, noul 13/14, score 3/7).", ""]
    if b["misses"]:
        md += ["| type | state | expected | got |", "|---|---|---|---|"]
        md += [f"| {m['type']} | {m['state']} | {m['expected']} | {m['got']} |" for m in b["misses"]]
        md.append("")

if report["bench"]:
    md += ["## Throughput (in process, `bench.jsonl`: one question, or a choice plus four yes/no questions)", "",
           "| clients | req/s | questions/s | p50 ms | p90 ms | p99 ms |", "|---|---|---|---|---|---|"]
    for k, r in report["bench"].items():
        if r:
            lat = r["latency_ms"]
            md.append(f"| {r['concurrency']} | {r['req_per_s']} | {r['questions_per_s']} | {lat['p50']} | {lat['p90']} | {lat['p99']} |")
    md.append("")

if stages:
    md += ["## Training stages", "", "| stage | steps | last loss | rows/s | dev accuracy | dev ECE |", "|---|---|---|---|---|---|"]
    for n, s in stages.items():
        f, l = s["final"] or {}, s["last_log"] or {}
        ev = f.get("eval_calibrated") or f.get("eval_raw") or {}
        md.append(f"| {n} | {f.get('step', '-')} | {l.get('loss', float('nan')):.4f} | {l.get('rows_per_s', 0):.1f} | "
                  f"{ev.get('accuracy', float('nan')):.3f} | {ev.get('ece', float('nan')):.3f} |")
    md.append("")

if times:
    md += ["## Wall time", "", "| step | minutes |", "|---|---|"]
    md += [f"| {k} | {v / 60:.1f} |" for k, v in times]
    md.append("")

d = report["data"]
if d:
    md += ["## Data", ""]
    md += [f"- `{k}`: {v['records']} records, {v['questions']} questions" for k, v in d["files"].items()]
    md += ["", "Sources: " + ", ".join(f"{k} (`{v['hf']}`)" for k, v in d["sources"].items()), ""]

with open(J("report.md"), "w") as f:
    f.write("\n".join(md))
print("\n".join(md))
