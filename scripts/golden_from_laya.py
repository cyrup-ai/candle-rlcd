"""Golden outputs from the published laya weights, for `real_laya_matches_pytorch`.

Needs network access to huggingface.co and a laya checkout for its reference code:

    git clone --depth 1 https://github.com/NandhaKishorM/laya /tmp/laya
    python scripts/golden_from_laya.py --laya-src /tmp/laya --out /tmp/laya-golden.json
    LAYA_DIR=$(python -c "from huggingface_hub import snapshot_download as s; print(s('convaiinnovations/laya', allow_patterns=['rl_agent_config.json','model.safetensors','tokenizer/*','encoder/*']))") \\
    LAYA_GOLDEN=/tmp/laya-golden.json cargo test --release real_laya -- --nocapture

The probe set is the same four requests the tiny fixture uses (choice with structured criteria,
score, noul with custom labels, a left-truncated conversation, head-budget overflow, and a
right-truncated long state).
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from make_tiny_fixture import REQUESTS, load_laya_common, make_golden  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--laya-src", required=True)
    ap.add_argument("--model", default="convaiinnovations/laya")
    ap.add_argument("--subfolder", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    common = load_laya_common(args.laya_src)

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    model_dir = args.model
    if not os.path.isdir(model_dir):
        prefix = f"{args.subfolder}/" if args.subfolder else ""
        model_dir = snapshot_download(args.model, allow_patterns=[
            prefix + p for p in ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")])
    if args.subfolder:
        model_dir = os.path.join(model_dir, args.subfolder)
    print("model dir:", model_dir)
    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    model = common.build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"), pretrained=False)
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    model = model.float().eval()
    cfg.setdefault("max_len", 512)
    cfg.setdefault("head_max_len", 192)
    cfg.setdefault("temperature", [1.0, 1.0, 1.0])
    cfg.setdefault("temperature_by_options", {})
    with torch.no_grad():
        golden = make_golden(common, model, tok, cfg, REQUESTS)
    with open(args.out, "w") as f:
        json.dump(golden, f, indent=1, ensure_ascii=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
