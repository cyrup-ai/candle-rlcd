"""Build a tiny random-weight laya checkpoint plus PyTorch golden outputs for parity tests.

The model and sequence code come from laya itself (`laya/common.py` in a laya checkout), so
the golden outputs are what laya's own Python produces, not our reimplementation of it.

    git clone --depth 1 https://github.com/NandhaKishorM/laya /tmp/laya
    python scripts/make_tiny_fixture.py --laya-src /tmp/laya --out tests/fixtures/tiny

Writes `rl_agent_config.json`, `model.safetensors`, `encoder/config.json`, `tokenizer/`, and
`golden.json` (per request: the token ids of every question row, raw option logits, act-head
probabilities, and laya's decoded answers).
"""
import argparse
import importlib.util
import json
import os
import random

import torch


def load_laya_common(src):
    path = os.path.join(src, "laya", "common.py")
    spec = importlib.util.spec_from_file_location("laya_common", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CORPUS = [
    "The invoice total is 1,240 dollars and it is overdue by 30 days.",
    "Customer says the package arrived damaged and wants a refund.",
    "User: my password reset link does not work. Agent: let me check that for you.",
    "Security alert: repeated failed logins from an unknown IP address.",
    "Great product, fast shipping, would buy again!",
    "The service was slow and the staff were rude.",
    "question: which category best fits this ticket? billing shipping account security other",
    "level 0 level 1 level 2 level 3 level 4 true false yes no statement holds does not hold",
    '{"order_id": 1234, "status": "delayed", "items": ["book", "lamp"], "priority": 2.5}',
    "choice score noul question instructions criteria option label description",
]


def build_tokenizer(out_dir):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    specials = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=600, special_tokens=specials,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(CORPUS * 20, trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", cls_token="[CLS]",
                                   sep_token="[SEP]", pad_token="[PAD]", mask_token="[MASK]")
    fast.save_pretrained(os.path.join(out_dir, "tokenizer"))
    return fast


REQUESTS = [
    {
        "state": "Customer says the package arrived damaged and wants a refund.",
        "questions": {
            "category": {"t": "choice", "ins": "Which team should handle this ticket?",
                         "crit": {"billing": "payments and refunds", "shipping": "delivery problems",
                                  "account": None, "security": {"desc": "fraud", "sev": 3}}},
            "urgency": {"t": "score", "ins": "How urgent is it?",
                        "crit": ["not urgent", "somewhat", "urgent", "critical"]},
            "refund": {"t": "noul", "ins": "Does the customer want a refund?"},
        },
    },
    {
        # Conversation list: serialized as JSON and truncated from the left.
        "state": [{"role": "user", "content": "my password reset link does not work " * 8},
                  {"role": "agent", "content": "let me check that for you [MASK] now"}],
        "questions": {
            "resolved": {"t": "noul", "ins": "Is the issue resolved?",
                         "crit": {"true": "the user confirms it works"},
                         "labels": {"false": "no", "true": "yes"}},
            "sentiment": {"t": "score", "ins": "Rate the user's mood.",
                          "crit": ["angry", "annoyed", "neutral", "happy", "delighted"]},
        },
    },
    {
        # Many long options overflow head_max_len and get cut per option.
        "state": {"order_id": 1234, "status": "delayed", "items": ["book", "lamp"], "priority": 2.5},
        "questions": {
            "action": {"t": "choice", "ins": "What should the agent do next with this order?",
                       "crit": {"opt%d" % i: "a fairly long description of option %d " % i * 3
                                for i in range(9)}},
        },
    },
    {
        # Long plain-text state is right-truncated to max_len.
        "state": " ".join(CORPUS) * 3,
        "questions": {
            "security": {"t": "noul", "ins": "Is this a security incident?"},
            "topic": {"t": "choice", "ins": "Pick one", "crit": {"a": "", "b": "invoice", "c": "security"}},
        },
    },
]


def make_golden(common, model, tok, cfg, requests):
    """Run laya's own sequence builder and DecisionModel over `requests`; return golden rows."""
    golden = []
    for req in requests:
        state, qs = req["state"], req["questions"]
        state_ids = tok(common.serialize_state(state).replace(tok.mask_token, " "),
                        add_special_tokens=False)["input_ids"]
        items = []
        for qid, q in qs.items():
            ids, markers = common.build_sequence(tok, state, q, cfg["max_len"], cfg["head_max_len"],
                                                 truncate_left=isinstance(state, list),
                                                 state_ids=state_ids)
            assert len(markers) == len(common.render_options(q)), qid
            items.append({"ids": ids, "markers": markers, "qtype": common.QTYPES[q["t"]]})
        b = common.collate_items([items], tok.pad_token_id)
        with torch.no_grad():
            logits, act = model(b["input_ids"], b["attention_mask"], b["marker_pos"],
                                b["marker_mask"], b["qtype"])
        act_p = torch.softmax(act.float(), -1)
        rows = []
        for j, (qid, it) in enumerate(zip(qs, items)):
            k = len(it["markers"])
            # Calibrated probabilities exactly as laya's Agent._decode_answers computes them.
            qt = it["qtype"]
            t = common.clamp_temperature(cfg["temperature_by_options"].get(
                common.temp_bucket(qt, k), cfg["temperature"][qt]))
            probs = torch.softmax(logits[j, :k] / t, -1)
            rows.append({"id": qid, "ids": it["ids"], "markers": it["markers"],
                         "logits": logits[j, :k].tolist(), "act_probs": act_p[j].tolist(),
                         "temperature": t, "probs": probs.tolist()})
        golden.append({"state": state, "questions": qs, "rows": rows})

    return golden


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--laya-src", required=True)
    ap.add_argument("--out", default="tests/fixtures/tiny")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    common = load_laya_common(args.laya_src)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    tok = build_tokenizer(args.out)

    from transformers import ModernBertConfig, ModernBertModel

    ecfg = ModernBertConfig(
        vocab_size=len(tok), hidden_size=128, intermediate_size=96, num_hidden_layers=4,
        num_attention_heads=4, max_position_embeddings=512, local_attention=16,
        global_attn_every_n_layers=3, pad_token_id=tok.pad_token_id, cls_token_id=tok.cls_token_id,
        sep_token_id=tok.sep_token_id, bos_token_id=tok.cls_token_id, eos_token_id=tok.sep_token_id,
    )
    # Distinct local theta so a swapped RoPE table would show up in the logits.
    ecfg.rope_parameters["sliding_attention"]["rope_theta"] = 20000.0
    enc = ModernBertModel._from_config(ecfg, attn_implementation="sdpa")
    ecfg.save_pretrained(os.path.join(args.out, "encoder"))

    cfg = {"encoder": "tiny", "head_layers": 2, "act_costs": {"act": 0.0}, "max_len": 128,
           "head_max_len": 48, "temperature": [1.3, 0.8, 1.0],
           "temperature_by_options": {"choice:3-5": 1.7, "score:3-5": 0.1, "noul:2": 2.0}}
    with open(os.path.join(args.out, "rl_agent_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    model = common.DecisionModel(enc, cfg["head_layers"], len(cfg["act_costs"]) + 1).eval()
    with torch.no_grad():
        # Randomise everything (norm weights/biases too) so a mis-loaded tensor cannot hide.
        # Matrices get unit-gain (1/sqrt(fan_in)) scale so option logits spread out and small
        # numerical differences are visible against them.
        for name, p in model.named_parameters():
            if p.dim() > 1:
                scale = 1.0 if "emb" in name else p.shape[1] ** -0.5
                p.copy_(torch.randn_like(p) * scale)
            else:
                base = 1.0 if ("norm" in name or name.startswith("scorer.0")) and name.endswith("weight") else 0.0
                p.copy_(base + 0.3 * torch.randn_like(p))
        model.temperature.copy_(torch.tensor([1.1, 0.9, 1.0]))
    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in model.state_dict().items()},
              os.path.join(args.out, "model.safetensors"))

    golden = make_golden(common, model, tok, cfg, REQUESTS)
    with open(os.path.join(args.out, "golden.json"), "w") as f:
        json.dump(golden, f, indent=1, ensure_ascii=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
