#!/usr/bin/env python3
"""Download public Hugging Face datasets and turn them into candle-rlcd training records.

Every record is Jev-shaped: a `state`, typed `questions` (`choice` / `score` / `noul`) and
`targets`. The output follows the staged recipe from the CLM comparison:

  stage1_general.jsonl  broad, easy-to-verify decisions: topics, intents, sentiment, NLI,
                        paraphrase, reading-comprehension and science multiple choice, STS
  stage2_hard.jsonl     near-miss negatives: intent/topic choices whose distractors are the
                        most confusable labels, "is this about X?" for the closest wrong X,
                        adversarial NLI (ANLI), PAWS paraphrases, HellaSwag endings
  stage3_task.jsonl     Jev-style work (routing, moderation, spam, toxicity, triage, plus any
                        --extra files) with 40% fresh general/hard rows mixed back in
  dev.jsonl             held-out mix for fitting temperatures (from each source's eval split)
  test/<source>.jsonl   held-out per-source test sets, disjoint from dev
  test/ag_news_test1k.jsonl  1,000 AG News test rows (random.seed(0)), the fixed comparison set
  bench.jsonl           requests for the throughput bench (one and several questions each)

Rows are drawn without replacement, so no training row repeats across stages, and eval rows
come from each dataset's own validation/test split (or a held-out tail when it has none).

  python scripts/fetch_data.py --out data/public [--scale 1.0] [--extra mine.jsonl ...]
"""
import argparse
import json
import os
import random
import re
import sys
import zlib
from collections import defaultdict

try:
    import datasets
except ImportError:
    sys.exit("needs the `datasets` package: python3 -m pip install datasets")

datasets.logging.set_verbosity_error()
datasets.disable_progress_bars()

# ------------------------------------------------------------------------------------------
# Record builders


def rec(state, questions, targets, source):
    return {"state": state, "questions": questions, "targets": targets, "source": source}


def choice(ins, crit):
    return {"t": "choice", "ins": ins, "crit": crit}


def noul(ins):
    return {"t": "noul", "ins": ins}


def score(ins, levels):
    return {"t": "score", "ins": ins, "crit": levels}


def words(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def pick_options(labels, gold, rng, k, hard):
    """`k` option keys including `gold`. Hard mode takes the labels sharing the most words with
    the gold label (e.g. `pto_request` next to `pto_balance`), easy mode takes random ones."""
    others = [x for x in labels if x != gold]
    if len(others) + 1 <= k:
        opts = others
    elif hard:
        g = words(gold + " " + labels[gold])
        rng.shuffle(others)
        others.sort(key=lambda x: -len(g & words(x + " " + labels[x])))
        opts = others[: k - 1]
    else:
        opts = rng.sample(others, k - 1)
    opts = opts[: k - 1] + [gold]
    rng.shuffle(opts)
    return {x: labels[x] for x in opts}


def closest_wrong(labels, gold, rng):
    g = words(gold + " " + labels[gold])
    others = [x for x in labels if x != gold]
    rng.shuffle(others)
    return max(others, key=lambda x: len(g & words(x + " " + labels[x])))


def snake(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def classify(state, ins_pool, about_pool, labels, gold, rng, hard, source, k_max=8, noul_p=0.3):
    """A `choice` over (a subset of) `labels`; sometimes an "is this about X?" `noul` on the
    same state, as Jev's fan-out asks several questions about one state."""
    k = len(labels) if len(labels) <= k_max else rng.randint(4, k_max)
    crit = pick_options(labels, gold, rng, k, hard)
    qs = {"q": choice(rng.choice(ins_pool), crit)}
    ts = {"q": gold}
    if about_pool and (hard or rng.random() < noul_p):
        yes = rng.random() < 0.5
        x = gold if yes else (closest_wrong(labels, gold, rng) if hard else
                              rng.choice([l for l in labels if l != gold]))
        qs["about"] = noul(rng.choice(about_pool).format(labels[x]))
        ts["about"] = yes
    return rec(state, qs, ts, source)


def fan_out(r, rng):
    """A bench request in Jev's fan-out shape: the record's choice plus four yes/no questions
    about the same state, so the encode-once layout's shared state shows in the timings."""
    qs = dict(r["questions"])
    opts = list(qs["q"]["crit"].values())
    for i in range(4):
        qs[f"about{i}"] = noul(f"Does this involve {rng.choice(opts)}?")
    return {"state": r["state"], "questions": qs}


LETTERS = "ABCDEFGHIJ"


def mc(state, ins, options, gold_idx, source):
    crit = {LETTERS[i]: o for i, o in enumerate(options)}
    return rec(state, {"q": choice(ins, crit)}, {"q": LETTERS[gold_idx]}, source)


# ------------------------------------------------------------------------------------------
# Sources: each yields records from one dataset row. `hard` selects near-miss options.

AG = {
    "world": "world news: politics, conflicts, diplomacy and international affairs",
    "sports": "sports: games, athletes, teams and competitions",
    "business": "business: companies, markets, the economy and finance",
    "science_tech": "science and technology: computing, the internet, research and space",
}
AG_KEYS = ["world", "sports", "business", "science_tech"]
ABOUT = ["Is this text about {}?", "Does this belong under: {}?", "Is the main subject {}?"]


def clean_ag(s):
    # AG News writes some characters as `#39;`; `candle-rlcd data ag-news` decodes them the same way.
    s = re.sub(r"#(\d+);", lambda m: chr(int(m.group(1))), s.replace("\\", " "))
    return " ".join(s.split())


def ag_news(r, rng, hard):
    return classify(clean_ag(r["text"]), ["Which section of a news site does this article belong in?",
                                "What is this news story about?", "Pick the topic of this article."],
                    ABOUT, AG, AG_KEYS[r["label"]], rng, hard, "ag_news")


def ag_news_fixed(r):
    # Same question as `candle-rlcd data ag-news`, so test1k matches earlier runs' framing.
    return rec(clean_ag(r["text"]), {"topic": choice("Which section of a news site does this article belong in?", AG)},
               {"topic": AG_KEYS[r["label"]]}, "ag_news")


DBP = ["Company", "EducationalInstitution", "Artist", "Athlete", "OfficeHolder",
       "MeanOfTransportation", "Building", "NaturalPlace", "Village", "Animal", "Plant", "Album",
       "Film", "WrittenWork"]
DBP_DESC = {snake(re.sub(r"(?<!^)([A-Z])", r" \1", x)): re.sub(r"(?<!^)([A-Z])", r" \1", x).lower()
            for x in DBP}
DBP_KEYS = list(DBP_DESC)


def dbpedia(r, rng, hard):
    return classify(f"{r['title']}\n{r['content'].strip()}",
                    ["What kind of thing is this article about?", "Which category fits this encyclopedia entry?"],
                    ["Is this article about a {}?"], DBP_DESC, DBP_KEYS[r["label"]], rng, hard, "dbpedia")


YAHOO = ["Society & Culture", "Science & Mathematics", "Health", "Education & Reference",
         "Computers & Internet", "Sports", "Business & Finance", "Entertainment & Music",
         "Family & Relationships", "Politics & Government"]
YAHOO_DESC = {snake(x): x.lower() for x in YAHOO}
YAHOO_KEYS = list(YAHOO_DESC)


def yahoo(r, rng, hard):
    state = "\n".join(x for x in [r["question_title"], r["question_content"], r["best_answer"]] if x)
    return classify(state, ["Which forum category should this question be posted in?",
                            "What category does this Q&A thread belong to?"],
                    ["Is this question about {}?"], YAHOO_DESC, YAHOO_KEYS[r["topic"]], rng, hard, "yahoo")


def intents(names, oos=None):
    d = {n: n.replace("_", " ") for n in names}
    if oos in d:
        d[oos] = "none of these / something else"
    return d


def clinc(names):
    labels = intents(names, "oos")

    def f(r, rng, hard):
        return classify(r["text"], ["What does the user want?", "Which intent does this message express?",
                                    "Route this request to the right skill."],
                        ["Is the user asking about {}?"], labels, names[r["intent"]], rng, hard, "clinc")
    return f


def banking(r, rng, hard):
    names = banking.names
    return classify(r["text"], ["Which support topic is this banking customer asking about?",
                                "Route this banking query to the right queue."],
                    ["Is the customer asking about {}?"], names, r["label_text"], rng, hard, "banking77")


EMO = {"sadness": "sad, down or hurt", "joy": "happy or pleased", "love": "loving or affectionate",
       "anger": "angry or irritated", "fear": "afraid, anxious or worried", "surprise": "surprised or amazed"}
EMO_KEYS = list(EMO)


def emotion(r, rng, hard):
    return classify(r["text"], ["Which emotion does the writer express?", "How does the author feel?"],
                    ["Does the writer feel {}?"], EMO, EMO_KEYS[r["label"]], rng, hard, "emotion")


SENT3 = {"negative": "unhappy, critical or angry", "neutral": "neither positive nor negative",
         "positive": "happy, approving or enthusiastic"}


def tweet_sentiment(r, rng, hard):
    return rec(r["text"], {"q": choice(rng.choice(["What is the sentiment of this tweet?", "What tone does this post take?"]), SENT3)},
               {"q": ["negative", "neutral", "positive"][r["label"]]}, "tweet_sentiment")


STARS = ["1 star: terrible", "2 stars: poor", "3 stars: average", "4 stars: good", "5 stars: excellent"]


def yelp(r, rng, hard):
    qs = {"q": score(rng.choice(["Rate this review from 1 to 5 stars.", "How many stars did the reviewer give?"]), STARS)}
    ts = {"q": r["label"]}
    if rng.random() < 0.3 and r["label"] != 2:
        qs["pos"] = noul("Is the customer satisfied?")
        ts["pos"] = r["label"] > 2
    return rec(r["text"], qs, ts, "yelp")


SST5 = ["very negative", "negative", "neutral", "positive", "very positive"]


def sst5(r, rng, hard):
    return rec(r["text"], {"q": score("How positive is this movie review snippet?", SST5)}, {"q": r["label"]}, "sst5")


NLI3 = {"entailment": "the statement must be true given the text",
        "neutral": "the statement might or might not be true",
        "contradiction": "the statement must be false given the text"}
NLI_KEYS = ["entailment", "neutral", "contradiction"]


def nli(source, a="premise", b="hypothesis"):
    def f(r, rng, hard):
        if r["label"] not in (0, 1, 2):
            return None
        if rng.random() < 0.5:
            return rec(r[a], {"q": choice(f'Given the text, is this statement true, false, or undetermined? "{r[b]}"', NLI3)},
                       {"q": NLI_KEYS[r["label"]]}, source)
        return rec(r[a], {"q": noul(f'Does the text imply that "{r[b].rstrip(".")}"?')}, {"q": r["label"] == 0}, source)
    return f


def rte(r, rng, hard):
    return rec(r["sentence1"], {"q": noul(f'Does the text imply that "{r["sentence2"].rstrip(".")}"?')},
               {"q": r["label"] == 0}, "rte")


def qnli(r, rng, hard):
    return rec(r["sentence"], {"q": noul(f"Does this sentence answer the question: {r['question']}")},
               {"q": r["label"] == 0}, "qnli")


def qqp(r, rng, hard):
    return rec(f"Question 1: {r['question1']}\nQuestion 2: {r['question2']}",
               {"q": noul("Are these two questions asking the same thing?")}, {"q": r["label"] == 1}, "qqp")


def paws(r, rng, hard):
    return rec(r["sentence1"], {"q": noul(f'Does this sentence mean the same as: "{r["sentence2"]}"?')},
               {"q": r["label"] == 1}, "paws")


STS = ["0: completely different meaning", "1: same topic, different meaning", "2: some details shared",
       "3: roughly equivalent, details differ", "4: mostly equivalent", "5: identical meaning"]


def stsb(r, rng, hard):
    y = max(0.0, min(5.0, r["label"]))
    lo = int(y)
    t = [0.0] * 6
    t[lo] = 1.0 - (y - lo)
    if lo < 5:
        t[lo + 1] = y - lo
    return rec(f"Sentence A: {r['sentence1']}\nSentence B: {r['sentence2']}",
               {"q": score("How similar in meaning are the two sentences?", STS)}, {"q": t}, "stsb")


def boolq(r, rng, hard):
    q = r["question"].strip()
    q = q[:1].upper() + q[1:] + ("" if q.endswith("?") else "?")
    return rec(r["passage"], {"q": noul(q)}, {"q": bool(r["answer"])}, "boolq")


def arc(source):
    def f(r, rng, hard):
        labels, texts = r["choices"]["label"], r["choices"]["text"]
        if r["answerKey"] not in labels:
            return None
        return mc(r["question"], "Which option correctly answers the question?", texts,
                  labels.index(r["answerKey"]), source)
    return f


def obqa(r, rng, hard):
    labels = r["choices"]["label"]
    return mc(r["question_stem"], "Which option best completes or answers this?", r["choices"]["text"],
              labels.index(r["answerKey"]), "openbookqa")


def csqa(r, rng, hard):
    labels = r["choices"]["label"]
    if r["answerKey"] not in labels:
        return None
    return mc(r["question"], "Which answer makes the most common sense?", r["choices"]["text"],
              labels.index(r["answerKey"]), "commonsense_qa")


def sciq(r, rng, hard):
    opts = [r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]]
    order = list(range(4))
    rng.shuffle(order)
    state = r["support"].strip() or r["question"]
    ins = r["question"] if r["support"].strip() else "Which option correctly answers the question?"
    return mc(state, ins, [opts[i] for i in order], order.index(0), "sciq")


def race(r, rng, hard):
    return mc(r["article"], r["question"].replace("_", "___"), r["options"], "ABCD".index(r["answer"]), "race")


def hellaswag(r, rng, hard):
    if r["label"] in ("", None):
        return None
    return mc(r["ctx"], "Which continuation is the most plausible?", [e.strip() for e in r["endings"]],
              int(r["label"]), "hellaswag")


def imdb(r, rng, hard):
    return rec(r["text"].replace("<br />", " "), {"q": noul("Is this review positive?")}, {"q": r["label"] == 1}, "imdb")


def trec(r, rng, hard):
    labels = trec.names
    return classify(r["text"], ["What kind of answer is this question looking for?"], None, labels,
                    snake(r["label_coarse_text"]), rng, hard, "trec")


def sms_spam(r, rng, hard):
    return rec(r["sms"].strip(), {"q": noul(rng.choice(["Is this message spam?", "Is this an unsolicited promotional or scam message?"]))},
               {"q": r["label"] == 1}, "sms_spam")


def enron_spam(r, rng, hard):
    text = f"Subject: {r['subject']}\n{r['message']}"[:3000]
    return rec(text, {"q": noul(rng.choice(["Is this email spam?", "Should this email go to the spam folder?"]))},
               {"q": r["label"] == 1}, "enron_spam")


def offensive(r, rng, hard):
    return rec(r["text"], {"q": noul(rng.choice(["Is this post offensive?", "Does this message contain insults or offensive language?"]))},
               {"q": r["label"] == 1}, "offensive")


def hate(r, rng, hard):
    return rec(r["text"], {"q": noul("Does this post express hate against a group of people?")}, {"q": r["label"] == 1}, "hate")


def civil(r, rng, hard):
    # Annotator agreement as a soft label: p(toxic) is the fraction of raters who said so.
    qs = {"q": noul(rng.choice(["Is this comment toxic or abusive?", "Should a moderator remove this comment for toxicity?"]))}
    ts = {"q": round(float(r["toxicity"]), 3)}
    if rng.random() < 0.3:
        qs["threat"] = noul("Does this comment contain a threat?")
        ts["threat"] = round(float(r["threat"]), 3)
    return rec(r["text"], qs, ts, "civil_comments")


def civil_keep(r):
    # Most comments are clean; keep every toxic-ish one and 1 in 8 of the rest.
    return r["toxicity"] >= 0.3 or zlib.crc32(r["text"].encode()) % 8 == 0


# name: (hf id, config, train split, eval split, builder, optional row filter)
SOURCES = {
    "ag_news": ("fancyzhx/ag_news", None, "train", "test", ag_news),
    "dbpedia": ("fancyzhx/dbpedia_14", None, "test", None, dbpedia),
    "yahoo": ("community-datasets/yahoo_answers_topics", None, "test", None, yahoo),
    "clinc": ("clinc/clinc_oos", "plus", "train", "test", None),
    "banking77": ("mteb/banking77", None, "train", "test", banking),
    "emotion": ("dair-ai/emotion", "split", "train", "test", emotion),
    "tweet_sentiment": ("cardiffnlp/tweet_eval", "sentiment", "train", "test", tweet_sentiment),
    "yelp": ("Yelp/yelp_review_full", None, "test", None, yelp),
    "sst5": ("SetFit/sst5", None, "train", "test", sst5),
    "mnli": ("nyu-mll/glue", "mnli", "train", "validation_matched", nli("mnli")),
    "anli": ("facebook/anli", None, "train_r1", "test_r1", nli("anli")),
    "rte": ("nyu-mll/glue", "rte", "train", "validation", rte),
    "qnli": ("nyu-mll/glue", "qnli", "train", "validation", qnli),
    "qqp": ("nyu-mll/glue", "qqp", "train", "validation", qqp),
    "paws": ("google-research-datasets/paws", "labeled_final", "train", "test", paws),
    "stsb": ("nyu-mll/glue", "stsb", "train", "validation", stsb),
    "boolq": ("google/boolq", None, "train", "validation", boolq),
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "train", "test", arc("arc_easy")),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "train", "test", arc("arc_challenge")),
    "openbookqa": ("allenai/openbookqa", "main", "train", "test", obqa),
    "commonsense_qa": ("tau/commonsense_qa", None, "train", "validation", csqa),
    "sciq": ("allenai/sciq", None, "train", "test", sciq),
    "race": ("ehovy/race", "all", "train", "test", race),
    "hellaswag": ("Rowan/hellaswag", None, "train", "validation", hellaswag),
    "imdb": ("stanfordnlp/imdb", None, "test", None, imdb),
    "trec": ("SetFit/TREC-QC", None, "train", "test", trec),
    "sms_spam": ("ucirvine/sms_spam", None, "train", None, sms_spam),
    "enron_spam": ("SetFit/enron_spam", None, "train", "test", enron_spam),
    "offensive": ("cardiffnlp/tweet_eval", "offensive", "train", "test", offensive),
    "hate": ("cardiffnlp/tweet_eval", "hate", "train", "test", hate),
    "civil_comments": ("google/civil_comments", None, "validation", "test", civil),
}

# Rows per source and stage at --scale 1. Stage 2 rows are built in hard mode.
STAGE1 = {"ag_news": 2000, "dbpedia": 2500, "yahoo": 2500, "clinc": 2000, "banking77": 1500,
          "emotion": 1500, "tweet_sentiment": 1500, "yelp": 2000, "sst5": 1000, "mnli": 3000,
          "rte": 800, "qnli": 1500, "qqp": 1500, "stsb": 1500, "boolq": 2500, "arc_easy": 1200,
          "arc_challenge": 600, "openbookqa": 1500, "commonsense_qa": 2000, "sciq": 2000,
          "race": 2500, "imdb": 1000, "trec": 1000}
STAGE2 = {"clinc": 2000, "banking77": 1500, "dbpedia": 1000, "yahoo": 1000, "ag_news": 500,
          "emotion": 500, "anli": 2500, "paws": 2000, "hellaswag": 2000, "mnli": 1000}
STAGE3_TASK = {"clinc": 1500, "banking77": 1500, "sms_spam": 1200, "enron_spam": 1200, "offensive": 1200,
               "hate": 1000, "civil_comments": 2000, "emotion": 600, "yelp": 800, "tweet_sentiment": 600,
               "ag_news": 800}
REPLAY = 0.4  # fraction of stage 3 that is general/hard data, as in CLM's post-training
DEV_PER_SOURCE = 40
TEST_PER_SOURCE = 150


def load(name):
    hf, cfg, tr, ev, _ = SOURCES[name]
    print(f"  {name}: {hf}{'/' + cfg if cfg else ''}", file=sys.stderr, flush=True)
    train = datasets.load_dataset(hf, cfg, split=tr)
    evs = datasets.load_dataset(hf, cfg, split=ev) if ev else None
    return train, evs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/public")
    ap.add_argument("--scale", type=float, default=1.0, help="multiplies every training row count")
    ap.add_argument("--test-scale", type=float, default=1.0, help="multiplies dev/test sizes")
    ap.add_argument("--extra", action="append", default=[], help="task JSONL added to stage 3 in full (repeatable)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(os.path.join(a.out, "test"), exist_ok=True)
    rng = random.Random(a.seed)
    names = sorted(set(STAGE1) | set(STAGE2) | set(STAGE3_TASK))
    q = lambda n: max(1, round(n * a.scale))
    nt = lambda n: max(4, round(n * a.test_scale))
    stage = {1: [], 2: [], 3: [], "replay": []}
    dev, bench = [], []
    counts = defaultdict(dict)
    print("downloading and converting:", file=sys.stderr)
    for name in names:
        train, evs = load(name)
        build = SOURCES[name][4]
        if name == "clinc":
            build = clinc(train.features["intent"].names)
        if name == "banking77":
            banking.names = intents(sorted(set(train["label_text"])))
        if name == "trec":
            trec.names = {snake(x): x for x in sorted(set(train["label_coarse_text"]))}
        keep = civil_keep if name == "civil_comments" else (lambda r: True)
        idx = list(range(len(train)))
        rng.shuffle(idx)
        # Eval rows: the eval split, or a held-out tail of the shuffled train pool.
        n_eval = nt(DEV_PER_SOURCE) + nt(TEST_PER_SOURCE)
        if evs is None:
            ev_rows = [train[i] for i in idx[-n_eval * 2:]]
            idx = idx[: -n_eval * 2]
        else:
            ev_idx = list(range(len(evs)))
            rng.shuffle(ev_idx)
            ev_rows = [evs[i] for i in ev_idx[: n_eval * 3]]
        pos = 0

        def take(n, hard):
            nonlocal pos
            out = []
            while len(out) < n and pos < len(idx):
                r = train[idx[pos]]
                pos += 1
                if keep(r):
                    x = build(r, rng, hard)
                    if x:
                        out.append(x)
            return out

        for s, quota in ((1, STAGE1), (2, STAGE2), (3, STAGE3_TASK)):
            if name in quota:
                got = take(q(quota[name]), hard=(s == 2))
                stage[s] += got
                counts[name][f"stage{s}"] = len(got)
        # Fresh rows for stage-3 replay, drawn from the general and hard mixes.
        if name in STAGE1 or name in STAGE2:
            share = (q(STAGE1.get(name, 0)) + q(STAGE2.get(name, 0)))
            stage["replay"] += take(share, hard=name in STAGE2 and name not in STAGE1)

        ev_recs = [x for x in (build(r, rng, name in STAGE2 and name not in STAGE1) for r in ev_rows if keep(r)) if x]
        dev += ev_recs[: nt(DEV_PER_SOURCE)]
        test = ev_recs[nt(DEV_PER_SOURCE): n_eval]
        with open(os.path.join(a.out, "test", f"{name}.jsonl"), "w") as f:
            for x in test:
                f.write(json.dumps(x) + "\n")
        counts[name]["dev"] = len(ev_recs[: nt(DEV_PER_SOURCE)])
        counts[name]["test"] = len(test)
        bench += test[:6] + [fan_out(x, rng) for x in test[6:12] if x["questions"]["q"]["t"] == "choice"]

        if name == "ag_news":
            # The fixed comparison set: random.seed(0) sample of 1,000 test rows.
            full = datasets.load_dataset("fancyzhx/ag_news", split="test")
            pick = random.Random(0).sample(range(len(full)), 1000)[: max(20, round(1000 * min(1.0, a.test_scale)))]
            with open(os.path.join(a.out, "test", "ag_news_test1k.jsonl"), "w") as f:
                for i in pick:
                    f.write(json.dumps(ag_news_fixed(full[i])) + "\n")

    extra = []
    for p in a.extra:
        with open(p) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        for r in rows:
            r.setdefault("source", os.path.splitext(os.path.basename(p))[0])
        extra += rows
        print(f"  extra: {len(rows)} records from {p}", file=sys.stderr)
    task = stage[3] + extra
    n_replay = round(len(task) * REPLAY / (1 - REPLAY))
    rng.shuffle(stage["replay"])
    replay = stage["replay"][:n_replay]
    if len(replay) < n_replay:
        print(f"  note: only {len(replay)} replay rows for {n_replay} wanted", file=sys.stderr)

    files = {"stage1_general.jsonl": stage[1], "stage2_hard.jsonl": stage[2],
             "stage3_task.jsonl": task + replay, "dev.jsonl": dev, "bench.jsonl": bench}
    for fname, rows in files.items():
        rng.shuffle(rows)
        with open(os.path.join(a.out, fname), "w") as f:
            for x in rows:
                f.write(json.dumps(x) + "\n")
    nq = lambda rows: sum(len(r["questions"]) for r in rows)
    summary = {
        "scale": a.scale, "seed": a.seed, "replay_fraction": REPLAY,
        "files": {k: {"records": len(v), "questions": nq(v)} for k, v in files.items()},
        "stage3": {"task_records": len(task), "extra_records": len(extra), "replay_records": len(replay)},
        "sources": {k: {"hf": SOURCES[k][0] + (f"/{SOURCES[k][1]}" if SOURCES[k][1] else ""), **v}
                    for k, v in counts.items()},
    }
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(summary, f, indent=1)
    for k, v in summary["files"].items():
        print(f"{k:22s} {v['records']:7d} records {v['questions']:7d} questions", file=sys.stderr)


if __name__ == "__main__":
    main()
