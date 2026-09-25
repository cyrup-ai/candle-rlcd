//! Training records, target parsing, augmentation, and dataset converters.
//!
//! One record per line of JSONL (the schema from the spec's §4):
//!
//! ```json
//! {"state": "...", "questions": {"topic": {"t": "choice", "ins": "...", "crit": {"a": "..", "b": ".."}}},
//!  "targets": {"topic": "b"}}
//! ```
//!
//! A target is an option key (`"b"`, `"true"`, `"2"`), an index, a boolean (`noul`), a
//! probability of `true` (`noul`), a list of per-option probabilities, or a `{key: prob}` map.

use std::io::{BufRead, Write};
use std::path::Path;

use anyhow::{bail, ensure, Context, Result};
use serde_json::{json, Map, Value};

use crate::sequence::{Criteria, QType, Question};

/// Small deterministic RNG (SplitMix64), so data order and augmentation are reproducible and
/// resumable from a saved seed.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n.max(1) as u64) as usize
    }

    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            v.swap(i, self.below(i + 1));
        }
    }
}

/// One (state, question, target) training row.
#[derive(Debug, Clone)]
pub struct Example {
    pub state: Value,
    pub question: Question,
    /// Target distribution over the question's options, in option order.
    pub target: Vec<f32>,
}

pub fn load_jsonl(path: &Path) -> Result<Vec<Example>> {
    let f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut out = Vec::new();
    for (i, line) in std::io::BufReader::new(f).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let rec: Value = serde_json::from_str(&line)
            .with_context(|| format!("{}:{}: bad JSON", path.display(), i + 1))?;
        out.extend(record_examples(&rec).with_context(|| format!("{}:{}", path.display(), i + 1))?);
    }
    Ok(out)
}

/// Splits a record into one example per question that has a target.
pub fn record_examples(rec: &Value) -> Result<Vec<Example>> {
    let state = rec.get("state").cloned().unwrap_or(Value::Null);
    let qs = rec
        .get("questions")
        .and_then(Value::as_object)
        .context("record needs a \"questions\" object")?;
    let targets = rec
        .get("targets")
        .and_then(Value::as_object)
        .context("record needs a \"targets\" object")?;
    let mut out = Vec::new();
    for (id, qv) in qs {
        let Some(tv) = targets.get(id) else { continue };
        let question = Question::from_json(qv).with_context(|| format!("question {id:?}"))?;
        let target = parse_target(&question, tv).with_context(|| format!("target {id:?}"))?;
        out.push(Example {
            state: state.clone(),
            question,
            target,
        });
    }
    Ok(out)
}

pub fn parse_target(q: &Question, v: &Value) -> Result<Vec<f32>> {
    let keys = q.option_keys();
    let k = keys.len();
    let one_hot = |i: usize| -> Result<Vec<f32>> {
        ensure!(i < k, "target index {i} out of range for {k} options");
        let mut t = vec![0f32; k];
        t[i] = 1.0;
        Ok(t)
    };
    let t = match v {
        Value::Bool(b) if q.t == QType::Noul => one_hot(*b as usize)?,
        Value::Number(n) if q.t == QType::Noul && n.is_f64() => {
            let p = n.as_f64().unwrap_or(0.0).clamp(0.0, 1.0) as f32;
            vec![1.0 - p, p]
        }
        Value::Number(n) => one_hot(
            n.as_u64()
                .context("index target must be a non-negative integer")? as usize,
        )?,
        Value::String(s) => match keys.iter().position(|x| x == s) {
            Some(i) => one_hot(i)?,
            None => bail!("target {s:?} is not one of {keys:?}"),
        },
        Value::Array(a) => {
            ensure!(
                a.len() == k,
                "soft target has {} entries for {k} options",
                a.len()
            );
            a.iter()
                .map(|x| {
                    x.as_f64()
                        .map(|f| f as f32)
                        .context("soft target must be numbers")
                })
                .collect::<Result<Vec<_>>>()?
        }
        Value::Object(m) => {
            let mut t = vec![0f32; k];
            for (key, p) in m {
                let i = keys
                    .iter()
                    .position(|x| x == key)
                    .with_context(|| format!("unknown option {key:?}"))?;
                t[i] = p.as_f64().context("probabilities must be numbers")? as f32;
            }
            t
        }
        _ => bail!("unsupported target {v}"),
    };
    let s: f32 = t.iter().sum();
    ensure!(
        s > 0.0 && t.iter().all(|x| *x >= 0.0),
        "target must be a non-negative distribution"
    );
    Ok(t.into_iter().map(|x| x / s).collect())
}

/// Anti-shortcut augmentation (spec §7.6), applied per example per epoch.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, clap::Args)]
pub struct AugmentConfig {
    /// Probability of shuffling a `choice` question's option order.
    #[arg(long, default_value_t = 1.0)]
    pub shuffle_options: f64,
    /// Probability of renaming `choice` labels to opaque codes (criteria are kept, and a label
    /// without a criterion becomes its own criterion), so the model reads options, not names.
    #[arg(long, default_value_t = 0.3)]
    pub opaque_labels: f64,
}

impl Default for AugmentConfig {
    fn default() -> Self {
        Self {
            shuffle_options: 1.0,
            opaque_labels: 0.3,
        }
    }
}

pub fn augment(ex: &Example, cfg: &AugmentConfig, rng: &mut Rng) -> Example {
    let Criteria::Choice(opts) = &ex.question.crit else {
        return ex.clone();
    };
    let mut idx: Vec<usize> = (0..opts.len()).collect();
    if rng.unit() < cfg.shuffle_options {
        rng.shuffle(&mut idx);
    }
    let opaque = rng.unit() < cfg.opaque_labels;
    let style = rng.below(3);
    let mut new = Vec::with_capacity(opts.len());
    for (pos, &i) in idx.iter().enumerate() {
        let (label, crit) = &opts[i];
        if opaque {
            let code = match style {
                0 => ((b'A' + (pos % 26) as u8) as char).to_string(),
                1 => format!("option {}", pos + 1),
                _ => format!("x{:02}", rng.below(100)),
            };
            let crit = match crit {
                Value::Null => Value::String(label.clone()),
                Value::String(s) if s.is_empty() => Value::String(label.clone()),
                c => c.clone(),
            };
            new.push((code, crit));
        } else {
            new.push((label.clone(), crit.clone()));
        }
    }
    // Keep labels unique (random codes can collide).
    for i in 0..new.len() {
        while new[..i].iter().any(|(l, _)| *l == new[i].0) {
            new[i].0.push('\'');
        }
    }
    let mut q = ex.question.clone();
    q.crit = Criteria::Choice(new);
    Example {
        state: ex.state.clone(),
        question: q,
        target: idx.iter().map(|&i| ex.target[i]).collect(),
    }
}

// ---------------------------------------------------------------------------------------------
// Converters

/// AG News labels in class-index order (1..=4 in the CSV), with the criteria we render.
const AG_NEWS: [(&str, &str); 4] = [
    (
        "world",
        "international news: politics, conflicts, elections and diplomacy",
    ),
    ("sports", "sports: games, teams, athletes and competitions"),
    (
        "business",
        "business: companies, markets, the economy and finance",
    ),
    (
        "science_tech",
        "science and technology: computing, the internet, research and space",
    ),
];

/// AG News CSV (`"class","title","description"`, as in the original release) to records.
pub fn convert_ag_news(input: &Path, out: &mut impl Write) -> Result<usize> {
    let text = std::fs::read_to_string(input)?;
    let crit: Map<String, Value> = AG_NEWS
        .iter()
        .map(|(k, v)| (k.to_string(), json!(v)))
        .collect();
    let mut n = 0;
    for (i, row) in parse_csv(&text).into_iter().enumerate() {
        ensure!(row.len() >= 3, "line {}: expected 3 columns", i + 1);
        let class: usize = row[0]
            .trim()
            .parse()
            .with_context(|| format!("line {}: class", i + 1))?;
        ensure!((1..=4).contains(&class), "line {}: class {class}", i + 1);
        let state = format!("{}\n{}", clean_ag(&row[1]), clean_ag(&row[2]));
        let rec = json!({
            "state": state,
            "questions": {"topic": {"t": "choice", "ins": "Which section of a news site does this article belong in?", "crit": crit}},
            "targets": {"topic": AG_NEWS[class - 1].0},
        });
        writeln!(out, "{rec}")?;
        n += 1;
    }
    Ok(n)
}

/// AG News text uses `\` for line breaks and `#36;`-style escapes for some characters.
fn clean_ag(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s.trim();
    while let Some(i) = rest.find(['\\', '#']) {
        out.push_str(&rest[..i]);
        rest = &rest[i..];
        if let Some(r) = rest.strip_prefix('\\') {
            out.push(' ');
            rest = r;
            continue;
        }
        let code = rest[1..].find(';').and_then(|j| {
            let n: u32 = rest[1..1 + j].parse().ok()?;
            Some((char::from_u32(n)?, j + 2))
        });
        match code {
            Some((c, len)) => {
                out.push(c);
                rest = &rest[len..];
            }
            None => {
                out.push('#');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// BoolQ JSONL (`question`, `passage`, `answer`, optional `title`) to `noul` records.
pub fn convert_boolq(input: &Path, out: &mut impl Write) -> Result<usize> {
    let f = std::fs::File::open(input)?;
    let mut n = 0;
    for line in std::io::BufReader::new(f).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(&line)?;
        let q = v["question"]
            .as_str()
            .context("question")?
            .trim()
            .to_string();
        let passage = v["passage"].as_str().context("passage")?;
        let state = match v.get("title").and_then(Value::as_str) {
            Some(t) => format!("{t}\n{passage}"),
            None => passage.to_string(),
        };
        let mut ins = q.clone();
        if let Some(c) = ins.get(..1) {
            ins = c.to_uppercase() + &ins[1..];
        }
        if !ins.ends_with('?') {
            ins.push('?');
        }
        let rec = json!({
            "state": state,
            "questions": {"answer": {"t": "noul", "ins": ins}},
            "targets": {"answer": v["answer"].as_bool().context("answer")?},
        });
        writeln!(out, "{rec}")?;
        n += 1;
    }
    Ok(n)
}

/// Minimal RFC 4180 CSV parser (quoted fields, doubled quotes, newlines inside quotes).
fn parse_csv(text: &str) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    let mut row = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match (quoted, c) {
            (true, '"') if chars.peek() == Some(&'"') => {
                field.push('"');
                chars.next();
            }
            (true, '"') => quoted = false,
            (true, c) => field.push(c),
            (false, '"') => quoted = true,
            (false, ',') => row.push(std::mem::take(&mut field)),
            (false, '\n') => {
                row.push(std::mem::take(&mut field));
                rows.push(std::mem::take(&mut row));
            }
            (false, '\r') => {}
            (false, c) => field.push(c),
        }
    }
    if !field.is_empty() || !row.is_empty() {
        row.push(field);
        rows.push(row);
    }
    rows
}

/// Synthetic multi-question records that can only be answered by reading the state and the
/// options: for smoke tests and CI where no dataset download is possible.
pub fn synthetic(n: usize, seed: u64, out: &mut impl Write) -> Result<()> {
    const TOPICS: [(&str, &str); 6] = [
        ("billing", "charges, invoices and refunds"),
        ("shipping", "delivery, tracking and damaged parcels"),
        ("account", "logins, passwords and profile settings"),
        ("security", "fraud, phishing and suspicious activity"),
        ("product", "features, bugs and how-to questions"),
        ("feedback", "praise, complaints and suggestions"),
    ];
    const MSG: [&str; 6] = [
        "I was charged twice on my last invoice and want a refund",
        "my parcel has not arrived and the tracking page is stuck",
        "I cannot log in and the password reset email never comes",
        "someone tried to phish me with a fake email from your company",
        "the export button crashes the app every time I press it",
        "your support team was wonderful, thank you so much",
    ];
    const CITIES: [&str; 5] = ["Paris", "Lagos", "Osaka", "Lima", "Oslo"];
    const LEVELS: [&str; 4] = ["low", "medium", "high", "critical"];
    const TONE: [&str; 4] = [
        "calm",
        "a bit annoyed",
        "angry",
        "furious and threatening to leave",
    ];
    let mut rng = Rng(seed);
    for _ in 0..n {
        let topic = rng.below(TOPICS.len());
        let city = rng.below(CITIES.len());
        let level = rng.below(LEVELS.len());
        let state = format!(
            "Customer from {} writes: {}. They sound {}.",
            CITIES[city], MSG[topic], TONE[level]
        );
        // A random subset of 3-6 topics that always includes the right one.
        let mut pool: Vec<usize> = (0..TOPICS.len()).filter(|&i| i != topic).collect();
        rng.shuffle(&mut pool);
        let k = 2 + rng.below(4);
        let mut opts = vec![topic];
        opts.extend(pool.into_iter().take(k));
        rng.shuffle(&mut opts);
        let crit: Map<String, Value> = opts
            .iter()
            .map(|&i| (TOPICS[i].0.to_string(), json!(TOPICS[i].1)))
            .collect();
        let asked_city = if rng.unit() < 0.5 {
            city
        } else {
            rng.below(CITIES.len())
        };
        let rec = json!({
            "state": state,
            "questions": {
                "team": {"t": "choice", "ins": "Which team should handle this message?", "crit": crit},
                "city": {"t": "noul", "ins": format!("Is the customer based in {}?", CITIES[asked_city])},
                "urgency": {"t": "score", "ins": "How urgent is this?", "crit": LEVELS},
            },
            "targets": {"team": TOPICS[topic].0, "city": asked_city == city, "urgency": level},
        });
        writeln!(out, "{rec}")?;
    }
    Ok(())
}

/// Trains a byte-level BPE tokenizer (the special tokens and pre-tokenizer ModernBERT uses) on
/// every state, instruction and option text in `files`.
pub fn train_tokenizer(files: &[std::path::PathBuf], out: &Path, vocab_size: usize) -> Result<()> {
    use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
    use tokenizers::models::TrainerWrapper;
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;
    use tokenizers::{AddedToken, Tokenizer};

    let mut texts = Vec::new();
    for f in files {
        for ex in load_jsonl(f)? {
            texts.push(crate::sequence::serialize_state(&ex.state));
            texts.push(format!(
                "{} question: {}",
                ex.question.t.name(),
                ex.question.ins
            ));
            texts.extend(ex.question.render_options());
        }
    }
    const SPECIALS: [&str; 5] = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"];
    let bpe = BPE::builder()
        .unk_token("[UNK]".into())
        .build()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut tok = Tokenizer::new(bpe);
    tok.with_pre_tokenizer(Some(ByteLevel::default().add_prefix_space(false)));
    tok.with_decoder(Some(ByteLevel::default()));
    let mut trainer: TrainerWrapper = BpeTrainerBuilder::new()
        .vocab_size(vocab_size)
        .min_frequency(2)
        .show_progress(false)
        .special_tokens(
            SPECIALS
                .iter()
                .map(|s| AddedToken::from(*s, true))
                .collect(),
        )
        .initial_alphabet(ByteLevel::alphabet().into_iter().collect())
        .build()
        .into();
    tok.train(&mut trainer, texts.into_iter())
        .map_err(|e| anyhow::anyhow!("training tokenizer: {e}"))?;
    std::fs::create_dir_all(out)?;
    tok.save(out.join("tokenizer.json"), false)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let cfg = json!({"cls_token": "[CLS]", "sep_token": "[SEP]", "pad_token": "[PAD]",
                     "mask_token": "[MASK]", "unk_token": "[UNK]"});
    std::fs::write(
        out.join("tokenizer_config.json"),
        serde_json::to_string_pretty(&cfg)?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn targets_parse_in_every_form() {
        let q = Question::from_json(
            &json!({"t": "choice", "ins": "x", "crit": {"a": null, "b": null, "c": null}}),
        )
        .unwrap();
        assert_eq!(parse_target(&q, &json!("b")).unwrap(), vec![0.0, 1.0, 0.0]);
        assert_eq!(parse_target(&q, &json!(2)).unwrap(), vec![0.0, 0.0, 1.0]);
        assert_eq!(
            parse_target(&q, &json!([1, 1, 2])).unwrap(),
            vec![0.25, 0.25, 0.5]
        );
        assert_eq!(
            parse_target(&q, &json!({"c": 1.0})).unwrap(),
            vec![0.0, 0.0, 1.0]
        );
        assert!(parse_target(&q, &json!("d")).is_err());
        let n = Question::from_json(&json!({"t": "noul", "ins": "x"})).unwrap();
        assert_eq!(parse_target(&n, &json!(true)).unwrap(), vec![0.0, 1.0]);
        assert_eq!(parse_target(&n, &json!(0.25)).unwrap(), vec![0.75, 0.25]);
    }

    #[test]
    fn augmentation_keeps_target_on_the_same_option() {
        let rec = json!({"state": "s", "questions": {"q": {"t": "choice", "ins": "x",
            "crit": {"a": "first", "b": "second", "c": null}}}, "targets": {"q": "c"}});
        let ex = record_examples(&rec).unwrap().remove(0);
        let mut rng = Rng(7);
        let cfg = AugmentConfig {
            shuffle_options: 1.0,
            opaque_labels: 1.0,
        };
        for _ in 0..20 {
            let a = augment(&ex, &cfg, &mut rng);
            let Criteria::Choice(opts) = &a.question.crit else {
                panic!()
            };
            let hot = a.target.iter().position(|&x| x == 1.0).unwrap();
            // "c" had no criterion, so its old label became the criterion.
            assert_eq!(opts[hot].1, json!("c"));
        }
    }

    #[test]
    fn ag_news_text_is_cleaned() {
        assert_eq!(
            clean_ag("a second\\team won  #36;10 million, it#39;s #x"),
            "a second team won $10 million, it's #x"
        );
    }

    #[test]
    fn csv_handles_quotes() {
        let rows =
            parse_csv("\"3\",\"A \"\"quoted\"\" title\",\"line, with comma\"\n\"1\",\"b\",\"c\"\n");
        assert_eq!(rows[0], vec!["3", "A \"quoted\" title", "line, with comma"]);
        assert_eq!(rows.len(), 2);
    }
}
