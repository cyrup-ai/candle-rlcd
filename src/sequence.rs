//! Question schema and token-sequence construction, matching laya's `build_sequence`.
//!
//! ```text
//! [CLS] "<t> question: <ins>" [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] <state> [SEP]
//! ```

use anyhow::{bail, ensure, Context, Result};
use serde_json::{Map, Value};
use tokenizers::Tokenizer;

/// Per-option token cap (laya: `truncation=True, max_length=48`).
const OPTION_MAX_TOKENS: usize = 48;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QType {
    Choice = 0,
    Score = 1,
    Noul = 2,
}

impl QType {
    pub fn name(self) -> &'static str {
        match self {
            QType::Choice => "choice",
            QType::Score => "score",
            QType::Noul => "noul",
        }
    }
}

/// One typed question, as in the Jev / laya request format.
#[derive(Debug, Clone)]
pub struct Question {
    pub t: QType,
    pub ins: String,
    pub crit: Criteria,
    /// `noul` only: display labels for `false` / `true`.
    pub labels: Option<(String, String)>,
}

#[derive(Debug, Clone)]
pub enum Criteria {
    /// `choice`: ordered `label -> description` (description may be null/empty).
    Choice(Vec<(String, Value)>),
    /// `score`: one criterion per level.
    Score(Vec<Value>),
    /// `noul`: optional `false` / `true` criteria.
    Noul {
        false_crit: Option<Value>,
        true_crit: Option<Value>,
    },
}

impl Question {
    pub fn from_json(v: &Value) -> Result<Self> {
        let o = v.as_object().context("question must be an object")?;
        // laya's short keys (`t` / `ins` / `crit`) or Jev's (`type` / `instructions` / `criteria`).
        let field = |short: &str, long: &str| o.get(short).or_else(|| o.get(long));
        let t = match field("t", "type").and_then(Value::as_str) {
            Some("choice") => QType::Choice,
            Some("score") => QType::Score,
            Some("noul") => QType::Noul,
            other => bail!("unknown question type {other:?}"),
        };
        let ins = match field("ins", "instructions") {
            Some(Value::String(s)) => s.clone(),
            Some(v) => python_str(v),
            None => String::new(),
        };
        let crit = field("crit", "criteria").cloned().unwrap_or(Value::Null);
        ensure!(
            t == QType::Noul || !o.contains_key("labels"),
            "labels is only supported for noul questions"
        );
        let crit = match t {
            QType::Choice => {
                let m = crit.as_object().context("choice crit must be an object")?;
                ensure!(!m.is_empty(), "choice needs at least one option");
                Criteria::Choice(m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            }
            QType::Score => {
                let a = crit.as_array().context("score crit must be a list")?;
                ensure!(!a.is_empty(), "score needs at least one level");
                Criteria::Score(a.clone())
            }
            QType::Noul => {
                let m = crit.as_object().cloned().unwrap_or_default();
                Criteria::Noul {
                    false_crit: m.get("false").cloned(),
                    true_crit: m.get("true").cloned(),
                }
            }
        };
        let labels = match o.get("labels") {
            None | Some(Value::Null) => None,
            Some(Value::Object(m)) => {
                let get = |k: &str| {
                    m.get(k)
                        .and_then(Value::as_str)
                        .map(|s| s.trim().to_string())
                };
                match (get("false"), get("true")) {
                    (Some(f), Some(t)) if m.len() == 2 && !f.is_empty() && !t.is_empty() && f != t => {
                        Some((f, t))
                    }
                    _ => bail!("noul labels must map exactly 'false' and 'true' to distinct non-empty strings"),
                }
            }
            _ => bail!("noul labels must be an object"),
        };
        Ok(Self {
            t,
            ins,
            crit,
            labels,
        })
    }

    /// Option labels in index order, as returned in answers.
    pub fn option_keys(&self) -> Vec<String> {
        match &self.crit {
            Criteria::Choice(c) => c.iter().map(|(k, _)| k.clone()).collect(),
            Criteria::Score(c) => (0..c.len()).map(|i| i.to_string()).collect(),
            Criteria::Noul { .. } => vec!["false".into(), "true".into()],
        }
    }

    /// Rendered option texts in label-index order (`render_options` in laya).
    pub fn render_options(&self) -> Vec<String> {
        match &self.crit {
            Criteria::Choice(c) => c
                .iter()
                .map(|(k, v)| {
                    if is_blank(v) {
                        k.clone()
                    } else {
                        format!("{k}: {}", render_criterion(v))
                    }
                })
                .collect(),
            Criteria::Score(c) => c
                .iter()
                .enumerate()
                .map(|(i, v)| format!("level {i}: {}", render_criterion(v)))
                .collect(),
            Criteria::Noul {
                false_crit,
                true_crit,
            } => {
                let (fl, tl) = self
                    .labels
                    .clone()
                    .unwrap_or_else(|| ("false".into(), "true".into()));
                let render = |c: &Option<Value>, dflt: &str| match c {
                    Some(v) if !is_blank(v) => render_criterion(v),
                    _ => dflt.to_string(),
                };
                vec![
                    format!(
                        "{fl}: {}",
                        render(false_crit, "no, the statement does not hold")
                    ),
                    format!("{tl}: {}", render(true_crit, "yes, the statement holds")),
                ]
            }
        }
    }
}

fn is_blank(v: &Value) -> bool {
    matches!(v, Value::Null) || matches!(v, Value::String(s) if s.is_empty())
}

/// Strings pass through; anything structured becomes compact JSON with Python's default
/// separators (`", "`, `": "`), matching laya's `render_criterion`.
pub fn render_criterion(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        v => python_json(v),
    }
}

/// `serialize_state`: strings pass through, everything else is `json.dumps(ensure_ascii=False)`.
pub fn serialize_state(v: &Value) -> String {
    render_criterion(v)
}

fn python_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Null => "None".into(),
        v => python_json(v),
    }
}

/// `json.dumps(v, ensure_ascii=False)` with Python's default separators.
///
/// Floats use Rust's shortest round-trip formatting, which matches Python's `repr` for ordinary
/// values but differs in exponent style (`1e-5` vs `1e-05`); states with such floats may tokenize
/// slightly differently from laya.
pub fn python_json(v: &Value) -> String {
    let mut out = String::new();
    write_json(v, &mut out);
    out
}

fn write_json(v: &Value, out: &mut String) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            if let Some(f) = n.as_f64().filter(|_| n.is_f64()) {
                if f.fract() == 0.0 && f.abs() < 1e16 {
                    out.push_str(&format!("{f:.1}"));
                } else {
                    out.push_str(&n.to_string());
                }
            } else {
                out.push_str(&n.to_string());
            }
        }
        Value::String(s) => write_str(s, out),
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_json(x, out);
            }
            out.push(']');
        }
        Value::Object(m) => write_obj(m, out),
    }
}

fn write_obj(m: &Map<String, Value>, out: &mut String) {
    out.push('{');
    for (i, (k, x)) in m.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        write_str(k, out);
        out.push_str(": ");
        write_json(x, out);
    }
    out.push('}');
}

fn write_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Special tokens the sequence format needs.
#[derive(Debug, Clone)]
pub struct Specials {
    pub cls: u32,
    pub sep: u32,
    pub mask: u32,
    pub pad: u32,
    pub mask_token: String,
}

impl Specials {
    /// Resolve from `tokenizer_config.json` token strings when given, else the BERT defaults.
    pub fn resolve(tok: &Tokenizer, tokenizer_config: Option<&Value>) -> Result<Self> {
        let name = |key: &str, dflt: &str| -> String {
            match tokenizer_config.and_then(|c| c.get(key)) {
                Some(Value::String(s)) => s.clone(),
                Some(Value::Object(o)) => o
                    .get("content")
                    .and_then(Value::as_str)
                    .unwrap_or(dflt)
                    .to_string(),
                _ => dflt.to_string(),
            }
        };
        let id = |s: &str| {
            tok.token_to_id(s)
                .with_context(|| format!("special token {s:?} missing from tokenizer"))
        };
        let mask_token = name("mask_token", "[MASK]");
        Ok(Self {
            cls: id(&name("cls_token", "[CLS]"))?,
            sep: id(&name("sep_token", "[SEP]"))?,
            mask: id(&mask_token)?,
            pad: id(&name("pad_token", "[PAD]"))?,
            mask_token,
        })
    }
}

/// One encoded question row.
#[derive(Debug, Clone)]
pub struct Encoded {
    pub ids: Vec<u32>,
    pub markers: Vec<usize>,
    pub qtype: QType,
    /// Prefix layout: number of leading state tokens (`[CLS] state [SEP]`); 0 in laya's layout.
    pub prefix_len: usize,
}

pub fn encode_text(tok: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    let enc = tok
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    Ok(enc.get_ids().to_vec())
}

/// Tokenize the state once for all questions (`state_ids` in laya).
pub fn encode_state(tok: &Tokenizer, sp: &Specials, state: &Value) -> Result<Vec<u32>> {
    encode_text(tok, &serialize_state(state).replace(&sp.mask_token, " "))
}

/// Tokenized question head: `"<t> question: <ins>"` and each option as `[MASK] opt`, with
/// laya's budget rules applied (options capped at 48 tokens, and cut to share `head_max_len`).
fn question_tokens(
    tok: &Tokenizer,
    sp: &Specials,
    q: &Question,
    head_max_len: usize,
) -> Result<(Vec<u32>, Vec<Vec<u32>>)> {
    let opts = q.render_options();
    let ins = q.ins.replace(&sp.mask_token, " ");
    let mut head_ids = encode_text(tok, &format!("{} question: {}", q.t.name(), ins))?;
    let mut opt_ids = Vec::with_capacity(opts.len());
    for o in &opts {
        let mut t = encode_text(tok, &format!(" {}", o.replace(&sp.mask_token, " ")))?;
        t.truncate(OPTION_MAX_TOKENS);
        let mut v = Vec::with_capacity(t.len() + 1);
        v.push(sp.mask);
        v.extend(t);
        opt_ids.push(v);
    }
    let total = |o: &[Vec<u32>]| o.iter().map(Vec::len).sum::<usize>() as isize;
    let mut opt_budget = head_max_len as isize - total(&opt_ids);
    if opt_budget < 16 {
        let per = 4.max(head_max_len.saturating_sub(16) / opt_ids.len().max(1));
        for o in &mut opt_ids {
            o.truncate(per);
        }
        opt_budget = head_max_len as isize - total(&opt_ids);
    }
    head_ids.truncate(8.max(opt_budget.max(0) as usize));
    Ok((head_ids, opt_ids))
}

/// A question's tokens with no budget applied: the head `"<t> question: <ins>"` and each
/// option as `[MASK] opt`, whole.
#[derive(Debug, Clone)]
pub struct QuestionParts {
    pub head: Vec<u32>,
    pub opts: Vec<Vec<u32>>,
}

impl QuestionParts {
    /// Tokens of `[CLS] head [SEP] opts.. [SEP]` for the options in `which`.
    pub fn len(&self, which: &[usize]) -> usize {
        self.head.len() + 3 + which.iter().map(|&i| self.opts[i].len()).sum::<usize>()
    }
}

pub fn question_parts(tok: &Tokenizer, sp: &Specials, q: &Question) -> Result<QuestionParts> {
    let ins = q.ins.replace(&sp.mask_token, " ");
    let head = encode_text(tok, &format!("{} question: {}", q.t.name(), ins))?;
    let opts = q
        .render_options()
        .iter()
        .map(|o| {
            let t = encode_text(tok, &format!(" {}", o.replace(&sp.mask_token, " ")))?;
            let mut v = Vec::with_capacity(t.len() + 1);
            v.push(sp.mask);
            v.extend(t);
            Ok(v)
        })
        .collect::<Result<_>>()?;
    Ok(QuestionParts { head, opts })
}

/// Appends `[CLS] head [SEP] [MASK] opt0 ... [SEP]` and returns the marker positions.
fn push_question<H, O>(ids: &mut Vec<u32>, sp: &Specials, head: H, opts: O) -> Vec<usize>
where
    H: AsRef<[u32]>,
    O: IntoIterator,
    O::Item: AsRef<[u32]>,
{
    ids.push(sp.cls);
    ids.extend_from_slice(head.as_ref());
    ids.push(sp.sep);
    let mut markers = Vec::new();
    for o in opts {
        markers.push(ids.len());
        ids.extend_from_slice(o.as_ref());
    }
    ids.push(sp.sep);
    markers
}

/// laya's layout row from fitted parts: `[CLS] head [SEP] opts.. [SEP] state [SEP]`, with
/// nothing cut.
pub fn joint_row(
    sp: &Specials,
    qtype: QType,
    head: &[u32],
    opts: &[&[u32]],
    state: &[u32],
) -> Encoded {
    let mut ids = Vec::new();
    let markers = push_question(&mut ids, sp, head, opts);
    ids.extend_from_slice(state);
    ids.push(sp.sep);
    Encoded {
        ids,
        markers,
        qtype,
        prefix_len: 0,
    }
}

/// Prefix layout row from fitted parts: `prefix` (`[CLS] state [SEP]`) then the question.
pub fn prefix_row(
    sp: &Specials,
    qtype: QType,
    head: &[u32],
    opts: &[&[u32]],
    prefix: &[u32],
) -> Encoded {
    let mut ids = prefix.to_vec();
    let markers = push_question(&mut ids, sp, head, opts);
    Encoded {
        ids,
        markers,
        qtype,
        prefix_len: prefix.len(),
    }
}

/// laya's `build_sequence` with pre-tokenized state ids.
pub fn build_sequence(
    tok: &Tokenizer,
    sp: &Specials,
    q: &Question,
    state_ids: &[u32],
    max_len: usize,
    head_max_len: usize,
    truncate_left: bool,
) -> Result<Encoded> {
    let (head_ids, opt_ids) = question_tokens(tok, sp, q, head_max_len)?;
    let k = opt_ids.len();
    let mut ids = Vec::with_capacity(max_len);
    let mut markers = push_question(&mut ids, sp, head_ids, opt_ids);
    let room = max_len.saturating_sub(ids.len() + 1);
    ids.extend_from_slice(truncate_state(state_ids, room, truncate_left));
    ids.push(sp.sep);
    ids.truncate(max_len);
    markers.retain(|&m| m < max_len);
    ensure!(
        markers.len() == k,
        "question options exceed head_max_len={head_max_len}"
    );
    Ok(Encoded {
        ids,
        markers,
        qtype: q.t,
        prefix_len: 0,
    })
}

fn truncate_state(state_ids: &[u32], room: usize, truncate_left: bool) -> &[u32] {
    if truncate_left {
        &state_ids[state_ids.len().saturating_sub(room)..]
    } else {
        &state_ids[..room.min(state_ids.len())]
    }
}

/// Prefix layout's shared state part, `[CLS] state [SEP]`, sized so any question fits after it.
pub fn build_state_prefix(
    sp: &Specials,
    state_ids: &[u32],
    max_len: usize,
    head_max_len: usize,
    truncate_left: bool,
) -> Vec<u32> {
    // The question part is at most `head_max_len` tokens plus [CLS], [SEP], [SEP].
    let room = max_len.saturating_sub(head_max_len + 3 + 2);
    let mut ids = Vec::with_capacity(room + 2);
    ids.push(sp.cls);
    ids.extend_from_slice(truncate_state(state_ids, room, truncate_left));
    ids.push(sp.sep);
    ids
}

/// Prefix layout row: `[CLS] state [SEP] [CLS] head [SEP] [MASK] opt0 ... [SEP]`.
///
/// The state comes first so its tokens and positions don't depend on the question: a model
/// trained with state tokens blind to the question part can encode the state once per request.
pub fn build_prefix_sequence(
    tok: &Tokenizer,
    sp: &Specials,
    q: &Question,
    state_prefix: &[u32],
    head_max_len: usize,
) -> Result<Encoded> {
    let (head_ids, opt_ids) = question_tokens(tok, sp, q, head_max_len)?;
    let mut ids = state_prefix.to_vec();
    let markers = push_question(&mut ids, sp, head_ids, opt_ids);
    Ok(Encoded {
        ids,
        markers,
        qtype: q.t,
        prefix_len: state_prefix.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn python_json_matches_dumps() {
        let v = json!({"a": [1, 2.5, "x\"y"], "b": {"c": null, "d": true}, "e": 3.0});
        assert_eq!(
            python_json(&v),
            r#"{"a": [1, 2.5, "x\"y"], "b": {"c": null, "d": true}, "e": 3.0}"#
        );
    }

    #[test]
    fn render_options_by_type() {
        let q = Question::from_json(&json!({"t": "choice", "ins": "x",
            "crit": {"a": "first", "b": null, "c": {"k": 1}}}))
        .unwrap();
        assert_eq!(q.render_options(), vec!["a: first", "b", r#"c: {"k": 1}"#]);
        let q = Question::from_json(&json!({"t": "noul", "ins": "x",
            "labels": {"false": "no", "true": "yes"}}))
        .unwrap();
        assert_eq!(
            q.render_options(),
            vec![
                "no: no, the statement does not hold",
                "yes: yes, the statement holds"
            ]
        );
        let q =
            Question::from_json(&json!({"t": "score", "ins": "x", "crit": ["lo", "hi"]})).unwrap();
        assert_eq!(q.render_options(), vec!["level 0: lo", "level 1: hi"]);
    }
}
