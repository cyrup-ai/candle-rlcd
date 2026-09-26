//! Request pipeline for serving: gets the whole question and the whole state in front of the
//! model, and says so when it can't.
//!
//! laya's own sequence builder ([`crate::sequence::build_sequence`], kept as-is for training
//! and PyTorch parity) gives the question head 192 tokens, each option 48, and the state
//! whatever is left of 512. Anything past that is dropped without a trace, so long
//! instructions lose the actual question, long options lose the words that tell them apart,
//! more than ~80 options are refused, and a long state loses its end. Here instead:
//!
//! - **The question comes first.** Instructions and options are kept whole. A row may grow
//!   past `max_len` (up to the encoder's position limit) so the state keeps at least
//!   `min_state` tokens next to a long question.
//! - **Large option lists run in batches.** A Choice whose options take more than laya's
//!   question budget is split into batches that fit. Each batch's leaders advance until one
//!   row holds every finalist. Logits are merged by chaining each eliminated option's logit
//!   difference to a leader of its batch (Luce's choice axiom: the odds between two options
//!   don't depend on what else is on the list), so every option still gets a probability.
//! - **A long state runs in chunks.** A state longer than the room left next to the question
//!   is split into overlapping chunks. Every question is asked of every chunk, and the answers
//!   are combined per question type ([`combine_chunks`]).
//! - **Truncation is reported.** Only a question too long for the encoder's positions is
//!   still cut; [`Plan::truncation`] says by how much, per question.

use anyhow::{ensure, Result};
use serde_json::Value;

use crate::agent::RowOutput;
use crate::model::Layout;
use crate::sequence::{
    encode_state, joint_row, prefix_row, question_parts, Encoded, QType, Question, QuestionParts,
};
use crate::Laya;

/// Token budgets for fitting a request into rows.
#[derive(Debug, Clone, PartialEq)]
pub struct Budget {
    /// Row length the model was trained on (`max_len` in `rl_agent_config.json`). Rows only
    /// grow past it when a question needs the room.
    pub max_len: usize,
    /// Hard ceiling on a row: the encoder's position limit.
    pub max_window: usize,
    /// Option tokens per row before a Choice is split into batches (laya's `head_max_len`,
    /// the question size it was trained on).
    pub option_budget: usize,
    /// State tokens every row keeps, however long the question.
    pub min_state: usize,
    /// Tokens shared by consecutive state chunks, so a fact on a boundary is whole in one.
    pub overlap: usize,
}

impl Budget {
    pub fn for_model(laya: &Laya) -> Self {
        let max_len = laya.cfg.max_len;
        Self {
            max_len,
            max_window: laya.encoder_cfg.max_position_embeddings.max(max_len),
            option_budget: laya.cfg.head_max_len,
            min_state: (max_len / 4).max(16),
            overlap: max_len / 8,
        }
    }
}

/// Tokens a question lost to the encoder's position limit. All zero unless its instructions
/// and options alone come close to `max_window`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Truncation {
    /// Instruction tokens cut (from the middle: the start and the end are kept).
    pub instructions: usize,
    /// Option tokens cut, from the ends of the longest options.
    pub options: usize,
    /// The question's tokens before cutting, and the most a row had room for.
    pub question_tokens: usize,
    pub limit: usize,
}

impl Truncation {
    pub fn any(&self) -> bool {
        self.instructions + self.options > 0
    }
}

/// One question's progress: the parts it is asked with, and its option tournament.
struct QPlan {
    q: Question,
    parts: QuestionParts,
    trunc: Truncation,
    /// Batches of option indices per round; the last round is running or done.
    rounds: Vec<Vec<Vec<usize>>>,
    /// Combined (over state chunks) logits and act probabilities per batch, per round.
    results: Vec<Vec<(Vec<f32>, Vec<f32>)>>,
}

impl QPlan {
    fn done(&self) -> bool {
        self.results.len() == self.rounds.len() && self.rounds.last().is_some_and(|r| r.len() == 1)
    }
}

/// A request's rows, run round by round:
///
/// ```ignore
/// let mut plan = Plan::new(&laya, &budget, &state, questions)?;
/// loop {
///     let rows = plan.rows();
///     if rows.is_empty() { break }
///     plan.feed(laya.forward(&rows)?)?;
/// }
/// let outs = plan.finish();
/// ```
///
/// Round one asks every question; only Choices whose options ran in batches need more.
pub struct Plan {
    qs: Vec<QPlan>,
    /// Each chunk of the state, as the tokens a row carries (in the prefix layout, the whole
    /// `[CLS] chunk [SEP]` prefix).
    chunks: Vec<Vec<u32>>,
    layout: Layout,
    budget: Budget,
    specials: crate::sequence::Specials,
    /// `(question, batch)` behind each row handed out by the last [`Self::rows`], per chunk.
    pending: Vec<(usize, usize)>,
    state_tokens: usize,
    input_tokens: usize,
}

impl Plan {
    pub fn new(
        laya: &Laya,
        budget: &Budget,
        state: &Value,
        questions: Vec<Question>,
    ) -> Result<Self> {
        let sp = &laya.specials;
        let state_ids = encode_state(&laya.tokenizer, sp, state)?;
        let mut qs = Vec::with_capacity(questions.len());
        for q in questions {
            let parts = question_parts(&laya.tokenizer, sp, &q)?;
            ensure!(!parts.opts.is_empty(), "question has no options");
            let batches = first_round(&q, &parts, budget);
            qs.push(QPlan {
                q,
                parts,
                trunc: Truncation::default(),
                rounds: vec![batches],
                results: vec![],
            });
        }
        // Row overhead besides the question part (`[CLS] head [SEP] opts [SEP]`, counted by
        // `QuestionParts::len`): the state's closing [SEP], plus its opening [CLS] in the
        // prefix layout.
        let overhead = match laya.cfg.layout {
            Layout::Laya => 1,
            Layout::Prefix => 2,
        };
        let floor = budget.min_state.min(state_ids.len());
        // Cut any question that can't fit next to `floor` state tokens within `max_window`.
        let cap = budget.max_window.saturating_sub(overhead + floor);
        for p in &mut qs {
            p.trunc = fit_question(&mut p.parts, &p.rounds[0], cap);
        }
        let longest = qs
            .iter()
            .map(|p| {
                p.rounds[0]
                    .iter()
                    .map(|b| p.parts.len(b))
                    .max()
                    .unwrap_or(0)
            })
            .max()
            .unwrap_or(0);
        let room = budget
            .max_len
            .saturating_sub(longest + overhead)
            .max(floor)
            .max(1);
        let spans = chunk_spans(state_ids.len(), room, budget.overlap);
        let chunks = spans
            .iter()
            .map(|&(a, b)| match laya.cfg.layout {
                Layout::Laya => state_ids[a..b].to_vec(),
                Layout::Prefix => {
                    let mut v = Vec::with_capacity(b - a + 2);
                    v.push(sp.cls);
                    v.extend_from_slice(&state_ids[a..b]);
                    v.push(sp.sep);
                    v
                }
            })
            .collect();
        Ok(Self {
            qs,
            chunks,
            layout: laya.cfg.layout,
            budget: budget.clone(),
            specials: sp.clone(),
            pending: vec![],
            state_tokens: state_ids.len(),
            input_tokens: 0,
        })
    }

    /// State chunks each question is asked against (1 when the state fits).
    pub fn state_chunks(&self) -> usize {
        self.chunks.len()
    }

    /// Tokens in the serialized state.
    pub fn state_tokens(&self) -> usize {
        self.state_tokens
    }

    pub fn truncation(&self, question: usize) -> Truncation {
        self.qs[question].trunc
    }

    /// Batches per round for a question (`[k]` for a question asked in one row).
    pub fn rounds(&self, question: usize) -> Vec<usize> {
        self.qs[question].rounds.iter().map(Vec::len).collect()
    }

    /// Tokens the model has read so far (each prefix-layout state chunk counted once).
    pub fn input_tokens(&self) -> usize {
        self.input_tokens
    }

    /// Tokens the next [`Self::rows`] call would hand out, without building them.
    pub fn next_tokens(&self) -> usize {
        let per_chunk: usize = self
            .unfinished()
            .map(|(i, b)| {
                self.qs[i]
                    .parts
                    .len(&self.qs[i].rounds.last().expect("round")[b])
            })
            .sum();
        if per_chunk == 0 {
            return 0;
        }
        self.chunks
            .iter()
            .map(|c| match self.layout {
                Layout::Laya => per_chunk + (c.len() + 1) * self.unfinished().count(),
                Layout::Prefix => per_chunk + c.len(),
            })
            .sum()
    }

    fn unfinished(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.qs
            .iter()
            .enumerate()
            .filter(|(_, p)| p.results.len() < p.rounds.len())
            .flat_map(|(i, p)| (0..p.rounds.last().expect("round").len()).map(move |b| (i, b)))
    }

    /// The next round's rows, grouped by state chunk (so prefix-layout rows sharing a chunk
    /// are adjacent). Empty when every question is answered.
    pub fn rows(&mut self) -> Vec<Encoded> {
        self.pending = self.unfinished().collect();
        let mut rows = Vec::with_capacity(self.pending.len() * self.chunks.len());
        for chunk in &self.chunks {
            for &(i, b) in &self.pending {
                let p = &self.qs[i];
                let batch = &p.rounds.last().expect("round")[b];
                let opts: Vec<&[u32]> = batch.iter().map(|&o| p.parts.opts[o].as_slice()).collect();
                rows.push(match self.layout {
                    Layout::Laya => joint_row(&self.specials, p.q.t, &p.parts.head, &opts, chunk),
                    Layout::Prefix => {
                        prefix_row(&self.specials, p.q.t, &p.parts.head, &opts, chunk)
                    }
                });
            }
        }
        self.input_tokens += rows_tokens(&rows);
        rows
    }

    /// Takes the outputs for the rows of the last [`Self::rows`] call and sets up the next
    /// round for any Choice still in its batches.
    pub fn feed(&mut self, outs: Vec<RowOutput>) -> Result<()> {
        let n = self.pending.len();
        ensure!(
            outs.len() == n * self.chunks.len(),
            "expected {} row outputs, got {}",
            n * self.chunks.len(),
            outs.len()
        );
        let mut per_batch: Vec<Vec<&RowOutput>> = vec![Vec::with_capacity(self.chunks.len()); n];
        for (j, o) in outs.iter().enumerate() {
            per_batch[j % n].push(o);
        }
        let pending = std::mem::take(&mut self.pending);
        for (&(i, b), chunk_outs) in pending.iter().zip(per_batch) {
            let p = &mut self.qs[i];
            let r = p.rounds.len() - 1;
            if p.results.len() == r {
                p.results.push(vec![(vec![], vec![]); p.rounds[r].len()]);
            }
            p.results[r][b] = combine_chunks(p.q.t, &chunk_outs);
        }
        for p in &mut self.qs {
            if p.results.len() == p.rounds.len() && !p.done() {
                let next = next_round(p, &self.budget);
                p.rounds.push(next);
            }
        }
        Ok(())
    }

    /// One output per question, in request order, with logits over all its options.
    pub fn finish(self) -> Vec<(Question, RowOutput)> {
        self.qs
            .into_iter()
            .map(|p| {
                debug_assert!(p.done(), "finish before every round ran");
                let out = merge_rounds(&p);
                (p.q, out)
            })
            .collect()
    }

    /// Runs every round with `forward` (a model call over rows).
    pub fn run(
        mut self,
        mut forward: impl FnMut(Vec<Encoded>) -> Result<Vec<RowOutput>>,
    ) -> Result<(Vec<(Question, RowOutput)>, usize)> {
        loop {
            let rows = self.rows();
            if rows.is_empty() {
                break;
            }
            let outs = forward(rows)?;
            self.feed(outs)?;
        }
        let tokens = self.input_tokens;
        Ok((self.finish(), tokens))
    }
}

/// Tokens the model reads for rows: each distinct prefix-layout state prefix once (rows
/// sharing one are adjacent), plus every row's own tokens.
fn rows_tokens(rows: &[Encoded]) -> usize {
    let mut total = 0;
    let mut last: Option<&[u32]> = None;
    for r in rows {
        let prefix = &r.ids[..r.prefix_len];
        if r.prefix_len > 0 && last != Some(prefix) {
            total += r.prefix_len;
            last = Some(prefix);
        }
        total += r.ids.len() - r.prefix_len;
    }
    total
}

/// Round one: the whole question in one row, or a Choice's options in batches that each fit
/// the option budget. Batches keep the request's option order and hold at least two options.
fn first_round(q: &Question, parts: &QuestionParts, budget: &Budget) -> Vec<Vec<usize>> {
    let all: Vec<usize> = (0..parts.opts.len()).collect();
    // Batches cost a round trip each, so a list that still fits one `max_len` row beside
    // `min_state` state tokens runs whole, even past laya's question budget.
    let whole = budget
        .max_len
        .saturating_sub(parts.head.len() + 5 + budget.min_state);
    if q.t != QType::Choice
        || all.len() <= 2
        || option_tokens(parts, &all) <= whole.max(option_limit(parts, budget))
    {
        return vec![all];
    }
    pack(parts, &all, option_limit(parts, budget))
}

/// Option tokens a row may take before batching: laya's question budget less the head, but
/// at least half of it so long instructions don't shrink batches to pairs.
fn option_limit(parts: &QuestionParts, budget: &Budget) -> usize {
    budget
        .option_budget
        .saturating_sub(parts.head.len())
        .max(budget.option_budget / 2)
}

fn option_tokens(parts: &QuestionParts, which: &[usize]) -> usize {
    which.iter().map(|&i| parts.opts[i].len()).sum()
}

/// Splits `which` into consecutive batches of even token size, each within `limit` where
/// possible and never a lone option.
fn pack(parts: &QuestionParts, which: &[usize], limit: usize) -> Vec<Vec<usize>> {
    let total = option_tokens(parts, which);
    let n = total.div_ceil(limit.max(1)).max(1);
    let target = total.div_ceil(n);
    let mut out: Vec<Vec<usize>> = vec![];
    let mut cur = vec![];
    let mut size = 0;
    for &i in which {
        let len = parts.opts[i].len();
        if cur.len() >= 2 && size + len > target {
            out.push(std::mem::take(&mut cur));
            size = 0;
        }
        cur.push(i);
        size += len;
    }
    if cur.len() < 2 {
        if let Some(last) = out.last_mut() {
            last.extend(cur);
            return out;
        }
    }
    out.push(cur);
    out
}

/// Candidates for the next round: every batch's leader, then runners-up rank by rank while
/// they still fit one row. Candidates that don't fit one row are batched again.
fn next_round(p: &QPlan, budget: &Budget) -> Vec<Vec<usize>> {
    let r = p.rounds.len() - 1;
    let ranked: Vec<Vec<usize>> = p.rounds[r]
        .iter()
        .zip(&p.results[r])
        .map(|(batch, (logits, _))| {
            let mut order: Vec<usize> = (0..batch.len()).collect();
            order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
            order.into_iter().map(|j| batch[j]).collect()
        })
        .collect();
    let limit = option_limit(&p.parts, budget);
    let mut keep: Vec<usize> = ranked.iter().map(|b| b[0]).collect();
    let mut size = option_tokens(&p.parts, &keep);
    'ranks: for rank in 1.. {
        let mut any = false;
        for b in &ranked {
            let Some(&o) = b.get(rank) else { continue };
            any = true;
            let len = p.parts.opts[o].len();
            if size + len > limit {
                break 'ranks;
            }
            keep.push(o);
            size += len;
        }
        if !any {
            break;
        }
    }
    // Back in request order, so the finalists read as the caller listed them.
    keep.sort_unstable();
    if keep.len() <= 2 || size <= limit {
        vec![keep]
    } else {
        pack(&p.parts, &keep, limit)
    }
}

/// Logits over every option: the final round's as they are, and each option eliminated in
/// an earlier round placed relative to its batch's leader, which has a logit from a later
/// round.
fn merge_rounds(p: &QPlan) -> RowOutput {
    let k = p.parts.opts.len();
    let mut score: Vec<Option<f32>> = vec![None; k];
    let last = p.rounds.len() - 1;
    let (final_logits, act) = &p.results[last][0];
    for (&o, &l) in p.rounds[last][0].iter().zip(final_logits) {
        score[o] = Some(l);
    }
    for r in (0..last).rev() {
        for (batch, (logits, _)) in p.rounds[r].iter().zip(&p.results[r]) {
            let lead = (0..batch.len())
                .filter(|&j| score[batch[j]].is_some())
                .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
                .expect("every batch sends its leader to the next round");
            let base = score[batch[lead]].expect("leader scored") - logits[lead];
            for (j, &o) in batch.iter().enumerate() {
                if score[o].is_none() {
                    score[o] = Some(base + logits[j]);
                }
            }
        }
    }
    RowOutput {
        logits: score
            .into_iter()
            .map(|s| s.expect("every option scored"))
            .collect(),
        act_probs: act.clone(),
    }
}

/// Splits `len` state tokens into spans of at most `room`, consecutive spans sharing
/// `overlap` tokens. One (possibly empty) span when the state fits.
pub fn chunk_spans(len: usize, room: usize, overlap: usize) -> Vec<(usize, usize)> {
    let room = room.max(1);
    if len <= room {
        return vec![(0, len)];
    }
    let overlap = overlap.min(room / 2);
    let stride = room - overlap;
    let n = (len - overlap).div_ceil(stride);
    // Even spans: spread the slack so the last chunk isn't a sliver.
    let stride = (len - overlap).div_ceil(n);
    (0..n)
        .map(|i| {
            let a = i * stride;
            (a, (a + stride + overlap).min(len))
        })
        .collect()
}

/// One answer from a question's answers on each state chunk.
///
/// - `noul`: the chunk most in favour of `true` decides. Questions about a long document are
///   mostly "does it contain / mention / express X", which one chunk can settle, while
///   chunks without X all say `false`; averaging would bury the one that found it.
/// - `choice` and `score`: logits are averaged (a product of the chunks' distributions), so
///   what the whole document leans to wins.
pub fn combine_chunks(qtype: QType, outs: &[&RowOutput]) -> (Vec<f32>, Vec<f32>) {
    if outs.len() == 1 {
        return (outs[0].logits.clone(), outs[0].act_probs.clone());
    }
    if qtype == QType::Noul {
        let best = outs
            .iter()
            .max_by(|a, b| {
                let m = |o: &RowOutput| o.logits[1] - o.logits[0];
                m(a).total_cmp(&m(b))
            })
            .expect("at least one chunk");
        return (best.logits.clone(), best.act_probs.clone());
    }
    let n = outs.len() as f32;
    let mean = |f: fn(&RowOutput) -> &Vec<f32>| -> Vec<f32> {
        let len = f(outs[0]).len();
        (0..len)
            .map(|j| outs.iter().map(|o| f(o)[j]).sum::<f32>() / n)
            .collect()
    };
    (mean(|o| &o.logits), mean(|o| &o.act_probs))
}

/// Shrinks a question so its longest round-one row fits `cap` tokens: instructions first
/// (keeping their start and end, where the actual question usually is), then options
/// evenly. Returns what was cut.
fn fit_question(parts: &mut QuestionParts, batches: &[Vec<usize>], cap: usize) -> Truncation {
    let longest = |parts: &QuestionParts| batches.iter().map(|b| parts.len(b)).max().unwrap_or(0);
    let mut t = Truncation {
        question_tokens: longest(parts),
        limit: cap,
        ..Default::default()
    };
    let over = longest(parts).saturating_sub(cap);
    if over == 0 {
        return t;
    }
    // Keep "<type> question:" plus the tail of the instructions.
    const KEEP_START: usize = 8;
    let cut = over.min(parts.head.len().saturating_sub(KEEP_START + 8));
    if cut > 0 {
        parts.head.drain(KEEP_START..KEEP_START + cut);
        t.instructions = cut;
    }
    let mut over = longest(parts).saturating_sub(cap);
    while over > 0 {
        // Trim the longest option of the longest batch by one token at a time (options keep
        // their [MASK] and at least one token).
        let b = batches
            .iter()
            .max_by_key(|b| parts.len(b))
            .expect("a batch");
        let Some(&o) = b
            .iter()
            .filter(|&&o| parts.opts[o].len() > 2)
            .max_by_key(|&&o| parts.opts[o].len())
        else {
            break;
        };
        parts.opts[o].pop();
        t.options += 1;
        over -= 1;
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parts(lens: &[usize]) -> QuestionParts {
        QuestionParts {
            head: vec![0; 10],
            opts: lens.iter().map(|&l| vec![1; l]).collect(),
        }
    }

    #[test]
    fn chunks_cover_the_state_with_overlap() {
        assert_eq!(chunk_spans(100, 200, 16), vec![(0, 100)]);
        assert_eq!(chunk_spans(0, 200, 16), vec![(0, 0)]);
        let s = chunk_spans(1000, 300, 40);
        assert_eq!(s.first().unwrap().0, 0);
        assert_eq!(s.last().unwrap().1, 1000);
        for w in s.windows(2) {
            assert!(w[1].0 < w[0].1, "chunks overlap: {s:?}");
            assert!(w[0].1 - w[1].0 <= 40);
        }
        assert!(s.iter().all(|(a, b)| b - a <= 300), "{s:?}");
    }

    #[test]
    fn packing_keeps_order_and_pairs() {
        let p = parts(&[10; 25]);
        let all: Vec<usize> = (0..25).collect();
        let b = pack(&p, &all, 60);
        assert!(b.iter().all(|b| b.len() >= 2));
        assert!(b.iter().all(|b| option_tokens(&p, b) <= 60), "{b:?}");
        assert_eq!(b.concat(), all);
        // One option longer than the limit still travels with a partner.
        let p = parts(&[100, 5, 5, 5]);
        let b = pack(&p, &[0, 1, 2, 3], 50);
        assert!(b.iter().all(|b| b.len() >= 2), "{b:?}");
    }

    fn plan_with(rounds: Vec<Vec<Vec<usize>>>, results: Vec<Vec<Vec<f32>>>, k: usize) -> QPlan {
        QPlan {
            q: Question::from_json(&serde_json::json!({"t": "noul", "ins": ""})).unwrap(),
            parts: parts(&vec![3; k]),
            trunc: Truncation::default(),
            rounds,
            results: results
                .into_iter()
                .map(|r| r.into_iter().map(|l| (l, vec![1.0, 0.0])).collect())
                .collect(),
        }
    }

    #[test]
    fn merged_logits_chain_through_leaders() {
        // Round 1: [0,1,2] and [3,4,5]; leaders 1 and 3 meet in the final.
        let p = plan_with(
            vec![vec![vec![0, 1, 2], vec![3, 4, 5]], vec![vec![1, 3]]],
            vec![
                vec![vec![0.0, 2.0, -1.0], vec![5.0, 4.0, 1.0]],
                vec![vec![1.0, 0.5]],
            ],
            6,
        );
        let l = merge_rounds(&p).logits;
        assert_eq!(l[1], 1.0);
        assert_eq!(l[3], 0.5);
        // Option 0 is 2 below its leader (1) in round one, so 2 below it in the merge.
        assert_eq!(l[0], -1.0);
        assert_eq!(l[2], -2.0);
        assert_eq!(l[4], -0.5);
        assert_eq!(l[5], -3.5);
    }

    #[test]
    fn noul_chunks_take_the_strongest_yes() {
        let o = |f: f32, t: f32| RowOutput {
            logits: vec![f, t],
            act_probs: vec![0.5, 0.5],
        };
        let (a, b, c) = (o(3.0, -3.0), o(0.0, 1.0), o(4.0, -4.0));
        let (l, _) = combine_chunks(QType::Noul, &[&a, &b, &c]);
        assert_eq!(l, vec![0.0, 1.0]);
        let (l, _) = combine_chunks(QType::Choice, &[&a, &c]);
        assert_eq!(l, vec![3.5, -3.5]);
    }

    #[test]
    fn long_instructions_keep_their_end() {
        let mut p = QuestionParts {
            head: (0..100).collect(),
            opts: vec![vec![1; 5], vec![1; 5]],
        };
        let t = fit_question(&mut p, &[vec![0, 1]], 60);
        assert_eq!(p.len(&[0, 1]), 60);
        assert_eq!(t.instructions, 53);
        assert_eq!(&p.head[..8], &(0..8).collect::<Vec<u32>>()[..]);
        assert_eq!(*p.head.last().unwrap(), 99);
    }
}
