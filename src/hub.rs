//! Resolve a model argument to a local checkpoint directory, downloading from the Hugging Face
//! Hub when it names a Hub repository.
//!
//! A model argument is either a local directory or a Hub id:
//!
//! ```text
//! runs/x/final                                # a directory
//! convaiinnovations/laya                      # a repository's root checkpoint
//! convaiinnovations/laya/multilingual         # a checkpoint in a sub-folder
//! convaiinnovations/laya@55cf4c4e…            # pinned to a revision (branch, tag or commit)
//! ```
//!
//! Downloads land in the Hugging Face cache (`$HF_HUB_CACHE`, else `$HF_HOME/hub`, else
//! `~/.cache/huggingface/hub`) in its `models--org--repo/snapshots/<commit>/` layout, so a
//! checkpoint fetched earlier with `huggingface-cli download` is reused, and only the files a
//! checkpoint needs are fetched (not the repository's images or other checkpoints). With
//! `HF_HUB_OFFLINE=1`, or when the Hub can't be reached, the cached snapshot is used.
//! `HF_TOKEN` (or the token `huggingface-cli login` saved) is sent for private repositories.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

/// Files a checkpoint directory needs, and ones it uses when present.
const REQUIRED: [&str; 4] = [
    "rl_agent_config.json",
    "model.safetensors",
    "encoder/config.json",
    "tokenizer/tokenizer.json",
];
const OPTIONAL: [&str; 2] = [
    "tokenizer/tokenizer_config.json",
    "tokenizer/special_tokens_map.json",
];

/// A Hub checkpoint reference: `org/repo[/sub/folder][@revision]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HubRef {
    /// `org/repo`.
    pub repo: String,
    /// Folder inside the repository holding the checkpoint (`""` for the root).
    pub subdir: String,
    /// Branch, tag or commit (`main` by default).
    pub revision: String,
}

impl HubRef {
    /// Parses `org/repo[/sub][@rev]`; `None` when the text can't be a Hub id.
    pub fn parse(s: &str) -> Option<Self> {
        let (path, revision) = match s.split_once('@') {
            Some((p, r)) if !r.is_empty() => (p, r.to_string()),
            Some(_) => return None,
            None => (s, "main".to_string()),
        };
        let parts: Vec<&str> = path.split('/').collect();
        let ok = |p: &str| {
            !p.is_empty()
                && p != "."
                && p != ".."
                && p.chars()
                    .all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c))
        };
        if parts.len() < 2 || !parts.iter().all(|p| ok(p)) {
            return None;
        }
        Some(Self {
            repo: format!("{}/{}", parts[0], parts[1]),
            subdir: parts[2..].join("/"),
            revision,
        })
    }

    /// A short name for the checkpoint: the repository name plus the sub-folder, e.g.
    /// `laya` or `laya-multilingual`.
    pub fn short_name(&self) -> String {
        let repo = self.repo.split('/').nth(1).unwrap_or(&self.repo);
        if self.subdir.is_empty() {
            repo.to_string()
        } else {
            format!("{repo}-{}", self.subdir.replace('/', "-"))
        }
    }

    fn file(&self, f: &str) -> String {
        if self.subdir.is_empty() {
            f.to_string()
        } else {
            format!("{}/{f}", self.subdir)
        }
    }
}

/// A model argument resolved to a directory.
#[derive(Debug, Clone)]
pub struct Resolved {
    pub dir: PathBuf,
    /// Default serving name: the directory name, or [`HubRef::short_name`].
    pub name: String,
    /// The Hub checkpoint and the commit it resolved to, for Hub ids.
    pub hub: Option<(HubRef, String)>,
}

/// Resolves a model argument: an existing directory is used as is, anything else is read as a
/// Hub id and downloaded (or found in the cache).
pub fn resolve(spec: &str) -> Result<Resolved> {
    let path = Path::new(spec);
    if path.is_dir() {
        let dir = std::fs::canonicalize(path)?;
        let name = dir
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "candle-rlcd".into());
        return Ok(Resolved {
            dir,
            name,
            hub: None,
        });
    }
    let Some(hub) = HubRef::parse(spec) else {
        bail!("model {spec:?} is neither a directory nor a Hub id like convaiinnovations/laya");
    };
    let (dir, commit) = fetch(&hub).with_context(|| {
        format!("{spec} is not a local directory, so it was looked up on the Hugging Face Hub")
    })?;
    Ok(Resolved {
        dir,
        name: hub.short_name(),
        hub: Some((hub, commit)),
    })
}

fn env_nonempty(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|v| !v.is_empty())
}

fn hf_home() -> PathBuf {
    env_nonempty("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let home = env_nonempty("HOME")
                .or_else(|| env_nonempty("USERPROFILE"))
                .unwrap_or_else(|| ".".into());
            PathBuf::from(home).join(".cache/huggingface")
        })
}

/// The Hugging Face hub cache directory.
pub fn cache_dir() -> PathBuf {
    env_nonempty("HF_HUB_CACHE")
        .or_else(|| env_nonempty("HUGGINGFACE_HUB_CACHE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| hf_home().join("hub"))
}

fn endpoint() -> String {
    env_nonempty("HF_ENDPOINT")
        .unwrap_or_else(|| "https://huggingface.co".into())
        .trim_end_matches('/')
        .to_string()
}

fn token() -> Option<String> {
    env_nonempty("HF_TOKEN")
        .or_else(|| env_nonempty("HUGGING_FACE_HUB_TOKEN"))
        .or_else(|| {
            let p = env_nonempty("HF_TOKEN_PATH")
                .map(PathBuf::from)
                .unwrap_or_else(|| hf_home().join("token"));
            std::fs::read_to_string(p)
                .ok()
                .map(|t| t.trim().to_string())
                .filter(|t| !t.is_empty())
        })
}

fn offline() -> bool {
    env_nonempty("HF_HUB_OFFLINE").is_some_and(|v| !matches!(v.as_str(), "0" | "false" | "False"))
}

fn is_commit(rev: &str) -> bool {
    rev.len() == 40 && rev.chars().all(|c| c.is_ascii_hexdigit())
}

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .try_proxy_from_env(true)
        .timeout_connect(std::time::Duration::from_secs(30))
        .timeout_read(std::time::Duration::from_secs(120))
        .user_agent(concat!("candle-rlcd/", env!("CARGO_PKG_VERSION")))
        .build()
}

fn get(agent: &ureq::Agent, url: &str) -> Result<ureq::Response> {
    let mut req = agent.get(url);
    if let Some(t) = token() {
        req = req.set("authorization", &format!("Bearer {t}"));
    }
    match req.call() {
        Ok(r) => Ok(r),
        Err(ureq::Error::Status(code @ (401 | 403), _)) => bail!(
            "{url}: HTTP {code}; for a private or gated repository set HF_TOKEN or run `huggingface-cli login`"
        ),
        Err(ureq::Error::Status(404, _)) => bail!("{url}: not found (HTTP 404)"),
        Err(e) => Err(e).with_context(|| format!("GET {url}")),
    }
}

/// Downloads (or finds in the cache) a Hub checkpoint; returns its directory and commit.
pub fn fetch(hub: &HubRef) -> Result<(PathBuf, String)> {
    let repo_dir = cache_dir().join(format!("models--{}", hub.repo.replace('/', "--")));
    let ref_file = repo_dir.join("refs").join(&hub.revision);
    let cached_commit = || -> Option<String> {
        if is_commit(&hub.revision) {
            return Some(hub.revision.clone());
        }
        std::fs::read_to_string(&ref_file)
            .ok()
            .map(|s| s.trim().to_string())
    };
    let have = |commit: &str| {
        let snap = repo_dir.join("snapshots").join(commit);
        REQUIRED.iter().all(|f| snap.join(hub.file(f)).is_file())
    };
    let checkpoint = |commit: &str| repo_dir.join("snapshots").join(commit).join(&hub.subdir);

    // Offline, or a commit already in the cache: no network needed.
    if let Some(c) = cached_commit().filter(|c| have(c)) {
        if offline() || is_commit(&hub.revision) {
            return Ok((checkpoint(&c), c));
        }
    }
    if offline() {
        bail!(
            "HF_HUB_OFFLINE is set and {} ({}) is not in the cache at {}",
            hub.repo,
            hub.revision,
            repo_dir.display()
        );
    }

    let agent = agent();
    let base = endpoint();
    let commit = if is_commit(&hub.revision) {
        hub.revision.clone()
    } else {
        let url = format!("{base}/api/models/{}/revision/{}", hub.repo, hub.revision);
        match get(&agent, &url).and_then(|r| Ok(r.into_json::<serde_json::Value>()?)) {
            Ok(v) => v["sha"]
                .as_str()
                .context("the Hub's revision info has no sha")?
                .to_string(),
            Err(e) => {
                // Unreachable Hub: fall back to what the cache has.
                if let Some(c) = cached_commit().filter(|c| have(c)) {
                    eprintln!(
                        "could not reach the Hub ({e:#}); using cached {} at {c}",
                        hub.repo
                    );
                    return Ok((checkpoint(&c), c));
                }
                return Err(e);
            }
        }
    };

    let snap = repo_dir.join("snapshots").join(&commit);
    for (f, required) in REQUIRED
        .iter()
        .map(|f| (f, true))
        .chain(OPTIONAL.iter().map(|f| (f, false)))
    {
        let rel = hub.file(f);
        let dest = snap.join(&rel);
        if dest.is_file() {
            continue;
        }
        let url = format!("{base}/{}/resolve/{commit}/{rel}", hub.repo);
        match download(&agent, &url, &dest) {
            Ok(()) => {}
            Err(e) if !required && format!("{e:#}").contains("HTTP 404") => {}
            Err(e) => return Err(e),
        }
    }
    std::fs::create_dir_all(ref_file.parent().expect("refs dir"))?;
    if !is_commit(&hub.revision) {
        std::fs::write(&ref_file, &commit)?;
    }
    Ok((checkpoint(&commit), commit))
}

/// Streams `url` to `dest` through a `.incomplete` file, with coarse progress on stderr.
fn download(agent: &ureq::Agent, url: &str, dest: &Path) -> Result<()> {
    std::fs::create_dir_all(dest.parent().expect("file has a parent"))?;
    let resp = get(agent, url)?;
    let total: Option<u64> = resp.header("content-length").and_then(|v| v.parse().ok());
    let tmp = dest.with_extension("incomplete");
    let mut out = std::io::BufWriter::new(
        std::fs::File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?,
    );
    let mut body = resp.into_reader();
    let mut buf = vec![0u8; 1 << 20];
    let (mut done, mut shown) = (0u64, 0u64);
    let name = dest
        .file_name()
        .map(|n| n.to_string_lossy())
        .unwrap_or_default();
    loop {
        let n = body
            .read(&mut buf)
            .with_context(|| format!("reading {url}"))?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
        done += n as u64;
        // Progress for large files only, every 10%.
        if let Some(t) = total.filter(|t| *t > 50 << 20) {
            let pct = done * 100 / t;
            if pct >= shown + 10 {
                shown = pct - pct % 10;
                eprintln!("downloading {name}: {shown}% of {} MB", t >> 20);
            }
        }
    }
    out.flush()?;
    drop(out);
    if let Some(t) = total {
        anyhow::ensure!(done == t, "{url}: got {done} of {t} bytes");
    }
    std::fs::rename(&tmp, dest)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_hub_ids() {
        let h = HubRef::parse("convaiinnovations/laya").unwrap();
        assert_eq!(
            (h.repo.as_str(), h.subdir.as_str(), h.revision.as_str()),
            ("convaiinnovations/laya", "", "main")
        );
        assert_eq!(h.short_name(), "laya");
        let h = HubRef::parse("convaiinnovations/laya/multilingual@v1").unwrap();
        assert_eq!(h.subdir, "multilingual");
        assert_eq!(h.revision, "v1");
        assert_eq!(h.short_name(), "laya-multilingual");
        assert_eq!(
            h.file("model.safetensors"),
            "multilingual/model.safetensors"
        );
        for bad in ["laya", "/abs/path", "./x/y", "a/../b", "a/b@", "a b/c"] {
            assert_eq!(HubRef::parse(bad), None, "{bad}");
        }
    }

    #[test]
    fn offline_uses_the_cache() {
        let root = std::env::temp_dir().join(format!("crlcd-hub-{}", std::process::id()));
        let commit = "0123456789abcdef0123456789abcdef01234567";
        let repo = root.join("models--org--m");
        for f in REQUIRED {
            let p = repo.join("snapshots").join(commit).join("sub").join(f);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, "{}").unwrap();
        }
        std::fs::create_dir_all(repo.join("refs")).unwrap();
        std::fs::write(repo.join("refs/main"), commit).unwrap();
        // Only this test touches these variables.
        std::env::set_var("HF_HUB_CACHE", &root);
        std::env::set_var("HF_HUB_OFFLINE", "1");
        let (dir, c) = fetch(&HubRef::parse("org/m/sub").unwrap()).unwrap();
        assert_eq!(c, commit);
        assert!(dir.join("rl_agent_config.json").is_file());
        assert!(fetch(&HubRef::parse("org/other").unwrap()).is_err());
        std::env::remove_var("HF_HUB_OFFLINE");
        std::env::remove_var("HF_HUB_CACHE");
        let _ = std::fs::remove_dir_all(root);
    }
}
