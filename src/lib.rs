//! ModernBERT + RLCD typed-decision model on Candle.
//!
//! Loads laya checkpoints (`convaiinnovations/laya`) and answers `choice` / `score` / `noul`
//! questions about a state in one encoder forward pass per request.

pub mod agent;
pub mod config;
pub mod head;
pub mod modernbert;
pub mod sequence;

pub use agent::Laya;
