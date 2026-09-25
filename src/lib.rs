//! ModernBERT + RLCD typed-decision model on Candle.
//!
//! Loads laya checkpoints (`convaiinnovations/laya`) and answers `choice` / `score` / `noul`
//! questions about a state in one encoder forward pass per request.

pub mod agent;
pub mod autograd;
pub mod bench;
pub mod config;
pub mod data;
pub mod head;
pub mod loss;
pub mod model;
pub mod modernbert;
pub mod optim;
pub mod sequence;
pub mod serve;
pub mod train;

pub use agent::Laya;
