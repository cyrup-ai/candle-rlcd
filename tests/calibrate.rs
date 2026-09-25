//! `calibrate` on the tiny fixture, and the 11+ option temperature floor.

use std::path::Path;

use candle_core::{DType, Device};
use candle_rlcd::calibrate::{calibrate, current_temperatures, CalibrateConfig};
use candle_rlcd::sequence::QType;
use candle_rlcd::Laya;
use serde_json::json;

fn tiny() -> Laya {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny");
    Laya::load(&dir, &Device::Cpu, DType::F32).unwrap()
}

#[test]
fn shipped_temperatures_do_not_sharpen_many_options() {
    let mut laya = tiny();
    // laya's shipped root config has choice:11+ = 0.1006.
    laya.cfg
        .temperature_by_options
        .insert("choice:11+".into(), 0.1006);
    laya.cfg
        .temperature_by_options
        .insert("choice:6-10".into(), 0.7);
    assert_eq!(laya.temperature(QType::Choice, 11), 1.0);
    assert_eq!(laya.temperature(QType::Choice, 200), 1.0);
    // Below 11 options the fitted value (clamped to [0.5, 5]) stands.
    assert_eq!(laya.temperature(QType::Choice, 7), 0.7);
    // Temperatures refit by candle-rlcd are trusted as they are.
    laya.apply_calibration(&json!({
        "temperature": [1.0, 1.0, 1.0],
        "temperature_by_options": {"choice:11+": 0.8},
    }))
    .unwrap();
    assert_eq!(laya.temperature(QType::Choice, 11), 0.8);
    assert!(laya.cfg.calibration.is_some());
}

#[test]
fn calibrate_refits_from_jev_shaped_labels() {
    let laya = tiny();
    let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/jev-labelled.jsonl");
    let before = current_temperatures(&laya);
    let cfg = CalibrateConfig {
        min_examples: 2,
        ..CalibrateConfig::default()
    };
    let (report, file) = calibrate(&laya, &data, &cfg).unwrap();
    assert_eq!(report["questions"], 6);
    assert_eq!(report["examples_by_bucket"]["choice"], 3);
    assert!(report["holdout"]["calibrated"]["nll"].is_number());
    // choice (3 questions) and noul (2) are refit; score (1) keeps the model's temperature.
    let t = file["temperature"].as_array().unwrap();
    assert_eq!(t[1].as_f64().unwrap(), before.by_type[1]);
    let fitted = t[0].as_f64().unwrap();
    assert!((0.5..=5.0).contains(&fitted));
    assert_eq!(file["temperature_by_options"]["choice:3-5"], t[0]);
    assert!(file["temperature_by_options"].get("choice:11+").is_none());
    assert_eq!(file["calibration"]["by"], "candle-rlcd calibrate");

    let mut applied = tiny();
    applied.apply_calibration(&file).unwrap();
    assert_eq!(applied.temperature(QType::Choice, 3), fitted);
}
