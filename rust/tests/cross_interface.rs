//! The Rust side of tests/python/test_cross_interface.py and test_rule_table.py:
//! `Loss::loss_and_grad_with_options` on the cases that the tests write to the directory
//! `SDCA_CROSS_INTERFACE_DIR` (cases.txt and raw little-endian buffers; see
//! tests/cpp/cross_interface.cpp for the format), for the float32 and float64 cases with
//! their class weights, label smoothing and ignored label, but for an upstream factor or a
//! negative ignored label, which the crate does not take. It writes `<id>.rust.values` (f64), `<id>.rust.grads` (the storage type) and
//! `<id>.rust.report` ("invalid_rows first_invalid_row", -1 for none), or the message of an
//! error in `<id>.rust.error`. The cases of `rust_cases.txt`, if any, run too; the targets
//! take their sizes from their files (labels: one per entry; probabilities and teacher
//! logits: n rows), so that their size errors reach the crate. Without the variable there
//! is nothing to do.

use std::fs;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use libsdca::{Error, Loss, Matrix, Options, Precision, Settings, Targets, Threads, TopKVariant};

struct Case {
    id: String,
    key: String,
    loss: Loss,
    dtype: String,
    n: usize,
    classes: usize,
    target: String,
    // An upstream factor or a negative ignored label, which the crate does not take.
    skip: bool,
    weight: bool,
    label_smoothing: f64,
    ignore_index: Option<usize>,
    threads: usize,
    float64: bool,
    check_rows: bool,
}

fn parse(line: &str) -> Case {
    let f: Vec<&str> = line.split_whitespace().collect();
    let k: usize = f[3].parse().unwrap();
    let gamma: f64 = f[4].parse().unwrap();
    let temperature: f64 = f[5].parse().unwrap();
    let loss = match f[2] {
        "0" => Loss::CrossEntropy,
        "1" => Loss::TopKCrossEntropy { k },
        "2" => Loss::MulticlassHinge { gamma },
        "3" => Loss::TopKHinge {
            k,
            variant: TopKVariant::Alpha,
            gamma,
        },
        "4" => Loss::TopKHinge {
            k,
            variant: TopKVariant::Beta,
            gamma,
        },
        "5" => Loss::MultilabelHinge { gamma },
        "6" => Loss::MultilabelCrossEntropy,
        "7" => Loss::ProximalCrossEntropy { gamma },
        "8" => Loss::Distillation { temperature },
        "9" => Loss::ProximalDistillation { temperature, gamma },
        other => panic!("unknown loss {other}"),
    };
    let label_smoothing: f64 = f[6].parse().unwrap();
    let ignore_index: i64 = f[8].parse().unwrap();
    let ignore = f[7] != "0";
    Case {
        id: f[0].to_string(),
        key: f[1].to_string(),
        loss,
        dtype: f[9].to_string(),
        n: f[10].parse().unwrap(),
        classes: f[11].parse().unwrap(),
        target: f[12].to_string(),
        skip: f[16] != "1.0" || (ignore && ignore_index < 0),
        weight: f[13] != "0",
        label_smoothing,
        ignore_index: ignore.then(|| usize::try_from(ignore_index).unwrap_or(usize::MAX)),
        threads: f[14].parse().unwrap(),
        float64: f[15] == "2",
        check_rows: f[17] == "1",
    }
}

fn read(dir: &Path, name: &str) -> Vec<u8> {
    fs::read(dir.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn int64s(bytes: &[u8]) -> Vec<i64> {
    bytes
        .chunks_exact(8)
        .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn float64s(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// The storage type of a case: reads and writes its raw buffers.
trait Storage: libsdca::Real {
    fn decode(bytes: &[u8]) -> Vec<Self>;
    fn encode(values: &[Self]) -> Vec<u8>;
    fn narrow(value: f64) -> Self;
}

impl Storage for f32 {
    fn decode(bytes: &[u8]) -> Vec<Self> {
        bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }
    fn encode(values: &[Self]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
    fn narrow(value: f64) -> Self {
        value as f32
    }
}

impl Storage for f64 {
    fn decode(bytes: &[u8]) -> Vec<Self> {
        float64s(bytes)
    }
    fn encode(values: &[Self]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
    fn narrow(value: f64) -> Self {
        value
    }
}

fn message(error: Error) -> String {
    error.to_string()
}

fn run<T: Storage>(dir: &Path, case: &Case) {
    if let Err(error) = evaluate::<T>(dir, case) {
        fs::write(dir.join(format!("{}.rust.error", case.id)), error).unwrap();
    }
}

fn evaluate<T: Storage>(dir: &Path, case: &Case) -> Result<(), String> {
    let scores = T::decode(&read(dir, &format!("{}.scores", case.key)));
    let logits = Matrix::new(&scores, case.n, case.classes).map_err(message)?;
    let threads = match NonZeroUsize::new(case.threads) {
        Some(count) => Threads::Count(count),
        None => Threads::FromEnv,
    };
    let mut settings = Settings::default()
        .with_threads(threads)
        .with_check_rows(case.check_rows);
    if case.float64 {
        settings = settings.with_precision(Precision::Float64);
    }
    let labels: Vec<usize>;
    let sets: Vec<Vec<usize>>;
    let matrix: Vec<T>;
    let targets: Targets<'_, T> = match case.target.as_str() {
        "labels" => {
            labels = int64s(&read(dir, &format!("{}.labels", case.key)))
                .into_iter()
                .map(usize::try_from)
                .collect::<Result<_, _>>()
                .map_err(|_| "a label below 0: Rust labels are usize".to_string())?;
            Targets::Labels(&labels)
        }
        "sets" => {
            let flat = int64s(&read(dir, &format!("{}.set_labels", case.key)));
            let offsets = int64s(&read(dir, &format!("{}.offsets", case.key)));
            sets = offsets
                .windows(2)
                .map(|w| {
                    flat[w[0] as usize..w[1] as usize]
                        .iter()
                        .map(|&y| y as usize)
                        .collect()
                })
                .collect();
            Targets::Sets(&sets)
        }
        "mask" => {
            let mask = read(dir, &format!("{}.mask", case.key));
            sets = mask
                .chunks_exact(case.classes)
                .map(|row| (0..case.classes).filter(|&j| row[j] != 0).collect())
                .collect();
            Targets::Sets(&sets)
        }
        "probs" => {
            matrix = float64s(&read(dir, &format!("{}.probs", case.key)))
                .into_iter()
                .map(T::narrow)
                .collect();
            let cols = matrix.len() / case.n.max(1);
            Targets::Probabilities(Matrix::new(&matrix, case.n, cols).map_err(message)?)
        }
        "teacher" => {
            matrix = T::decode(&read(dir, &format!("{}.teacher", case.key)));
            let cols = matrix.len() / case.n.max(1);
            Targets::Teacher(Matrix::new(&matrix, case.n, cols).map_err(message)?)
        }
        other => panic!("unknown target {other}"),
    };
    let weight = if case.weight {
        float64s(&read(dir, &format!("{}.weight", case.key)))
    } else {
        Vec::new()
    };
    let mut options = Options::default().with_label_smoothing(case.label_smoothing);
    if case.weight {
        options = options.with_weight(&weight);
    }
    if let Some(index) = case.ignore_index {
        options = options.with_ignore_index(index);
    }
    let (values, grads, report) = case
        .loss
        .loss_and_grad_with_options(logits, targets, &options, &settings)
        .map_err(message)?;
    let first = report.first_invalid_row.map_or(-1, |i| i as i64);
    let report = format!("{} {first}\n", report.invalid_rows);
    fs::write(dir.join(format!("{}.rust.report", case.id)), report).unwrap();
    let values: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(dir.join(format!("{}.rust.values", case.id)), values).unwrap();
    fs::write(
        dir.join(format!("{}.rust.grads", case.id)),
        T::encode(&grads),
    )
    .unwrap();
    Ok(())
}

#[test]
fn cross_interface_cases() {
    let Some(dir) = std::env::var_os("SDCA_CROSS_INTERFACE_DIR").map(PathBuf::from) else {
        return;
    };
    let mut manifest = fs::read_to_string(dir.join("cases.txt")).unwrap();
    manifest += &fs::read_to_string(dir.join("rust_cases.txt")).unwrap_or_default();
    for case in manifest.lines().filter(|l| !l.is_empty()).map(parse) {
        if case.skip {
            continue;
        }
        match case.dtype.as_str() {
            "float32" => run::<f32>(&dir, &case),
            "float64" => run::<f64>(&dir, &case),
            _ => {}
        }
    }
}
