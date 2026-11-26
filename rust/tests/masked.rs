//! The rule table (masked_cases.txt, written by tests/python/masked_cases.py;
//! docs/API.md, Masked classes) through `Loss::loss_and_grad`, in f64 and f32.
//! A valid row gives its expected loss and gradient, the exact entries (zeros,
//! +0.0, and infinities) to the byte; an invalid one is an invalid row
//! (docs/API.md, Errors): NaN outputs and a report of both copies, and with
//! `Settings::with_check_rows` an `Error::InvalidArgument` that names their count
//! and the first.

use libsdca::{Error, Loss, LossReport, Matrix, Precision, Real, Settings, Targets, TopKVariant};

struct Case {
    name: String,
    loss: Loss,
    valid: bool,
    label: i64,
    logits: Vec<f64>,
    positive: Vec<bool>,
    teacher: Vec<f64>,
    value: f64,
    gradient: Vec<f64>,
    error: String,
}

fn numbers(field: &str) -> Vec<f64> {
    field
        .split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect()
}

fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for line in include_str!("masked_cases.txt").lines() {
        let fields: Vec<&str> = line.split(" | ").collect();
        let head: Vec<&str> = fields[0].split_whitespace().collect();
        let k: usize = head[5].parse().unwrap();
        let number = |i: usize| -> f64 { head[i].parse().unwrap() };
        let (gamma, alpha, temperature) = (number(6), number(7), number(8));
        let loss = match head[1] {
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
            "7" => Loss::ProximalCrossEntropy { gamma: alpha },
            "8" => Loss::Distillation { temperature },
            "9" => Loss::ProximalDistillation {
                temperature,
                gamma: alpha,
            },
            other => panic!("unknown loss {other}"),
        };
        cases.push(Case {
            name: head[0].to_string(),
            loss,
            valid: head[3] == "1",
            label: head[4].parse().unwrap(),
            logits: numbers(fields[1]),
            positive: fields[2].split_whitespace().map(|v| v == "1").collect(),
            teacher: numbers(fields[3]),
            value: fields[4].trim().parse().unwrap(),
            gradient: numbers(fields[5]),
            error: fields.get(6).map_or("", |e| e.trim()).to_string(),
        });
    }
    cases
}

/// The loss and gradient of two copies of the row in the data type `T`.
fn evaluate<T: Real>(
    case: &Case,
    settings: &Settings,
    cast: fn(f64) -> T,
) -> Result<(Vec<f64>, Vec<f64>, LossReport), Error> {
    let classes = case.logits.len();
    let twice =
        |values: &[f64]| -> Vec<T> { values.iter().chain(values).map(|&v| cast(v)).collect() };
    let (z, t) = (twice(&case.logits), twice(&case.teacher));
    let logits = Matrix::new(&z, 2, classes)?;
    let distillation = matches!(
        case.loss,
        Loss::Distillation { .. } | Loss::ProximalDistillation { .. }
    );
    let (values, gradients, report) = if distillation {
        let teacher = Matrix::new(&t, 2, classes)?;
        case.loss
            .loss_and_grad_with(logits, Targets::Teacher(teacher), settings)?
    } else if case.label >= 0 {
        let labels = [case.label as usize; 2];
        case.loss
            .loss_and_grad_with(logits, &labels[..], settings)?
    } else {
        let set: Vec<usize> = (0..classes).filter(|&j| case.positive[j]).collect();
        let sets = [set.clone(), set];
        case.loss.loss_and_grad_with(logits, &sets, settings)?
    };
    Ok((values, gradients.iter().map(|g| g.f64()).collect(), report))
}

/// Whether a computed entry is the expected one: by bytes where that is exact
/// (a zero, with its sign, or an infinity), else within `tol` relative to 1 +
/// its size.
fn same(actual: f64, expected: f64, tol: f64) -> bool {
    if expected == 0.0 || expected.is_infinite() {
        actual.to_bits() == expected.to_bits()
    } else {
        (actual - expected).abs() <= tol * (1.0 + expected.abs())
    }
}

fn check<T: Real>(case: &Case, settings: &Settings, cast: fn(f64) -> T, tol: f64) {
    let what = format!("{} ({})", case.name, std::any::type_name::<T>());
    let result = evaluate(case, settings, cast);
    if !case.valid {
        assert!(
            case.error.is_empty(),
            "{what}: the table's rows are invalid rows"
        );
        let (values, gradients, report) =
            result.unwrap_or_else(|e| panic!("{what} is not an invalid row: {e}"));
        assert!(
            (report.invalid_rows, report.first_invalid_row) == (2, Some(0)),
            "{what}: {report:?}"
        );
        assert!(
            values.iter().chain(&gradients).all(|v| v.is_nan()),
            "{what}: not NaN rows"
        );
        let checked = (*settings).with_check_rows(true);
        let prefix = "2 of 2 rows are invalid (first: row 0): ";
        let result = evaluate(case, &checked, cast);
        assert!(
            matches!(&result, Err(Error::InvalidArgument(e)) if e.starts_with(prefix)),
            "{what} with check_rows is not rejected with {prefix:?}: {result:?}"
        );
        return;
    }
    let (values, gradients, _) = result.unwrap_or_else(|e| panic!("{what} failed: {e}"));
    let classes = case.logits.len();
    for copy in 0..2 {
        let value = values[copy];
        assert!(
            same(value, case.value, tol),
            "{what}: loss {value:?}, not {:?}",
            case.value
        );
        for j in 0..classes {
            let g = gradients[copy * classes + j];
            let expected = case.gradient[j];
            assert!(
                same(g, expected, tol),
                "{what}: gradient entry {j} is {g:?}, not {expected:?}"
            );
        }
    }
}

#[test]
fn masked_classes_follow_the_shared_table() {
    let cases = cases();
    assert!(cases.len() > 200);
    let default = Settings::default();
    let double = Settings::default().with_precision(Precision::Float64);
    for case in &cases {
        check::<f64>(case, &default, |v| v, 1e-12);
        check::<f32>(case, &default, |v| v as f32, 4e-6);
        check::<f32>(case, &double, |v| v as f32, 4e-6);
    }
}
