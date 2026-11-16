//! The contracts of the Rust interface, mirroring tests/python/test_api_contracts.py.
use std::num::NonZeroUsize;

use libsdca::{
    Error, Evaluation, Input, Loss, Matrix, Model, Precision, ProjectionPrecision, Real, Settings,
    Shrinking, Solver, Start, Status, Targets, Threads,
};

const X: [f64; 8] = [1.0, 0.2, -0.3, 1.0, 0.7, -0.4, -0.9, -0.8];
const Y: [usize; 4] = [0, 1, 2, 0];

/// Every training objective with k = 2 where it has one, nonsmooth.
const OBJECTIVES: [Loss; 7] = [
    Loss::CrossEntropy,
    Loss::TopKCrossEntropy { k: 2 },
    Loss::MulticlassHinge { gamma: 0.0 },
    Loss::TopKHingeAlpha { k: 2, gamma: 0.0 },
    Loss::TopKHingeBeta { k: 2, gamma: 0.0 },
    Loss::MultilabelHinge { gamma: 0.0 },
    Loss::MultilabelCrossEntropy,
];

fn x() -> Matrix<'static, f64> {
    Matrix::new(&X, 4, 2).unwrap()
}

fn softmax(z: &[f64]) -> Vec<f64> {
    let m = z.iter().cloned().fold(f64::MIN, f64::max);
    let e: Vec<f64> = z.iter().map(|v| (v - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|v| v / s).collect()
}

fn invalid(result: Result<impl std::fmt::Debug, Error>, message: &str) {
    match result {
        Err(Error::InvalidArgument(text)) => assert!(text.contains(message), "{text}"),
        other => panic!("expected InvalidArgument({message}), got {other:?}"),
    }
}

/// The loss with the smoothing `gamma` if it has one.
fn smoothed(loss: Loss, gamma: f64) -> Loss {
    match loss {
        Loss::MulticlassHinge { .. } => Loss::MulticlassHinge { gamma },
        Loss::TopKHingeAlpha { k, .. } => Loss::TopKHingeAlpha { k, gamma },
        Loss::TopKHingeBeta { k, .. } => Loss::TopKHingeBeta { k, gamma },
        Loss::MultilabelHinge { .. } => Loss::MultilabelHinge { gamma },
        other => other,
    }
}

fn last(model: &Model<f64>) -> Evaluation {
    *model.history().last().unwrap()
}

#[test]
fn shared_vocabulary() {
    let names: Vec<_> = OBJECTIVES.iter().map(|o| o.name()).collect();
    assert_eq!(
        names,
        [
            "cross_entropy",
            "topk_cross_entropy",
            "multiclass_hinge",
            "topk_hinge_alpha",
            "topk_hinge_beta",
            "multilabel_hinge",
            "multilabel_cross_entropy"
        ]
    );
    assert_eq!(
        "proximal_cross_entropy".parse::<Loss>(),
        Ok(Loss::ProximalCrossEntropy { alpha: 1.0 })
    );
    assert_eq!(
        "multilabel_cross_entropy".parse::<Loss>(),
        Ok(Loss::MultilabelCrossEntropy)
    );
    assert_eq!("float32".parse::<Precision>(), Ok(Precision::Float32));
    assert_eq!("float64".parse::<Precision>(), Ok(Precision::Float64));
    // One spelling per name.
    for alias in ["fp32", "single", "fp64", "double", "mixed"] {
        invalid(alias.parse::<Precision>(), "Unknown precision");
    }
    invalid("typo".parse::<Loss>(), "Unknown loss");
    assert_eq!(Status::Solved.to_string(), "solved");
    assert_eq!(libsdca::version(), env!("CARGO_PKG_VERSION"));
}

#[test]
fn losses_carry_their_parameters() {
    assert_eq!(Loss::cross_entropy(), Loss::CrossEntropy);
    assert_eq!(Loss::topk_cross_entropy(3), Loss::TopKCrossEntropy { k: 3 });
    assert_eq!(
        Loss::multiclass_hinge(),
        Loss::MulticlassHinge { gamma: 0.0 }
    );
    assert_eq!(
        Loss::topk_hinge(2),
        Loss::TopKHingeBeta { k: 2, gamma: 0.0 }
    );
    assert_eq!(Loss::topk_hinge_beta(2), Loss::topk_hinge(2));
    assert_eq!(
        Loss::topk_hinge_alpha(2),
        Loss::TopKHingeAlpha { k: 2, gamma: 0.0 }
    );
    assert_eq!(
        Loss::multilabel_hinge(),
        Loss::MultilabelHinge { gamma: 0.0 }
    );
    assert_eq!(
        Loss::multilabel_cross_entropy(),
        Loss::MultilabelCrossEntropy
    );
    assert_eq!(
        Loss::proximal_cross_entropy(),
        Loss::ProximalCrossEntropy { alpha: 1.0 }
    );
    assert_eq!(
        Loss::distillation(),
        Loss::Distillation { temperature: 1.0 }
    );
    assert_eq!(
        Loss::proximal_distillation(),
        Loss::ProximalDistillation {
            temperature: 1.0,
            alpha: 1.0
        }
    );
    assert!(
        Loss::multilabel_hinge().is_multilabel() && Loss::MultilabelCrossEntropy.is_multilabel()
    );
    assert!(!Loss::cross_entropy().is_multilabel() && !Loss::topk_hinge(2).is_multilabel());
    // The parameters reach the model.
    let loss = Loss::TopKHingeAlpha { k: 2, gamma: 0.25 };
    let model = Solver::new(loss).with_c(3.0).fit(x(), &Y).unwrap();
    assert_eq!((model.loss(), model.c()), (loss, 3.0));
    assert!(!model.has_probability_targets());
    // And change the values: smoothing lowers the hinge.
    let z = Matrix::new(&[0.5, 0.0, 0.3], 1, 3).unwrap();
    let (rough, _) = Loss::topk_hinge_alpha(2).loss_and_grad(z, &[0]).unwrap();
    let (smooth, _) = loss.loss_and_grad(z, &[0]).unwrap();
    assert!(smooth[0] < rough[0]);
}

#[test]
fn precision_must_match_the_data_type() {
    let (z64, z32) = ([2.0, -0.5, 1.0], [2.0f32, -0.5, 1.0]);
    let (z64, z32) = (
        Matrix::new(&z64, 1, 3).unwrap(),
        Matrix::new(&z32, 1, 3).unwrap(),
    );
    let settings = |precision| Settings::default().with_precision(precision);
    let loss = Loss::cross_entropy();
    invalid(
        loss.loss_and_grad_with(z64, &[0], &settings(Precision::Float32)),
        "does not match f64 data",
    );
    for precision in [Precision::Float32, Precision::Float64] {
        assert!(loss
            .loss_and_grad_with(z32, &[0], &settings(precision))
            .is_ok());
    }
    assert!(loss
        .loss_and_grad_with(z64, &[0], &settings(Precision::Float64))
        .is_ok());
    let solver = Solver::default().with_precision(Precision::Float32);
    invalid(
        solver.coordinate_update(z64, &[0], 1.0, None),
        "does not match f64 data",
    );
    invalid(solver.fit(x(), &Y), "does not match f64 data");
    assert_eq!(f64::DEFAULT_PRECISION, Precision::Float64);
    assert_eq!(f32::DEFAULT_PRECISION, Precision::Float32);
}

#[test]
fn f32_data_trains_in_every_precision() {
    let x32: Vec<f32> = X.iter().map(|&v| v as f32).collect();
    let x32 = Matrix::new(&x32, 4, 2).unwrap();
    let solver = Solver::default().with_c(10.0);
    let exact = solver.fit(x(), &Y).unwrap();
    let default = solver.fit(x32, &Y).unwrap();
    let float = solver.clone().with_precision(Precision::Float32);
    assert_eq!(default.dual(), float.fit(x32, &Y).unwrap().dual());
    let double = solver
        .with_precision(Precision::Float64)
        .fit(x32, &Y)
        .unwrap();
    for model in [default, double] {
        assert_eq!(model.status(), Status::Solved);
        assert_eq!(model.predict(x32).unwrap(), exact.predict(x()).unwrap());
    }
}

#[test]
fn public_types_are_send_and_sync() {
    fn check<T: Send + Sync>() {}
    check::<Matrix<'static, f32>>();
    check::<Targets<'static, f64>>();
    check::<Input<'static, f32>>();
    check::<Start<'static, f64>>();
    check::<Model<f64>>();
    check::<Evaluation>();
    check::<Solver>();
    check::<Settings>();
    check::<Error>();
    check::<(
        Loss,
        Precision,
        Status,
        Threads,
        Shrinking,
        ProjectionPrecision,
    )>();
}

#[test]
fn the_handshake_runs_once_across_threads() {
    let threads: Vec<_> = (0..4)
        .map(|i| {
            std::thread::spawn(move || {
                let z = [i as f64, 0.0, 1.0];
                Loss::cross_entropy().loss_and_grad(Matrix::new(&z, 1, 3).unwrap(), &[1])
            })
        })
        .collect();
    for thread in threads {
        assert!(thread.join().unwrap().is_ok());
    }
}

#[test]
fn fit_solves_and_weights_are_x_transpose_dual() {
    let model = Solver::default().with_c(10.0).fit(x(), &Y).unwrap();
    assert_eq!(model.status(), Status::Solved);
    assert!(last(&model).relative_gap <= 1e-3);
    let w = model.weights().unwrap();
    for l in 0..2 {
        for j in 0..3 {
            let expected: f64 = (0..4).map(|i| X[i * 2 + l] * model.dual()[i * 3 + j]).sum();
            assert!((w[l * 3 + j] - expected).abs() < 1e-12);
        }
    }
    for row in model.dual().chunks(3) {
        assert!(row.iter().sum::<f64>().abs() < 1e-14);
    }
    assert_eq!(model.predict(x()).unwrap().len(), 4);
}

#[test]
fn zero_rows_take_the_zero_score_optimum() {
    let mut data = X;
    data[0] = 0.0;
    data[1] = 0.0;
    let solver = Solver::default().with_tol(1e-10).with_eval_epoch(1);
    let model = solver.fit(Matrix::new(&data, 4, 2).unwrap(), &Y).unwrap();
    assert_eq!(model.status(), Status::Solved);
    // Entropy at zero scores: C (e_y - 1/m) with label 0.
    let expected = [2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0];
    for (a, e) in model.dual()[..3].iter().zip(expected) {
        assert!((a - e).abs() < 1e-15, "{a} {e}");
    }
    let negative = Matrix::new(&[-1.0, 0.0, 0.0, 1.0], 2, 2).unwrap();
    invalid(solver.fit(Input::Kernel(negative), &Y[..2]), "nonnegative");
    // All-zero rows take a block of size C, which must fit f32 storage.
    let zeros = Matrix::new(&[0.0f32; 2], 2, 1).unwrap();
    for c in [1e100, 1e-100] {
        let solver = Solver::default().with_c(c).with_num_classes(2);
        invalid(solver.fit(zeros, &Y[..2]), "must be normal");
    }
}

#[test]
fn kernel_and_feature_models_agree() {
    let gram: Vec<f64> = (0..16)
        .map(|e| {
            (0..2)
                .map(|l| X[(e / 4) * 2 + l] * X[(e % 4) * 2 + l])
                .sum()
        })
        .collect();
    let gram = Matrix::new(&gram, 4, 4).unwrap();
    let solver = Solver::default().with_max_epoch(20).with_tol(0.0);
    let feature = solver.fit(x(), &Y).unwrap();
    let kernel = solver.fit(Input::Kernel(gram), &Y).unwrap();
    assert!(kernel.weights().is_none() && kernel.is_kernel());
    assert_eq!(kernel.num_features(), None);
    assert_eq!(
        (feature.num_features(), feature.is_kernel()),
        (Some(2), false)
    );
    let a = feature.decision_function(x()).unwrap();
    let b = kernel.decision_function(gram).unwrap();
    for (u, v) in a.iter().zip(&b) {
        assert!((u - v).abs() < 1e-10, "{u} {v}");
    }
    invalid(
        kernel.decision_function(x()),
        "4 columns (the training examples",
    );
}

#[test]
fn every_objective_trains() {
    let sets = vec![vec![0, 1], vec![1], vec![2], vec![0]];
    for objective in OBJECTIVES {
        let solver = Solver::new(objective).with_max_epoch(50);
        let model = if objective.is_multilabel() {
            solver.fit(x(), &sets)
        } else {
            solver.fit(x(), &Y)
        }
        .unwrap();
        assert_eq!(model.loss(), objective);
        assert!(model.history().iter().all(|e| e.primal.is_finite()));
    }
}

#[test]
fn probability_targets_train_cross_entropy() {
    let q_data = [0.8, 0.1, 0.1, 0.1, 0.7, 0.2, 0.2, 0.2, 0.6, 0.6, 0.3, 0.1];
    let q = Matrix::new(&q_data, 4, 3).unwrap();
    let solver = Solver::default().with_c(2.0).with_tol(1e-6);
    let model = solver.fit(x(), Targets::Probabilities(q)).unwrap();
    assert_eq!(model.status(), Status::Solved);
    assert!(model.has_probability_targets());
    assert_eq!((model.loss(), model.num_classes()), (Loss::CrossEntropy, 3));
    // The optimum is a fixed point of the coordinate steps.
    let scores = model.decision_function(x()).unwrap();
    let z = Matrix::new(&scores, 4, 3).unwrap();
    let a = Solver::default()
        .coordinate_update(z, Targets::Probabilities(q), 1.0, None)
        .unwrap();
    for row in a.chunks(3) {
        assert!(row.iter().sum::<f64>().abs() < 1e-14);
    }
    // f32 targets sum to one to f32 rounding.
    let x32: Vec<f32> = X.iter().map(|&v| v as f32).collect();
    let q32: Vec<f32> = q_data.iter().map(|&v| v as f32).collect();
    let model32 = solver
        .fit(
            Matrix::new(&x32, 4, 2).unwrap(),
            Targets::Probabilities(Matrix::new(&q32, 4, 3).unwrap()),
        )
        .unwrap();
    assert_eq!(model32.status(), Status::Solved);
    // A warm start from the duals of the optimum, and an infeasible one.
    let warm = solver
        .fit_from(
            x(),
            Targets::Probabilities(q),
            &Start::default().with_initial_dual(model.dual()),
        )
        .unwrap();
    assert_eq!(warm.status(), Status::Solved);
    let outside = [3.0, -1.5, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    invalid(
        solver.fit_from(
            x(),
            Targets::Probabilities(q),
            &Start::default().with_initial_dual(&outside),
        ),
        "feasible set of the distillation objective",
    );
    // Rows that are not distributions, and the wrong number of classes.
    let mut bad = q_data;
    bad[0] = 0.9;
    invalid(
        solver.fit(
            x(),
            Targets::Probabilities(Matrix::new(&bad, 4, 3).unwrap()),
        ),
        "sum to one",
    );
    bad[0] = -0.1;
    bad[1] = 1.0;
    invalid(
        solver.fit(
            x(),
            Targets::Probabilities(Matrix::new(&bad, 4, 3).unwrap()),
        ),
        "nonnegative",
    );
    invalid(
        solver
            .clone()
            .with_num_classes(4)
            .fit(x(), Targets::Probabilities(q)),
        "num_classes = 4 columns",
    );
}

#[test]
fn every_loss_checks_its_target_kind() {
    let z = Matrix::new(&X[..6], 2, 3).unwrap();
    let sets = [vec![0, 1], vec![2]];
    let q = Matrix::new(&[0.5, 0.5, 0.0, 0.0, 0.5, 0.5], 2, 3).unwrap();
    let labels = [0, 2];
    let single = [
        Loss::cross_entropy(),
        Loss::topk_cross_entropy(2),
        Loss::multiclass_hinge(),
        Loss::topk_hinge_alpha(2),
        Loss::topk_hinge(2),
        Loss::proximal_cross_entropy(),
    ];
    for loss in single {
        assert!(loss.loss_and_grad(z, &labels).is_ok(), "{loss}");
        invalid(loss.loss_and_grad(z, &sets), "takes one label per example");
        invalid(
            loss.loss_and_grad(z, Targets::Teacher(z)),
            "not Targets::Teacher",
        );
        invalid(
            loss.loss_and_grad(z, Targets::Probabilities(q)),
            "not Targets::Probabilities",
        );
    }
    invalid(
        Loss::cross_entropy().loss_and_grad(z, &sets),
        "use Loss::MultilabelCrossEntropy",
    );
    for loss in [Loss::multilabel_hinge(), Loss::multilabel_cross_entropy()] {
        assert!(loss.loss_and_grad(z, &sets).is_ok(), "{loss}");
        invalid(
            loss.loss_and_grad(z, &labels),
            "takes label sets, not Targets::Labels",
        );
        invalid(
            loss.loss_and_grad(z, Targets::Teacher(z)),
            "takes label sets",
        );
    }
    for loss in [Loss::distillation(), Loss::proximal_distillation()] {
        assert!(loss.loss_and_grad(z, Targets::Teacher(z)).is_ok(), "{loss}");
        invalid(loss.loss_and_grad(z, &labels), "takes teacher logits");
        invalid(
            loss.loss_and_grad(z, Targets::Probabilities(q)),
            "takes teacher logits",
        );
    }
    // Training: the same kinds, plus probabilities for cross-entropy.
    invalid(
        Solver::default().fit(x(), &[vec![0, 1], vec![1], vec![2], vec![0]]),
        "label sets",
    );
    invalid(
        Solver::new(Loss::multilabel_hinge()).fit(x(), &Y),
        "takes label sets",
    );
    let q4 = Matrix::new(&[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 4, 2).unwrap();
    invalid(
        Solver::new(Loss::topk_hinge(1)).fit(x(), Targets::Probabilities(q4)),
        "does not take Targets::Probabilities",
    );
    invalid(
        Solver::default().fit(x(), Targets::Teacher(q4)),
        "not Targets::Teacher",
    );
    invalid(
        Solver::default().coordinate_update(z, Targets::Teacher(z), 1.0, None),
        "not Targets::Teacher",
    );
}

#[test]
fn losses_that_are_not_objectives_do_not_train() {
    for loss in [
        Loss::proximal_cross_entropy(),
        Loss::distillation(),
        Loss::proximal_distillation(),
    ] {
        invalid(Solver::new(loss).fit(x(), &Y), "not a training objective");
        let z = Matrix::new(&X[..6], 2, 3).unwrap();
        invalid(
            Solver::new(loss).coordinate_update(z, &[0, 1], 1.0, None),
            "not a training objective",
        );
    }
    invalid(
        Solver::new(Loss::distillation()).fit(x(), &Y),
        "Targets::Probabilities",
    );
}

#[test]
fn distillation_losses() {
    let z = Matrix::new(&[2.0, -0.5, 1.0, 0.3, 0.1, -2.0], 2, 3).unwrap();
    let t = Matrix::new(&[1.0, 0.0, 0.5, -1.0, 2.0, 0.0], 2, 3).unwrap();
    // Equal logits: a zero gradient.
    let (dv, dg) = Loss::distillation()
        .loss_and_grad(z, Targets::Teacher(z))
        .unwrap();
    assert!(dg.iter().all(|&g| g == 0.0));
    let proximal = Loss::ProximalDistillation {
        temperature: 1.0,
        alpha: 0.5,
    };
    let (pv, pg) = proximal.loss_and_grad(z, Targets::Teacher(z)).unwrap();
    assert!(pv.iter().zip(&dv).all(|(p, d)| (p - d).abs() < 1e-14));
    assert!(pg.iter().all(|&g: &f64| g.abs() < 1e-15));
    // The cross-entropy of the softmaxes at T, and the gradient (q - p) / T.
    let temperature = 2.0;
    let (v, g) = Loss::Distillation { temperature }
        .loss_and_grad(z, Targets::Teacher(t))
        .unwrap();
    for i in 0..2 {
        let scaled =
            |m: Matrix<'_, f64>| -> Vec<f64> { m.row(i).iter().map(|v| v / temperature).collect() };
        let (p, q) = (softmax(&scaled(t)), softmax(&scaled(z)));
        let h: f64 = -p.iter().zip(&q).map(|(p, q)| p * q.ln()).sum::<f64>();
        assert!((v[i] - h).abs() < 1e-14, "{} {h}", v[i]);
        for j in 0..3 {
            assert!((g[i * 3 + j] - (q[j] - p[j]) / temperature).abs() < 1e-15);
        }
    }
    // Proximal distillation lies below plain distillation.
    let (pv, _) = Loss::proximal_distillation()
        .loss_and_grad(z, Targets::Teacher(t))
        .unwrap();
    let (dv, _) = Loss::distillation()
        .loss_and_grad(z, Targets::Teacher(t))
        .unwrap();
    assert!(pv.iter().zip(&dv).all(|(p, d)| p < d));
    // Shapes and parameters.
    let short = Matrix::new(&[1.0, 0.0, 0.5], 1, 3).unwrap();
    invalid(
        Loss::distillation().loss_and_grad(z, Targets::Teacher(short)),
        "shape of the logits",
    );
    let nan = Matrix::new(&[f64::NAN; 6], 2, 3).unwrap();
    invalid(
        Loss::distillation().loss_and_grad(z, Targets::Teacher(nan)),
        "Teacher logits must be finite",
    );
    let bad = [
        Loss::Distillation { temperature: 0.0 },
        Loss::ProximalDistillation {
            temperature: f64::NAN,
            alpha: 1.0,
        },
    ];
    for loss in bad {
        invalid(
            loss.loss_and_grad(z, Targets::Teacher(t)),
            "temperature must be",
        );
    }
    let no_step = Loss::ProximalDistillation {
        temperature: 1.0,
        alpha: 0.0,
    };
    invalid(
        no_step.loss_and_grad(z, Targets::Teacher(t)),
        "alpha must be",
    );
}

#[test]
fn proximal_distillation_rejects_float32_work() {
    let z32 = [2.0f32, -0.5, 1.0];
    let z32 = Matrix::new(&z32, 1, 3).unwrap();
    let loss = Loss::proximal_distillation();
    let float = Settings::default().with_precision(Precision::Float32);
    invalid(
        loss.loss_and_grad_with(z32, Targets::Teacher(z32), &float),
        "float64",
    );
    // The default precision of f32 data and Float64 compute the same, in double.
    let double = Settings::default().with_precision(Precision::Float64);
    let a = loss.loss_and_grad(z32, Targets::Teacher(z32)).unwrap();
    let b = loss
        .loss_and_grad_with(z32, Targets::Teacher(z32), &double)
        .unwrap();
    assert_eq!(a, b);
    // Plain distillation computes in float on request.
    let plain = Loss::distillation();
    assert!(plain
        .loss_and_grad_with(z32, Targets::Teacher(z32), &float)
        .is_ok());
}

#[test]
fn coordinate_update_is_the_entropy_fixed_point() {
    let z = [2.0, -0.5, 1.0, 0.3, 0.1, -2.0];
    let (c, norm2) = (3.0, 0.7);
    let solver = Solver::default().with_c(c);
    let a = solver
        .coordinate_update(Matrix::new(&z, 2, 3).unwrap(), &[0, 2], norm2, None)
        .unwrap();
    for (i, label) in [0usize, 2].into_iter().enumerate() {
        let u: Vec<f64> = (0..3)
            .map(|j| z[i * 3 + j] + norm2 * a[i * 3 + j])
            .collect();
        let p = softmax(&u);
        for j in 0..3 {
            let target = if j == label { 1.0 } else { 0.0 };
            assert!((a[i * 3 + j] / c - (target - p[j])).abs() < 1e-12);
        }
    }
    // Zero previous duals are the default.
    let zeros = [0.0; 6];
    let from_zeros = solver
        .coordinate_update(Matrix::new(&z, 2, 3).unwrap(), &[0, 2], norm2, Some(&zeros))
        .unwrap();
    assert_eq!(a, from_zeros);
    // The step of a multilabel objective takes label sets.
    let sets = [vec![0, 1], vec![2]];
    let a = Solver::new(Loss::multilabel_cross_entropy())
        .coordinate_update(Matrix::new(&z, 2, 3).unwrap(), &sets, 1.0, None)
        .unwrap();
    assert!(a.chunks(3).all(|row| row.iter().sum::<f64>().abs() < 1e-14));
    invalid(
        Solver::default().with_num_classes(4).coordinate_update(
            Matrix::new(&z, 2, 3).unwrap(),
            &[0, 2],
            1.0,
            None,
        ),
        "num_classes = 4 columns",
    );
}

#[test]
fn loss_and_gradient_match_softmax() {
    let z = [2.0, -0.5, 1.0];
    let (values, grad) = Loss::cross_entropy()
        .loss_and_grad(Matrix::new(&z, 1, 3).unwrap(), &[2])
        .unwrap();
    let p = softmax(&z);
    assert!((values[0] + p[2].ln()).abs() < 1e-14);
    for j in 0..3 {
        assert!((grad[j] - (p[j] - if j == 2 { 1.0 } else { 0.0 })).abs() < 1e-14);
    }
}

#[test]
fn float32_losses_are_close_to_float64() {
    let z: Vec<f32> = (0..30).map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.7).collect();
    let y: Vec<usize> = (0..6).map(|i| i % 5).collect();
    let eval = |loss: Loss, precision| {
        let settings = Settings::default().with_precision(precision);
        loss.loss_and_grad_with(Matrix::new(&z, 6, 5).unwrap(), &y, &settings)
            .unwrap()
    };
    let losses = [
        Loss::CrossEntropy,
        Loss::TopKCrossEntropy { k: 2 },
        Loss::TopKHingeAlpha { k: 2, gamma: 0.5 },
        Loss::ProximalCrossEntropy { alpha: 1.0 },
    ];
    for loss in losses {
        let (v32, g32) = eval(loss, Precision::Float32);
        let (v64, g64) = eval(loss, Precision::Float64);
        assert!(v32.iter().zip(&v64).all(|(a, b)| (a - b).abs() < 1e-5));
        assert!(g32.iter().zip(&g64).all(|(a, b)| (a - b).abs() < 1e-5));
    }
}

#[test]
fn float32_and_float64_compute_on_f32_storage() {
    let z: Vec<f32> = (0..36).map(|i| ((i * 7 % 11) as f32 - 5.0) * 3.0).collect();
    let y: Vec<usize> = (0..6).collect();
    let solver = Solver::default().with_c(10.0);
    let z64: Vec<f64> = z.iter().map(|&v| f64::from(v)).collect();
    let exact = solver
        .coordinate_update(Matrix::new(&z64, 6, 6).unwrap(), &y, 1.0, None)
        .unwrap();
    let update = |solver: Solver| {
        solver
            .coordinate_update(Matrix::new(&z, 6, 6).unwrap(), &y, 1.0, None)
            .unwrap()
    };
    assert_eq!(
        update(solver.clone()),
        update(solver.clone().with_precision(Precision::Float32))
    );
    for precision in [Precision::Float32, Precision::Float64] {
        let a = update(solver.clone().with_precision(precision));
        let error = a
            .iter()
            .zip(&exact)
            .map(|(u, v)| (f64::from(*u) - v).abs())
            .fold(0.0, f64::max);
        assert!(error < 1e-4 * 10.0, "{precision}: {error}");
    }
}

#[test]
fn warm_start_and_its_validation() {
    let a = [0.4, -0.2, -0.2];
    let solver = Solver::default().with_num_classes(3);
    let one = Matrix::new(&[2.0], 1, 1).unwrap();
    let start = Start::default().with_initial_dual(&a);
    let model = solver
        .clone()
        .with_max_epoch(0)
        .fit_from(one, &[0], &start)
        .unwrap();
    assert_eq!(model.dual(), a);
    assert_eq!(model.weights().unwrap(), [0.8, -0.4, -0.4]);
    let bad = [1.1, -0.6, -0.5];
    invalid(
        solver.fit_from(
            Matrix::new(&[1.0], 1, 1).unwrap(),
            &[0],
            &Start::default().with_initial_dual(&bad),
        ),
        "feasible set of the cross-entropy objective",
    );
}

#[test]
fn invalid_input() {
    invalid(Matrix::new(&X, 3, 2), "not 3x2");
    let solver = Solver::default();
    invalid(
        solver.fit(x(), &[0, 1, 3]),
        "one entry per example (4), not 3",
    );
    invalid(
        solver.clone().with_num_classes(3).fit(x(), &[0, 1, 5, 0]),
        "labels must lie in [0, 3)",
    );
    invalid(
        Solver::new(Loss::topk_hinge(3)).fit(x(), &Y),
        "k must be in [1, 2]",
    );
    invalid(
        Solver::new(Loss::topk_cross_entropy(0)).fit(x(), &Y),
        "k must be in",
    );
    invalid(solver.clone().with_c(0.0).fit(x(), &Y), "C must be finite");
    invalid(
        solver.fit(Input::Kernel(x()), &Y),
        "must be a square Gram matrix",
    );
    // Asymmetric and indefinite (|K_01| > sqrt(K_00 K_11)) kernels.
    for gram in [[1.0, 0.5, 0.4, 1.0], [1.0, 2.0, 2.0, 1.0]] {
        invalid(
            solver.fit(Input::Kernel(Matrix::new(&gram, 2, 2).unwrap()), &[0, 1]),
            "Training kernel must be",
        );
    }
    let full = [vec![0, 1, 2], vec![1], vec![2], vec![0]];
    invalid(
        Solver::new(Loss::multilabel_hinge()).fit(x(), &full),
        "between 1 and 2 labels",
    );
    invalid(
        Solver::new(Loss::multilabel_cross_entropy())
            .fit(x(), Targets::Sets(&[vec![0], vec![1], vec![1, 1], vec![0]])),
        "duplicates",
    );
    let loss = Loss::cross_entropy();
    invalid(
        loss.loss_and_grad(Matrix::new(&[1.0], 1, 1).unwrap(), &[0]),
        "logits must have at least one row and two columns",
    );
    invalid(
        loss.loss_and_grad(Matrix::new(&[1.0, f64::NAN], 1, 2).unwrap(), &[0]),
        "logits must be finite",
    );
    invalid(
        loss.loss_and_grad(Matrix::new(&[1.0, 0.0], 1, 2).unwrap(), &[2]),
        "labels must lie in [0, 2)",
    );
    invalid(
        loss.loss_and_grad(Matrix::new(&[1.0, 0.0], 1, 2).unwrap(), &[0, 1]),
        "one entry per example (1), not 2",
    );
    invalid(
        solver.coordinate_update(
            Matrix::new(&[1.0, f64::INFINITY], 1, 2).unwrap(),
            &[0],
            1.0,
            None,
        ),
        "finite",
    );
    // Multilabel losses need a non-label class.
    invalid(
        Loss::multilabel_hinge()
            .loss_and_grad(Matrix::new(&[0.0, 0.0], 1, 2).unwrap(), &[vec![0, 1]]),
        "between 1 and 1 labels",
    );
}

#[test]
fn oversized_epoch_limits_are_rejected_before_the_native_call() {
    for max_epoch in [usize::MAX - 1, usize::MAX / 20] {
        let solver = Solver::default().with_max_epoch(max_epoch).with_tol(1e300);
        invalid(solver.fit(x(), &Y), "too large for the history buffer");
    }
    invalid(
        Solver::default().fit(Matrix::new(&[1.0], 1, 1).unwrap(), &[usize::MAX]),
        "native BLAS integer limit",
    );
    invalid(
        Solver::default().with_num_classes(usize::MAX).fit(x(), &Y),
        "native BLAS integer limit",
    );
}

#[test]
fn warm_start_signs_are_exact() {
    let a = [-1e-16, 1e-16];
    let start = Start::default().with_initial_dual(&a);
    invalid(
        Solver::default().with_num_classes(2).fit_from(
            Matrix::new(&[1.0], 1, 1).unwrap(),
            &[0],
            &start,
        ),
        "the other classes must be nonpositive",
    );
}

#[test]
fn label_sets_need_a_class_outside_them() {
    // The steps of the multilabel objectives need one, and the losses follow them.
    let z = Matrix::new(&[0.0f64, 0.0, 0.0], 1, 3).unwrap();
    for loss in [Loss::multilabel_cross_entropy(), Loss::multilabel_hinge()] {
        invalid(loss.loss_and_grad(z, &[vec![0, 1, 2]]), "between 1 and 2");
        let (value, grad) = loss.loss_and_grad(z, &[vec![0, 1]]).unwrap();
        assert_eq!((value.len(), grad.len()), (1, 3));
    }
    // Multilabel cross-entropy at equal scores: the log of the number of classes.
    let (value, _) = Loss::multilabel_cross_entropy()
        .loss_and_grad(z, &[vec![0, 1]])
        .unwrap();
    assert!((value[0] - 3f64.ln()).abs() < 1e-15);
}

#[test]
fn prediction_checks_the_columns() {
    let model = Solver::default().with_max_epoch(2).fit(x(), &Y).unwrap();
    assert_eq!(model.num_classes(), 3);
    invalid(
        model.decision_function(Matrix::new(&[1.0, 2.0, 3.0], 1, 3).unwrap()),
        "2 columns",
    );
    // An empty matrix can carry any column count without allocating. Checking
    // that shape must not overflow before reporting the invalid dimension.
    invalid(
        model.decision_function(Matrix::new(&[], 0, usize::MAX).unwrap()),
        "2 columns",
    );
    assert!(model
        .decision_function(Matrix::new(&[], 0, 2).unwrap())
        .unwrap()
        .is_empty());
}

#[test]
fn errors_display_their_message() {
    let error = Error::InvalidArgument("bad".into());
    assert_eq!(error.to_string(), "bad");
    let native: Box<dyn std::error::Error> = Box::new(Error::Native("failed".into()));
    assert_eq!(native.to_string(), "failed");
}

#[test]
fn history_and_model_accessors() {
    let solver = Solver::default()
        .with_max_epoch(4)
        .with_eval_epoch(2)
        .with_eval_on_start(true)
        .with_tol(0.0);
    let model = solver.fit(x(), &Y).unwrap();
    let history = model.history();
    assert!(history.len() >= 3);
    let epochs: Vec<_> = history.iter().map(|e| e.epoch).collect();
    assert_eq!(epochs[..3], [0, 2, 4]);
    let end = last(&model);
    assert!((end.primal - (end.regularizer + end.primal_loss)).abs() <= 1e-12 * end.primal);
    assert!(end.dual <= end.primal && end.relative_gap >= 0.0);
    assert!((0.0..=1.0).contains(&end.accuracy) && end.wall_time >= 0.0 && end.cpu_time >= 0.0);
    assert_eq!(
        (model.epochs(), model.is_kernel(), model.status()),
        (4, false, Status::MaxEpoch)
    );
    // The rank loss of the multilabel losses.
    let sets = vec![vec![0, 1], vec![1], vec![2], vec![0]];
    let multilabel = Solver::new(Loss::multilabel_cross_entropy())
        .with_max_epoch(4)
        .fit(x(), &sets)
        .unwrap();
    assert!((0.0..=1.0).contains(&last(&multilabel).accuracy));
}

#[test]
fn solver_defaults_and_getters() {
    let solver = Solver::default();
    assert_eq!(solver.loss(), Loss::CrossEntropy);
    assert_eq!((solver.c(), solver.tol()), (1.0, 1e-3));
    assert_eq!((solver.max_epoch(), solver.eval_epoch()), (1000, 10));
    assert!(!solver.eval_on_start() && solver.recompute_primal());
    assert_eq!(solver.shrinking(), Shrinking::Auto);
    assert_eq!((solver.seed(), solver.num_classes()), (0, None));
    assert_eq!(solver.settings(), Settings::default());
    let settings = Settings::default();
    assert_eq!(settings.precision, None);
    assert_eq!(settings.threads, Threads::FromEnv);
    assert_eq!(settings.projection_precision, ProjectionPrecision::Float64);
    let two = NonZeroUsize::new(2).unwrap();
    let tuned = Solver::new(Loss::topk_hinge(2))
        .with_c(5.0)
        .with_tol(1e-6)
        .with_max_epoch(7)
        .with_eval_epoch(0)
        .with_eval_on_start(true)
        .with_recompute_primal(false)
        .with_shrinking(Shrinking::Off)
        .with_seed(7)
        .with_num_classes(4)
        .with_precision(Precision::Float64)
        .with_threads(Threads::Count(two))
        .with_projection_precision(ProjectionPrecision::Extended);
    assert_eq!(tuned.loss(), Loss::topk_hinge(2));
    assert_eq!(
        (
            tuned.c(),
            tuned.tol(),
            tuned.max_epoch(),
            tuned.eval_epoch()
        ),
        (5.0, 1e-6, 7, 0)
    );
    assert!(tuned.eval_on_start() && !tuned.recompute_primal());
    assert_eq!(
        (tuned.shrinking(), tuned.seed(), tuned.num_classes()),
        (Shrinking::Off, 7, Some(4))
    );
    assert_eq!(
        tuned.settings(),
        Settings::default()
            .with_precision(Precision::Float64)
            .with_threads(Threads::Count(two))
            .with_projection_precision(ProjectionPrecision::Extended)
    );
    assert_eq!(
        Solver::default().with_settings(tuned.settings()).settings(),
        tuned.settings()
    );
}

#[test]
fn the_seed_orders_the_epochs() {
    let solver = Solver::default().with_max_epoch(3).with_tol(0.0);
    let fit = |seed| solver.clone().with_seed(seed).fit(x(), &Y).unwrap();
    // Seeds equal modulo 2^31 - 2 give the same order.
    assert_eq!(fit(5).dual(), fit(5 + (1 << 31) - 2).dual());
    assert_ne!(fit(0).dual(), fit(1).dual());
}

#[test]
fn coordinate_steps_check_previous_duals() {
    let z = Matrix::new(&[1.0, 2.0, 3.0], 1, 3).unwrap();
    let solver = Solver::default();
    invalid(
        solver.coordinate_update(z, &[0], 1.0, Some(&[2.0, -1.0, -1.0])),
        "previous is outside the feasible set",
    );
    let a = solver.coordinate_update(z, &[0], 1.0, None).unwrap();
    assert!(solver.coordinate_update(z, &[0], 1.0, Some(&a)).is_ok());
}

#[test]
fn recompute_primal_is_on_by_default_and_keeps_the_duals() {
    let on = Solver::default().with_max_epoch(5).with_eval_epoch(5);
    let off = on.clone().with_recompute_primal(false);
    assert!(on.recompute_primal());
    let (a, b) = (on.fit(x(), &Y).unwrap(), off.fit(x(), &Y).unwrap());
    assert_eq!(a.dual(), b.dual());
}

#[test]
fn projection_precision_is_float64_by_default_and_extended_agrees() {
    assert_eq!(ProjectionPrecision::default(), ProjectionPrecision::Float64);
    assert_eq!(
        "extended".parse::<ProjectionPrecision>(),
        Ok(ProjectionPrecision::Extended)
    );
    invalid(
        "double".parse::<ProjectionPrecision>(),
        "projection_precision",
    );
    let sets = vec![vec![0usize, 1], vec![1], vec![0, 2], vec![2]];
    let loss = Loss::MultilabelHinge { gamma: 0.5 };
    let fits: Vec<_> = ProjectionPrecision::ALL
        .iter()
        .map(|&mode| {
            Solver::new(loss)
                .with_tol(1e-10)
                .with_projection_precision(mode)
                .fit(x(), &sets)
                .unwrap()
        })
        .collect();
    // Where long double is double (Apple arm64, MSVC), the extended solver rounds differently
    // from the float64 one and the fits stop at different solved points: each within
    // sqrt(2 gap) of the optimum, as the objective is 1-strongly convex in W.
    let radius = |m: &Model<f64>| {
        let end = last(m);
        (2.0 * (end.relative_gap * end.primal.abs()).max(0.0)).sqrt()
    };
    let narrow = cfg!(any(
        target_env = "msvc",
        all(target_vendor = "apple", target_arch = "aarch64")
    ));
    let tolerance = if narrow {
        radius(&fits[0]) + radius(&fits[1])
    } else {
        1e-12
    };
    assert!(fits.iter().all(|m| m.status() == Status::Solved));
    let close = |a: &[f64], b: &[f64]| a.iter().zip(b).all(|(u, v)| (u - v).abs() <= tolerance);
    assert!(close(
        fits[0].weights().unwrap(),
        fits[1].weights().unwrap()
    ));
    let z = [0.3, -0.2, 0.9, 0.1, 0.4, -0.7];
    let losses: Vec<_> = ProjectionPrecision::ALL
        .iter()
        .map(|&mode| {
            let settings = Settings::default().with_projection_precision(mode);
            loss.loss_and_grad_with(Matrix::new(&z, 2, 3).unwrap(), &sets[..2], &settings)
                .unwrap()
        })
        .collect();
    let close_loss = |a: &[f64], b: &[f64]| a.iter().zip(b).all(|(u, v)| (u - v).abs() <= 1e-12);
    assert!(close_loss(&losses[0].0, &losses[1].0) && close_loss(&losses[0].1, &losses[1].1));
}

#[test]
fn targets_convert_from_slices_arrays_and_vectors() {
    let labels = vec![0usize, 1];
    let sets = vec![vec![0usize], vec![1, 2]];
    assert_eq!(Targets::<f64>::from(&labels).len(), 2);
    assert_eq!(Targets::<f64>::from(labels.as_slice()).len(), 2);
    assert_eq!(Targets::<f64>::from(&sets).len(), 2);
    assert_eq!(Targets::<f64>::from(sets.as_slice()).len(), 2);
    assert_eq!(Targets::<f64>::from(&[vec![0usize]]).len(), 1);
    assert!(Targets::<f64>::Labels(&[]).is_empty());
    let q = Matrix::new(&[0.5f32, 0.5], 1, 2).unwrap();
    assert_eq!(
        (Targets::Probabilities(q).len(), Targets::Teacher(q).len()),
        (1, 1)
    );
}

#[test]
fn names_parse_and_display() {
    for &status in Status::ALL {
        assert_eq!(status.name().parse::<Status>(), Ok(status));
    }
    let names: Vec<_> = Status::ALL.iter().map(|s| s.name()).collect();
    assert_eq!(names, ["solved", "no_progress", "max_epoch", "failed"]);
    for state in ["none", "solving", "max_cpu_time", "max_wall_time"] {
        invalid(state.parse::<Status>(), "Unknown status");
    }
    for &precision in Precision::ALL {
        assert_eq!(precision.to_string().parse::<Precision>(), Ok(precision));
    }
    let every = [
        Loss::cross_entropy(),
        Loss::topk_cross_entropy(1),
        Loss::multiclass_hinge(),
        Loss::topk_hinge_alpha(1),
        Loss::topk_hinge_beta(1),
        Loss::multilabel_hinge(),
        Loss::multilabel_cross_entropy(),
        Loss::proximal_cross_entropy(),
        Loss::distillation(),
        Loss::proximal_distillation(),
    ];
    for loss in every {
        assert_eq!(loss.name().parse::<Loss>(), Ok(loss));
        assert_eq!(loss.to_string(), loss.name());
    }
    // Display and the parsed defaults leave out the parameters.
    let smooth = Loss::TopKHingeBeta { k: 2, gamma: 0.5 };
    assert_eq!(smooth.to_string(), "topk_hinge_beta");
    // Canonical names only; smoothing is gamma, not a name.
    for alias in [
        "topk_hinge",
        "ce",
        "softmax",
        "svm",
        "topk_hinge_beta_smooth",
    ] {
        invalid(alias.parse::<Loss>(), "Unknown loss");
    }
}

#[test]
fn eval_epoch_zero_evaluates_at_the_end() {
    // As in C++ and MATLAB: one evaluation, after max_epoch; it decides the status.
    let model = Solver::default()
        .with_eval_epoch(0)
        .with_max_epoch(5)
        .fit(x(), &Y)
        .unwrap();
    assert_eq!(model.epochs(), 5);
    assert_eq!(model.history().len(), 1);
}

#[test]
fn invalid_options() {
    let solver = Solver::default();
    invalid(
        solver.clone().with_tol(-1.0).fit(x(), &Y),
        "tol must be finite",
    );
    invalid(
        solver.clone().with_num_classes(1).fit(x(), &Y),
        "num_classes must be at least 2",
    );
    invalid(solver.clone().with_c(1e-320).fit(x(), &Y), "must be normal");
    let center = Start::default().with_center(&[0.0; 5]);
    invalid(solver.fit_from(x(), &Y, &center), "Start::center must have");
    let nan = [f64::NAN; 6];
    invalid(
        solver.fit_from(x(), &Y, &Start::default().with_center(&nan)),
        "center must be finite",
    );
    let dual = Start::default().with_initial_dual(&[0.0; 5]);
    invalid(solver.fit_from(x(), &Y, &dual), "initial_dual must have");
    invalid(
        solver.fit(Matrix::<f64>::new(&[], 0, 2).unwrap(), Targets::Labels(&[])),
        "nonempty matrix",
    );
    let gram = Matrix::new(&[1.0, 0.0, 0.0, 1.0], 2, 2).unwrap();
    let kernel_center = Start::default().with_center(&[0.0; 4]);
    invalid(
        solver.fit_from(Input::Kernel(gram), &[0, 1], &kernel_center),
        "Start::center is for Input::Features, not Input::Kernel",
    );
    let z = Matrix::new(&[1.0, 2.0, 3.0], 1, 3).unwrap();
    let rough = Loss::MulticlassHinge { gamma: -1.0 };
    invalid(
        rough.loss_and_grad(z, &[0]),
        "gamma must be finite and nonnegative",
    );
    invalid(Solver::new(rough).fit(x(), &Y), "gamma must be");
    let step = Loss::ProximalCrossEntropy {
        alpha: f64::INFINITY,
    };
    invalid(
        step.loss_and_grad(z, &[0]),
        "alpha must be finite and positive",
    );
    invalid(
        solver.coordinate_update(z, &[0], 1.0, Some(&[0.0; 2])),
        "previous must have the 1x3 entries",
    );
    invalid(
        solver.coordinate_update(z, &[0], 1.0, Some(&[0.0, f64::NAN, 0.0])),
        "previous must be finite",
    );
    invalid(solver.coordinate_update(z, &[0], f64::NAN, None), "norm2");
    invalid(
        solver
            .clone()
            .with_c(f64::MAX)
            .coordinate_update(z, &[0], 2.0, None),
        "C, norm2, and C*norm2 must be normal",
    );
}

#[test]
fn warm_start_feasible_sets() {
    let x1 = Matrix::new(&[1.0], 1, 1).unwrap();
    let cases: [(Loss, [f64; 3], &str); 3] = [
        (
            Loss::topk_hinge_beta(1),
            [1.1, -0.6, -0.5],
            "outside the feasible set",
        ),
        (Loss::topk_hinge_alpha(2), [0.6, -0.4, -0.2], "top-k cone"),
        (Loss::topk_hinge_beta(2), [0.9, -0.6, -0.3], "top-k cap"),
    ];
    for (loss, a, message) in cases {
        let start = Start::default().with_initial_dual(&a);
        let solver = Solver::new(loss).with_num_classes(3);
        invalid(solver.fit_from(x1, &[0], &start), message);
    }
    let unbalanced = [0.5, -0.2, -0.2];
    let start = Start::default().with_initial_dual(&unbalanced);
    invalid(
        Solver::default()
            .with_num_classes(3)
            .fit_from(x1, &[0], &start),
        "sum to zero",
    );
}

#[test]
fn cold_and_warm_starts_from_a_center() {
    // Row-major (d, classes).
    let center = [0.5, -0.25, 0.0, 0.1, 0.2, -0.3];
    let cold = Start::default().with_center(&center);
    let model = Solver::default()
        .with_max_epoch(0)
        .fit_from(x(), &Y, &cold)
        .unwrap();
    assert_eq!(model.weights().unwrap(), center);
    assert!(model.dual().iter().all(|&a| a == 0.0));
    let solver = Solver::default().with_c(10.0);
    let solved = solver.fit_from(x(), &Y, &cold).unwrap();
    let warm = cold.with_initial_dual(solved.dual());
    let model = solver.with_max_epoch(0).fit_from(x(), &Y, &warm).unwrap();
    for l in 0..2 {
        for j in 0..3 {
            let xa: f64 = (0..4).map(|i| X[i * 2 + l] * model.dual()[i * 3 + j]).sum();
            let w = model.weights().unwrap()[l * 3 + j];
            assert!((w - center[l * 3 + j] - xa).abs() < 1e-12);
        }
    }
}

#[test]
fn label_sets_borrow_slices() {
    let rows: [&[usize]; 4] = [&[0, 1], &[1], &[2], &[0]];
    let vector = rows.to_vec();
    assert_eq!(Targets::<f64>::from(&rows), Targets::SetSlices(&rows));
    assert_eq!(Targets::<f64>::from(rows.as_slice()).len(), 4);
    assert_eq!(Targets::<f64>::from(&vector).len(), 4);
    let sets: Vec<Vec<usize>> = rows.iter().map(|row| row.to_vec()).collect();
    let solver = Solver::new(Loss::multilabel_cross_entropy()).with_max_epoch(20);
    let borrowed = solver.fit(x(), &rows).unwrap();
    let owned = solver.fit(x(), &sets).unwrap();
    assert_eq!(borrowed.loss(), Loss::MultilabelCrossEntropy);
    assert_eq!(borrowed.dual(), owned.dual());
    let scores = Matrix::new(&X[..6], 2, 3).unwrap();
    let pair: [&[usize]; 2] = [&[2, 0], &[1]];
    let loss = Loss::multilabel_cross_entropy();
    assert_eq!(
        loss.loss_and_grad(scores, &pair).unwrap(),
        loss.loss_and_grad(scores, &[vec![0, 2], vec![1]]).unwrap()
    );
    let duplicate: [&[usize]; 4] = [&[1, 1], &[1], &[2], &[0]];
    invalid(solver.fit(x(), &duplicate), "duplicates");
}

#[test]
#[should_panic(expected = "row 2 out of range for 2 rows")]
fn rows_are_bounds_checked_without_columns() {
    let empty: [f64; 0] = [];
    let _ = Matrix::new(&empty, 2, 0).unwrap().row(2);
}

#[test]
fn stress_cases_match_independent_oracles() {
    // Warm entropy cancellation at A = 2^80 against 100-digit KKT oracle.
    let a80 = 2.0f64.powi(80);
    let prev = [a80, -a80 / 2.0, -a80 / 2.0, 0.0];
    let logits = [a80, -a80 / 2.0, -a80 / 2.0, 2.0];
    let oracle = [
        40.144_988_117_492_43,
        -12.760_436_816_689_61,
        -12.760_436_816_689_61,
        -14.624_114_484_113_21,
    ];
    for loss in [Loss::cross_entropy(), Loss::topk_cross_entropy(2)] {
        let out = Solver::new(loss)
            .with_c(a80)
            .coordinate_update(Matrix::new(&logits, 1, 4).unwrap(), &[0], 1.0, Some(&prev))
            .unwrap();
        let err = out
            .iter()
            .zip(oracle)
            .map(|(u, v)| (u - v).abs())
            .fold(0.0, f64::max)
            / oracle[0];
        assert!(err <= 32.0 * f64::EPSILON, "{loss}: {err}");
    }
    // Tiny hinge margins [1, 2^-26] and [1, 1e-20], and huge norm2 = 1e20.
    let h = 2.0f64.powi(-26);
    let z32 = [1.0f32, h as f32];
    for gamma in [0.0, 1.0] {
        for loss in [
            Loss::TopKHingeAlpha { k: 1, gamma },
            Loss::TopKHingeBeta { k: 1, gamma },
        ] {
            let s = h / (2.0 + gamma);
            let u = Solver::new(loss)
                .with_precision(Precision::Float64)
                .coordinate_update(Matrix::new(&z32, 1, 2).unwrap(), &[0], 1.0, None)
                .unwrap();
            assert!(((f64::from(u[0]) / s) - 1.0).abs() <= 4.0 * f64::from(f32::EPSILON));
            assert!(((f64::from(u[1]) / -s) - 1.0).abs() <= 4.0 * f64::from(f32::EPSILON));
        }
    }
    // Smoothed hinge value at [0, 1e200], gamma = 1e300 is 5e99, not NaN.
    let (v, g) = Loss::TopKHingeBeta { k: 1, gamma: 1e300 }
        .loss_and_grad(Matrix::new(&[0.0, 1e200], 1, 2).unwrap(), &[0])
        .unwrap();
    assert!((v[0] / 5e99 - 1.0).abs() < 1e-14);
    assert!((g[1] / 1e-100 - 1.0_f64).abs() < 1e-14);
    // proximal_cross_entropy at a tiny step: the cross-entropy less
    // alpha / 2 |gradient|^2, to first order.
    let z = [2.0, -0.5, 1.0];
    let alpha = 1e-6;
    let (vp, gp) = Loss::ProximalCrossEntropy { alpha }
        .loss_and_grad(Matrix::new(&z, 1, 3).unwrap(), &[2])
        .unwrap();
    let (vc, gc) = Loss::cross_entropy()
        .loss_and_grad(Matrix::new(&z, 1, 3).unwrap(), &[2])
        .unwrap();
    let squared: f64 = gc.iter().map(|g| g * g).sum();
    assert!((vp[0] - (vc[0] - alpha / 2.0 * squared)).abs() < 1e-10);
    assert!(gp.iter().zip(&gc).all(|(p, c)| (p - c).abs() < 1e-5));
}

/// Scores of several examples and classes, no two margins tied.
fn scores(n: usize, m: usize) -> Vec<f64> {
    (0..n * m)
        .map(|i| 2.0 * ((i * i + 3 * i) as f64 * 0.37).sin())
        .collect()
}

#[test]
fn hinge_losses_match_closed_forms_on_several_examples() {
    let (n, m) = (5, 4);
    let z = scores(n, m);
    let y: Vec<usize> = (0..n).map(|i| (3 * i + 1) % m).collect();
    // Two positive labels per example: its label and the next class.
    let sets: Vec<Vec<usize>> = y
        .iter()
        .map(|&l| vec![l.min((l + 1) % m), l.max((l + 1) % m)])
        .collect();
    for loss in [
        Loss::multiclass_hinge(),
        Loss::topk_hinge_alpha(2),
        Loss::topk_hinge_beta(2),
        Loss::multilabel_hinge(),
    ] {
        let matrix = Matrix::new(&z, n, m).unwrap();
        let (values, grad) = if loss.is_multilabel() {
            loss.loss_and_grad(matrix, &sets)
        } else {
            loss.loss_and_grad(matrix, &y)
        }
        .unwrap();
        let k = match loss {
            Loss::TopKHingeAlpha { k, .. } | Loss::TopKHingeBeta { k, .. } => k,
            _ => 1,
        };
        for i in 0..n {
            let row = &z[i * m..(i + 1) * m];
            let mut expected = vec![0.0; m];
            let value;
            if loss.is_multilabel() {
                let positive = |j: usize| sets[i].contains(&j);
                let low = (0..m)
                    .filter(|&j| positive(j))
                    .min_by(|&a, &b| row[a].total_cmp(&row[b]))
                    .unwrap();
                let high = (0..m)
                    .filter(|&j| !positive(j))
                    .max_by(|&a, &b| row[a].total_cmp(&row[b]))
                    .unwrap();
                value = (1.0 + row[high] - row[low]).max(0.0);
                if value > 0.0 {
                    expected[high] = 1.0;
                    expected[low] = -1.0;
                }
            } else {
                // Margins 1 + z_j - z_y of the other classes, largest first.
                let mut others: Vec<usize> = (0..m).filter(|&j| j != y[i]).collect();
                others.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
                let margins: Vec<f64> = others[..k]
                    .iter()
                    .map(|&j| 1.0 + row[j] - row[y[i]])
                    .collect();
                let beta = matches!(loss, Loss::TopKHingeBeta { .. });
                value = if beta {
                    margins.iter().map(|h| h.max(0.0)).sum::<f64>() / k as f64
                } else {
                    (margins.iter().sum::<f64>() / k as f64).max(0.0)
                };
                for (&j, h) in others[..k].iter().zip(&margins) {
                    let active = if beta { *h > 0.0 } else { value > 0.0 };
                    if active {
                        expected[j] = 1.0 / k as f64;
                        expected[y[i]] -= 1.0 / k as f64;
                    }
                }
            }
            assert!(
                (values[i] - value).abs() < 1e-14,
                "{loss} {i}: {} {value}",
                values[i]
            );
            for j in 0..m {
                assert!(
                    (grad[i * m + j] - expected[j]).abs() < 1e-14,
                    "{loss} {i} {j}"
                );
            }
        }
    }
}

#[test]
fn every_objective_solves_at_its_reported_primal() {
    let (n, d, m) = (12, 3, 4);
    let x: Vec<f64> = (0..n * d)
        .map(|i| ((i * 7 + 2) as f64 * 0.61).sin())
        .collect();
    let y: Vec<usize> = (0..n).map(|i| i % m).collect();
    let sets: Vec<Vec<usize>> = y
        .iter()
        .map(|&l| if l == 0 { vec![0, 2] } else { vec![l] })
        .collect();
    for objective in OBJECTIVES {
        for gamma in [0.0, 0.5] {
            let loss = smoothed(objective, gamma);
            let solver = Solver::new(loss)
                .with_c(2.0)
                .with_tol(1e-8)
                .with_max_epoch(100_000);
            let matrix = Matrix::new(&x, n, d).unwrap();
            let model = if loss.is_multilabel() {
                solver.fit(matrix, &sets)
            } else {
                solver.fit(matrix, &y)
            }
            .unwrap();
            assert_eq!(model.status(), Status::Solved, "{loss:?}");
            let w = model.weights().unwrap();
            // W = X' A, and the primal of the returned weights from loss_and_grad.
            for l in 0..d {
                for j in 0..m {
                    let xa: f64 = (0..n).map(|r| x[r * d + l] * model.dual()[r * m + j]).sum();
                    assert!((w[l * m + j] - xa).abs() < 1e-12, "{loss:?}");
                }
            }
            let z: Vec<f64> = (0..n * m)
                .map(|e| (0..d).map(|l| x[(e / m) * d + l] * w[l * m + e % m]).sum())
                .collect();
            let scores = Matrix::new(&z, n, m).unwrap();
            let (values, _) = if loss.is_multilabel() {
                loss.loss_and_grad(scores, &sets)
            } else {
                loss.loss_and_grad(scores, &y)
            }
            .unwrap();
            let primal = 0.5 * w.iter().map(|v| v * v).sum::<f64>()
                + solver.c() * values.iter().sum::<f64>();
            let end = last(&model);
            assert!(
                (primal - end.primal).abs() <= 1e-12 * primal.abs(),
                "{loss:?}: {primal} {}",
                end.primal
            );
            assert!(
                primal - end.dual <= 1e-8 * primal.abs() && end.dual <= primal * (1.0 + 1e-14),
                "{loss:?}"
            );
        }
    }
}

#[test]
fn shrinking_is_auto_by_default_and_every_mode_agrees() {
    assert_eq!(Shrinking::default(), Shrinking::Auto);
    assert_eq!(Solver::default().shrinking(), Shrinking::Auto);
    assert_eq!("safe".parse::<Shrinking>(), Ok(Shrinking::Safe));
    invalid("on".parse::<Shrinking>(), "shrinking");
    let names: Vec<_> = Shrinking::ALL.iter().map(|s| s.name()).collect();
    assert_eq!(names, ["auto", "off", "heuristic", "safe", "both"]);
    let solver = Solver::new(Loss::multiclass_hinge())
        .with_tol(1e-8)
        .with_max_epoch(50000);
    let fits: Vec<_> = Shrinking::ALL
        .iter()
        .map(|&mode| solver.clone().with_shrinking(mode).fit(x(), &Y).unwrap())
        .collect();
    for model in &fits {
        assert_eq!(model.status(), Status::Solved);
        let (p, p0) = (last(model).primal, last(&fits[1]).primal);
        assert!((p - p0).abs() <= 2e-8 * p0.abs(), "{p} {p0}");
    }
    // The entropy losses update every class of every example: Auto only.
    let entropy = Solver::new(Loss::cross_entropy());
    assert!(entropy.fit(x(), &Y).is_ok());
    for &mode in &Shrinking::ALL[1..] {
        invalid(
            entropy.clone().with_shrinking(mode).fit(x(), &Y),
            "takes no shrinking",
        );
    }
}

#[test]
fn threads_do_not_change_a_fit() {
    let serial = Solver::default().with_threads(Threads::Count(NonZeroUsize::MIN));
    let a = serial.fit(x(), &Y).unwrap();
    for threads in [
        Threads::FromEnv,
        Threads::Count(NonZeroUsize::new(4).unwrap()),
    ] {
        let b = serial.clone().with_threads(threads).fit(x(), &Y).unwrap();
        assert_eq!(a.dual(), b.dual());
    }
}
