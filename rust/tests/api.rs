//! The contracts of the Rust interface, mirroring tests/python/test_api_contracts.py.
use libsdca::{
    coordinate_update, coordinate_update_distillation, distillation_loss_and_grad, fit,
    fit_distillation, loss_and_grad, proximal_distillation_loss_and_grad, DistillationOptions,
    Error, FitOptions, History, Loss, LossOptions, Matrix, Model, Objective, Precision,
    ProjectionPrecision, Real, Shrinking, Status, Targets, UpdateOptions, HISTORY,
};

const X: [f64; 8] = [1.0, 0.2, -0.3, 1.0, 0.7, -0.4, -0.9, -0.8];
const Y: [usize; 4] = [0, 1, 2, 0];

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

#[test]
fn shared_vocabulary() {
    let names: Vec<_> = Objective::ALL.iter().map(|o| o.name()).collect();
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
        Ok(Loss::ProximalCrossEntropy)
    );
    assert_eq!(
        "multilabel_cross_entropy".parse::<Loss>(),
        Ok(Loss::MultilabelCrossEntropy)
    );
    assert_eq!("float32".parse::<Precision>(), Ok(Precision::Float32));
    assert_eq!("fp32".parse::<Precision>(), Ok(Precision::Float32));
    assert_eq!("single".parse::<Precision>(), Ok(Precision::Float32));
    assert_eq!("float64".parse::<Precision>(), Ok(Precision::Float64));
    assert_eq!("fp64".parse::<Precision>(), Ok(Precision::Float64));
    assert_eq!("double".parse::<Precision>(), Ok(Precision::Float64));
    assert!("mixed".parse::<Precision>().is_err());
    assert!("typo".parse::<Objective>().is_err());
    assert_eq!(Status::Solved.to_string(), "solved");
    assert_eq!(HISTORY.len(), 10);
    assert_eq!(libsdca::version(), env!("CARGO_PKG_VERSION"));
}

#[test]
fn precision_must_match_the_data_type() {
    let (z64, z32) = ([2.0, -0.5, 1.0], [2.0f32, -0.5, 1.0]);
    let (z64, z32) = (
        Matrix::new(&z64, 1, 3).unwrap(),
        Matrix::new(&z32, 1, 3).unwrap(),
    );
    let loss = |precision| LossOptions {
        precision: Some(precision),
        ..Default::default()
    };
    invalid(
        loss_and_grad(z64, &[0], &loss(Precision::Float32)),
        "does not match f64 data",
    );
    for precision in [Precision::Float32, Precision::Float64] {
        assert!(loss_and_grad(z32, &[0], &loss(precision)).is_ok());
    }
    assert!(loss_and_grad(z64, &[0], &loss(Precision::Float64)).is_ok());
    invalid(
        coordinate_update(
            z64,
            &[0],
            &UpdateOptions {
                precision: Some(Precision::Float32),
                ..Default::default()
            },
        ),
        "does not match f64 data",
    );
    invalid(
        fit(
            x(),
            &Y,
            &FitOptions {
                precision: Some(Precision::Float32),
                ..Default::default()
            },
        ),
        "does not match f64 data",
    );
    assert_eq!(f64::DEFAULT_PRECISION, Precision::Float64);
    assert_eq!(f32::DEFAULT_PRECISION, Precision::Float64);
}

#[test]
fn f32_data_trains_in_every_precision() {
    let x32: Vec<f32> = X.iter().map(|&v| v as f32).collect();
    let x32 = Matrix::new(&x32, 4, 2).unwrap();
    let train = |precision| {
        fit(
            x32,
            &Y,
            &FitOptions {
                c: 10.0,
                precision,
                ..Default::default()
            },
        )
        .unwrap()
    };
    let exact = fit(
        x(),
        &Y,
        &FitOptions {
            c: 10.0,
            ..Default::default()
        },
    )
    .unwrap();
    let default = train(None);
    assert_eq!(default.dual(), train(Some(Precision::Float64)).dual());
    for model in [default, train(Some(Precision::Float32))] {
        assert_eq!(model.status(), Status::Solved);
        assert_eq!(model.predict(x32).unwrap(), exact.predict(x()).unwrap());
    }
}

#[test]
fn public_types_are_send_and_sync() {
    fn check<T: Send + Sync>() {}
    check::<Matrix<'static, f32>>();
    check::<Targets<'static>>();
    check::<Model<f64>>();
    check::<History>();
    check::<FitOptions<'static, f64>>();
    check::<LossOptions>();
    check::<UpdateOptions<'static, f32>>();
    check::<Error>();
    check::<(Objective, Loss, Precision, Status)>();
}

#[test]
fn the_handshake_runs_once_across_threads() {
    let threads: Vec<_> = (0..4)
        .map(|i| {
            std::thread::spawn(move || {
                let z = [i as f64, 0.0, 1.0];
                loss_and_grad(
                    Matrix::new(&z, 1, 3).unwrap(),
                    &[1],
                    &LossOptions::default(),
                )
            })
        })
        .collect();
    for thread in threads {
        assert!(thread.join().unwrap().is_ok());
    }
}

#[test]
fn fit_solves_and_weights_are_x_transpose_dual() {
    let model = fit(
        x(),
        &Y,
        &FitOptions {
            c: 10.0,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(model.status(), Status::Solved);
    assert!(model.history().get("relative_gap").unwrap().last().unwrap() <= &1e-3);
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
    let opts = FitOptions {
        tol: 1e-10,
        eval_epoch: 1,
        ..Default::default()
    };
    let model = fit(Matrix::new(&data, 4, 2).unwrap(), &Y, &opts).unwrap();
    assert_eq!(model.status(), Status::Solved);
    // Entropy at zero scores: C (e_y - 1/m) with label 0.
    let expected = [2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0];
    for (a, e) in model.dual()[..3].iter().zip(expected) {
        assert!((a - e).abs() < 1e-15, "{a} {e}");
    }
    let negative = [-1.0, 0.0, 0.0, 1.0];
    let kernel = FitOptions {
        kernel: true,
        ..opts
    };
    invalid(
        fit(Matrix::new(&negative, 2, 2).unwrap(), &Y[..2], &kernel),
        "nonnegative",
    );
    // All-zero rows take a block of size C, which must fit f32 storage.
    let zeros = [0.0f32; 2];
    for c in [1e100, 1e-100] {
        let opts = FitOptions::<f32> {
            c,
            num_classes: Some(2),
            ..Default::default()
        };
        invalid(
            fit(Matrix::new(&zeros, 2, 1).unwrap(), &Y[..2], &opts),
            "normal in the data dtype",
        );
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
    let opts = FitOptions {
        max_epoch: 20,
        tol: 0.0,
        ..Default::default()
    };
    let feature = fit(x(), &Y, &opts).unwrap();
    let kernel = fit(
        Matrix::new(&gram, 4, 4).unwrap(),
        &Y,
        &FitOptions {
            kernel: true,
            ..opts
        },
    )
    .unwrap();
    assert!(kernel.weights().is_none());
    let a = feature.decision_function(x()).unwrap();
    let b = kernel
        .decision_function(Matrix::new(&gram, 4, 4).unwrap())
        .unwrap();
    for (u, v) in a.iter().zip(&b) {
        assert!((u - v).abs() < 1e-10, "{u} {v}");
    }
}

#[test]
fn every_objective_trains() {
    let sets = vec![vec![0, 1], vec![1], vec![2], vec![0]];
    for &objective in Objective::ALL {
        let opts = FitOptions {
            objective,
            k: 2,
            max_epoch: 50,
            ..Default::default()
        };
        let model = if objective.is_multilabel() {
            fit(x(), &sets, &opts)
        } else {
            fit(x(), &Y, &opts)
        }
        .unwrap();
        assert_eq!(model.objective(), objective);
        assert!(model
            .history()
            .get("primal")
            .unwrap()
            .iter()
            .all(|v| v.is_finite()));
    }
    let model = fit(x(), &sets, &FitOptions::default()).unwrap();
    assert_eq!(model.objective(), Objective::MultilabelCrossEntropy);
    let q_data = [0.8, 0.1, 0.1, 0.1, 0.7, 0.2, 0.2, 0.2, 0.6, 0.6, 0.3, 0.1];
    let q = Matrix::new(&q_data, 4, 3).unwrap();
    let dmodel = fit_distillation(
        x(),
        q,
        &FitOptions {
            c: 2.0,
            tol: 1e-6,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(dmodel.status(), Status::Solved);
    let scores = dmodel.decision_function(x()).unwrap();
    let z = Matrix::new(&scores, 4, 3).unwrap();
    let a = coordinate_update_distillation(z, q, &UpdateOptions::default()).unwrap();
    for row in a.chunks(3) {
        assert!(row.iter().sum::<f64>().abs() < 1e-14);
    }
    let (dv, dg) = distillation_loss_and_grad(z, z, &DistillationOptions::default()).unwrap();
    assert!(dg.iter().all(|&g| g.abs() < 1e-15));
    let (pv, pg) = proximal_distillation_loss_and_grad(
        z,
        z,
        &DistillationOptions {
            temperature: 1.0,
            alpha: 0.5,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(pv.iter().zip(&dv).all(|(p, d)| (p - d).abs() < 1e-14));
    assert!(pg.iter().all(|&g| g.abs() < 1e-15));
}

#[test]
fn coordinate_update_is_the_entropy_fixed_point() {
    let z = [2.0, -0.5, 1.0, 0.3, 0.1, -2.0];
    let (c, norm2) = (3.0, 0.7);
    let opts = UpdateOptions {
        c,
        norm2,
        ..Default::default()
    };
    let a = coordinate_update(Matrix::new(&z, 2, 3).unwrap(), &[0, 2], &opts).unwrap();
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
}

#[test]
fn loss_and_gradient_match_softmax() {
    let z = [2.0, -0.5, 1.0];
    let (values, grad) = loss_and_grad(
        Matrix::new(&z, 1, 3).unwrap(),
        &[2],
        &LossOptions::default(),
    )
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
    let eval = |loss, precision| {
        let options = LossOptions {
            loss,
            k: 2,
            gamma: 0.5,
            precision,
            ..Default::default()
        };
        loss_and_grad(Matrix::new(&z, 6, 5).unwrap(), &y, &options).unwrap()
    };
    let losses = [
        Loss::CrossEntropy,
        Loss::TopKCrossEntropy,
        Loss::TopKHingeAlpha,
        Loss::ProximalCrossEntropy,
    ];
    for loss in losses {
        let (v32, g32) = eval(loss, Some(Precision::Float32));
        let (v64, g64) = eval(loss, Some(Precision::Float64));
        assert!(v32.iter().zip(&v64).all(|(a, b)| (a - b).abs() < 1e-5));
        assert!(g32.iter().zip(&g64).all(|(a, b)| (a - b).abs() < 1e-5));
    }
}

#[test]
fn float32_and_float64_compute_on_f32_storage() {
    let z: Vec<f32> = (0..36).map(|i| ((i * 7 % 11) as f32 - 5.0) * 3.0).collect();
    let y: Vec<usize> = (0..6).collect();
    let exact = coordinate_update(
        Matrix::new(&z.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(), 6, 6).unwrap(),
        &y,
        &UpdateOptions {
            c: 10.0,
            ..Default::default()
        },
    )
    .unwrap();
    let update = |precision| {
        coordinate_update(
            Matrix::new(&z, 6, 6).unwrap(),
            &y,
            &UpdateOptions {
                c: 10.0,
                precision,
                ..Default::default()
            },
        )
        .unwrap()
    };
    assert_eq!(update(None), update(Some(Precision::Float64)));
    for precision in [Precision::Float32, Precision::Float64] {
        let a = update(Some(precision));
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
    let model = fit(
        Matrix::new(&[2.0], 1, 1).unwrap(),
        &[0],
        &FitOptions {
            num_classes: Some(3),
            initial_dual: Some(&a),
            max_epoch: 0,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(model.dual(), a);
    assert_eq!(model.weights().unwrap(), [0.8, -0.4, -0.4]);
    let bad = [1.1, -0.6, -0.5];
    invalid(
        fit(
            Matrix::new(&[1.0], 1, 1).unwrap(),
            &[0],
            &FitOptions {
                num_classes: Some(3),
                initial_dual: Some(&bad),
                ..Default::default()
            },
        ),
        "entropy feasible set",
    );
}

#[test]
fn invalid_input() {
    invalid(Matrix::new(&X, 3, 2), "not 3x2");
    invalid(
        fit(x(), &[0, 1, 3], &FitOptions::default()),
        "one target per example",
    );
    invalid(
        fit(
            x(),
            &[0, 1, 5, 0],
            &FitOptions {
                num_classes: Some(3),
                ..Default::default()
            },
        ),
        "Labels must lie",
    );
    invalid(
        fit(
            x(),
            &Y,
            &FitOptions {
                k: 3,
                ..Default::default()
            },
        ),
        "k must be an integer",
    );
    invalid(
        fit(
            x(),
            &Y,
            &FitOptions {
                c: 0.0,
                ..Default::default()
            },
        ),
        "C must be finite",
    );
    invalid(
        fit(
            x(),
            &Y,
            &FitOptions {
                kernel: true,
                ..Default::default()
            },
        ),
        "must be square",
    );
    // Asymmetric and indefinite (|K_01| > sqrt(K_00 K_11)) kernels.
    for gram in [[1.0, 0.5, 0.4, 1.0], [1.0, 2.0, 2.0, 1.0]] {
        invalid(
            fit(
                Matrix::new(&gram, 2, 2).unwrap(),
                &[0, 1],
                &FitOptions {
                    kernel: true,
                    ..Default::default()
                },
            ),
            "Training kernel must be",
        );
    }
    let full = [vec![0, 1, 2], vec![1], vec![2], vec![0]];
    invalid(
        fit(
            x(),
            &full,
            &FitOptions {
                objective: Objective::MultilabelHinge,
                ..Default::default()
            },
        ),
        "between 1 and",
    );
    invalid(
        fit(
            x(),
            Targets::Sets(&[vec![0], vec![1], vec![1, 1], vec![0]]),
            &FitOptions::default(),
        ),
        "duplicates",
    );
    let loss = LossOptions::default();
    invalid(
        loss_and_grad(Matrix::new(&[1.0], 1, 1).unwrap(), &[0], &loss),
        "logits must have shape",
    );
    invalid(
        loss_and_grad(Matrix::new(&[1.0, f64::NAN], 1, 2).unwrap(), &[0], &loss),
        "finite",
    );
    invalid(
        coordinate_update(
            Matrix::new(&[1.0, f64::INFINITY], 1, 2).unwrap(),
            &[0],
            &UpdateOptions::default(),
        ),
        "finite",
    );
}

#[test]
fn oversized_epoch_limits_are_rejected_before_the_native_call() {
    for max_epoch in [usize::MAX - 1, usize::MAX / 20] {
        let opts = FitOptions {
            max_epoch,
            tol: 1e300,
            ..Default::default()
        };
        invalid(fit(x(), &Y, &opts), "too large for the history buffer");
    }
    invalid(
        fit(
            Matrix::new(&[1.0], 1, 1).unwrap(),
            &[usize::MAX],
            &FitOptions::default(),
        ),
        "native BLAS integer limit",
    );
    invalid(
        fit(
            x(),
            &Y,
            &FitOptions {
                num_classes: Some(usize::MAX),
                ..Default::default()
            },
        ),
        "native BLAS integer limit",
    );
}

#[test]
fn warm_start_signs_are_exact() {
    let a = [-1e-16, 1e-16];
    let opts = FitOptions {
        initial_dual: Some(&a),
        num_classes: Some(2),
        ..Default::default()
    };
    invalid(
        fit(Matrix::new(&[1.0], 1, 1).unwrap(), &[0], &opts),
        "non-label entries",
    );
}

#[test]
fn entropy_loss_takes_every_class_as_positive() {
    let z = [0.0f64, 0.0];
    let sets = [vec![0, 1]];
    let (value, grad) = loss_and_grad(
        Matrix::new(&z, 1, 2).unwrap(),
        &sets,
        &LossOptions::default(),
    )
    .unwrap();
    assert!((value[0] - 2f64.ln()).abs() < 1e-15);
    assert!(grad.iter().all(|g| g.abs() < 1e-15));
    let hinge = LossOptions {
        loss: Loss::MultilabelHinge,
        ..Default::default()
    };
    assert!(matches!(
        loss_and_grad(Matrix::new(&z, 1, 2).unwrap(), &sets, &hinge),
        Err(Error::InvalidArgument(_))
    ));
}

#[test]
fn prediction_checks_the_columns() {
    let model = fit(
        x(),
        &Y,
        &FitOptions {
            max_epoch: 2,
            ..Default::default()
        },
    )
    .unwrap();
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
    let opts = FitOptions {
        max_epoch: 4,
        eval_epoch: 2,
        eval_on_start: true,
        tol: 0.0,
        ..Default::default()
    };
    let model = fit(x(), &Y, &opts).unwrap();
    let history = model.history();
    assert!(history.len() >= 3 && !history.is_empty());
    assert_eq!(history.get("epoch").unwrap()[..3], [0.0, 2.0, 4.0]);
    assert!(history.get("unknown").is_none());
    let last: Vec<_> = history.last().unwrap().map(|(name, _)| name).collect();
    assert_eq!(last, HISTORY);
    assert_eq!(
        (model.epochs(), model.kernel(), model.status()),
        (4, false, Status::MaxEpoch)
    );
    let sets = vec![vec![0, 1], vec![1], vec![2], vec![0]];
    let multilabel = fit(x(), &sets, &opts).unwrap();
    assert_eq!(
        multilabel.history().get("rank_loss"),
        multilabel.history().get("accuracy")
    );
}

#[test]
fn recompute_primal_is_on_by_default_and_keeps_the_duals() {
    let on = FitOptions::<f64> {
        max_epoch: 5,
        eval_epoch: 5,
        ..Default::default()
    };
    let off = FitOptions {
        recompute_primal: false,
        ..on
    };
    assert!(on.recompute_primal);
    let (a, b) = (fit(x(), &Y, &on).unwrap(), fit(x(), &Y, &off).unwrap());
    assert_eq!(a.dual(), b.dual());
}

#[test]
fn projection_precision_is_float64_by_default_and_extended_agrees() {
    assert_eq!(
        FitOptions::<f64>::default().projection_precision,
        ProjectionPrecision::Float64
    );
    assert_eq!(
        LossOptions::default().projection_precision,
        ProjectionPrecision::Float64
    );
    assert_eq!(
        UpdateOptions::<f64>::default().projection_precision,
        ProjectionPrecision::Float64
    );
    assert_eq!(
        "extended".parse::<ProjectionPrecision>(),
        Ok(ProjectionPrecision::Extended)
    );
    invalid(
        "double".parse::<ProjectionPrecision>(),
        "projection_precision",
    );
    let sets = vec![vec![0usize, 1], vec![1], vec![0, 2], vec![2]];
    let fits: Vec<_> = ProjectionPrecision::ALL
        .iter()
        .map(|&mode| {
            let opts = FitOptions::<f64> {
                objective: Objective::MultilabelHinge,
                gamma: 0.5,
                tol: 1e-10,
                projection_precision: mode,
                ..Default::default()
            };
            fit(x(), &sets, &opts).unwrap()
        })
        .collect();
    let close = |a: &[f64], b: &[f64]| a.iter().zip(b).all(|(u, v)| (u - v).abs() < 1e-12);
    assert!(close(
        fits[0].weights().unwrap(),
        fits[1].weights().unwrap()
    ));
    let z = [0.3, -0.2, 0.9, 0.1, 0.4, -0.7];
    let losses: Vec<_> = ProjectionPrecision::ALL
        .iter()
        .map(|&mode| {
            let opts = LossOptions {
                loss: Loss::MultilabelHinge,
                gamma: 0.5,
                projection_precision: mode,
                ..Default::default()
            };
            loss_and_grad(Matrix::new(&z, 2, 3).unwrap(), &sets[..2], &opts).unwrap()
        })
        .collect();
    assert!(close(&losses[0].0, &losses[1].0) && close(&losses[0].1, &losses[1].1));
}

#[test]
fn targets_convert_from_slices_arrays_and_vectors() {
    let labels = vec![0usize, 1];
    let sets = vec![vec![0usize], vec![1, 2]];
    assert_eq!(Targets::from(&labels).len(), 2);
    assert_eq!(Targets::from(labels.as_slice()).len(), 2);
    assert_eq!(Targets::from(&sets).len(), 2);
    assert_eq!(Targets::from(sets.as_slice()).len(), 2);
    assert_eq!(Targets::from(&[vec![0usize]]).len(), 1);
    assert!(Targets::Labels(&[]).is_empty());
}

#[test]
fn names_parse_and_display() {
    for &status in Status::ALL {
        assert_eq!(status.name().parse::<Status>(), Ok(status));
    }
    for &precision in Precision::ALL {
        assert_eq!(precision.to_string().parse::<Precision>(), Ok(precision));
    }
    assert_eq!(
        Loss::ProximalCrossEntropy.to_string(),
        "proximal_cross_entropy"
    );
    invalid("typo".parse::<Status>(), "Unknown status");
    for &objective in Objective::ALL {
        assert_eq!(objective.name().parse::<Objective>(), Ok(objective));
        assert_eq!(
            objective.name().parse::<Loss>().unwrap().name(),
            objective.name()
        );
    }
    invalid(
        "proximal_cross_entropy".parse::<Objective>(),
        "Unknown objective",
    );
    assert_eq!(
        "topk_hinge".parse::<Objective>(),
        Ok(Objective::TopKHingeBeta)
    );
    assert_eq!("topk_hinge".parse::<Loss>(), Ok(Loss::TopKHingeBeta));
    assert_eq!(Objective::TopKHinge, Objective::TopKHingeBeta);
    assert_eq!(Loss::TopKHinge, Loss::TopKHingeBeta);
    // Canonical names (plus topk_hinge = topk_hinge_beta); smoothing is gamma, not a name.
    for alias in ["ce", "softmax", "svm", "topk_hinge_beta_smooth"] {
        invalid(alias.parse::<Loss>(), "Unknown loss");
        invalid(alias.parse::<Objective>(), "Unknown objective");
    }
}

#[test]
fn invalid_options() {
    let fit_x = |opts: FitOptions<'_, f64>| fit(x(), &Y, &opts);
    invalid(
        fit_x(FitOptions {
            tol: -1.0,
            ..Default::default()
        }),
        "tol must be finite",
    );
    invalid(
        fit_x(FitOptions {
            eval_epoch: 0,
            ..Default::default()
        }),
        "eval_epoch must be >=1",
    );
    invalid(
        fit_x(FitOptions {
            num_classes: Some(1),
            ..Default::default()
        }),
        "num_classes must be >=2",
    );
    invalid(
        fit_x(FitOptions {
            c: 1e-320,
            ..Default::default()
        }),
        "must be normal",
    );
    invalid(
        fit_x(FitOptions {
            center: Some(&[0.0; 5]),
            ..Default::default()
        }),
        "center must be finite",
    );
    invalid(
        fit_x(FitOptions {
            initial_dual: Some(&[0.0; 5]),
            ..Default::default()
        }),
        "initial_dual must have shape",
    );
    invalid(
        fit(
            Matrix::<f64>::new(&[], 0, 2).unwrap(),
            Targets::Labels(&[]),
            &FitOptions::default(),
        ),
        "nonempty matrix",
    );
    let gram = [1.0, 0.0, 0.0, 1.0];
    let kernel = FitOptions {
        kernel: true,
        center: Some(&[0.0; 4]),
        ..Default::default()
    };
    invalid(
        fit(Matrix::new(&gram, 2, 2).unwrap(), &[0, 1], &kernel),
        "only supported for feature",
    );
    let z = Matrix::new(&[1.0, 2.0, 3.0], 1, 3).unwrap();
    invalid(
        loss_and_grad(
            z,
            &[0],
            &LossOptions {
                gamma: -1.0,
                ..Default::default()
            },
        ),
        "gamma must be",
    );
    invalid(
        coordinate_update(
            z,
            &[0],
            &UpdateOptions {
                previous: Some(&[0.0; 2]),
                ..Default::default()
            },
        ),
        "previous must match",
    );
    invalid(
        coordinate_update(
            z,
            &[0],
            &UpdateOptions {
                norm2: f64::NAN,
                ..Default::default()
            },
        ),
        "norm2",
    );
}

#[test]
fn warm_start_feasible_sets() {
    let x1 = Matrix::new(&[1.0], 1, 1).unwrap();
    let cases: [(Objective, [f64; 3], usize, &str); 3] = [
        (
            Objective::TopKHingeBeta,
            [1.1, -0.6, -0.5],
            1,
            "outside the feasible set",
        ),
        (
            Objective::TopKHingeAlpha,
            [0.6, -0.4, -0.2],
            2,
            "top-k cone",
        ),
        (Objective::TopKHingeBeta, [0.9, -0.6, -0.3], 2, "top-k cap"),
    ];
    for (objective, a, k, message) in cases {
        let opts = FitOptions {
            objective,
            k,
            num_classes: Some(3),
            initial_dual: Some(&a),
            ..Default::default()
        };
        invalid(fit(x1, &[0], &opts), message);
    }
    let unbalanced = [0.5, -0.2, -0.2];
    let opts = FitOptions {
        num_classes: Some(3),
        initial_dual: Some(&unbalanced),
        ..Default::default()
    };
    invalid(fit(x1, &[0], &opts), "sum to zero");
}

#[test]
fn cold_and_warm_starts_from_a_center() {
    // Row-major (d, classes).
    let center = [0.5, -0.25, 0.0, 0.1, 0.2, -0.3];
    let cold = FitOptions {
        center: Some(&center),
        max_epoch: 0,
        ..Default::default()
    };
    let model = fit(x(), &Y, &cold).unwrap();
    assert_eq!(model.weights().unwrap(), center);
    assert!(model.dual().iter().all(|&a| a == 0.0));
    let solved = fit(
        x(),
        &Y,
        &FitOptions {
            c: 10.0,
            center: Some(&center),
            ..Default::default()
        },
    )
    .unwrap();
    let warm = FitOptions {
        c: 10.0,
        initial_dual: Some(solved.dual()),
        ..cold
    };
    let model = fit(x(), &Y, &warm).unwrap();
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
    assert_eq!(Targets::from(&rows), Targets::SetSlices(&rows));
    assert_eq!(Targets::from(rows.as_slice()).len(), 4);
    assert_eq!(Targets::from(&vector).len(), 4);
    let sets: Vec<Vec<usize>> = rows.iter().map(|row| row.to_vec()).collect();
    let options = FitOptions {
        max_epoch: 20,
        ..Default::default()
    };
    let borrowed = fit(x(), &rows, &options).unwrap();
    let owned = fit(x(), &sets, &options).unwrap();
    assert_eq!(borrowed.objective(), Objective::MultilabelCrossEntropy);
    assert_eq!(borrowed.dual(), owned.dual());
    let scores = Matrix::new(&X[..6], 2, 3).unwrap();
    let pair: [&[usize]; 2] = [&[2, 0], &[1]];
    assert_eq!(
        loss_and_grad(scores, &pair, &LossOptions::default()).unwrap(),
        loss_and_grad(scores, &[vec![0, 2], vec![1]], &LossOptions::default()).unwrap()
    );
    let duplicate: [&[usize]; 4] = [&[1, 1], &[1], &[2], &[0]];
    invalid(fit(x(), &duplicate, &options), "duplicates");
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
    for (objective, k) in [
        (Objective::CrossEntropy, 1),
        (Objective::TopKCrossEntropy, 2),
    ] {
        let out = coordinate_update(
            Matrix::new(&logits, 1, 4).unwrap(),
            &[0],
            &UpdateOptions {
                previous: Some(&prev),
                c: a80,
                objective,
                k,
                ..Default::default()
            },
        )
        .unwrap();
        let err = out
            .iter()
            .zip(oracle)
            .map(|(u, v)| (u - v).abs())
            .fold(0.0, f64::max)
            / oracle[0];
        assert!(err <= 32.0 * f64::EPSILON, "{objective}: {err}");
    }
    // Tiny hinge margins [1, 2^-26] and [1, 1e-20], and huge norm2 = 1e20.
    let h = 2.0f64.powi(-26);
    let z32 = [1.0f32, h as f32];
    for (obj, width) in [
        (Objective::TopKHingeAlpha, 0.0),
        (Objective::TopKHingeBeta, 0.0),
        (Objective::TopKHingeAlpha, 1.0),
        (Objective::TopKHingeBeta, 1.0),
    ] {
        let s = h / (2.0 + width);
        let u = coordinate_update(
            Matrix::new(&z32, 1, 2).unwrap(),
            &[0],
            &UpdateOptions {
                objective: obj,
                gamma: width,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(((f64::from(u[0]) / s) - 1.0).abs() <= 4.0 * f64::from(f32::EPSILON));
        assert!(((f64::from(u[1]) / -s) - 1.0).abs() <= 4.0 * f64::from(f32::EPSILON));
    }
    // Smoothed hinge value at [0, 1e200], gamma = 1e300 is 5e99, not NaN.
    let (v, g) = loss_and_grad(
        Matrix::new(&[0.0, 1e200], 1, 2).unwrap(),
        &[0],
        &LossOptions {
            loss: Loss::TopKHingeBeta,
            gamma: 1e300,
            ..Default::default()
        },
    )
    .unwrap();
    assert!((v[0] / 5e99 - 1.0).abs() < 1e-14);
    assert!((g[1] / 1e-100 - 1.0_f64).abs() < 1e-14);
    // proximal_entropy cancellation at [1e16, 0, 0], alpha = 1e50 -> log(2).
    let (vp, _) = loss_and_grad(
        Matrix::new(&[1e16, 0.0, 0.0], 1, 3).unwrap(),
        &[vec![0, 1]],
        &LossOptions {
            loss: Loss::ProximalCrossEntropy,
            alpha: 1e50,
            ..Default::default()
        },
    )
    .unwrap();
    assert!((vp[0] / 2.0f64.ln() - 1.0).abs() < 1e-14);
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
    for (loss, k) in [
        (Loss::MulticlassHinge, 1),
        (Loss::TopKHingeAlpha, 2),
        (Loss::TopKHingeBeta, 2),
        (Loss::MultilabelHinge, 1),
    ] {
        let options = LossOptions {
            loss,
            k,
            ..Default::default()
        };
        let matrix = Matrix::new(&z, n, m).unwrap();
        let (values, grad) = if loss == Loss::MultilabelHinge {
            loss_and_grad(matrix, &sets, &options)
        } else {
            loss_and_grad(matrix, &y, &options)
        }
        .unwrap();
        for i in 0..n {
            let row = &z[i * m..(i + 1) * m];
            let mut expected = vec![0.0; m];
            let value;
            if loss == Loss::MultilabelHinge {
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
                let beta = loss == Loss::TopKHingeBeta;
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
    for &objective in Objective::ALL {
        for gamma in [0.0, 0.5] {
            let opts = FitOptions {
                objective,
                k: 2,
                gamma,
                c: 2.0,
                tol: 1e-8,
                max_epoch: 100_000,
                ..Default::default()
            };
            let matrix = Matrix::new(&x, n, d).unwrap();
            let model = if objective.is_multilabel() {
                fit(matrix, &sets, &opts)
            } else {
                fit(matrix, &y, &opts)
            }
            .unwrap();
            assert_eq!(model.status(), Status::Solved, "{objective} {gamma}");
            let w = model.weights().unwrap();
            // W = X' A, and the primal of the returned weights from loss_and_grad.
            for l in 0..d {
                for j in 0..m {
                    let xa: f64 = (0..n).map(|r| x[r * d + l] * model.dual()[r * m + j]).sum();
                    assert!((w[l * m + j] - xa).abs() < 1e-12, "{objective}");
                }
            }
            let z: Vec<f64> = (0..n * m)
                .map(|e| (0..d).map(|l| x[(e / m) * d + l] * w[l * m + e % m]).sum())
                .collect();
            let loss: Loss = objective.name().parse().unwrap();
            let options = LossOptions {
                loss,
                k: 2,
                gamma,
                ..Default::default()
            };
            let scores = Matrix::new(&z, n, m).unwrap();
            let (values, _) = if objective.is_multilabel() {
                loss_and_grad(scores, &sets, &options)
            } else {
                loss_and_grad(scores, &y, &options)
            }
            .unwrap();
            let primal =
                0.5 * w.iter().map(|v| v * v).sum::<f64>() + opts.c * values.iter().sum::<f64>();
            let reported = *model.history().get("primal").unwrap().last().unwrap();
            assert!(
                (primal - reported).abs() <= 1e-12 * primal.abs(),
                "{objective} {gamma}: {primal} {reported}"
            );
            let dual = *model.history().get("dual").unwrap().last().unwrap();
            assert!(
                primal - dual <= 1e-8 * primal.abs() && dual <= primal * (1.0 + 1e-14),
                "{objective}"
            );
        }
    }
}

#[test]
fn shrinking_is_auto_by_default_and_every_mode_agrees() {
    assert_eq!(FitOptions::<f64>::default().shrinking, Shrinking::Auto);
    assert_eq!("safe".parse::<Shrinking>(), Ok(Shrinking::Safe));
    invalid("on".parse::<Shrinking>(), "shrinking");
    let names: Vec<_> = Shrinking::ALL.iter().map(|s| s.name()).collect();
    assert_eq!(names, ["auto", "off", "heuristic", "safe", "both"]);
    let primal = |model: &Model<f64>| *model.history().get("primal").unwrap().last().unwrap();
    for objective in [Objective::MulticlassHinge, Objective::CrossEntropy] {
        let fits: Vec<_> = Shrinking::ALL
            .iter()
            .map(|&mode| {
                let opts = FitOptions::<f64> {
                    objective,
                    tol: 1e-8,
                    max_epoch: 50000,
                    shrinking: mode,
                    ..Default::default()
                };
                fit(x(), &Y, &opts).unwrap()
            })
            .collect();
        for model in &fits {
            assert_eq!(model.status(), Status::Solved);
            let (p, p0) = (primal(model), primal(&fits[1]));
            assert!((p - p0).abs() <= 2e-8 * p0.abs(), "{objective}: {p} {p0}");
        }
        if objective == Objective::CrossEntropy {
            assert!(fits.iter().all(|m| m.dual() == fits[0].dual()));
        }
    }
}
