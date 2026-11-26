//! Native rows on several threads: bitwise the results of one thread.
use std::num::NonZeroUsize;

use libsdca::{Loss, Matrix, Precision, Real, Settings, Solver, Threads, TopKVariant};

// 2600 x 101 scores take milliseconds per call: the default (automatic) and
// the explicit counts add threads.
const N: usize = 2600;
const C: usize = 101;

fn threads() -> [Threads; 4] {
    let count = |n| Threads::Count(NonZeroUsize::new(n).unwrap());
    [Threads::FromEnv, count(2), count(3), count(8)]
}

fn serial() -> Threads {
    Threads::Count(NonZeroUsize::MIN)
}

/// Every loss but distillation, with k = 3, gamma = 0.5 (0.7 for the proximal losses).
const LOSSES: [Loss; 8] = [
    Loss::CrossEntropy,
    Loss::TopKCrossEntropy { k: 3 },
    Loss::MulticlassHinge { gamma: 0.5 },
    Loss::TopKHinge {
        k: 3,
        variant: TopKVariant::Alpha,
        gamma: 0.5,
    },
    Loss::TopKHinge {
        k: 3,
        variant: TopKVariant::Beta,
        gamma: 0.5,
    },
    Loss::MultilabelHinge { gamma: 0.5 },
    Loss::MultilabelCrossEntropy,
    Loss::ProximalCrossEntropy { gamma: 0.7 },
];

/// Scores in about [-3, 3] (every 97th row shifted by 1e6), labels and
/// label sets of 1 to C - 1 labels, from a fixed linear congruential sequence.
fn problem<T: Real + From<f32>>() -> (Vec<T>, Vec<usize>, Vec<Vec<usize>>) {
    let mut state = 12345u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 40) as f32 / (1u64 << 24) as f32
    };
    let mut z = Vec::with_capacity(N * C);
    for i in 0..N * C {
        let shift = if (i / C) % 97 == 0 { 1e6 } else { 0.0 };
        z.push(T::from(6.0 * next() - 3.0 + shift));
    }
    let labels = (0..N).map(|_| (next() * C as f32) as usize % C).collect();
    let sets = (0..N)
        .map(|_| {
            let mut set = vec![0];
            set.extend((2..C).filter(|_| next() < 0.1));
            set
        })
        .collect();
    (z, labels, sets)
}

/// Bitwise equality: Debug prints the shortest round-trip form, and -0.0 and NaN.
fn same<T: Real>(a: &[T], b: &[T]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| format!("{x:?}") == format!("{y:?}"))
}

fn losses_match<T: Real + From<f32>>(precision: Precision) {
    let (z, labels, sets) = problem::<T>();
    let z = Matrix::new(&z, N, C).unwrap();
    let teacher: Vec<T> = (0..N * C)
        .map(|i| T::from(((i % 13) as f32 - 6.0) * 0.5))
        .collect();
    let teacher = Matrix::new(&teacher, N, C).unwrap();
    let distillation = [
        Loss::Distillation { temperature: 2.0 },
        Loss::ProximalDistillation {
            temperature: 2.0,
            gamma: 0.7,
        },
    ];
    for loss in LOSSES.into_iter().chain(distillation) {
        // Proximal distillation computes in float64 only.
        let precision = match loss {
            Loss::ProximalDistillation { .. } => Precision::Float64,
            _ => precision,
        };
        let eval = |threads| {
            let settings = Settings::default()
                .with_precision(precision)
                .with_threads(threads);
            match loss {
                Loss::Distillation { .. } | Loss::ProximalDistillation { .. } => {
                    loss.loss_and_grad_with(z, libsdca::Targets::Teacher(teacher), &settings)
                }
                _ if loss.is_multilabel() => loss.loss_and_grad_with(z, &sets, &settings),
                _ => loss.loss_and_grad_with(z, &labels, &settings),
            }
            .unwrap()
        };
        let one = eval(serial());
        for threads in threads() {
            let many = eval(threads);
            assert!(
                same(&one.0, &many.0) && same(&one.1, &many.1),
                "{loss} {precision} {threads:?}"
            );
        }
    }
}

fn updates_match<T: Real + From<f32>>(precision: Precision) {
    let (z, labels, sets) = problem::<T>();
    let z = Matrix::new(&z, N, C).unwrap();
    // The training objectives.
    for loss in &LOSSES[..7] {
        let update = |threads| {
            let solver = Solver::new(*loss)
                .with_c(2.0)
                .with_precision(precision)
                .with_threads(threads);
            if loss.is_multilabel() {
                solver.coordinate_update(z, &sets, 1.0, None).unwrap()
            } else {
                solver.coordinate_update(z, &labels, 1.0, None).unwrap()
            }
        };
        let one = update(serial());
        for threads in threads() {
            let many = update(threads);
            assert!(same(&one, &many), "{loss} {precision} {threads:?}");
        }
    }
}

#[test]
fn losses_do_not_depend_on_the_thread_count() {
    losses_match::<f64>(Precision::Float64);
    losses_match::<f32>(Precision::Float64);
    losses_match::<f32>(Precision::Float32);
}

#[test]
fn coordinate_steps_do_not_depend_on_the_thread_count() {
    updates_match::<f64>(Precision::Float64);
    updates_match::<f32>(Precision::Float64);
    updates_match::<f32>(Precision::Float32);
}

#[test]
fn concurrent_callers_get_the_serial_results() {
    let (z, labels, _) = problem::<f64>();
    let loss = Loss::topk_hinge(2);
    let z = Matrix::new(&z, N, C).unwrap();
    let settings = Settings::default().with_threads(serial());
    let serial = loss.loss_and_grad_with(z, &labels, &settings).unwrap();
    let auto = Settings::default().with_threads(Threads::FromEnv);
    std::thread::scope(|scope| {
        let callers: Vec<_> = (0..4)
            .map(|_| scope.spawn(|| loss.loss_and_grad_with(z, &labels, &auto).unwrap()))
            .collect();
        for caller in callers {
            assert!(caller.join().unwrap() == serial);
        }
    });
}
