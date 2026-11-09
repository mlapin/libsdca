//! Native rows on several threads: bitwise the results of one thread.
use libsdca::{
    coordinate_update, loss_and_grad, Loss, LossOptions, Matrix, Objective, Precision, Real,
    UpdateOptions,
};

// 2600 x 101 scores take milliseconds per call: the default (automatic) and
// the explicit counts add threads.
const N: usize = 2600;
const C: usize = 101;
const THREADS: [Option<usize>; 5] = [None, Some(2), Some(3), Some(8), Some(0)];

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
    for &loss in Loss::ALL {
        let sets_only = matches!(
            loss,
            Loss::MultilabelHinge | Loss::MultilabelCrossEntropy | Loss::ProximalCrossEntropy
        );
        let options = |num_threads| LossOptions {
            loss,
            k: 3,
            gamma: 0.5,
            alpha: 0.7,
            precision: Some(precision),
            num_threads,
            ..Default::default()
        };
        let one = if sets_only {
            loss_and_grad(z, &sets, &options(Some(1))).unwrap()
        } else {
            loss_and_grad(z, &labels, &options(Some(1))).unwrap()
        };
        for threads in THREADS {
            let many = if sets_only {
                loss_and_grad(z, &sets, &options(threads)).unwrap()
            } else {
                loss_and_grad(z, &labels, &options(threads)).unwrap()
            };
            assert!(
                same(&one.0, &many.0) && same(&one.1, &many.1),
                "{loss} {precision} {threads:?} threads"
            );
        }
    }
}

fn updates_match<T: Real + From<f32>>(precision: Precision) {
    let (z, labels, sets) = problem::<T>();
    let z = Matrix::new(&z, N, C).unwrap();
    for &objective in Objective::ALL {
        let options = |num_threads| UpdateOptions::<T> {
            c: 2.0,
            objective,
            k: 3,
            gamma: 0.5,
            precision: Some(precision),
            num_threads,
            ..Default::default()
        };
        let update = |threads| {
            if objective.is_multilabel() {
                coordinate_update(z, &sets, &options(threads)).unwrap()
            } else {
                coordinate_update(z, &labels, &options(threads)).unwrap()
            }
        };
        let one = update(Some(1));
        for threads in THREADS {
            let many = update(threads);
            assert!(
                same(&one, &many),
                "{objective} {precision} {threads:?} threads"
            );
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
    let options = LossOptions {
        loss: Loss::TopKHingeBeta,
        k: 2,
        num_threads: Some(0),
        ..Default::default()
    };
    let z = Matrix::new(&z, N, C).unwrap();
    let serial = loss_and_grad(
        z,
        &labels,
        &LossOptions {
            num_threads: Some(1),
            ..options
        },
    )
    .unwrap();
    std::thread::scope(|scope| {
        let callers: Vec<_> = (0..4)
            .map(|_| scope.spawn(|| loss_and_grad(z, &labels, &options).unwrap()))
            .collect();
        for caller in callers {
            assert!(caller.join().unwrap() == serial);
        }
    });
}
