//! Train a softmax classifier and a top-2 entropy classifier on toy data.
use libsdca::{fit, loss_and_grad, FitOptions, Loss, LossOptions, Matrix, Objective};

fn main() -> libsdca::Result<()> {
    // Three classes around the vertices of a triangle, row-major (n, d).
    let x = [
        1.0, 0.1, 0.9, -0.1, -0.5, 0.9, -0.4, 0.8, -0.5, -0.9, -0.6, -0.8,
    ];
    let y = [0, 0, 1, 1, 2, 2];
    let x = Matrix::new(&x, 6, 2)?;
    for objective in [Objective::CrossEntropy, Objective::TopKCrossEntropy] {
        let options = FitOptions {
            c: 10.0,
            objective,
            k: 2,
            ..Default::default()
        };
        let model = fit(x, &y, &options)?;
        let scores = model.decision_function(x)?;
        let options = LossOptions {
            loss: Loss::TopKHingeAlpha,
            ..Default::default()
        };
        let (losses, _) = loss_and_grad(Matrix::new(&scores, 6, 3)?, &y, &options)?;
        println!(
            "{objective}: {} after {} epochs, predictions {:?}, mean hinge {:.3}",
            model.status(),
            model.epochs(),
            model.predict(x)?,
            losses.iter().sum::<f64>() / losses.len() as f64
        );
    }
    Ok(())
}
