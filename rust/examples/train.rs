//! Train a softmax classifier and a top-2 SVM on toy data.
use libsdca::{Loss, Matrix, Solver};

fn main() -> libsdca::Result<()> {
    // Three classes around the vertices of a triangle, row-major (n, d).
    let x = [
        1.0, 0.1, 0.9, -0.1, -0.5, 0.9, -0.4, 0.8, -0.5, -0.9, -0.6, -0.8,
    ];
    let y = [0, 0, 1, 1, 2, 2];
    let x = Matrix::new(&x, 6, 2)?;
    for loss in [Loss::cross_entropy(), Loss::topk_hinge(2)] {
        let model = Solver::new(loss).with_c(10.0).fit(x, &y)?;
        let scores = model.decision_function(x)?;
        let (losses, _) = loss.loss_and_grad(Matrix::new(&scores, 6, 3)?, &y)?;
        println!(
            "{loss}: {} after {} epochs, predictions {:?}, mean loss {:.3}",
            model.status(),
            model.epochs(),
            model.predict(x)?,
            losses.iter().sum::<f64>() / losses.len() as f64
        );
    }
    Ok(())
}
