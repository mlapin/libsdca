use crate::error::{invalid, Result};

/// A borrowed row-major matrix.
///
/// ```
/// use libsdca::Matrix;
///
/// let x = Matrix::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)?;
/// assert_eq!((x.rows(), x.cols(), x.row(1)), (2, 3, &[4.0, 5.0, 6.0][..]));
/// assert!(Matrix::new(&[1.0, 2.0], 2, 3).is_err());
/// # Ok::<(), libsdca::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Matrix<'a, T> {
    data: &'a [T],
    rows: usize,
    cols: usize,
}

impl<'a, T> Matrix<'a, T> {
    /// A view of `data` as `rows` rows of `cols` entries.
    ///
    /// # Errors
    /// [`crate::Error::InvalidArgument`] if `data.len() != rows * cols`.
    pub fn new(data: &'a [T], rows: usize, cols: usize) -> Result<Self> {
        if rows.checked_mul(cols) != Some(data.len()) {
            return invalid(format!(
                "Matrix data has {} entries, not {rows}x{cols}",
                data.len()
            ));
        }
        Ok(Matrix { data, rows, cols })
    }

    /// The number of rows.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The number of columns.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// The entries in row-major order.
    #[must_use]
    pub fn data(&self) -> &'a [T] {
        self.data
    }

    /// Row `i`.
    ///
    /// # Panics
    /// If `i >= self.rows()`.
    #[must_use]
    pub fn row(&self, i: usize) -> &'a [T] {
        assert!(i < self.rows, "row {i} out of range for {} rows", self.rows);
        &self.data[i * self.cols..(i + 1) * self.cols]
    }
}
