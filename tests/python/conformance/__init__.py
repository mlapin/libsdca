"""Cross-backend conformance: every loss on every backend against the same mpmath references.

references: the losses on rows given by their distinct entries, in mpmath.
inputs: the rows of each input family, as represented in each dtype.
contract: the documented error bound of each value and gradient entry.
backends: one call of a loss on NumPy, PyTorch (CPU, CUDA) or JAX (CPU, GPU).
cells: the matrix (losses x backends x dtypes x precisions x inputs x reductions) and the
documented limitations, as tables of one line per entry.
"""
