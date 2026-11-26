"""Behaviour that open work changes, in one place for test_cross_interface.py, rule_table.py and
test_rule_table.py: what each change touches, and the flag that switches its tests on once the
change lands (until then they are generated without those rows, or xfail).

ENABLED holds the flags by name, as rule_table's rows and errors refer to them
(Row.pending, Row.until, Error.pending).
"""

from __future__ import annotations

# A masked label, a masked positive class and a student class with teacher mass give the +inf
# limit.
MASKED_LIMITS = True
# The row kinds of test_cross_interface.py that MASKED_LIMITS covers.
MASKED_LIMIT_ROW_KINDS = (
    "masked_label",  # a score of -inf at the label of a single-label loss
    "masked_positive",  # a score of -inf at a positive of a multilabel loss
    "masked_target_mass",  # -inf at a class of positive target probability or teacher mass
    "topk_few_finite",  # fewer than k finite classes beside the label (top-k losses)
)
# Labels out of range are argument errors ("Target 4 is out of bounds.", as in PyTorch): the C
# ABI, NumPy, Rust and sdca/loss.h raise them, as do synchronous PyTorch and JAX CPU calls.
LABEL_ERRORS = True

ENABLED = {"masked_limits": MASKED_LIMITS, "label_errors": LABEL_ERRORS}
