"""The native tensor entry points: sdca_cross_entropy_forward/backward and
sdca_loss_gradient_rows, on all four storage dtypes.

References: the float64 native entry points (sdca_entropy_forward/gradient and
sdca_loss_gradient, bitwise), float64 PyTorch, and 60-digit mpmath values of the
represented logits for the float32 arithmetic.
"""

import ctypes
import math
from typing import Any

import mpmath as mp
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from libsdca import _core, loss_and_grad

CODES = {torch.float32: 0, torch.float64: 1, torch.float16: 2, torch.bfloat16: 3}
HALVES = (torch.float16, torch.bfloat16)
FLOATS = (torch.float32, *HALVES)
DTYPES = (*FLOATS, torch.float64)
STATS = _core.ENTROPY_STATS
# The float32 contract of TORCH.md, as test_fused_entropy: relative loss and target errors
# in float32 eps and eps of the logits dtype, competitor errors relative to the row's largest
# entry.
VALUE_EPS, TARGET_EPS, ROW_EPS = 8, 8, 8


def forward(
    z: torch.Tensor,
    y: torch.Tensor,
    fp32: bool = False,
    ignore: int = -100,
    weight: torch.Tensor | None = None,
    threads: int | None = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n, c = z.shape
    values = torch.empty(n, dtype=torch.float64)
    stats = torch.empty(n, STATS, dtype=torch.float64)
    totals = torch.empty(2, dtype=torch.float64)
    _core._cross_entropy_forward(
        z.data_ptr(),
        CODES[z.dtype],
        n,
        c,
        y.data_ptr(),
        ignore,
        None if weight is None else weight.data_ptr(),
        values.data_ptr(),
        stats.data_ptr(),
        totals.data_ptr(),
        threads,
        fp32,
    )
    return values, stats, totals


def backward(
    z: torch.Tensor,
    y: torch.Tensor,
    stats: torch.Tensor,
    scale: object,
    fp32: bool = False,
    ignore: int = -100,
    weight: torch.Tensor | None = None,
    threads: int | None = 1,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    n, c = z.shape
    s = torch.as_tensor(scale, dtype=torch.float64).reshape(-1).contiguous()
    g = torch.empty_like(z) if out is None else out
    _core._cross_entropy_backward(
        z.data_ptr(),
        CODES[z.dtype],
        n,
        c,
        y.data_ptr(),
        ignore,
        None if weight is None else weight.data_ptr(),
        stats.data_ptr(),
        s.data_ptr(),
        0 if s.numel() == 1 else 1,
        g.data_ptr(),
        threads,
        fp32,
    )
    return g


def round_into(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """float64 x rounded once to nearest even into a 16-bit dtype (PyTorch's bfloat16
    conversion of float64 rounds twice, through float32)."""
    first = x.to(torch.float32).to(dtype)
    inf = torch.full_like(first, math.inf)
    try:
        candidates = [torch.nextafter(first, -inf), first, torch.nextafter(first, inf)]
    except RuntimeError as error:  # PyTorch 2.1 has no CPU float16 nextafter
        pytest.skip(f"this PyTorch lacks the reference's {dtype} nextafter: {error}")
    best = candidates[1]
    error = (best.double() - x).abs()
    for c in (candidates[0], candidates[2]):
        e = (c.double() - x).abs()
        bits = c.view(torch.int16) & 1
        better = (e < error) | ((e == error) & (bits == 0) & (c != best))
        best = torch.where(better, c, best)
        error = torch.where(better, e, error)
    # Beyond the largest value plus half its unit: infinity (ties too, the largest is odd).
    largest = torch.tensor(torch.finfo(dtype).max, dtype=dtype)
    below = torch.nextafter(largest, torch.zeros_like(largest))
    limit = largest.double() + (largest.double() - below.double()) / 2
    best = torch.where(x.abs() >= limit, x.sign().to(dtype) * math.inf, best)
    return torch.where(x.isnan(), x.to(dtype), best)


def native64(
    z: torch.Tensor, y: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Values, stats and scaled gradients of the float64 native entry points on z (float32
    for float32 storage, else float64 copies of the scores)."""
    x = z.float() if z.dtype == torch.float32 else z.double()
    values, stats = _core._entropy_forward(x.numpy(), y.numpy(), 1, True)
    assert stats is not None
    g = torch.empty_like(x)
    s = scale.expand(len(z)).contiguous()
    _core._entropy_gradient(x.numpy(), y.numpy(), stats, s.numpy(), 1, g.numpy())
    return torch.from_numpy(values), torch.from_numpy(stats), g


def mp_reference(z: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> tuple[list, list]:
    """60-digit losses and scaled gradient rows of the represented logits."""
    values, grads = [], []
    with mp.workdps(60):
        for i, row in enumerate(z.double().tolist()):
            t = int(y[i])
            e = [mp.exp(mp.mpf(v) - row[t]) if j != t else mp.mpf(0) for j, v in enumerate(row)]
            tail = mp.fsum(e)
            values.append(float(mp.log1p(tail)))
            u = mp.mpf(float(scale[i]))
            grad = [u * v / (1 + tail) for v in e]
            grad[t] = -u * tail / (1 + tail)
            grads.append([float(v) for v in grad])
    return values, grads


def assert_contract(z: torch.Tensor, y: torch.Tensor, scale: torch.Tensor, msg: str = "") -> None:
    """The float32 arithmetic within the float32 contract of TORCH.md."""
    scale = scale.expand(len(z))
    values, stats, _ = forward(z, y, fp32=True)
    grad = backward(z, y, stats, scale, fp32=True)
    ref_values, ref_grads = mp_reference(z, y, scale)
    info = torch.finfo(z.dtype)
    eps, tiny = info.eps, info.tiny
    eps32, tiny32 = torch.finfo(torch.float32).eps, torch.finfo(torch.float32).tiny
    for i, (a, b) in enumerate(zip(values.tolist(), ref_values, strict=True)):
        assert abs(a - b) <= max(VALUE_EPS * eps32 * abs(b), tiny32 * eps32), f"{msg} value {i}"
    for i, (row, ref) in enumerate(zip(grad.double().tolist(), ref_grads, strict=True)):
        big = max(abs(v) for v in ref)
        t = int(y[i])
        for j, (a, b) in enumerate(zip(row, ref, strict=True)):
            if abs(b) > info.max:
                # Beyond the storage range (an AMP scale): the rounded reference.
                assert a == round_into(torch.tensor(b, dtype=torch.float64), z.dtype).item()
                continue
            if j == t:
                bound = max(TARGET_EPS * eps * abs(b), tiny * eps)
            else:
                bound = max(ROW_EPS * eps * big, tiny * eps)
            assert abs(a - b) <= bound, f"{msg} row {i} entry {j}: {a!r} vs {b!r}"


def rows(generator: torch.Generator, n: int, c: int, spread: float, dtype: torch.dtype):
    z = (spread * torch.randn(n, c, generator=generator, dtype=torch.float64)).to(dtype)
    return z, torch.randint(0, c, (n,), generator=generator)


class NativeCrossEntropyTests:
    def setup_method(self):
        self.gen = torch.Generator().manual_seed(0)

    def test_double_arithmetic_matches_the_float64_entry_points(self):
        for dtype in DTYPES:
            for n, c, spread in ((7, 2, 1.0), (9, 32, 3.0), (33, 37, 3.0), (5, 1000, 10.0)):
                z, y = rows(self.gen, n, c, spread, dtype)
                scale = torch.randn(n, generator=self.gen, dtype=torch.float64)
                self.check_double(z, y, scale)
            z, y = rows(self.gen, 3, 40000, 2.0, dtype)
            self.check_double(z, y, torch.ones(1, dtype=torch.float64))

    def check_double(self, z: torch.Tensor, y: torch.Tensor, scale: torch.Tensor) -> None:
        """Rows over 32 classes as the float64 entry points, bit for bit; shorter rows (one
        term at a time) within a few double eps."""
        dtype, msg = z.dtype, f"{z.dtype} {z.shape}"
        values, stats, totals = forward(z, y)
        ref_values, ref_stats, ref_grad = native64(z, y, scale)
        expected = ref_grad if dtype in (torch.float32, torch.float64) else None
        expected = round_into(ref_grad, dtype) if expected is None else expected
        assert totals.tolist() == [math.fsum(values.tolist()), len(z)]
        if z.shape[1] > 32:
            assert torch.equal(values, ref_values), msg
            assert torch.equal(stats, ref_stats), msg
            assert torch.equal(backward(z, y, stats, scale), expected), msg
            return
        eps = torch.finfo(torch.float64).eps
        torch.testing.assert_close(values, ref_values, rtol=4 * eps, atol=1e-300, msg=msg)
        assert torch.equal(stats[:, [0, 1, 4]], ref_stats[:, [0, 1, 4]]), msg
        torch.testing.assert_close(stats, ref_stats, rtol=4 * eps, atol=1e-300, msg=msg)
        # Rounded once from nearly equal doubles: a few units of the storage apart.
        grad, unit = backward(z, y, stats, scale).double(), torch.finfo(dtype).eps
        bound = 4 * unit * expected.double().abs() + torch.finfo(dtype).smallest_normal * unit
        assert bool(((grad - expected.double()).abs() <= bound).all()), msg

    def test_sixteen_bit_gradients_round_once(self):
        # Scales spread the products over the normal, subnormal and overflowing ranges.
        for dtype in HALVES:
            z, y = rows(self.gen, 64, 300, 4.0, torch.float64)
            z = z.to(dtype)
            _, stats, _ = forward(z, y)
            info = torch.finfo(dtype)
            for scale in (1.0, -0.37, 65536.0, 1e-30, info.tiny, info.max, 3e38 * 4):
                ones = torch.ones(1, dtype=torch.float64)
                _, _, g64 = native64(z, y, ones)
                expected = round_into(g64 * scale, dtype)
                assert torch.equal(backward(z, y, stats, scale), expected), f"{dtype} {scale}"
            per_row = torch.logspace(-40, 30, 64, dtype=torch.float64)
            per_row[::3] *= -1
            _, _, g64 = native64(z, y, torch.ones(1, dtype=torch.float64))
            expected = round_into(g64 * per_row[:, None], dtype)
            assert torch.equal(backward(z, y, stats, per_row), expected), dtype

    def test_float32_arithmetic_contract(self):
        for dtype in FLOATS:
            for c, spread in ((2, 3.0), (37, 3.0), (1000, 3.0), (1000, 0.01), (3000, 10.0)):
                z, y = rows(self.gen, 3, c, spread, dtype)
                for scale in (0.75, -1.0, 65536.0, 1e-20):
                    assert_contract(z, y, torch.tensor([scale], dtype=torch.float64), f"{c}")

    def test_float32_arithmetic_tails_and_boundaries(self):
        # Correct predictions near the float32 storage limits, wrong ones, offsets, ties.
        gaps = (0.0, 1.0, 16.7, 36.5, 37.5, 60.0, 87.0, 88.0, 100.0, 103.9)
        lists = [[0.0, -gap] for gap in gaps]
        lists += [[0.0, 30.0], [0.0, 1e4], [1e4, 1e4 - 0.5], [-1e4, -1e4 + 3], [3e38, -3e38]]
        z = torch.tensor(lists)
        y = torch.zeros(len(lists), dtype=torch.long)
        for scale in (1.0, -1e-30, 3e38):
            assert_contract(z, y, torch.tensor([scale], dtype=torch.float64), "two classes")
        for gap in (20.0, 40.0, 90.0, 110.8):
            z = torch.full((2, 1000), -gap)
            z[:, 0] = 0
            z[1, 1:] -= torch.linspace(0, 5, 999)
            y = torch.zeros(2, dtype=torch.long)
            for scale in (1.0, 3e38):
                assert_contract(z, y, torch.tensor([scale], dtype=torch.float64), f"gap {gap}")
        mixed = [[0, 0, 0, 0], [0, 0.25, -0.125, 0.5], [0, -36, -37, -38], [0, 1, -80, -90]]
        for dtype in FLOATS:
            z = torch.tensor(mixed, dtype=dtype)
            y = torch.zeros(len(mixed), dtype=torch.long)
            info = torch.finfo(dtype)
            for scale in (0.0, 1.0, -1.0, info.tiny * info.eps, info.max):
                assert_contract(z, y, torch.tensor([scale], dtype=torch.float64), str(dtype))

    def test_float32_arithmetic_wide_rows(self):
        eps = torch.finfo(torch.float32).eps
        for dtype in FLOATS:
            for c, spread in ((32000, 1.0), (128256, 3.0), (50000, 30.0)):
                z, y = rows(self.gen, 4, c, spread, dtype)
                x = z.double().clone().requires_grad_()
                expected = F.cross_entropy(x, y, reduction="none")
                expected.sum().backward()
                assert x.grad is not None
                values, stats, _ = forward(z, y, fp32=True)
                torch.testing.assert_close(
                    values, expected.detach(), rtol=VALUE_EPS * eps, atol=0, msg=f"{c}"
                )
                grad = backward(z, y, stats, 1.0, fp32=True).double()
                bound = ROW_EPS * torch.finfo(dtype).eps * x.grad.abs().amax(1, keepdim=True)
                assert bool(((grad - x.grad).abs() <= bound).all()), f"{dtype} {c}"

    def test_ignore_index_and_class_weights(self):
        for dtype in DTYPES:
            for fp32 in (False, True) if dtype != torch.float64 else (False,):
                for ignore in (-100, 3):
                    z, y = rows(self.gen, 50, 11, 3.0, dtype)
                    y[::4] = ignore
                    y[1::4] = -100 if ignore == 3 else y[1::4]
                    weight = torch.rand(11, generator=self.gen, dtype=torch.float64)
                    weight[[2, 5]] = 0
                    self.check_weighted(z, y, ignore, weight, fp32)
                    self.check_weighted(z, y, ignore, None, fp32)

    def check_weighted(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        ignore: int,
        weight: torch.Tensor | None,
        fp32: bool,
    ) -> None:
        msg = f"{z.dtype} fp32={fp32} ignore={ignore} weighted={weight is not None}"
        if ignore == 3:
            # Rows of -100 are labels out of range here: they fail.
            with pytest.raises(ValueError, match=r"Labels must lie in \[0, num_classes\)"):
                forward(z, y, fp32, ignore, weight)
            y = torch.where(y == -100, 0, y)
        scale = torch.randn(len(z), generator=self.gen, dtype=torch.float64)
        x = z.double().clone().requires_grad_()
        w = None if weight is None else weight
        expected = F.cross_entropy(x, y, w, ignore_index=ignore, reduction="none")
        (expected * scale).sum().backward()
        assert x.grad is not None
        values, stats, totals = forward(z, y, fp32, ignore, weight)
        kept = y != ignore
        eps = torch.finfo(torch.float32 if fp32 else torch.float64).eps
        # The float64 reference has absolute errors of a few double eps.
        torch.testing.assert_close(
            values, expected.detach(), rtol=VALUE_EPS * eps, atol=1e-14, msg=msg
        )
        assert bool((values[~kept] == 0).all()) and bool((stats[~kept] == 0).all()), msg
        wy = torch.ones(len(z), dtype=torch.float64) if w is None else w[y.clamp_min(0)]
        assert totals[1].item() == math.fsum(wy[kept].tolist()), msg
        assert totals[0].item() == pytest.approx(math.fsum(values.tolist()), rel=1e-15), msg
        grad = backward(z, y, stats, scale, fp32, ignore, weight).double()
        assert bool((grad[~kept] == 0).all()) and not bool(grad[~kept].signbit().any()), msg
        # PyTorch's float64 softmax minus one errs by double eps (of the row's scale).
        slack = 4 * torch.finfo(torch.float64).eps * (scale * wy).abs()[:, None]
        bound = 8 * torch.finfo(z.dtype).eps * x.grad.abs().amax(1, keepdim=True) + slack
        assert bool(((grad - x.grad).abs() <= bound).all()), msg
        # Weighted rows equal the unweighted ones with the weight folded into the scale.
        if weight is not None:
            _, stats1, _ = forward(z, y, fp32, ignore)
            folded = backward(z, y, stats1, scale * wy, fp32, ignore)
            assert torch.equal(backward(z, y, stats, scale, fp32, ignore, weight), folded), msg

    def test_all_ignored_and_empty_batches(self):
        for dtype in DTYPES:
            z, y = rows(self.gen, 6, 5, 1.0, dtype)
            z[0, 0] = math.nan  # ignored rows are not read
            y[:] = -100
            for fp32 in (False, True) if dtype != torch.float64 else (False,):
                values, stats, totals = forward(z, y, fp32)
                assert values.tolist() == [0.0] * 6 and totals.tolist() == [0.0, 0.0]
                g = torch.full_like(z, 7.0)
                assert bool((backward(z, y, stats, 1.0, fp32, out=g) == 0).all())
            empty = torch.empty(0, 5, dtype=dtype)
            labels = torch.empty(0, dtype=torch.long)
            assert forward(empty, labels)[2].tolist() == [0.0, 0.0]
            assert (
                backward(empty, labels, torch.empty(0, STATS, dtype=torch.float64), 1.0).numel()
                == 0
            )

    def test_extreme_rows(self):
        # Huge spreads, confident rows with tiny losses, ties and offsets.
        for dtype in DTYPES:
            info = torch.finfo(dtype)
            big = min(info.max / 4, 1e300)
            lists = [
                [0.0, -50.0, -60.0, -700.0],
                [big, -big, 0.0, 1.0],
                [-big, big, 0.0, 1.0],
                [1e4, 1e4, 1e4, 1e4],
                [0.0, -1e4, -2e4, -3e4],
                [5.0, 5.0 - 2.0**-6, 5.0, -5.0],
            ]
            z = torch.tensor(lists, dtype=torch.float64).to(dtype)
            y = torch.tensor([0, 0, 0, 1, 0, 3])
            scale = torch.tensor([1.0, -2.0, 0.5, 3.0, 1e10, -1.0], dtype=torch.float64)
            self.check_double(z, y, scale)
            # Wide copies (each row's last score repeated) take the lane kernels.
            self.check_double(torch.cat([z, z[:, -1:].expand(-1, 60)], 1), y, scale)
            if dtype != torch.float64:
                assert_contract(z, y, scale, f"extreme {dtype}")
        # The tiny flag: derivatives below the normal doubles.
        z = torch.tensor([[0.0, -800.0], [0.0, -1.0]], dtype=torch.float64)
        stats = forward(z, torch.zeros(2, dtype=torch.long))[1]
        assert stats[:, STATS - 1].tolist() == [1.0, 0.0]

    def test_thread_counts_give_the_same_bits(self):
        for dtype in DTYPES:
            z, y = rows(self.gen, 300, 777, 3.0, dtype)
            y[::5] = -100
            weight = torch.rand(777, generator=self.gen, dtype=torch.float64)
            scale = torch.randn(300, generator=self.gen, dtype=torch.float64)
            for fp32 in (False, True) if dtype != torch.float64 else (False,):
                results = []
                for threads in (1, 2, 7, 0):
                    values, stats, totals = forward(z, y, fp32, -100, weight, threads)
                    grad = backward(z, y, stats, scale, fp32, -100, weight, threads)
                    results.append((values, stats, totals, grad))
                for other in results[1:]:
                    for a, b in zip(results[0], other, strict=True):
                        assert torch.equal(a, b), f"{dtype} fp32={fp32}"

    def test_invalid_inputs(self):
        z, y = rows(self.gen, 4, 6, 1.0, torch.float32)
        n, c = z.shape
        for label in (-1, 6, -101):
            bad = y.clone()
            bad[2] = label
            self.assert_untouched(z, bad, r"Labels must lie in \[0, num_classes\)")
        for value in (math.nan, math.inf, -math.inf):
            for column in (0, int(y[1])):
                bad = z.clone()
                bad[1, column] = value
                self.assert_untouched(bad, y, "Logits must be finite")
        for dtype in HALVES:
            for value in (math.nan, math.inf):
                bad = z.to(dtype)
                bad[3, 2] = value
                for fp32 in (False, True):
                    with pytest.raises(ValueError, match="Logits must be finite"):
                        forward(bad, y, fp32)
        with pytest.raises(ValueError, match="float64 storage needs double arithmetic"):
            forward(z.double(), y, fp32=True)
        values = torch.empty(n, dtype=torch.float64)
        lib, function = _core._tensor_entry("sdca_cross_entropy_forward")
        address = _core._tensor_options(False, 1)[1]
        args = (z.data_ptr(), y.data_ptr(), -100, None, values.data_ptr(), None, None)
        assert function(address, 4, n, c, *args) == -2
        assert lib.sdca_last_error() == b"Invalid dtype"
        assert function(address, 0, n, 1, *args) == -2
        assert function(address, 0, n, c, 0, *args[1:]) == -2
        # Backward checks the labels before writing.
        _, stats, _ = forward(z, y)
        bad = y.clone()
        bad[3] = 6
        g = torch.full_like(z, 7.0)
        with pytest.raises(ValueError, match="Labels must"):
            backward(z, bad, stats, 1.0, out=g)
        assert bool((g == 7).all())
        with pytest.raises(ValueError, match="scale_stride"):
            _core._cross_entropy_backward(
                z.data_ptr(), 0, n, c, y.data_ptr(), -100, None, stats.data_ptr(),
                values.data_ptr(), 2, g.data_ptr(),
            )  # fmt: skip
        with pytest.raises(ValueError, match="num_threads"):
            forward(z, y, threads=-1)

    def assert_untouched(self, z: torch.Tensor, y: torch.Tensor, message: str) -> None:
        values = torch.full((len(z),), 5.0, dtype=torch.float64)
        stats = torch.full((len(z), STATS), 5.0, dtype=torch.float64)
        totals = torch.full((2,), 5.0, dtype=torch.float64)
        for fp32 in (False, True):
            with pytest.raises(ValueError, match=message):
                n, c = z.shape
                _core._cross_entropy_forward(
                    z.data_ptr(), 0, n, c, y.data_ptr(), -100, None, values.data_ptr(),
                    stats.data_ptr(), totals.data_ptr(), 3, fp32,
                )  # fmt: skip
            assert bool((values == 5).all()) and bool((stats == 5).all())
            assert totals.tolist() == [5.0, 5.0]


LOSS_OPTIONS: list[tuple[str, dict[str, Any]]] = [
    ("cross_entropy", {}),
    ("topk_cross_entropy", dict(k=1)),
    ("topk_cross_entropy", dict(k=3)),
    ("multiclass_hinge", {}),
    ("multiclass_hinge", dict(gamma=0.5)),
    ("topk_hinge_alpha", dict(k=2)),
    ("topk_hinge_alpha", dict(k=3, gamma=1e-3)),
    ("topk_hinge_beta", dict(k=2)),
    ("topk_hinge_beta", dict(k=2, gamma=30.0)),
    ("multilabel_hinge", {}),
    ("multilabel_hinge", dict(gamma=0.6)),
    ("multilabel_cross_entropy", {}),
    ("proximal_cross_entropy", dict(alpha=0.3)),
]
MULTILABEL = ("multilabel_hinge", "multilabel_cross_entropy", "proximal_cross_entropy")


def loss_rows(
    z: torch.Tensor,
    target: torch.Tensor,
    loss: str,
    fp32: bool,
    threads: int | None = 1,
    **options: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    n, c = z.shape
    values = torch.empty(n, dtype=torch.float64)
    grad_dtype = torch.float32 if z.dtype in HALVES else z.dtype
    gradients = torch.empty(n, c, dtype=grad_dtype)
    mask = target.dtype == torch.bool
    positive = target.to(torch.uint8) if mask else None
    _core._loss_gradient_rows(
        z.data_ptr(),
        CODES[z.dtype],
        n,
        c,
        None if mask else target.data_ptr(),
        None if positive is None else positive.data_ptr(),
        values.data_ptr(),
        gradients.data_ptr(),
        loss,
        num_threads=threads,
        fp32=fp32,
        **options,
    )
    return values, gradients


def loss_gradient(
    z: np.ndarray, target: np.ndarray, loss: str, precision: str, **options: Any
) -> tuple[np.ndarray, np.ndarray]:
    """sdca_loss_gradient itself (loss_and_grad takes sdca_loss_gradient_rows)."""
    solver = _core._solver(z, precision)
    positive = solver._positive_labels(target, *z.shape)
    values, gradients = np.empty(len(z)), np.empty_like(z)
    fields = _core._options(precision=solver.code, loss=_core._IDS[loss], **options)
    pointers = map(_core._ptr, (z, positive, values, gradients))
    solver._call("sdca_loss_gradient", ctypes.byref(fields), *z.shape, *pointers)
    return values, gradients


class NativeLossRowsTests:
    def setup_method(self):
        self.gen = torch.Generator().manual_seed(1)

    def targets(self, loss: str, n: int, c: int) -> list[torch.Tensor]:
        labels = torch.randint(0, c, (n,), generator=self.gen)
        mask = torch.rand(n, c, generator=self.gen) < 0.3
        mask[torch.arange(n), labels] = True
        mask[torch.arange(n), (labels + 1) % c] = False
        if loss in MULTILABEL:
            return [labels, mask]
        if loss == "cross_entropy":
            return [labels, mask, F.one_hot(labels, c).bool()]
        return [labels, F.one_hot(labels, c).bool()]

    def test_float_storage_matches_sdca_loss_gradient(self):
        for loss, options in LOSS_OPTIONS:
            for dtype in (torch.float32, torch.float64):
                z = (3 * torch.randn(40, 9, generator=self.gen)).to(dtype)
                for target in self.targets(loss, 40, 9):
                    for fp32 in (False, True) if dtype == torch.float32 else (False,):
                        precision = "float32" if fp32 else "float64"
                        y = target.numpy()
                        expected = loss_gradient(z.numpy(), y, loss, precision, **options)
                        rows = [t.numpy() for t in loss_rows(z, target, loss, fp32, **options)]
                        given = loss_and_grad(z.numpy(), y, loss, precision=precision, **options)
                        msg = f"{loss} {options} {dtype} fp32={fp32}"
                        for values, grad in (rows, given):
                            assert np.array_equal(values, expected[0]), msg
                            assert np.array_equal(grad, expected[1]), msg

    def test_sixteen_bit_storage_equals_float32_of_the_upcast_scores(self):
        for loss, options in LOSS_OPTIONS:
            for dtype in HALVES:
                z = (3 * torch.randn(30, 7, generator=self.gen)).to(dtype)
                for target in self.targets(loss, 30, 7):
                    for fp32 in (False, True):
                        expected = loss_rows(z.float(), target, loss, fp32, **options)
                        values, grad = loss_rows(z, target, loss, fp32, **options)
                        msg = f"{loss} {dtype} fp32={fp32}"
                        assert torch.equal(values, expected[0]), msg
                        assert torch.equal(grad, expected[1]), msg

    def test_every_finite_sixteen_bit_value_converts_exactly(self):
        for dtype in HALVES:
            bits = torch.arange(-(2**15), 2**15, dtype=torch.int32).to(torch.int16)
            z = bits.view(dtype)
            z = z[z.isfinite()]
            z = z[: len(z) // 8 * 8].reshape(-1, 8)
            y = torch.zeros(len(z), dtype=torch.long)
            expected = loss_rows(z.float(), y, "multiclass_hinge", True)
            values, grad = loss_rows(z, y, "multiclass_hinge", True)
            assert torch.equal(values, expected[0]) and torch.equal(grad, expected[1]), dtype
            # The loss is max(0, 1 + max_j z_j - z_0): the row values pass through exactly.
            hinge = (1 + z.double()[:, 1:].amax(1) - z.double()[:, 0]).clamp_min(0)
            assert torch.equal(values, hinge.float().double()), dtype

    def test_thread_counts_give_the_same_bits(self):
        for loss, options in LOSS_OPTIONS[::2]:
            for dtype in DTYPES:
                z = (3 * torch.randn(200, 200, generator=self.gen)).to(dtype)
                target = self.targets(loss, 200, 200)[-1]
                for fp32 in (False, True) if dtype != torch.float64 else (False,):
                    first = loss_rows(z, target, loss, fp32, 1, **options)
                    for threads in (3, 0):
                        other = loss_rows(z, target, loss, fp32, threads, **options)
                        assert torch.equal(first[0], other[0]), loss
                        assert torch.equal(first[1], other[1]), loss

    def test_invalid_rows(self):
        z = torch.randn(5, 4, generator=self.gen).bfloat16()
        labels = torch.tensor([0, 1, 2, 3, 0])
        mask = F.one_hot(labels, 4).bool()
        bad = z.clone()
        bad[2, 1] = math.inf
        cases = [
            (bad, labels, "cross_entropy", "Logits must be finite"),
            (z, torch.tensor([0, 1, 4, 3, 0]), "cross_entropy", r"Labels must lie in \[0"),
            (z, torch.tensor([0, 1, -1, 3, 0]), "multiclass_hinge", r"Labels must lie in \[0"),
            (bad.float(), labels, "multiclass_hinge", "Logits must be finite"),
            (bad.float(), torch.tensor([0, 1, 4, 3, 0]), "topk_hinge_beta", "must be finite"),
            (z.float(), torch.tensor([0, 1, 4, 3, 0]), "multiclass_hinge", r"must lie in \[0"),
            (z, mask & (torch.arange(5) != 3)[:, None], "cross_entropy", "a positive label"),
            (z, mask | (torch.arange(4) == 2), "topk_hinge_beta", "one positive label per row"),
            (z, mask | (torch.arange(5) == 1)[:, None], "multilabel_hinge", "a negative label"),
            (z, mask | (torch.arange(5) == 4)[:, None], "proximal_cross_entropy", "a negative"),
        ]
        for logits, target, loss, message in cases:
            with pytest.raises(ValueError, match=message):
                loss_rows(logits, target, loss, False, k=1)
        # Every class positive: cross-entropy takes the uniform target.
        values, _ = loss_rows(z, torch.ones(5, 4, dtype=torch.bool), "cross_entropy", False)
        assert bool(values.isfinite().all())
        n, c = z.shape
        values = torch.empty(n, dtype=torch.float64)
        g = torch.empty(n, c)
        common = (z.data_ptr(), 3, n, c)
        with pytest.raises(ValueError, match="either labels or positive"):
            _core._loss_gradient_rows(*common, None, None, values.data_ptr(), g.data_ptr())
        with pytest.raises(ValueError, match="Invalid loss dimensions"):
            _core._loss_gradient_rows(
                *common, labels.data_ptr(), None, values.data_ptr(), g.data_ptr(), k=4
            )
        with pytest.raises(ValueError, match="Unknown loss"):
            _core._loss_gradient_rows(
                *common, labels.data_ptr(), None, values.data_ptr(), g.data_ptr(), "hinge"
            )


def test_topk_entropy_rows_across_the_cap_boundary():
    """Top-k cross-entropy rows whose largest competitor share is just below, at and above
    1/k (entropy's pass serves the uncapped rows) against the tensor operations."""
    from libsdca.torch import sdca_loss

    k, c = 5, 40
    rows = []
    for share in (0.15, 0.199, 0.2, 0.201, 0.25, 0.6):
        # One competitor with the given share of the competitors, the rest equal.
        z = torch.zeros(c, dtype=torch.float64)
        z[1] = math.log(share * (c - 2) / (1 - share))
        z[0] = -1.0
        rows.append(z)
    # The label first, in the middle and last (the native rows skip it in place).
    z = torch.cat([torch.stack(rows).roll(shift, -1) for shift in (0, 7, c - 1)])
    y = torch.tensor([0, 7, c - 1]).repeat_interleave(len(rows))
    for fp32 in (False, True):
        x = z.float() if fp32 else z
        values, grad = loss_rows(x, y, "topk_cross_entropy", fp32, k=k)
        ref = z.clone().requires_grad_()
        expected = sdca_loss(ref, y, "topk_cross_entropy", k=k, backend="torch", reduction="none")
        expected.sum().backward()
        assert ref.grad is not None
        tol = 1e-5 if fp32 else 1e-13
        torch.testing.assert_close(values, expected.detach(), rtol=tol, atol=0)
        torch.testing.assert_close(grad.double(), ref.grad, rtol=0, atol=tol)


def distill(
    s: torch.Tensor,
    t: torch.Tensor,
    temperature: float,
    fp32: bool,
    upstream: torch.Tensor | None = None,
    threads: int | None = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """sdca_distillation_forward and _backward: values, stats and the gradient."""
    n, c = s.shape
    values = torch.empty(n, dtype=torch.float64)
    stats = torch.empty(n, 7, dtype=torch.float64)
    args = (s.data_ptr(), t.data_ptr(), CODES[s.dtype], n, c, temperature)
    _core._distillation_forward(*args, values.data_ptr(), stats.data_ptr(), threads, fp32)
    scale = torch.ones((), dtype=torch.float64) if upstream is None else upstream
    gradient = torch.empty_like(s)
    stride = 1 if scale.ndim else 0
    _core._distillation_backward(
        *args, stats.data_ptr(), scale.data_ptr(), stride, gradient.data_ptr(), threads, fp32
    )
    return values, stats, gradient


def distill64(s: torch.Tensor, t: torch.Tensor, temperature: float) -> tuple[Any, Any]:
    """float64 PyTorch values and gradients of the represented logits."""
    x = s.double().requires_grad_()
    p = torch.softmax(t.double() / temperature, -1)
    value = -(p * torch.log_softmax(x / temperature, -1)).sum(-1)
    value.sum().backward()
    assert x.grad is not None
    return value.detach(), x.grad


class NativeDistillationTests:
    """sdca_distillation_forward/_backward and sdca_proximal_distillation_rows."""

    def test_float_and_double_rows(self):
        gen = torch.Generator().manual_seed(3)
        for dtype in DTYPES:
            for c in (2, 7, 33, 1000):
                for temperature in (0.5, 2.0):
                    s = (3 * torch.randn(5, c, generator=gen)).to(dtype)
                    t = (3 * torch.randn(5, c, generator=gen)).to(dtype)
                    ref_v, ref_g = distill64(s, t, temperature)
                    up = torch.rand(5, generator=gen, dtype=torch.float64) + 0.5
                    for fp32 in (False, True) if dtype != torch.float64 else (False,):
                        msg = f"{dtype} C={c} T={temperature} fp32={fp32}"
                        values, stats, grad = distill(s, t, temperature, fp32, up)
                        assert bool((stats[:, 0] == (1 if fp32 else -1)).all()), msg
                        work = 64 * (torch.finfo(torch.float32).eps if fp32 else 2.0**-52)
                        torch.testing.assert_close(values, ref_v, rtol=work, atol=0, msg=msg)
                        expected = ref_g * up[:, None]
                        # One rounding into the dtype past the working error.
                        scale = expected.abs().amax(-1, keepdim=True)
                        bound = work + torch.finfo(dtype).eps
                        error = ((grad.double() - expected).abs() / scale).max()
                        assert float(error) <= bound, msg

    def test_equal_logits_give_zero_gradients(self):
        gen = torch.Generator().manual_seed(4)
        for dtype in DTYPES:
            z = (5 * torch.randn(4, 300, generator=gen)).to(dtype) + 1e3
            for fp32 in (False, True) if dtype != torch.float64 else (False,):
                _, _, grad = distill(z, z, 2.0, fp32)
                assert bool((grad == 0).all()), (dtype, fp32)

    def test_thread_counts_give_the_same_bits(self):
        gen = torch.Generator().manual_seed(5)
        s, t = torch.randn(64, 2000, generator=gen), torch.randn(64, 2000, generator=gen)
        for fp32 in (False, True):
            one = distill(s, t, 2.0, fp32, threads=1)
            many = distill(s, t, 2.0, fp32, threads=8)
            for a, b in zip(one, many, strict=True):
                assert torch.equal(a, b)

    def test_proximal_rows_converge_to_distillation(self):
        gen = torch.Generator().manual_seed(6)
        for dtype in DTYPES:
            s = (2 * torch.randn(3, 50, generator=gen)).to(dtype)
            t = (2 * torch.randn(3, 50, generator=gen)).to(dtype)
            n, c = s.shape
            values = torch.empty(n, dtype=torch.float64)
            grad = torch.empty(
                n, c, dtype=torch.float64 if dtype == torch.float64 else torch.float32
            )
            _core._proximal_distillation_rows(
                s.data_ptr(),
                t.data_ptr(),
                CODES[dtype],
                n,
                c,
                2.0,
                1e-9,
                values.data_ptr(),
                grad.data_ptr(),
                1,
            )
            ref_v, ref_g = distill64(s, t, 2.0)
            torch.testing.assert_close(values, ref_v, rtol=1e-6, atol=0)
            torch.testing.assert_close(grad.double(), ref_g, rtol=0, atol=1e-6)

    def test_invalid_inputs(self):
        s = torch.zeros(2, 3)
        values = torch.empty(2, dtype=torch.float64)
        call = (s.data_ptr(), s.data_ptr(), CODES[s.dtype], 2, 3)
        for temperature in (0.0, -1.0, math.inf, math.nan):
            with pytest.raises(ValueError, match="temperature"):
                _core._distillation_forward(*call, temperature, values.data_ptr())
        with pytest.raises(ValueError, match="alpha"):
            _core._proximal_distillation_rows(
                *call, 1.0, 0.0, values.data_ptr(), torch.empty(2, 3).data_ptr()
            )
        bad = torch.tensor([[0.0, math.inf, 1.0], [0.0, 1.0, 2.0]])
        for fp32 in (False, True):
            with pytest.raises(ValueError, match="Logits must be finite"):
                _core._distillation_forward(
                    bad.data_ptr(), s.data_ptr(), 0, 2, 3, 1.0, values.data_ptr(), None, 1, fp32
                )
        with pytest.raises(ValueError, match="float64 storage"):
            d = torch.zeros(2, 3, dtype=torch.float64)
            _core._distillation_forward(
                d.data_ptr(), d.data_ptr(), 1, 2, 3, 1.0, values.data_ptr(), None, 1, True
            )
