"""A known numerical limitation must not hide a new failure or a fixed entry."""

import pytest
from conformance import known
from conformance.contract import Failure


@pytest.fixture
def expected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        known,
        "manifest",
        lambda: {
            "case": [[1, "gradient 0", 5.0, "finite", "mask-entropy"]],
        },
    )


def test_expected_entry_is_tracked(expected: None) -> None:
    failure = Failure(1, "gradient 0", 1.0, "1.000001", 4.0)
    assert known.verify("case", [failure]) == ["mask-entropy"]


def test_fixed_entry_requires_narrowing(expected: None) -> None:
    with pytest.raises(AssertionError, match="Fixed conformance"):
        known.verify("case", [])


def test_new_entry_inside_a_known_row_fails(expected: None) -> None:
    failures = [
        Failure(1, "gradient 0", 1.0, "1.000001", 4.0),
        Failure(1, "gradient 1", 1.0, "1.000001", 2.0),
    ]
    with pytest.raises(AssertionError, match="New conformance"):
        known.verify("case", failures)


@pytest.mark.parametrize("value,ratio", [(1.0, 6.0), (float("nan"), 4.0)])
def test_worsening_a_known_entry_fails(expected: None, value: float, ratio: float) -> None:
    with pytest.raises(AssertionError):
        known.verify("case", [Failure(1, "gradient 0", value, "1.000001", ratio)])


def test_compiler_variant_requires_an_exact_reviewed_profile(
    expected: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        known,
        "variants",
        lambda: {
            "case": [
                [
                    [1, "gradient 1", 3.0, "finite", "near-mask"],
                    [1, "gradient 2", 3.0, "finite", "near-mask"],
                ]
            ]
        },
    )
    pair = [Failure(1, f"gradient {i}", 1.0, "1.000001", 2.0) for i in (1, 2)]
    assert known.verify("case", pair) == ["near-mask"]
    with pytest.raises(AssertionError):
        known.verify("case", pair[:1])
    with pytest.raises(AssertionError):
        known.verify("case", pair + [Failure(1, "gradient 3", 1.0, "1.000001", 2.0)])
    with pytest.raises(AssertionError):
        known.verify("case", [Failure(1, "gradient 1", 1.0, "1.000001", 4.0), pair[1]])
