"""The suite in a source distribution: the sdist ships tests/python but not tests/cpp or rust/,
so the compiled cross-interface helpers report their sources missing and their cases skip,
instead of failing every case of the session."""

from __future__ import annotations

from pathlib import Path

import pytest

import interfaces


def test_compiled_helpers_without_their_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(interfaces, "CPP_SOURCE", tmp_path / "tests/cpp/cross_interface.cpp")
    monkeypatch.setattr(interfaces, "RUST_MANIFEST", tmp_path / "rust/Cargo.toml")
    outcome = interfaces.run_compiled(tmp_path)
    assert outcome["failed"] == {}
    assert set(outcome["missing"]) == {"cpp", "rust"}
    assert all("not in this tree" in reason for reason in outcome["missing"].values())
    session = interfaces.Session(tmp_path, outcome["missing"], outcome["failed"])
    for interface in ("cpp_kernel", "cpp_loss", "rust"):
        with pytest.raises(pytest.skip.Exception, match="source distribution"):
            session.require(interface)

