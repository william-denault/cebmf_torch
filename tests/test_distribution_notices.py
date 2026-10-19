"""Prevent declared license notices from silently disappearing from builds."""

import tomllib
from pathlib import Path


def test_declared_distribution_notices_exist():
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    declared = config["tool"]["setuptools"]["license-files"]
    assert "THIRD_PARTY_NOTICES.md" in declared
    for name in declared:
        assert (root / name).is_file(), f"Missing declared license notice: {name}"
    notice = (root / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert "fsusieR/R/hmm_routines.R" in notice
    assert "Copyright (c) 2023-2025" in notice
    assert "Redistribution and use in source and binary forms" in notice
    assert "THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS" in notice
