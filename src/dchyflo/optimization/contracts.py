"""Typed Checkpoint 1 contract loading and immutable-input verification."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def registered_hash_matches(root: Path, artifact: dict) -> bool:
    """Match a frozen hash, allowing only the audited CP2 PWL dual-hash migration."""
    path = root / artifact["path"]
    actual = sha256_file(path)
    if actual == artifact["sha256"]:
        return True
    migration_path = root / "config/v5/cp2_pwl_hash_migration.json"
    if not migration_path.is_file():
        return False
    migration = json.loads(migration_path.read_text(encoding="utf-8"))
    if not (
        artifact["path"] == migration["path"]
        and artifact["sha256"] == migration["legacy_sha256"]
        and actual == migration["current_sha256"]
    ):
        return False
    for key, relative in migration["replay_evidence"].items():
        evidence = root / relative
        if not evidence.is_file():
            return False
        if key.endswith("checks"):
            import pandas as pd
            if not pd.read_csv(evidence)["passed"].all():
                return False
    return True


@dataclass(frozen=True)
class FixedCase:
    """The pre-registered deterministic design and operating case."""

    s0_reference: str
    year: int
    capacity_mwac: float
    cold_scenario: str
    corridor_multiplier: float
    lexicographic_epsilon: float
    max_slp_iterations: int
    stall_window: int
    damping: float


@dataclass(frozen=True)
class InputArtifact:
    """One immutable Checkpoint 1 input or baseline artifact."""

    name: str
    path: str
    sha256: str


@dataclass(frozen=True)
class ReproductionContract:
    """Pre-registered case, tolerances, and input hashes for Gate O."""

    artifact_id: str
    checkpoint: int
    purpose: str
    case: FixedCase
    inputs: tuple[InputArtifact, ...]
    categorical_columns: tuple[str, ...]
    integer_columns: tuple[str, ...]
    numeric_absolute_tolerances: dict[str, float]
    physics_tolerances: dict[str, float]
    claim_boundary: str

    def artifact(self, name: str) -> InputArtifact:
        matches = [artifact for artifact in self.inputs if artifact.name == name]
        if len(matches) != 1:
            raise KeyError(f"Expected one contract artifact named {name!r}")
        return matches[0]

    def verify_inputs(self, root: Path) -> pd.DataFrame:
        """Verify every registered path and digest and return an audit table."""

        rows: list[dict[str, Any]] = []
        for artifact in self.inputs:
            path = root / Path(artifact.path)
            exists = path.is_file()
            actual = sha256_file(path) if exists else ""
            rows.append(
                {
                    "artifact_name": artifact.name,
                    "relative_path": artifact.path,
                    "exists": exists,
                    "expected_sha256": artifact.sha256,
                    "actual_sha256": actual,
                    "hash_match": exists and actual.lower() == artifact.sha256.lower(),
                }
            )
        return pd.DataFrame(rows)


def load_reproduction_contract(path: Path) -> ReproductionContract:
    """Load and validate a Checkpoint 1 JSON contract."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if int(raw["checkpoint"]) != 1:
        raise ValueError("The reproduction contract must declare checkpoint=1")
    case = FixedCase(**raw["case"])
    if case.capacity_mwac < 0.0:
        raise ValueError("Fixed FPV capacity must be non-negative")
    if not 0.0 < case.damping <= 1.0:
        raise ValueError("SLP damping must lie in (0, 1]")
    if case.max_slp_iterations < 2 or case.stall_window < 4:
        raise ValueError("SLP iteration and stall-window settings are invalid")
    inputs = tuple(
        InputArtifact(name=name, path=value["path"], sha256=value["sha256"])
        for name, value in raw["inputs"].items()
    )
    comparison = raw["comparison"]
    tolerances = {
        str(name): float(value)
        for name, value in comparison["numeric_absolute_tolerances"].items()
    }
    if any(value < 0.0 for value in tolerances.values()):
        raise ValueError("Comparison tolerances must be non-negative")
    physics = {
        str(name): float(value)
        for name, value in raw["physics_tolerances"].items()
    }
    if any(value < 0.0 for value in physics.values()):
        raise ValueError("Physics tolerances must be non-negative")
    return ReproductionContract(
        artifact_id=str(raw["artifact_id"]),
        checkpoint=1,
        purpose=str(raw["purpose"]),
        case=case,
        inputs=inputs,
        categorical_columns=tuple(comparison["categorical_columns"]),
        integer_columns=tuple(comparison["integer_columns"]),
        numeric_absolute_tolerances=tolerances,
        physics_tolerances=physics,
        claim_boundary=str(raw["claim_boundary"]),
    )
