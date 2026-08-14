"""Fail-closed audit for the public clean-room repository."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".toml", ".yml", ".yaml", ".json", ".csv", ".cff"}
FORBIDDEN = {
    "real_site_name_en": re.compile(r"\bNie" + r"rji\b", re.IGNORECASE),
    "real_site_name_zh": re.compile("\u5c3c\u5c14\u57fa"),
    "windows_absolute_path": re.compile(r"[A-Za-z]:\\\\"),
    "github_token": re.compile(r"\bgh[opusr]_[A-Za-z0-9_]{30,}\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}
REQUIRED = {
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "pyproject.toml",
    "docs/DATA_AVAILABILITY.md",
    "examples/synthetic_reservoir/run_demo.py",
    ".github/workflows/ci.yml",
}


def main() -> int:
    missing = sorted(name for name in REQUIRED if not (ROOT / name).is_file())
    hits: list[tuple[str, str]] = []
    oversized: list[str] = []
    audited_files = 0
    for path in ROOT.rglob("*"):
        parts = path.relative_to(ROOT).parts
        if not path.is_file() or any(part in SKIP_PARTS or part.startswith(".venv") for part in parts):
            continue
        audited_files += 1
        rel = path.relative_to(ROOT).as_posix()
        if path.stat().st_size > 10 * 1024 * 1024:
            oversized.append(rel)
        if path.suffix.lower() in TEXT_SUFFIXES and path.stat().st_size < 2 * 1024 * 1024:
            content = path.read_text(encoding="utf-8", errors="ignore")
            for label, pattern in FORBIDDEN.items():
                if pattern.search(content):
                    hits.append((rel, label))
    if missing or hits or oversized:
        print({"status": "FAIL", "missing": missing, "indicators": hits, "oversized": oversized})
        return 1
    print({"status": "PASS", "files": audited_files})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
