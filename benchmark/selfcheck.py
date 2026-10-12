"""
E1 · Self-containment check selfcheck.py
========================================

Purpose
-------
To **codify as an executable check** the project paradigm that "**any test branch should be
self-contained**", rather than leaving it as a verbal convention.

The full paradigm text, the rationale for the two-level definition, the asset checklist (§4),
CI integration, and how to apply it to other branches are given in
**`docs/E1_self_contained_branch_paradigm.md`**.

Definition of self-contained
----------------------------
A test branch is self-contained = on a **clean checkout**, without depending on any file outside
the branch, it can complete **all of the branch's tests**.

Since the inference stack contains large binary assets (weights / vector caches, see §ASSETS),
full self-containment conflicts with "keeping the repository size manageable", so this tool
measures self-containment **at two levels**:

    ┌──────────────┬────────────────────────────────────────────┬──────────┐
    │ Level        │ Meaning                                    │ Applies  │
    ├──────────────┼────────────────────────────────────────────┼──────────┤
    │ CODE level   │ All .py and configuration files present;   │ All      │
    │              │ logic tests can pass                       │ branches │
    │              │ (`benchmark/test_e1_contracts.py` all      │ Mandatory│
    │              │ green)                                     │ gate     │
    ├──────────────┼────────────────────────────────────────────┼──────────┤
    │ RUNTIME level│ Additionally all binary assets present, so │ Branches │
    │              │ real inference and stress tests can run    │ requiring│
    │              │ (`--backend real`)                         │ real-    │
    │              │                                            │ machine  │
    │              │                                            │ accept.  │
    └──────────────┴────────────────────────────────────────────┴──────────┘

Decision rules (executable)
---------------------------
1. **Static import-graph closure**: starting from the entry files, parse every `import` /
   `from ... import`; every **in-package module** (one not on the stdlib / third-party whitelist)
   must physically exist, otherwise the CODE level fails.
2. **Entry points executable**: `py_compile` all entry points; if the contract test can run, run
   it directly and verify the exit code.
3. **Asset checklist completeness**: check the binary assets against the `ASSETS` checklist to
   decide whether the RUNTIME level can be reached.
4. **Placeholder-file detection**: identify XML error pages / empty files left behind by failed
   downloads from the network drive (guarding against silent corruption).

Usage
-----
    python benchmark/selfcheck.py            # human-readable report
    python benchmark/selfcheck.py --json     # machine-readable, for CI consumption
    python benchmark/selfcheck.py --strict-runtime   # treat missing RUNTIME as failure too

CI integration suggestion
-------------------------
    # CODE-level gate: must pass before any test branch is committed
    python benchmark/selfcheck.py || exit 1
"""  # noqa: E501

from __future__ import annotations

import argparse
import ast
import json
import py_compile
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
# Entry files: the starting point of the self-containment check, and what "this branch promises can be run"
ENTRYPOINTS = [
    "app.py",
    "config.py",
    "benchmark/benchmark_e1.py",
    "benchmark/test_e1_contracts.py",
]

# Binary asset checklist: not kept in the version control (size), but the RUNTIME level of
# self-containment requires them all.
# `source` states where to obtain them, so a new branch can be completed quickly.
ASSETS = [
    {
        "path": "Service/ObjectDetector/fire.pt",
        "required_for": "runtime",
        "source": "项目盘 IGNITE/Service/ObjectDetector/fire.pt",
        "note": "Flame detection weights (after fine-tuning)",
    },
    {
        "path": "Service/ObjectDetector/obj1.pt",
        "required_for": "runtime",
        "source": "项目盘 IGNITE/Service/ObjectDetector/obj1.pt",
        "note": "Environmental object detection weights (after fine-tuning)",
    },
    {
        "path": "Service/Semantic/triplets_cache.pkl",
        "required_for": "runtime",
        "source": "项目盘 IGNITE/Service/Semantic/triplets_cache.pkl",
        "note": "Triplet vector cache; when missing it is rebuilt by common_knowledge.py (slower, and requires transformers)",
    },
]

# Third-party / system package whitelist: these are allowed to be "not in the package", as long as
# the runtime environment has them installed.
# Note: this only decides "whether it is an in-package module"; it does not check whether they are
# actually installed (that is an environment concern, handled by README/requirement.txt;
# self-containment only cares about "whether the files are all present").
STDLIB_HINT = set(sys.stdlib_module_names) if hasattr(sys, "stdlib_module_names") else set()
THIRD_PARTY = {
    "torch", "torchvision", "cv2", "numpy", "ultralytics", "av",
    "transformers", "yaml", "PIL", "matplotlib", "pandas",
}


# ==========================================================================
# Result container
# ==========================================================================
class CheckReport:
    def __init__(self):
        self.missing_modules: list[str] = []
        self.compile_errors: list[tuple[str, str]] = []
        self.placeholder_files: list[str] = []
        self.assets_present: list[str] = []
        self.assets_missing: list[dict] = []
        self.entrypoints: list[str] = []
        self.contract_test_ok: bool | None = None
        self.contract_test_output: str = ""

    @property
    def code_level_ok(self) -> bool:
        """CODE-level self-containment: modules complete + compiles + no placeholder files."""
        return not self.missing_modules and not self.compile_errors and not self.placeholder_files

    @property
    def runtime_level_ok(self) -> bool:
        """RUNTIME-level self-containment: CODE level passes and assets are complete."""
        return self.code_level_ok and not self.assets_missing

    def to_dict(self) -> dict:
        return {
            "code_level": {
                "ok": self.code_level_ok,
                "missing_modules": self.missing_modules,
                "compile_errors": [{"file": f, "error": e} for f, e in self.compile_errors],
                "placeholder_files": self.placeholder_files,
                "entrypoints": self.entrypoints,
            },
            "runtime_level": {
                "ok": self.runtime_level_ok,
                "assets_present": self.assets_present,
                "assets_missing": self.assets_missing,
            },
            "contract_test": {
                "ran": self.contract_test_ok is not None,
                "ok": self.contract_test_ok,
                "tail": self.contract_test_output,
            },
        }


# ==========================================================================
# Checks
# ==========================================================================
def collect_local_modules() -> set[str]:
    """
    Scan every .py in the package and return the set of importable module names (including
    package names and submodules).

    Used to decide "whether a given import should exist inside the package".
    """
    names: set[str] = set()
    for py in ROOT.rglob("*.py"):
        rel = py.relative_to(ROOT)
        if any(part.startswith(".") for part in rel.parts):   # skip hidden directories such as .baseline
            continue
        if "__pycache__" in rel.parts:
            continue
        parts = list(rel.with_suffix("").parts)
        names.add(parts[0])                       # top-level package / module
        if len(parts) > 1:
            names.add(".".join(parts))            # dotted full path
            names.add(parts[-1])                  # leaf module name (for from utils import x)
    return names


def extract_imports(path: Path) -> set[str]:
    """Use AST to extract all top-level import targets of a .py file (without executing the file)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return set()

    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            # Handle absolute imports only; relative imports are treated as in-package and inherently present
            if node.level == 0 and node.module:
                found.add(node.module)
    return found


def check_import_closure(report: CheckReport) -> None:
    """
    Static import-graph closure check: the in-package modules of the entry points and their
    transitive dependencies must physically exist.

    Only **in-package** modules are tracked (stdlib and third-party are skipped) to avoid false positives.
    """
    local_modules = collect_local_modules()
    visited: set[str] = set()
    queue: list[tuple[str, Path]] = []

    # Entry files are enqueued directly
    for ep in ENTRYPOINTS:
        p = ROOT / ep
        if p.exists():
            report.entrypoints.append(ep)
            queue.append((ep, p))
        else:
            report.missing_modules.append(f"{ep} (entry point missing)")

    while queue:
        modname, path = queue.pop(0)
        if modname in visited:
            continue
        visited.add(modname)

        for target in extract_imports(path):
            top = target.split(".")[0]
            # Skip stdlib and third-party
            if top in STDLIB_HINT or top in THIRD_PARTY:
                continue
            # Decide whether it is an in-package module
            if top not in local_modules:
                # A module that should be in the package is not found -> record as missing
                if top[0].isupper() or top in {"utils", "Service", "service", "config",
                                               "benchmark", "app"}:
                    entry = f"{target}  (referenced by {modname})"
                    if entry not in report.missing_modules:
                        report.missing_modules.append(entry)
                continue

            # Resolve to an actual file path and keep tracking
            cand = _resolve_module_path(target)
            if cand is None:
                entry = f"{target}  (referenced by {modname}, file not locatable)"
                if entry not in report.missing_modules:
                    report.missing_modules.append(entry)
            elif target not in visited:
                queue.append((target, cand))


def _resolve_module_path(target: str) -> Path | None:
    """Resolve a dotted module name to a .py file path under ROOT."""
    parts = target.split(".")
    # Form one: a/b/c.py
    p1 = ROOT.joinpath(*parts).with_suffix(".py")
    if p1.exists():
        return p1
    # Form two: a/b/c/__init__.py
    p2 = ROOT.joinpath(*parts) / "__init__.py"
    if p2.exists():
        return p2
    return None


def check_compile(report: CheckReport) -> None:
    """All produced files must compile."""
    for py in sorted(ROOT.rglob("*.py")):
        rel = py.relative_to(ROOT)
        if any(part.startswith(".") for part in rel.parts) or "__pycache__" in rel.parts:
            continue
        try:
            py_compile.compile(str(py), doraise=True)
        except py_compile.PyCompileError as exc:
            report.compile_errors.append((str(rel), str(exc)))


def check_placeholders(report: CheckReport) -> None:
    """
    Detect placeholder/corrupted files: a failed network-drive download leaves behind an XML
    error page or a 0-byte file. Such files pass the "existence" check but raise errors at
    runtime, so they must be explicitly blocked.
    """
    for py in sorted(ROOT.rglob("*.py")):
        rel = py.relative_to(ROOT)
        if any(part.startswith(".") for part in rel.parts) or "__pycache__" in rel.parts:
            continue
        if py.stat().st_size == 0:
            report.placeholder_files.append(f"{rel} (0 bytes)")
            continue
        head = py.read_text(encoding="utf-8", errors="ignore")[:200]
        if "<?xml" in head and "<Error>" in head:
            report.placeholder_files.append(f"{rel} (cloud-drive error page)")


def check_assets(report: CheckReport) -> None:
    """Binary asset checklist verification."""
    for asset in ASSETS:
        p = ROOT / asset["path"]
        if p.exists() and p.stat().st_size > 0:
            report.assets_present.append(asset["path"])
        else:
            report.assets_missing.append(asset)


def check_contract_test(report: CheckReport, run: bool = True) -> None:
    """
    Run the contract test. It is the **operational proof** of CODE-level self-containment —
    having the files is not enough; it must actually run through.
    """
    if not run:
        return
    script = ROOT / "benchmark" / "test_e1_contracts.py"
    if not script.exists():
        report.contract_test_ok = False
        report.contract_test_output = "Contract test script missing"
        return
    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True, text=True, timeout=300, cwd=str(ROOT),
        )
        report.contract_test_ok = proc.returncode == 0
        report.contract_test_output = "\n".join(proc.stdout.strip().splitlines()[-6:])
    except subprocess.TimeoutExpired:
        report.contract_test_ok = False
        report.contract_test_output = "Contract test timed out"
    except Exception as exc:
        report.contract_test_ok = False
        report.contract_test_output = f"Contract test raised an exception: {exc}"


# ==========================================================================
# Report rendering
# ==========================================================================
def render(report: CheckReport, strict_runtime: bool) -> str:
    ok, warn, bad = "✅", "⚠️ ", "❌"
    L: list[str] = []
    L.append("=" * 74)
    L.append("Test branch self-containment check (project paradigm: any test branch should be self-contained)")
    L.append("=" * 74)
    L.append(f"Branch root: {ROOT}")
    L.append("")

    L.append("── CODE-level self-containment (mandatory gate)" + " " + "─" * 40)
    L.append(f"  Entry files ({len(report.entrypoints)}/{len(ENTRYPOINTS)}): "
             f"{', '.join(report.entrypoints) if report.entrypoints else 'none'}")
    L.append(f"  {ok if not report.missing_modules else bad} In-package module closure: "
             f"{'all present' if not report.missing_modules else f'{len(report.missing_modules)} missing'}")
    for m in report.missing_modules:
        L.append(f"      - {m}")
    L.append(f"  {ok if not report.compile_errors else bad} Syntax compilation: "
             f"{'all passed' if not report.compile_errors else f'{len(report.compile_errors)} file(s) failed'}")
    for f, e in report.compile_errors:
        L.append(f"      - {f}: {e.splitlines()[0] if e else ''}")
    L.append(f"  {ok if not report.placeholder_files else bad} Placeholder/corrupted files: "
             f"{'none' if not report.placeholder_files else f'{len(report.placeholder_files)}'}")
    for f in report.placeholder_files:
        L.append(f"      - {f}")
    if report.contract_test_ok is not None:
        L.append(f"  {ok if report.contract_test_ok else bad} Contract test: "
                 f"{'PASS' if report.contract_test_ok else 'FAIL'}")
        for line in report.contract_test_output.splitlines():
            L.append(f"      {line}")
    L.append(f"  → CODE-level self-containment: {ok + ' passes' if report.code_level_ok else bad + ' does not pass'}")
    L.append("")

    L.append("── RUNTIME-level self-containment (needed by real-machine acceptance branches)" + " " * 26)
    L.append(f"  {ok} Assets present ({len(report.assets_present)}): "
             f"{', '.join(report.assets_present) if report.assets_present else 'none'}")
    if report.assets_missing:
        L.append(f"  {warn} Assets pending ({len(report.assets_missing)}):")
        for a in report.assets_missing:
            L.append(f"      - {a['path']}")
            L.append(f"          Purpose: {a['note']}")
            L.append(f"          Source: {a['source']}")
    else:
        L.append(f"  {ok} Assets complete")
    L.append(f"  → RUNTIME-level self-containment: "
             f"{ok + ' passes' if report.runtime_level_ok else warn + ' does not pass (does not affect the CODE level)'}")
    L.append("")

    L.append("=" * 74)
    if report.code_level_ok:
        L.append(f"{ok} This test branch reaches CODE-level self-containment — it can be checked out and its logic tests run directly.")
    else:
        L.append(f"{bad} This test branch does **not reach** CODE-level self-containment; please complete the items above before committing.")
    if not report.runtime_level_ok and not report.assets_missing:
        L.append(f"{ok} It also reaches RUNTIME-level self-containment.")
    elif report.assets_missing:
        L.append(f"{warn} The RUNTIME level requires {len(report.assets_missing)} more binary asset(s) "
                 f"(see docs/E1_self_contained_branch_paradigm.md §4).")
    L.append("=" * 74)

    if strict_runtime and not report.runtime_level_ok:
        L.append("")
        L.append(f"{bad} --strict-runtime is enabled: failing the RUNTIME level counts as a failure.")
    return "\n".join(L)


# ==========================================================================
# Main flow
# ==========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="Test branch self-containment check")
    ap.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    ap.add_argument("--strict-runtime", action="store_true",
                    help="Return a non-zero exit code if the RUNTIME level does not pass")
    ap.add_argument("--skip-contract", action="store_true", help="Skip execution of the contract test")
    args = ap.parse_args()

    report = CheckReport()
    check_import_closure(report)
    check_compile(report)
    check_placeholders(report)
    check_assets(report)
    check_contract_test(report, run=not args.skip_contract)

    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(render(report, args.strict_runtime))

    if not report.code_level_ok:
        return 1
    if args.strict_runtime and not report.runtime_level_ok:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
