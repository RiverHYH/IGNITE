"""
datasetGate.py
==============
DHSO dataset-integrity pre-flight gate.  Deliverable of improvement item **D1 /
T0-2**; it turns the one-off T0-1b manual audit into a check that runs
automatically whenever the evaluation harness starts.

Why this exists
---------------
The DHSO freeze set (N = 1,285; 889 hazardous / 396 benign) is the evaluation
oracle behind every downstream IGNITE claim, so any silent modification of it
invalidates every metric computed on top of it.  The T0-1b audit established
three facts a re-run must not quietly lose:

  1. **ids are not contiguous.**  The source pool was ``test_1 .. test_1300``;
     CSP dropped 15 frames.  ``test_1 .. test_1285`` does *not* exist as a
     block, so reconstructing the benchmark by numbering silently samples the
     wrong set -- and the total still comes out at 1,285, so a bare count check
     cannot detect it.
  2. **``safe/`` holds 2 genuine duplicate pairs** (byte-identical), i.e. the
     effective number of independent samples is 1,283, not 1,285.
  3. **zero cross-label duplication**, hence the oracle contains no same-frame
     label contradiction.

Completeness argument (why size-grouping is enough)
---------------------------------------------------
Content duplication necessarily implies identical byte size.  Therefore
grouping files by size is a *complete* -- not sampled -- capture of duplicate
candidates.  SHA-256 is then only required on those candidates, so hashing cost
is O(number of collisions) rather than O(dataset size).

Severity model
--------------
HARD (raises ``DatasetIntegrityError``)
    Count drift, id-set drift, effective-N drift, cross-label identical
    content, same id under both labels.  These mean the frozen set was changed;
    results are no longer comparable with the recorded baseline.
WARN (logged, non-blocking)
    Individual duplicate frames and extra unknown duplicates.  Decided policy:
    keep all 1,285 frames frozen, record the duplication, and additionally
    report a de-duplicated variant (effective N = 1,283).

Standalone use
--------------
    python datasetGate.py --safe-dir <dir> --fire-dir <dir>
    python datasetGate.py --safe-dir <dir> --fire-dir <dir> --no-hash
    python datasetGate.py --safe-dir <dir> --fire-dir <dir> --baseline my.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Frozen baseline -- DHSO N = 1,285.
# Source: Planning/D1_评测Oracle重建_执行计划_v1.1.md  section 1.4
#         D1_audit/T0-1b_DHSO_完整性审计报告.md
# Everything here is a parameter, never a scattered literal.
# ---------------------------------------------------------------------------
BASELINE: Dict[str, Any] = {
    "n_safe": 396,
    "n_fire": 889,
    "n_total": 1285,
    "source_pool_range": [1, 1300],
    "expected_missing_ids": [
        48, 203, 211, 279, 315, 467, 486, 721, 751, 766,
        844, 990, 1118, 1186, 1282,
    ],
    "known_duplicate_pairs": [["test_584", "test_1240"], ["test_696", "test_967"]],
    "expected_effective_n": 1283,
}

FRAME_ID_RE = re.compile(r"^test_(\d+)$", re.IGNORECASE)
VALID_EXTS = (".jpg", ".jpeg", ".png")
_HASH_CHUNK = 1 << 20  # 1 MiB


class DatasetIntegrityError(RuntimeError):
    """Raised when the frozen DHSO set has drifted from the recorded baseline."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class GateReport:
    """Outcome of one gate run.  ``ok`` is False only for HARD failures."""

    n_safe: int = 0
    n_fire: int = 0
    n_total: int = 0
    n_ids: int = 0
    effective_n: int = 0
    redundant_copies: int = 0
    # True once SHA-256 has confirmed the duplicate candidates.  When False the
    # duplicate list is a size-level candidate list and effective_n is unverified.
    duplicates_confirmed: bool = True
    # (label, stem_a, stem_b, size) -- identical content within one label
    duplicate_pairs: List[Tuple[str, str, str, int]] = field(default_factory=list)
    # (safe_stem, fire_stem, size) -- identical content across labels
    cross_label_pairs: List[Tuple[str, str, int]] = field(default_factory=list)
    # size-only cross-label candidates, unresolved when --no-hash is used
    cross_label_candidates: List[Tuple[str, str, int]] = field(default_factory=list)
    # baseline pairs re-confirmed by this run
    known_pairs_recovered: List[str] = field(default_factory=list)
    # same numeric id appearing under both labels
    conflicting_ids: List[int] = field(default_factory=list)
    missing_ids: List[int] = field(default_factory=list)
    unexpected_ids: List[int] = field(default_factory=list)
    unnamed_files: List[str] = field(default_factory=list)
    hashed_files: int = 0
    hash_skipped: bool = False
    hard_failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.hard_failures


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _log(logger: Any, level: str, msg: str) -> None:
    """Emit through the caller's logger, degrading gracefully to ``print``."""
    fn = getattr(logger, level, None) if logger is not None else None
    if fn is None and logger is not None:
        fn = getattr(logger, "info", None)
    (fn or print)(msg)


def _stem(fname: str) -> str:
    return os.path.splitext(fname)[0]


def _frame_id(fname: str) -> Optional[int]:
    m = FRAME_ID_RE.match(_stem(fname))
    return int(m.group(1)) if m else None


def _list_frames(directory: str) -> Dict[str, int]:
    """Return ``{filename: byte_size}`` for every image directly inside ``directory``."""
    if not os.path.isdir(directory):
        raise DatasetIntegrityError("directory not found: %s" % directory)
    out: Dict[str, int] = {}
    for fname in os.listdir(directory):
        if fname.lower().endswith(VALID_EXTS):
            out[fname] = os.path.getsize(os.path.join(directory, fname))
    return out


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _size_groups(frames: Dict[str, int]) -> Dict[int, List[str]]:
    """Sizes shared by more than one file inside a single label."""
    groups: Dict[int, List[str]] = defaultdict(list)
    for fname, size in frames.items():
        groups[size].append(fname)
    return {size: sorted(names) for size, names in groups.items() if len(names) > 1}


def _merge_baseline(override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    base = json.loads(json.dumps(BASELINE))  # deep copy
    if override:
        base.update(override)
    return base


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
def audit(
    safe_dir: str,
    fire_dir: str,
    baseline: Optional[Dict[str, Any]] = None,
    do_hash: bool = True,
) -> GateReport:
    """Audit a DHSO-style freeze set and return a :class:`GateReport`.

    Raises :class:`DatasetIntegrityError` only if a directory cannot be read at
    all.  Drift is *reported* in ``hard_failures`` so callers can decide whether
    to enforce; use :func:`enforce` for the enforcing path.
    """
    base = _merge_baseline(baseline)
    rep = GateReport()

    safe = _list_frames(safe_dir)
    fire = _list_frames(fire_dir)
    rep.n_safe, rep.n_fire = len(safe), len(fire)
    rep.n_total = rep.n_safe + rep.n_fire

    # --- 1. size-level candidates (a COMPLETE capture, not a sample) --------
    safe_by_size: Dict[int, List[str]] = defaultdict(list)
    for fname, size in safe.items():
        safe_by_size[size].append(fname)
    fire_by_size: Dict[int, List[str]] = defaultdict(list)
    for fname, size in fire.items():
        fire_by_size[size].append(fname)

    def _pairs(label: str, names: List[str], size: int):
        st = sorted(_stem(n) for n in names)
        for i in range(len(st)):
            for j in range(i + 1, len(st)):
                yield (label, st[i], st[j], size)

    within_cand: List[Tuple[str, str, str, int]] = []
    for size, names in sorted(safe_by_size.items()):
        if len(names) > 1:
            within_cand.extend(_pairs("safe", names, size))
    for size, names in sorted(fire_by_size.items()):
        if len(names) > 1:
            within_cand.extend(_pairs("fire", names, size))
    cand_redundant = sum(
        len(v) - 1
        for v in list(safe_by_size.values()) + list(fire_by_size.values())
        if len(v) > 1
    )

    for size in sorted(set(safe_by_size) & set(fire_by_size)):
        for s_f in sorted(safe_by_size[size]):
            for f_f in sorted(fire_by_size[size]):
                rep.cross_label_candidates.append((_stem(s_f), _stem(f_f), size))

    # only files whose size is shared are worth hashing
    cand: Dict[str, Tuple[str, str, int]] = {}  # path -> (label, fname, size)
    for fname, size in safe.items():
        if len(safe_by_size[size]) > 1 or size in fire_by_size:
            cand[os.path.join(safe_dir, fname)] = ("safe", fname, size)
    for fname, size in fire.items():
        if len(fire_by_size[size]) > 1 or size in safe_by_size:
            cand[os.path.join(fire_dir, fname)] = ("fire", fname, size)

    # --- 2. hash confirmation (cost is O(collisions), not O(dataset)) -------
    rep.hash_skipped = not do_hash
    rep.duplicates_confirmed = bool(do_hash)
    if do_hash:
        if cand:
            by_digest: Dict[str, List[Tuple[str, str, int]]] = defaultdict(list)
            for path, (label, fname, size) in cand.items():
                by_digest[_sha256(path)].append((label, fname, size))
            rep.hashed_files = len(cand)

            for members in by_digest.values():
                if len(members) < 2:
                    continue
                labels = {lbl for lbl, _nm, _sz in members}
                size = max(sz for _l, _n, sz in members)
                if len(labels) > 1:
                    safe_names = [nm for lbl, nm, _ in members if lbl == "safe"]
                    fire_names = [nm for lbl, nm, _ in members if lbl == "fire"]
                    for s_nm in sorted(safe_names):
                        for f_nm in sorted(fire_names):
                            rep.cross_label_pairs.append((_stem(s_nm), _stem(f_nm), size))
                else:
                    seg_label = members[0][0]
                    names = sorted(nm for _l, nm, _sz in members)
                    rep.redundant_copies += len(names) - 1
                    for i in range(len(names)):
                        for j in range(i + 1, len(names)):
                            rep.duplicate_pairs.append(
                                (seg_label, _stem(names[i]), _stem(names[j]), size)
                            )
        else:
            rep.warnings.append(
                "no duplicate candidates found by size -- nothing to hash "
                "(expected for a clean set)"
            )
    else:
        # Size level only.  The candidates are still reported so that a skipped
        # hash can never masquerade as "no duplicates found".
        rep.duplicate_pairs = within_cand
        rep.redundant_copies = cand_redundant
        rep.warnings.append(
            "--no-hash: candidate list rests on byte size alone and is NOT "
            "cryptographically confirmed; effective N is therefore unverified"
        )
        if rep.cross_label_candidates:
            rep.warnings.append(
                "cross-label size collision(s) %s unresolved without hashing; "
                "SHA-256 is required to decide whether a same-frame label "
                "contradiction exists" % (rep.cross_label_candidates,)
            )

    # --- 3. effective sample count -----------------------------------------
    rep.effective_n = rep.n_total - rep.redundant_copies

    # --- 4. id coverage -----------------------------------------------------
    safe_ids = {i for i in (_frame_id(f) for f in safe) if i is not None}
    fire_ids = {i for i in (_frame_id(f) for f in fire) if i is not None}
    rep.conflicting_ids = sorted(safe_ids & fire_ids)

    lo, hi = base["source_pool_range"]
    expected_present = set(range(lo, hi + 1)) - set(base["expected_missing_ids"])
    present = safe_ids | fire_ids
    rep.n_ids = len(present)
    rep.missing_ids = sorted(expected_present - present)
    rep.unexpected_ids = sorted(present - expected_present)

    unnamed = len(safe) + len(fire) - (len(safe_ids) + len(fire_ids))
    if unnamed > 0:
        rep.unnamed_files.append("%d file(s)" % unnamed)
        rep.warnings.append(
            "%d file(s) do not match the 'test_<id>' naming pattern and were "
            "excluded from the id-coverage check" % unnamed
        )

    # --- 5. known-duplicate bookkeeping ------------------------------------
    if do_hash:
        found = {frozenset((a, b)) for _lbl, a, b, _s in rep.duplicate_pairs}
        for pair in base["known_duplicate_pairs"]:
            if frozenset(pair) in found:
                rep.known_pairs_recovered.append("%s <-> %s" % tuple(pair))
            else:
                rep.warnings.append(
                    "baseline duplicate pair %s <-> %s was NOT re-confirmed "
                    "(dataset content changed?)" % tuple(pair)
                )

    # --- 6. severity --------------------------------------------------------
    for key, label in (("n_safe", "safe/"), ("n_fire", "fire/")):
        got = rep.__dict__[key]
        if got != base[key]:
            rep.hard_failures.append(
                "count drift in %s: got %d, baseline %d" % (label, got, base[key])
            )
    if rep.n_total != base["n_total"]:
        rep.hard_failures.append(
            "total count drift: got %d, baseline %d" % (rep.n_total, base["n_total"])
        )
    if not do_hash:
        rep.warnings.append(
            "effective-sample-count check skipped: certifying effective N "
            "requires SHA-256 confirmation"
        )
    elif rep.effective_n != base["expected_effective_n"]:
        rep.hard_failures.append(
            "effective sample count drift: got %d, baseline %d"
            % (rep.effective_n, base["expected_effective_n"])
        )
    if rep.missing_ids:
        rep.hard_failures.append(
            "missing id(s) %s -- the expected present set is source pool %d..%d "
            "minus the recorded 15 dropped frames" % (rep.missing_ids, lo, hi)
        )
    if rep.unexpected_ids:
        rep.hard_failures.append(
            "unexpected id(s) %s outside the recorded present set" % rep.unexpected_ids
        )
    if rep.conflicting_ids:
        rep.hard_failures.append(
            "id(s) %s appear under BOTH labels -- oracle label contradiction"
            % rep.conflicting_ids
        )
    if rep.cross_label_pairs:
        rep.hard_failures.append(
            "cross-label byte-identical content: %s" % (rep.cross_label_pairs,)
        )

    for _lbl, a, b, size in rep.duplicate_pairs:
        rep.warnings.append(
            "duplicate frame within %s: %s == %s (%d B, SHA-256 identical)"
            % (_lbl, a, b, size)
        )

    return rep


def format_report(rep: GateReport, baseline: Optional[Dict[str, Any]] = None) -> str:
    """Render a :class:`GateReport` as a fixed-width text block."""
    base = _merge_baseline(baseline)
    lines: List[str] = []
    add = lines.append
    add("=" * 71)
    add(" D1 / T0-2   DHSO DATASET INTEGRITY GATE")
    add("=" * 71)
    add(" Counts        : safe %d / %d | fire %d / %d | total %d / %d   (got / baseline)"
        % (rep.n_safe, base["n_safe"], rep.n_fire, base["n_fire"],
           rep.n_total, base["n_total"]))
    conf = ("CONFIRMED (SHA-256)" if rep.duplicates_confirmed
            else "UNCONFIRMED (size-level only)")
    add(" Effective N   : %d / %d%s   [%s]"
        % (rep.effective_n, base["expected_effective_n"],
           "   [DRIFT]" if (rep.duplicates_confirmed
                           and rep.effective_n != base["expected_effective_n"]) else "",
           "verified" if rep.duplicates_confirmed else "unverified"))
    add(" Redundant     : %d cop%s from %d within-label pair(s)  -- %s"
        % (rep.redundant_copies, "y" if rep.redundant_copies == 1 else "ies",
           len(rep.duplicate_pairs), conf))
    add(" Files hashed  : %d%s"
        % (rep.hashed_files, "   (--no-hash: size-level only)" if rep.hash_skipped else ""))
    add("-" * 71)
    add(" Id coverage   : %d id(s) present | missing %s | unexpected %s"
        % (rep.n_ids, rep.missing_ids or "none", rep.unexpected_ids or "none"))
    if rep.cross_label_pairs:
        add(" Cross-label   : %s   <-- LABEL CONTRADICTION" % (rep.cross_label_pairs,))
    elif rep.cross_label_candidates and not rep.duplicates_confirmed:
        add(" Cross-label   : size-collision candidate(s) %s -- unresolved without hashing"
            % (rep.cross_label_candidates,))
    else:
        add(" Cross-label   : none -- no same-frame label contradiction")
    add(" Same id in both labels : %s" % (rep.conflicting_ids or "none"))
    if rep.duplicate_pairs:
        add(" Duplicate pairs within a label (%s):" % conf)
        for lbl, a, b, size in rep.duplicate_pairs:
            add("   * [%s] %s == %s   (%d B)" % (lbl, a, b, size))
    else:
        add(" Duplicate pairs within a label : none")
    if rep.known_pairs_recovered:
        add(" Baseline pairs re-confirmed : %s" % ", ".join(rep.known_pairs_recovered))
    if rep.warnings:
        add("-" * 71)
        add(" WARNINGS (non-blocking)")
        for w in rep.warnings:
            add("   * " + w)
    if rep.hard_failures:
        add("-" * 71)
        add(" HARD FAILURES")
        for h in rep.hard_failures:
            add("   ! " + h)
    add("-" * 71)
    add(" VERDICT       : %s" % ("PASS (warnings only)" if rep.ok else "FAIL"))
    add("=" * 71)
    return "\n".join(lines)


def enforce(
    rep: GateReport,
    logger: Any = None,
    baseline: Optional[Dict[str, Any]] = None,
) -> GateReport:
    """Log the report and raise if there were HARD failures."""
    _log(logger, "report", "\n" + format_report(rep, baseline))
    for h in rep.hard_failures:
        _log(logger, "error", "DATASET GATE FAILURE: " + h)
    if not rep.ok:
        raise DatasetIntegrityError(
            "DHSO freeze set drifted from baseline (%d hard failure(s)); refusing "
            "to run evaluation on a non-comparable set." % len(rep.hard_failures)
        )
    _log(logger, "info", "Dataset integrity gate: PASS (warnings only)")
    return rep


def run_gate(
    safe_dir: str,
    fire_dir: str,
    logger: Any = None,
    baseline: Optional[Dict[str, Any]] = None,
    do_hash: bool = True,
    raise_on_failure: bool = True,
) -> GateReport:
    """One-call entry point for the evaluation harness.

    Called at the top of ``test.py``'s entry points so the T0-1b audit is re-run
    automatically on every evaluation start instead of being a one-off manual
    check.
    """
    _log(logger, "info", "Running D1/T0-2 dataset integrity gate...")
    rep = audit(safe_dir, fire_dir, baseline=baseline, do_hash=do_hash)
    if raise_on_failure:
        return enforce(rep, logger=logger, baseline=baseline)
    _log(logger, "report", "\n" + format_report(rep, baseline))
    return rep


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="DHSO dataset-integrity gate (D1/T0-2)")
    ap.add_argument("--safe-dir", default="Dataset//evaluation_slices//safe")
    ap.add_argument(
        "--fire-dir", default="Dataset//evaluation_slices//fire_present_set//images"
    )
    ap.add_argument("--baseline", default=None, help="JSON file overriding baseline keys")
    ap.add_argument(
        "--no-hash", action="store_true", help="size-level check only (fast, weaker)"
    )
    ap.add_argument(
        "--no-fail", action="store_true", help="report only, never exit non-zero"
    )
    args = ap.parse_args(argv)

    baseline = None
    if args.baseline:
        with open(args.baseline, "r", encoding="utf-8") as fh:
            baseline = json.load(fh)

    rep = audit(args.safe_dir, args.fire_dir, baseline=baseline, do_hash=not args.no_hash)
    print(format_report(rep, baseline))
    if not args.no_fail and not rep.ok:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
