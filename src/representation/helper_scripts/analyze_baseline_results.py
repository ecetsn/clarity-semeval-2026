
# Scans:
#   <project_root>/experiments/representation/baseline/weightedclasses_<task>_<encpool>_<decpool>/
#
# For each experiment folder:
# - Reads summary.json -> best_macro_f1
# - Scans val_epoch_*.json and finds the one whose metrics.macro_f1 == best_macro_f1 (within tolerance)
# - Extracts metrics.accuracy and metrics.macro_f1 from that val file
# - Ranks all experiments by macro_f1 (descending)
# - Prints a table
# - Writes the ranking outputs under:
#     <project_root>/experiments/logs/representation/
#   Also writes to:
#     <project_root>/experiment/logs/representation/
#   if that directory exists (to support your requested path spelling).
#
# Usage:
#   python src/representation/scripts/rank_baseline_experiments.py

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


NAME_RE = re.compile(
    r"^weightedclasses_(?P<task>clarity|evasion)_(?P<enc>masked_mean|max|cls|last_non_pad)_(?P<dec>masked_mean|max|last_non_pad)$"
)


@dataclass
class Row:
    rank: int
    name: str
    task: str
    enc_pool: str
    dec_pool: str
    accuracy: Optional[float]
    macro_f1: Optional[float]
    best_macro_f1: Optional[float]
    matched_val_file: Optional[str]
    notes: str
    exp_dir: str


def project_root_from_script(script_path: Path) -> Path:
    # <project_root>/src/representation/scripts/this_file.py -> parents[3] is project_root
    return script_path.resolve().parents[3]


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return obj


def as_float(x: Any) -> Optional[float]:
    if isinstance(x, (int, float)):
        return float(x)
    return None


def extract_val_metrics(val_obj: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    m = val_obj.get("metrics")
    if not isinstance(m, dict):
        return None, None
    acc = as_float(m.get("accuracy"))
    mf1 = as_float(m.get("macro_f1"))
    return acc, mf1


def extract_best_macro_f1(summary_obj: Dict[str, Any]) -> Optional[float]:
    return as_float(summary_obj.get("best_macro_f1"))


def find_matching_val_file(
    exp_dir: Path,
    best_macro_f1: float,
    tol: float = 1e-9,
) -> Tuple[Optional[Path], Optional[float], Optional[float], str]:
    """
    Returns: (val_path, accuracy, macro_f1, note)

    Logic:
    1) exact match within tol
    2) fallback to closest macro_f1 by absolute difference
    """
    val_files = sorted(exp_dir.glob("val_epoch_*.json"))
    if not val_files:
        return None, None, None, "No val_epoch_*.json files found"

    # exact match
    for vf in val_files:
        try:
            obj = read_json(vf)
        except Exception:
            continue
        acc, mf1 = extract_val_metrics(obj)
        if mf1 is None:
            continue
        if abs(mf1 - best_macro_f1) <= tol:
            return vf, acc, mf1, "Matched best_macro_f1"

    # closest match fallback
    closest_vf = None
    closest_acc = None
    closest_mf1 = None
    closest_diff = None

    for vf in val_files:
        try:
            obj = read_json(vf)
        except Exception:
            continue
        acc, mf1 = extract_val_metrics(obj)
        if mf1 is None:
            continue
        diff = abs(mf1 - best_macro_f1)
        if closest_diff is None or diff < closest_diff:
            closest_vf = vf
            closest_acc = acc
            closest_mf1 = mf1
            closest_diff = diff

    if closest_vf is None:
        return None, None, None, "Could not parse metrics.macro_f1 from any val_epoch file"

    return closest_vf, closest_acc, closest_mf1, f"No exact match; selected closest (|Δ|={closest_diff:.12g})"


def fmt(x: Optional[float]) -> str:
    return "NA" if x is None else f"{x:.6f}"


def write_outputs(rows: List[Row], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"baseline_rankings_{ts}.json"
    jsonl_path = out_dir / f"baseline_rankings_{ts}.jsonl"
    csv_path = out_dir / f"baseline_rankings_{ts}.csv"
    txt_path = out_dir / f"baseline_rankings_{ts}.txt"

    # JSON
    json_path.write_text(
        json.dumps([r.__dict__ for r in rows], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # JSONL
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")

    # CSV (minimal, no external deps)
    header = [
        "rank",
        "name",
        "task",
        "enc_pool",
        "dec_pool",
        "accuracy",
        "macro_f1",
        "best_macro_f1",
        "matched_val_file",
        "notes",
        "exp_dir",
    ]
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            vals = [
                str(r.rank),
                r.name,
                r.task,
                r.enc_pool,
                r.dec_pool,
                "" if r.accuracy is None else str(r.accuracy),
                "" if r.macro_f1 is None else str(r.macro_f1),
                "" if r.best_macro_f1 is None else str(r.best_macro_f1),
                "" if r.matched_val_file is None else r.matched_val_file,
                r.notes.replace("\n", " ").replace(",", ";"),
                r.exp_dir.replace("\n", " ").replace(",", ";"),
            ]
            f.write(",".join(vals) + "\n")

    # TXT (table)
    lines = []
    lines.append(
        "rank  name                                               task     enc_pool      dec_pool      accuracy    macro_f1    val_file           notes"
    )
    lines.append(
        "----  -------------------------------------------------  -------  ------------  ------------  ----------  ---------  -----------------  ------------------------------------------------"
    )
    for r in rows:
        lines.append(
            f"{str(r.rank).ljust(4)}  "
            f"{r.name.ljust(49)}  "
            f"{r.task.ljust(7)}  "
            f"{r.enc_pool.ljust(12)}  "
            f"{r.dec_pool.ljust(12)}  "
            f"{fmt(r.accuracy).rjust(10)}  "
            f"{fmt(r.macro_f1).rjust(9)}  "
            f"{(r.matched_val_file or 'NA').ljust(17)}  "
            f"{r.notes}"
        )
    txt_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    project_root = project_root_from_script(Path(__file__))
    base_dir = project_root / "experiments" / "representation" / "baseline"
    if not base_dir.exists():
        raise FileNotFoundError(f"Not found: {base_dir}")

    exp_dirs = [p for p in base_dir.iterdir() if p.is_dir() and NAME_RE.match(p.name)]
    if not exp_dirs:
        raise FileNotFoundError(f"No weightedclasses_* folders found under: {base_dir}")

    tmp_rows: List[Tuple[str, Row]] = []

    for exp_dir in sorted(exp_dirs, key=lambda p: p.name):
        m = NAME_RE.match(exp_dir.name)
        assert m is not None

        task = m.group("task")
        enc = m.group("enc")
        dec = m.group("dec")

        summary_path = exp_dir / "summary.json"
        if not summary_path.exists():
            tmp_rows.append(
                (
                    exp_dir.name,
                    Row(
                        rank=-1,
                        name=exp_dir.name,
                        task=task,
                        enc_pool=enc,
                        dec_pool=dec,
                        accuracy=None,
                        macro_f1=None,
                        best_macro_f1=None,
                        matched_val_file=None,
                        notes="Missing summary.json",
                        exp_dir=str(exp_dir),
                    ),
                )
            )
            continue

        summary = read_json(summary_path)
        best_mf1 = extract_best_macro_f1(summary)
        if best_mf1 is None:
            tmp_rows.append(
                (
                    exp_dir.name,
                    Row(
                        rank=-1,
                        name=exp_dir.name,
                        task=task,
                        enc_pool=enc,
                        dec_pool=dec,
                        accuracy=None,
                        macro_f1=None,
                        best_macro_f1=None,
                        matched_val_file=None,
                        notes="summary.json missing best_macro_f1",
                        exp_dir=str(exp_dir),
                    ),
                )
            )
            continue

        val_path, acc, mf1, note = find_matching_val_file(exp_dir, best_mf1, tol=1e-9)

        tmp_rows.append(
            (
                exp_dir.name,
                Row(
                    rank=-1,
                    name=exp_dir.name,
                    task=task,
                    enc_pool=enc,
                    dec_pool=dec,
                    accuracy=acc,
                    macro_f1=mf1,
                    best_macro_f1=best_mf1,
                    matched_val_file=(val_path.name if val_path else None),
                    notes=note,
                    exp_dir=str(exp_dir),
                ),
            )
        )

    # Rank by macro_f1 (prefer mf1 from matched val file; else -inf)
    def score(r: Row) -> float:
        return r.macro_f1 if isinstance(r.macro_f1, float) else float("-inf")

    rows_sorted = sorted((r for _, r in tmp_rows), key=score, reverse=True)
    for i, r in enumerate(rows_sorted, start=1):
        r.rank = i

    # Print to console
    print(
        "rank  name                                               task     enc_pool      dec_pool      accuracy    macro_f1    val_file           notes"
    )
    print(
        "----  -------------------------------------------------  -------  ------------  ------------  ----------  ---------  -----------------  ------------------------------------------------"
    )
    for r in rows_sorted:
        print(
            f"{str(r.rank).ljust(4)}  "
            f"{r.name.ljust(49)}  "
            f"{r.task.ljust(7)}  "
            f"{r.enc_pool.ljust(12)}  "
            f"{r.dec_pool.ljust(12)}  "
            f"{fmt(r.accuracy).rjust(10)}  "
            f"{fmt(r.macro_f1).rjust(9)}  "
            f"{(r.matched_val_file or 'NA').ljust(17)}  "
            f"{r.notes}"
        )

    # Write outputs (as requested)
    out_dir_1 = project_root / "experiments" / "logs" / "representation"
    write_outputs(rows_sorted, out_dir_1)

    # Also write to <project_root>/experiment/logs/representation if it already exists
    out_dir_2 = project_root / "experiment" / "logs" / "representation"
    if (project_root / "experiment").exists():
        write_outputs(rows_sorted, out_dir_2)


if __name__ == "__main__":
    main()
