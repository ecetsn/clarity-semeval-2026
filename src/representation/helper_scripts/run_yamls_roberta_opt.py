# src/representation/scripts/run_yamls_roberta_opt.py
#
# Runs train_from_yaml sequentially on all YAMLs under:
#   <project_root>/src/representation/configs/roberta-opt/*.yaml
#
# Runner stdout/stderr logs + JSONL summary are written to:
#   <project_root>/experiments/logs/run_logs/
#
# Usage:
#   python src/representation/scripts/run_yamls_roberta_opt.py

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def find_repo_root(script_path: Path) -> Path:
    script_path = script_path.resolve()
    return script_path.parents[3]


def safe_relpath(p: Path, root: Path) -> str:
    try:
        return str(p.resolve().relative_to(root.resolve()))
    except Exception:
        return str(p.resolve())


def main() -> None:
    script_path = Path(__file__).resolve()
    repo_root = find_repo_root(script_path)

    yamls_dir = repo_root / "src" / "representation" / "configs" / "roberta-opt"
    if not yamls_dir.exists():
        raise FileNotFoundError(f"YAML directory not found: {yamls_dir}")

    yamls = sorted(yamls_dir.glob("*.yaml"))
    if len(yamls) == 0:
        raise FileNotFoundError(f"No YAML files found in: {yamls_dir}")

    # Runner logs (stdout/stderr) + summary JSONL
    run_logs_dir = repo_root / "experiments" / "logs" / "run_logs"
    run_logs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_logs_dir / "roberta_opt_runs.jsonl"

    print(f"Repo root: {repo_root}")
    print(f"YAMLs dir : {yamls_dir}  (found {len(yamls)})")
    print(f"Run logs : {run_logs_dir}")
    print(f"Summary  : {summary_path}")

    for i, ypath in enumerate(yamls, start=1):
        run_id = ypath.stem

        out_log = run_logs_dir / f"roberta_opt__{i:02d}__{run_id}.out.log"
        err_log = run_logs_dir / f"roberta_opt__{i:02d}__{run_id}.err.log"

        cmd = [
            sys.executable,
            "-m",
            "src.representation.training.train_from_yaml",
            str(ypath),
        ]

        started_at = datetime.now().isoformat(timespec="seconds")
        t0 = time.time()

        print(f"[{i}/{len(yamls)}] RUN: {run_id}")

        with out_log.open("w", encoding="utf-8") as f_out, err_log.open("w", encoding="utf-8") as f_err:
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(repo_root),
                    stdout=f_out,
                    stderr=f_err,
                    check=False,
                )
                rc = proc.returncode
            except Exception as e:
                rc = -1
                f_err.write(f"\n[runner-error] {repr(e)}\n")

        ended_at = datetime.now().isoformat(timespec="seconds")
        duration_sec = round(time.time() - t0, 3)

        row = {
            "idx": i,
            "total": len(yamls),
            "run_id": run_id,
            "yaml_path": safe_relpath(ypath, repo_root),
            "cmd": " ".join(cmd),
            "return_code": rc,
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_sec": duration_sec,
            "stdout_log": safe_relpath(out_log, repo_root),
            "stderr_log": safe_relpath(err_log, repo_root),
        }

        with summary_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        status = "OK" if rc == 0 else f"FAIL(rc={rc})"
        print(f"[{i}/{len(yamls)}] {status} ({duration_sec}s)")

    print("DONE")


if __name__ == "__main__":
    main()
