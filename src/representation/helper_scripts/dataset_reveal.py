# src/representation/scripts/inspect_qevasion_splits.py
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

from datasets import load_dataset


def _row_to_printable(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make row JSON-printable and not insanely long.
    Truncates long strings and lists for console readability.
    """
    out = {}
    for k, v in row.items():
        if isinstance(v, str):
            out[k] = (v[:200] + " ...") if len(v) > 200 else v
        elif isinstance(v, list):
            # short preview
            out[k] = v[:20] if len(v) > 20 else v
        else:
            out[k] = v
    return out


def print_split_summary(ds, split_name: str, n_rows: int) -> None:
    print("\n" + "=" * 100)
    print(f"SPLIT: {split_name}")
    print("=" * 100)

    print(f"Num rows: {ds.num_rows}")
    print(f"Columns ({len(ds.column_names)}): {ds.column_names}")

    n = min(n_rows, ds.num_rows)
    print(f"\nFirst {n} rows (truncated fields):\n")

    for i in range(n):
        row = ds[i]
        row = _row_to_printable(row)
        print(f"--- row[{i}] ---")
        print(json.dumps(row, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="ailsntua/QEvasion")
    parser.add_argument("--n_rows", type=int, default=2)
    args = parser.parse_args()

    ds_dict = load_dataset(args.dataset_name)

    # Show what splits exist
    print("=" * 100)
    print(f"Dataset: {args.dataset_name}")
    print(f"Splits found: {list(ds_dict.keys())}")
    print("=" * 100)

    # Print train/test specifically if they exist
    if "train" in ds_dict:
        print_split_summary(ds_dict["train"], "train", args.n_rows)
    else:
        print("\n[WARN] No 'train' split found.")

    if "test" in ds_dict:
        print_split_summary(ds_dict["test"], "test", args.n_rows)
    else:
        print("\n[WARN] No 'test' split found.")


if __name__ == "__main__":
    main()
