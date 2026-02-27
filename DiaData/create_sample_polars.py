from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic sample datasets from maindatabase.csv using Polars."
    )
    parser.add_argument(
        "--input",
        default="DiaData/datasets for T1D/maindatabase.csv",
        help="Path to source CSV.",
    )
    parser.add_argument(
        "--output",
        default="DiaData/datasets for T1D/maindatabase_half.csv",
        help="Path to sampled CSV.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.5,
        help="Sample fraction in (0, 1].",
    )
    parser.add_argument(
        "--mode",
        choices=["by_patient", "by_row"],
        default="by_patient",
        help="Sampling strategy.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for deterministic hash-based sampling.",
    )
    parser.add_argument(
        "--summary-json",
        default="DiaData/datasets for T1D/maindatabase_half_summary.json",
        help="Where to save summary stats.",
    )
    return parser.parse_args()


def sampled_lazyframe(
    input_path: str,
    fraction: float,
    mode: str,
    seed: int,
) -> pl.LazyFrame:
    if not 0 < fraction <= 1:
        raise ValueError("--fraction must be in (0, 1].")

    lf = pl.scan_csv(input_path, infer_schema_length=2_000, ignore_errors=True)
    mod = 1_000_000
    keep = max(1, int(mod * fraction))

    if mode == "by_patient":
        return lf.filter((pl.col("PtID").hash(seed=seed) % mod) < keep)
    return lf.filter((pl.struct(pl.all()).hash(seed=seed) % mod) < keep)


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    summary_path = Path(args.summary_json)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    base = pl.scan_csv(str(in_path), infer_schema_length=2_000, ignore_errors=True)
    sample = sampled_lazyframe(
        input_path=str(in_path),
        fraction=args.fraction,
        mode=args.mode,
        seed=args.seed,
    )

    sample.sink_csv(str(out_path))

    base_rows = base.select(pl.len().alias("n")).collect(engine="streaming").item()
    base_pts = base.select(pl.col("PtID").n_unique().alias("n")).collect(
        engine="streaming"
    ).item()
    sample_rows = sample.select(pl.len().alias("n")).collect(engine="streaming").item()
    sample_pts = sample.select(pl.col("PtID").n_unique().alias("n")).collect(
        engine="streaming"
    ).item()

    payload = {
        "input": str(in_path),
        "output": str(out_path),
        "mode": args.mode,
        "fraction_requested": args.fraction,
        "rows_total": int(base_rows),
        "rows_sampled": int(sample_rows),
        "rows_ratio": float(sample_rows / max(1, base_rows)),
        "patients_total": int(base_pts),
        "patients_sampled": int(sample_pts),
        "patients_ratio": float(sample_pts / max(1, base_pts)),
        "seed": int(args.seed),
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
