from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build train/val/test model dataset from maindatabase.csv using Polars streaming."
    )
    parser.add_argument(
        "--input",
        default="DiaData/datasets for T1D/maindatabase.csv",
        help="Path to raw integrated full CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="DiaData/model_training/full_dataset",
        help="Directory where parquet splits and metadata are written.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=500_000)
    return parser.parse_args()


def _prep_batch(
    df: pl.DataFrame,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    has_database: bool,
) -> pl.DataFrame:
    if val_ratio <= 0 or test_ratio <= 0 or val_ratio + test_ratio >= 0.9:
        raise ValueError("Invalid split ratios. Ensure 0 < val/test and val+test < 0.9")

    train_ratio = 1.0 - val_ratio - test_ratio
    bucket_mod = 10_000
    train_cut = int(train_ratio * bucket_mod)
    val_cut = int((train_ratio + val_ratio) * bucket_mod)

    select_cols = ["ts", "PtID", "GlucoseCGM"] + (["Database"] if has_database else [])
    return (
        df.select(select_cols)
        .with_columns(
            [
                pl.col("ts").str.to_datetime(strict=False).alias("ts"),
                pl.col("PtID").cast(pl.Utf8),
                pl.col("GlucoseCGM").cast(pl.Float32),
            ]
        )
        .with_columns(
            [
                (
                    (pl.col("ts").dt.hour() * 60) + pl.col("ts").dt.minute()
                ).cast(pl.Int16).alias("minute_of_day"),
                pl.col("ts").dt.weekday().cast(pl.Int8).alias("weekday"),
                pl.lit(0.0).cast(pl.Float32).alias("insulin_consumed"),
            ]
        )
        .drop_nulls(["ts", "PtID", "GlucoseCGM"])
        .filter(pl.col("GlucoseCGM").is_between(40.0, 400.0))
        .with_columns((pl.col("PtID").hash(seed=seed) % bucket_mod).cast(pl.Int32).alias("split_bucket"))
        .with_columns(
            pl.when(pl.col("split_bucket") < train_cut)
            .then(pl.lit("train"))
            .when(pl.col("split_bucket") < val_cut)
            .then(pl.lit("val"))
            .otherwise(pl.lit("test"))
            .alias("split")
        )
        .drop("split_bucket")
    )


def build_and_write_chunked(
    input_path: Path,
    out_dir: Path,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    batch_size: int,
) -> dict[str, dict[str, float | int | list[str]]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict[str, float | int | list[str]]] = {
        "train": {"rows": 0, "patients": 0, "files": []},
        "val": {"rows": 0, "patients": 0, "files": []},
        "test": {"rows": 0, "patients": 0, "files": []},
    }
    patient_sets: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    part_idx = {"train": 0, "val": 0, "test": 0}

    reader = pl.read_csv_batched(
        str(input_path),
        infer_schema_length=2_000,
        batch_size=batch_size,
        ignore_errors=True,
        low_memory=True,
    )

    has_database = False
    first_batches = reader.next_batches(1)
    if first_batches:
        has_database = "Database" in first_batches[0].columns
        # process the first batch before continuing
        batches_iter = [first_batches[0]]
    else:
        batches_iter = []

    while True:
        if not batches_iter:
            batches = reader.next_batches(1)
            if not batches:
                break
            batch_df = batches[0]
        else:
            batch_df = batches_iter.pop()

        prepped = _prep_batch(
            df=batch_df,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            has_database=has_database,
        )
        if prepped.is_empty():
            continue

        for split in ["train", "val", "test"]:
            split_df = prepped.filter(pl.col("split") == split)
            if split_df.is_empty():
                continue

            part_path = out_dir / f"{split}_part_{part_idx[split]:05d}.parquet"
            split_df.write_parquet(part_path, compression="zstd")
            part_idx[split] += 1

            stats[split]["rows"] = int(stats[split]["rows"]) + split_df.height
            stats[split]["files"].append(str(part_path))
            patient_sets[split].update(split_df.get_column("PtID").unique().to_list())

    for split in ["train", "val", "test"]:
        stats[split]["patients"] = len(patient_sets[split])

    return stats


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    stats = build_and_write_chunked(
        input_path=input_path,
        out_dir=out_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        batch_size=args.batch_size,
    )

    meta = {
        "input": str(input_path),
        "output_dir": str(out_dir),
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "splits": stats,
    }
    meta_path = out_dir / "dataset_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
