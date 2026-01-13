#!/usr/bin/env python3
"""
Fetch CDC Influenza A wastewater surveillance data.

Dataset: CDC Wastewater Data for Influenza A (ymmh-divb)
Source: https://data.cdc.gov/Public-Health-Surveillance/CDC-Wastewater-Data-for-Influenza-A/ymmh-divb
"""

import argparse
import logging
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from sodapy import Socrata

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def validate_date(date_str: str) -> str:
    """
    Validate date string format to prevent SOQL injection.

    Args:
        date_str: Date string in YYYY-MM-DD format

    Returns:
        Validated date string

    Raises:
        ValueError: If date format is invalid
    """
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        raise ValueError(f"Invalid date format: {date_str}. Expected YYYY-MM-DD")

    # Parse to ensure it's a real date
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Invalid date: {date_str}")

    return date_str


def validate_output_path(output_dir: str | Path, project_root: Path) -> Path:
    """
    Validate output directory to prevent path traversal attacks.

    Args:
        output_dir: User-provided output directory
        project_root: Absolute path to project root

    Returns:
        Validated absolute path

    Raises:
        ValueError: If path escapes project root
    """
    output_path = Path(output_dir).resolve()
    project_root = project_root.resolve()

    # Ensure path is within project root
    try:
        output_path.relative_to(project_root)
    except ValueError:
        raise ValueError(f"Output directory {output_path} is outside project root {project_root}")

    return output_path


def validate_limit(value: str) -> int:
    """Validate limit argument."""
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid integer: {value}")

    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"Limit must be positive, got: {ivalue}")

    if ivalue > 10_000_000:
        raise argparse.ArgumentTypeError(f"Limit too large: {ivalue} (max: 10M)")

    return ivalue

CDC_DOMAIN = "data.cdc.gov"
FLU_DATASET_ID = "ymmh-divb"


def fetch_flu_wastewater(
    start_date: str = "2024-01-01", limit: int = 500000, output_dir: Path | None = None
) -> pd.DataFrame:
    """
    Fetch Influenza A wastewater data from CDC.

    Args:
        start_date: Start date filter (YYYY-MM-DD)
        limit: Maximum records to fetch
        output_dir: Directory to save data

    Returns:
        DataFrame with flu wastewater data
    """
    logger.info(f"Fetching Influenza A wastewater data...")

    client = Socrata(CDC_DOMAIN, None)

    # Validate date to prevent SOQL injection
    validated_date = validate_date(start_date)
    where_clause = f"sample_collect_date >= '{validated_date}'"
    logger.info(f"Filter: {where_clause}")

    results = client.get(
        FLU_DATASET_ID, limit=limit, where=where_clause, order="sample_collect_date DESC"
    )

    df = pd.DataFrame.from_records(results)
    logger.info(f"Fetched {len(df):,} records")

    # Convert types
    df["sample_collect_date"] = pd.to_datetime(df["sample_collect_date"])

    numeric_cols = [
        "population_served",
        "pcr_target_avg_conc",
        "pcr_target_avg_conc_lin",
        "pcr_target_flowpop_lin",
        "flow_rate",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Save if output specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d")
        parquet_path = output_dir / f"flu_wastewater_{timestamp}.parquet"
        csv_path = output_dir / f"flu_wastewater_{timestamp}.csv"

        df.to_parquet(parquet_path, index=False)
        df.to_csv(csv_path, index=False)

        logger.info(f"Saved to {parquet_path}")

    return df


def aggregate_to_state_week(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate flu wastewater data to state-week level.
    """
    df = df.copy()

    # Create week ending date (Saturday)
    df["week_end"] = df["sample_collect_date"].dt.to_period("W-SAT").dt.end_time.dt.normalize()

    # Aggregate by state and week
    # Use concentration as the main signal
    agg = (
        df.groupby(["wwtp_jurisdiction", "week_end"])
        .agg(
            {
                "pcr_target_avg_conc_lin": "mean",  # Linear concentration
                "pcr_target_flowpop_lin": "mean",  # Flow-population normalized
                "population_served": "sum",
                "sewershed_id": "nunique",
            }
        )
        .reset_index()
    )

    agg = agg.rename(
        columns={
            "wwtp_jurisdiction": "state",
            "pcr_target_avg_conc_lin": "flu_concentration",
            "pcr_target_flowpop_lin": "flu_flowpop",
            "sewershed_id": "flu_n_sites",
        }
    )

    logger.info(f"Aggregated to {len(agg):,} state-week records")
    return agg


def main():
    parser = argparse.ArgumentParser(description="Fetch Influenza A wastewater data")
    parser.add_argument("--start-date", default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--output", default="data/raw/flu_wastewater", help="Output directory")
    parser.add_argument("--limit", type=validate_limit, default=500000, help="Max records (1-10,000,000)")

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Validate output path to prevent path traversal
    try:
        output_dir = validate_output_path(project_root / args.output, project_root)
    except ValueError as e:
        logger.error(f"Invalid output path: {e}")
        exit(1)

    df = fetch_flu_wastewater(start_date=args.start_date, limit=args.limit, output_dir=output_dir)

    print("\n" + "=" * 60)
    print("FLU WASTEWATER DATA SUMMARY")
    print("=" * 60)
    print(f"Total records: {len(df):,}")
    print(f"Date range: {df['sample_collect_date'].min()} to {df['sample_collect_date'].max()}")
    print(f"States: {df['wwtp_jurisdiction'].nunique()}")
    print(f"Sites: {df['sewershed_id'].nunique()}")


if __name__ == "__main__":
    main()
