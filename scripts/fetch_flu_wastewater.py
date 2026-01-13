#!/usr/bin/env python3
"""
Fetch CDC Influenza A wastewater surveillance data.

Dataset: CDC Wastewater Data for Influenza A (ymmh-divb)
Source: https://data.cdc.gov/Public-Health-Surveillance/CDC-Wastewater-Data-for-Influenza-A/ymmh-divb
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from sodapy import Socrata

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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

    where_clause = f"sample_collect_date >= '{start_date}'"
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
    parser.add_argument("--limit", type=int, default=500000, help="Max records")

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_dir = project_root / args.output

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
