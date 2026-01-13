#!/usr/bin/env python3
"""
Fetch CDC NWSS wastewater surveillance data.

Data Sources:
- SARS-CoV-2 Wastewater Metrics: https://data.cdc.gov/resource/2ew6-ywp6
- SARS-CoV-2 Concentration Data: https://data.cdc.gov/resource/g653-rqe2

Usage:
    python scripts/fetch_nwss_data.py
    python scripts/fetch_nwss_data.py --start-date 2024-01-01 --output data/raw/nwss
"""

import argparse
import logging
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from sodapy import Socrata

# Allowed domains for SSRF prevention
ALLOWED_DOMAINS = ['data.cdc.gov', 'healthdata.gov']

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
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
    if not re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        raise ValueError(f"Invalid date format: {date_str}. Expected YYYY-MM-DD")

    # Parse to ensure it's a real date
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
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
        raise ValueError(
            f"Output directory {output_path} is outside project root {project_root}"
        )

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


# CDC Socrata API endpoints
CDC_DOMAIN = "data.cdc.gov"
DATASETS = {
    "metrics": {
        "id": "2ew6-ywp6",
        "name": "NWSS Public SARS-CoV-2 Wastewater Metric Data",
        "description": "Percentiles and percent changes for wastewater viral activity"
    },
    "concentration": {
        "id": "g653-rqe2",
        "name": "NWSS Public SARS-CoV-2 Concentration Data",
        "description": "Raw concentration values with normalization factors"
    }
}


def fetch_nwss_data(
    dataset_key: str = "metrics",
    start_date: str | None = None,
    limit: int = 500000,
    output_dir: Path | None = None
) -> pd.DataFrame:
    """
    Fetch NWSS data from CDC Socrata API.

    Args:
        dataset_key: Which dataset to fetch ("metrics" or "concentration")
        start_date: Optional start date filter (YYYY-MM-DD)
        limit: Maximum number of records to fetch
        output_dir: Directory to save the data

    Returns:
        DataFrame with NWSS data
    """
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_key}. Choose from: {list(DATASETS.keys())}")

    dataset = DATASETS[dataset_key]
    logger.info(f"Fetching {dataset['name']}...")

    # Initialize Socrata client (no authentication needed for public data)
    client = Socrata(CDC_DOMAIN, None)

    try:
        # Build query - use date_end for filtering (the actual column name in NWSS data)
        where_clause = None
        if start_date:
            # Validate date to prevent SOQL injection
            validated_date = validate_date(start_date)
            where_clause = f"date_end >= '{validated_date}'"
            logger.info(f"Filtering to records from {validated_date} onwards")

        # Fetch data
        logger.info(f"Fetching up to {limit:,} records...")
        results = client.get(
            dataset["id"],
            limit=limit,
            where=where_clause,
            order="date_end DESC"
        )

        df = pd.DataFrame.from_records(results)
        logger.info(f"Fetched {len(df):,} records")
    finally:
        # Always close the client to prevent resource leaks
        client.close()

    # Convert date columns
    for date_col in ["date_start", "date_end", "first_sample_date"]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])

    # Convert numeric columns
    numeric_cols = ["percentile", "ptc_15d", "detect_prop_15d", "population_served"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Save if output directory specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"nwss_{dataset_key}_{timestamp}.parquet"
        output_path = output_dir / filename

        df.to_parquet(output_path, index=False)
        logger.info(f"Saved to {output_path}")

        # Also save a CSV for easy inspection
        csv_path = output_dir / f"nwss_{dataset_key}_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Also saved CSV to {csv_path}")

    return df


def fetch_all_pathogens(
    start_date: str | None = None,
    output_dir: Path | None = None
) -> dict[str, pd.DataFrame]:
    """
    Fetch wastewater data for all available pathogens.

    Note: As of 2026, CDC NWSS tracks:
    - SARS-CoV-2 (COVID-19)
    - Influenza A
    - RSV

    Returns:
        Dictionary of DataFrames keyed by pathogen
    """
    results = {}

    # Fetch metrics data (has all pathogens)
    df = fetch_nwss_data("metrics", start_date=start_date, output_dir=output_dir)

    # Split by pathogen if pcr_target column exists
    if "pcr_target" in df.columns:
        for pathogen in df["pcr_target"].unique():
            results[pathogen] = df[df["pcr_target"] == pathogen].copy()
            logger.info(f"  {pathogen}: {len(results[pathogen]):,} records")
    else:
        results["sars-cov-2"] = df

    return results


def print_data_summary(df: pd.DataFrame) -> None:
    """Print summary statistics for the fetched data."""
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)

    print(f"\nTotal records: {len(df):,}")

    if "date_end" in df.columns:
        print(f"Date range: {df['date_end'].min()} to {df['date_end'].max()}")

    if "wwtp_jurisdiction" in df.columns:
        print(f"States/Territories: {df['wwtp_jurisdiction'].nunique()}")

    if "wwtp_id" in df.columns:
        print(f"Unique sites: {df['wwtp_id'].nunique()}")

    if "pcr_target" in df.columns:
        print(f"\nPathogens tracked:")
        for pathogen, count in df["pcr_target"].value_counts().items():
            print(f"  {pathogen}: {count:,} records")

    if "population_served" in df.columns:
        total_pop = df.groupby("wwtp_id")["population_served"].first().sum()
        print(f"\nTotal population served: {total_pop:,.0f}")

    print("\nColumns available:")
    for col in sorted(df.columns):
        print(f"  - {col}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch CDC NWSS wastewater surveillance data"
    )
    parser.add_argument(
        "--dataset",
        choices=["metrics", "concentration", "all"],
        default="metrics",
        help="Which dataset to fetch"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2023-01-01",
        help="Start date for data (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/nwss",
        help="Output directory"
    )
    parser.add_argument(
        "--limit",
        type=validate_limit,
        default=500000,
        help="Maximum records to fetch (1-10,000,000)"
    )

    args = parser.parse_args()

    # Resolve output path relative to project root with path traversal protection
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    try:
        output_dir = validate_output_path(project_root / args.output, project_root)
    except ValueError as e:
        logger.error(f"Invalid output path: {e}")
        exit(1)

    if args.dataset == "all":
        # Fetch all datasets
        for key in DATASETS:
            df = fetch_nwss_data(
                dataset_key=key,
                start_date=args.start_date,
                limit=args.limit,
                output_dir=output_dir
            )
            print_data_summary(df)
    else:
        df = fetch_nwss_data(
            dataset_key=args.dataset,
            start_date=args.start_date,
            limit=args.limit,
            output_dir=output_dir
        )
        print_data_summary(df)


if __name__ == "__main__":
    main()
