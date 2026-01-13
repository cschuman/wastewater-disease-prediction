#!/usr/bin/env python3
"""
Fetch NHSN hospital respiratory data (COVID-19, Influenza, RSV admissions).

Data Sources:
- Weekly Hospital Respiratory Data: https://healthdata.gov/CDC/Weekly-Hospital-Respiratory-Data-HRD-Metrics-by-Ju/n3kj-exp9
- Historical HHS Protect (pre-May 2024): https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Capa/g62h-syeh

Usage:
    python scripts/fetch_nhsn_data.py
    python scripts/fetch_nhsn_data.py --start-date 2024-01-01 --output data/raw/nhsn
"""

import argparse
import logging
from datetime import datetime
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

# Security constants
ALLOWED_DOMAINS = ['data.cdc.gov', 'healthdata.gov']
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB limit
DEFAULT_TIMEOUT = 120
MAX_TIMEOUT = 600

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def validate_url(url: str) -> str:
    """
    Validate URL to prevent SSRF attacks.

    Args:
        url: URL to validate

    Returns:
        Validated URL

    Raises:
        ValueError: If URL is invalid or from untrusted domain
    """
    parsed = urlparse(url)

    # Ensure HTTPS
    if parsed.scheme != 'https':
        raise ValueError(f"Only HTTPS URLs allowed, got: {parsed.scheme}")

    # Check domain whitelist
    if parsed.netloc not in ALLOWED_DOMAINS:
        raise ValueError(f"Untrusted domain: {parsed.netloc}. Allowed: {ALLOWED_DOMAINS}")

    # Prevent localhost/internal network access
    if parsed.hostname in ['localhost', '127.0.0.1', '0.0.0.0']:
        raise ValueError("Localhost access forbidden")

    return url


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

    try:
        output_path.relative_to(project_root)
    except ValueError:
        raise ValueError(
            f"Output directory {output_path} is outside project root {project_root}"
        )

    return output_path


def fetch_csv_with_limit(url: str, timeout: int = DEFAULT_TIMEOUT, max_size: int = MAX_FILE_SIZE) -> pd.DataFrame:
    """
    Fetch CSV with size limit to prevent DoS.

    Args:
        url: URL to fetch
        timeout: Request timeout in seconds
        max_size: Maximum file size in bytes

    Returns:
        DataFrame

    Raises:
        ValueError: If file exceeds size limit
        requests.exceptions.RequestException: If request fails
    """
    # Validate URL
    validated_url = validate_url(url)

    response = requests.get(validated_url, timeout=timeout, stream=True)
    response.raise_for_status()

    # Check Content-Length header
    content_length = response.headers.get('Content-Length')
    if content_length and int(content_length) > max_size:
        raise ValueError(f"File too large: {content_length} bytes (max: {max_size})")

    # Stream download with size check
    downloaded = BytesIO()
    for chunk in response.iter_content(chunk_size=8192):
        downloaded.write(chunk)
        if downloaded.tell() > max_size:
            raise ValueError(f"File exceeds {max_size} bytes")

    downloaded.seek(0)
    return pd.read_csv(downloaded)


# Data source URLs
DATASETS = {
    "weekly_respiratory": {
        "url": "https://data.cdc.gov/api/views/ua7e-t2fy/rows.csv?accessType=DOWNLOAD",
        "name": "Weekly Hospital Respiratory Data (HRD) Metrics by Jurisdiction",
        "description": "Weekly COVID-19, Influenza, and RSV hospital admissions by state"
    },
    "historical_capacity": {
        "url": "https://healthdata.gov/api/views/g62h-syeh/rows.csv?accessType=DOWNLOAD",
        "name": "COVID-19 Reported Patient Impact and Hospital Capacity",
        "description": "Historical hospital capacity data (through May 2024)"
    }
}


def fetch_nhsn_data(
    dataset_key: str = "weekly_respiratory",
    start_date: str | None = None,
    output_dir: Path | None = None
) -> pd.DataFrame:
    """
    Fetch NHSN hospital data.

    Args:
        dataset_key: Which dataset to fetch
        start_date: Optional start date filter (YYYY-MM-DD)
        output_dir: Directory to save the data

    Returns:
        DataFrame with hospital data
    """
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_key}. Choose from: {list(DATASETS.keys())}")

    dataset = DATASETS[dataset_key]
    logger.info(f"Fetching {dataset['name']}...")
    logger.info(f"URL: {dataset['url']}")

    # Fetch data with security controls (URL validation, size limits)
    try:
        df = fetch_csv_with_limit(dataset["url"])
        logger.info(f"Fetched {len(df):,} records")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch {dataset['name']}: {e}")
        raise
    except ValueError as e:
        logger.error(f"Validation error fetching {dataset['name']}: {e}")
        raise

    # Identify and convert date columns
    date_cols = [col for col in df.columns if "date" in col.lower() or "week" in col.lower()]
    for col in date_cols:
        try:
            df[col] = pd.to_datetime(df[col])
            logger.info(f"Converted {col} to datetime")
        except (ValueError, TypeError) as e:
            logger.debug(f"Could not convert {col} to datetime: {e}")

    # Filter by start date if provided
    if start_date:
        start_dt = pd.to_datetime(start_date)
        # Find the main date column
        date_col = next((col for col in date_cols if df[col].dtype == "datetime64[ns]"), None)
        if date_col:
            original_len = len(df)
            df = df[df[date_col] >= start_dt]
            logger.info(f"Filtered from {original_len:,} to {len(df):,} records (from {start_date})")

    # Save if output directory specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"nhsn_{dataset_key}_{timestamp}.parquet"
        output_path = output_dir / filename

        df.to_parquet(output_path, index=False)
        logger.info(f"Saved to {output_path}")

        # Also save a CSV for easy inspection
        csv_path = output_dir / f"nhsn_{dataset_key}_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Also saved CSV to {csv_path}")

    return df


def print_data_summary(df: pd.DataFrame) -> None:
    """Print summary statistics for the fetched data."""
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)

    print(f"\nTotal records: {len(df):,}")

    # Find date column
    date_cols = [col for col in df.columns if df[col].dtype == "datetime64[ns]"]
    if date_cols:
        date_col = date_cols[0]
        print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")

    # Count jurisdictions
    jurisdiction_cols = [col for col in df.columns if "jurisdiction" in col.lower() or "state" in col.lower()]
    if jurisdiction_cols:
        jur_col = jurisdiction_cols[0]
        print(f"Jurisdictions: {df[jur_col].nunique()}")

    # Identify admission columns
    admission_cols = [col for col in df.columns if "admission" in col.lower() or "confirm" in col.lower()]
    if admission_cols:
        print(f"\nAdmission columns found:")
        for col in admission_cols[:10]:  # Limit to first 10
            print(f"  - {col}")

    print(f"\nAll columns ({len(df.columns)}):")
    for col in sorted(df.columns)[:30]:  # Show first 30
        print(f"  - {col}")
    if len(df.columns) > 30:
        print(f"  ... and {len(df.columns) - 30} more")


def create_combined_target(
    df: pd.DataFrame,
    output_dir: Path | None = None
) -> pd.DataFrame:
    """
    Create the combined respiratory burden target variable.

    This aggregates COVID-19 + Influenza + RSV admissions by state and week.
    """
    logger.info("Creating combined respiratory burden target variable...")

    # This will depend on the actual column names in the data
    # Common patterns in NHSN data:
    covid_cols = [col for col in df.columns if "covid" in col.lower() and "admission" in col.lower()]
    flu_cols = [col for col in df.columns if "flu" in col.lower() and "admission" in col.lower()]
    rsv_cols = [col for col in df.columns if "rsv" in col.lower() and "admission" in col.lower()]

    logger.info(f"COVID columns: {covid_cols}")
    logger.info(f"Flu columns: {flu_cols}")
    logger.info(f"RSV columns: {rsv_cols}")

    # Create combined metric if columns found
    result = df.copy()

    # Convert admission columns to numeric
    for cols in [covid_cols, flu_cols, rsv_cols]:
        for col in cols:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    # Sum admissions (take first column of each type for now)
    if covid_cols and flu_cols and rsv_cols:
        result["total_respiratory_admissions"] = (
            result[covid_cols[0]].fillna(0) +
            result[flu_cols[0]].fillna(0) +
            result[rsv_cols[0]].fillna(0)
        )
        logger.info("Created total_respiratory_admissions column")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"combined_respiratory_target_{timestamp}.parquet"
        output_path = output_dir / filename

        result.to_parquet(output_path, index=False)
        logger.info(f"Saved combined target to {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Fetch NHSN hospital respiratory data"
    )
    parser.add_argument(
        "--dataset",
        choices=["weekly_respiratory", "historical_capacity", "all"],
        default="weekly_respiratory",
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
        default="data/raw/nhsn",
        help="Output directory"
    )
    parser.add_argument(
        "--create-target",
        action="store_true",
        help="Also create combined respiratory burden target variable"
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
        for key in DATASETS:
            try:
                df = fetch_nhsn_data(
                    dataset_key=key,
                    start_date=args.start_date,
                    output_dir=output_dir
                )
                print_data_summary(df)
            except (requests.exceptions.RequestException, ValueError) as e:
                logger.error(f"Failed to fetch {key}: {e}")
    else:
        df = fetch_nhsn_data(
            dataset_key=args.dataset,
            start_date=args.start_date,
            output_dir=output_dir
        )
        print_data_summary(df)

        if args.create_target:
            processed_dir = project_root / "data" / "processed"
            create_combined_target(df, output_dir=processed_dir)


if __name__ == "__main__":
    main()
