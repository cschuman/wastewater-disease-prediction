"""Tests for data fetching utilities."""

import pytest
from pathlib import Path
from datetime import datetime
import re


class TestDateValidation:
    """Tests for date validation in data fetching."""

    def test_valid_date_format(self):
        """Test that valid dates pass validation."""
        from scripts.fetch_nwss_data import validate_date

        valid_dates = ["2024-01-01", "2023-12-31", "2020-06-15"]
        for date in valid_dates:
            result = validate_date(date)
            assert result == date

    def test_invalid_date_format_rejected(self):
        """Test that invalid date formats are rejected."""
        from scripts.fetch_nwss_data import validate_date

        invalid_dates = [
            "01-01-2024",  # Wrong order
            "2024/01/01",  # Wrong separator
            "2024-1-1",    # Missing leading zeros
            "not-a-date",
            "2024-13-01",  # Invalid month
            "2024-01-32",  # Invalid day
        ]
        for date in invalid_dates:
            with pytest.raises(ValueError):
                validate_date(date)

    def test_sql_injection_prevented(self):
        """Test that SQL injection attempts are rejected."""
        from scripts.fetch_nwss_data import validate_date

        injection_attempts = [
            "2024-01-01'; DROP TABLE--",
            "2024-01-01 OR 1=1",
            "'; DELETE FROM data;--",
        ]
        for attempt in injection_attempts:
            with pytest.raises(ValueError):
                validate_date(attempt)


class TestPathValidation:
    """Tests for path validation."""

    def test_valid_path_within_project(self, temp_data_dir):
        """Test that paths within project are accepted."""
        from scripts.fetch_nwss_data import validate_output_path

        output_dir = temp_data_dir / "data" / "raw"
        output_dir.mkdir(parents=True, exist_ok=True)

        result = validate_output_path(output_dir, temp_data_dir)
        assert result == output_dir.resolve()

    def test_path_traversal_rejected(self, temp_data_dir):
        """Test that path traversal attempts are rejected."""
        from scripts.fetch_nwss_data import validate_output_path

        # Attempt to escape project root
        malicious_path = temp_data_dir / ".." / ".." / "etc"

        with pytest.raises(ValueError, match="outside project root"):
            validate_output_path(malicious_path, temp_data_dir)


class TestURLValidation:
    """Tests for URL validation in NHSN fetching."""

    def test_valid_cdc_url_accepted(self):
        """Test that valid CDC URLs are accepted."""
        from scripts.fetch_nhsn_data import validate_url

        valid_urls = [
            "https://data.cdc.gov/api/views/test/rows.csv",
            "https://healthdata.gov/api/views/test/rows.csv",
        ]
        for url in valid_urls:
            result = validate_url(url)
            assert result == url

    def test_invalid_domain_rejected(self):
        """Test that non-allowed domains are rejected."""
        from scripts.fetch_nhsn_data import validate_url

        invalid_urls = [
            "https://malicious.com/data.csv",
            "https://fake-cdc.gov/api/data",
            "http://data.cdc.gov/api/data",  # HTTP not HTTPS
        ]
        for url in invalid_urls:
            with pytest.raises(ValueError):
                validate_url(url)


class TestFileNaming:
    """Tests for file naming conventions."""

    def test_parquet_filename_format(self):
        """Test that parquet files follow naming convention."""
        # Pattern: name_YYYY-MM-DD.parquet
        pattern = r"^[a-z_]+_\d{4}-\d{2}-\d{2}\.parquet$"

        valid_names = [
            "nwss_metrics_2024-01-15.parquet",
            "nhsn_weekly_respiratory_2023-12-01.parquet",
        ]
        for name in valid_names:
            assert re.match(pattern, name), f"{name} doesn't match pattern"

    def test_get_latest_file_returns_most_recent(self, temp_data_dir):
        """Test that get_latest_file returns the most recent file."""
        from src.analysis.health_equity_ratios import get_latest_file

        # Create test files
        nwss_dir = temp_data_dir / "data" / "raw" / "nwss"
        nwss_dir.mkdir(parents=True)

        files = [
            "nwss_metrics_2024-01-01.parquet",
            "nwss_metrics_2024-01-15.parquet",
            "nwss_metrics_2024-02-01.parquet",
        ]
        for f in files:
            (nwss_dir / f).touch()

        result = get_latest_file(nwss_dir, "nwss_metrics_*.parquet")

        assert result.name == "nwss_metrics_2024-02-01.parquet"

    def test_get_latest_file_raises_on_no_match(self, temp_data_dir):
        """Test that get_latest_file raises error when no files match."""
        from src.analysis.health_equity_ratios import get_latest_file

        empty_dir = temp_data_dir / "empty"
        empty_dir.mkdir()

        with pytest.raises(FileNotFoundError):
            get_latest_file(empty_dir, "*.parquet")
