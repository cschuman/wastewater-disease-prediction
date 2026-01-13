"""
Wastewater Surveillance Equity Gap Trajectory Analysis

This analysis examines whether the equity gap in wastewater surveillance
is closing, stable, or widening over time by tracking site expansion
patterns across state SVI quartiles.

Author: Public Health Data Analyst
Date: 2026-01-11
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Set up plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["font.size"] = 10


class EquityGapTrajectory:
    """Analyzes the trajectory of the wastewater surveillance equity gap over time."""

    def __init__(self, wastewater_path: str, svi_path: str):
        """
        Initialize the analysis with data paths.

        Args:
            wastewater_path: Path to NWSS metrics parquet file
            svi_path: Path to SVI CSV file
        """
        self.wastewater_path = wastewater_path
        self.svi_path = svi_path
        self.sites = None
        self.state_svi = None
        self.quarterly_data = None

    def load_data(self):
        """Load and prepare wastewater and SVI data."""
        print("Loading wastewater data...")
        df = pd.read_parquet(self.wastewater_path)

        # Get unique sites with their characteristics
        self.sites = (
            df.groupby("key_plot_id")
            .agg(
                {
                    "reporting_jurisdiction": "first",
                    "first_sample_date": "first",
                    "population_served": "first",
                }
            )
            .reset_index()
        )

        # Convert dates
        self.sites["first_sample_date"] = pd.to_datetime(self.sites["first_sample_date"])
        self.sites["year"] = self.sites["first_sample_date"].dt.year
        self.sites["quarter"] = self.sites["first_sample_date"].dt.to_period("Q")

        print(f"Loaded {len(self.sites):,} unique monitoring sites")

        # Load and aggregate SVI data to state level
        print("\nLoading and aggregating SVI data to state level...")
        svi = pd.read_csv(self.svi_path)

        # Calculate state-level SVI as population-weighted average
        self.state_svi = (
            svi.groupby("ST_ABBR")
            .apply(
                lambda x: pd.Series(
                    {
                        "STATE": x["STATE"].iloc[0],
                        "total_population": x["E_TOTPOP"].sum(),
                        "RPL_THEMES": np.average(x["RPL_THEMES"], weights=x["E_TOTPOP"]),
                    }
                )
            )
            .reset_index()
        )

        print(f"Calculated SVI for {len(self.state_svi)} states")

    def create_state_mapping(self):
        """Create mapping between state names and abbreviations for wastewater data."""
        # Common state name to abbreviation mapping
        state_mapping = {
            "Alabama": "AL",
            "Alaska": "AK",
            "Arizona": "AZ",
            "Arkansas": "AR",
            "California": "CA",
            "Colorado": "CO",
            "Connecticut": "CT",
            "Delaware": "DE",
            "Florida": "FL",
            "Georgia": "GA",
            "Hawaii": "HI",
            "Idaho": "ID",
            "Illinois": "IL",
            "Indiana": "IN",
            "Iowa": "IA",
            "Kansas": "KS",
            "Kentucky": "KY",
            "Louisiana": "LA",
            "Maine": "ME",
            "Maryland": "MD",
            "Massachusetts": "MA",
            "Michigan": "MI",
            "Minnesota": "MN",
            "Mississippi": "MS",
            "Missouri": "MO",
            "Montana": "MT",
            "Nebraska": "NE",
            "Nevada": "NV",
            "New Hampshire": "NH",
            "New Jersey": "NJ",
            "New Mexico": "NM",
            "New York": "NY",
            "North Carolina": "NC",
            "North Dakota": "ND",
            "Ohio": "OH",
            "Oklahoma": "OK",
            "Oregon": "OR",
            "Pennsylvania": "PA",
            "Rhode Island": "RI",
            "South Carolina": "SC",
            "South Dakota": "SD",
            "Tennessee": "TN",
            "Texas": "TX",
            "Utah": "UT",
            "Vermont": "VT",
            "Virginia": "VA",
            "Washington": "WA",
            "West Virginia": "WV",
            "Wisconsin": "WI",
            "Wyoming": "WY",
            "District of Columbia": "DC",
            "Puerto Rico": "PR",
            "Virgin Islands": "VI",
            "Guam": "GU",
        }

        # Map state names to abbreviations
        self.sites["state_abbr"] = self.sites["reporting_jurisdiction"].map(state_mapping)

        # Check for unmapped states
        unmapped = self.sites[self.sites["state_abbr"].isna()]["reporting_jurisdiction"].unique()
        if len(unmapped) > 0:
            print(f"\nWarning: {len(unmapped)} unmapped jurisdictions: {unmapped}")

    def assign_svi_quartiles(self):
        """Assign SVI quartiles to states."""
        # Calculate quartiles
        self.state_svi["svi_quartile"] = pd.qcut(
            self.state_svi["RPL_THEMES"],
            q=4,
            labels=["Q1 (Lowest SVI)", "Q2 (Low-Mid SVI)", "Q3 (Mid-High SVI)", "Q4 (Highest SVI)"],
        )

        print("\n=== SVI QUARTILE DISTRIBUTION ===")
        print(
            self.state_svi.groupby("svi_quartile").agg(
                {"STATE": "count", "total_population": "sum", "RPL_THEMES": ["min", "max", "mean"]}
            )
        )

    def merge_svi_to_sites(self):
        """Merge SVI quartiles to site data."""
        self.sites = self.sites.merge(
            self.state_svi[["ST_ABBR", "svi_quartile", "RPL_THEMES"]],
            left_on="state_abbr",
            right_on="ST_ABBR",
            how="left",
        )

        # Check merge success
        print(
            f"\nSites with SVI data: {self.sites['svi_quartile'].notna().sum():,} / {len(self.sites):,}"
        )

    def calculate_quarterly_expansion(self):
        """Calculate site expansion metrics by quarter and SVI quartile."""
        # Filter to sites with SVI data
        sites_with_svi = self.sites[self.sites["svi_quartile"].notna()].copy()

        # Group by quarter and SVI quartile
        quarterly = (
            sites_with_svi.groupby(["quarter", "svi_quartile"]).size().reset_index(name="new_sites")
        )

        # Calculate cumulative sites
        quarterly = quarterly.sort_values("quarter")
        quarterly["cumulative_sites"] = quarterly.groupby("svi_quartile")["new_sites"].cumsum()

        # Calculate quarter-over-quarter growth rate
        quarterly["pct_growth"] = (
            quarterly.groupby("svi_quartile")["cumulative_sites"].pct_change() * 100
        )

        # Convert quarter to datetime for plotting
        quarterly["date"] = quarterly["quarter"].dt.to_timestamp()

        self.quarterly_data = quarterly

        return quarterly

    def calculate_expansion_stats(self):
        """Calculate key expansion statistics by SVI quartile."""
        sites_with_svi = self.sites[self.sites["svi_quartile"].notna()].copy()

        # Overall statistics by quartile
        stats = (
            sites_with_svi.groupby("svi_quartile")
            .agg({"key_plot_id": "count", "population_served": "sum"})
            .rename(columns={"key_plot_id": "total_sites"})
        )

        # Split by time period
        sites_with_svi["period"] = sites_with_svi["year"].apply(
            lambda x: "Pre-2024" if x < 2024 else "2024+"
        )

        period_stats = sites_with_svi.pivot_table(
            index="svi_quartile",
            columns="period",
            values="key_plot_id",
            aggfunc="count",
            fill_value=0,
        )

        stats = stats.join(period_stats)
        stats["pct_new_2024"] = (stats["2024+"] / stats["total_sites"] * 100).round(1)

        # Calculate per capita rates (sites per 100k population)
        state_pop = self.state_svi.groupby(
            self.state_svi["ST_ABBR"].map(
                dict(
                    zip(
                        self.state_svi["ST_ABBR"],
                        pd.qcut(
                            self.state_svi["RPL_THEMES"],
                            q=4,
                            labels=[
                                "Q1 (Lowest SVI)",
                                "Q2 (Low-Mid SVI)",
                                "Q3 (Mid-High SVI)",
                                "Q4 (Highest SVI)",
                            ],
                        ),
                    )
                )
            )
        )["total_population"].sum()

        stats["sites_per_100k"] = (stats["total_sites"] / state_pop * 100000).round(2)

        return stats

    def calculate_gap_metrics(self):
        """Calculate gap metrics over time."""
        quarterly = self.quarterly_data.copy()

        # Pivot to wide format for easier comparison
        gap_data = quarterly.pivot(
            index="date",
            columns="svi_quartile",
            values=["new_sites", "cumulative_sites", "pct_growth"],
        ).fillna(0)

        # Calculate gap ratios (Q1 vs Q4)
        gap_metrics = pd.DataFrame(
            {
                "date": gap_data.index,
                "cumulative_gap_ratio": (
                    gap_data[("cumulative_sites", "Q1 (Lowest SVI)")]
                    / gap_data[("cumulative_sites", "Q4 (Highest SVI)")].replace(0, np.nan)
                ),
                "new_sites_gap_ratio": (
                    gap_data[("new_sites", "Q1 (Lowest SVI)")]
                    / gap_data[("new_sites", "Q4 (Highest SVI)")].replace(0, np.nan)
                ),
            }
        )

        return gap_metrics

    def plot_cumulative_expansion(self, save_path: str = None):
        """Plot cumulative site expansion over time by SVI quartile."""
        fig, ax = plt.subplots(figsize=(14, 8))

        # Define colors for quartiles
        colors = {
            "Q1 (Lowest SVI)": "#2ecc71",  # Green
            "Q2 (Low-Mid SVI)": "#3498db",  # Blue
            "Q3 (Mid-High SVI)": "#f39c12",  # Orange
            "Q4 (Highest SVI)": "#e74c3c",  # Red
        }

        for quartile in self.quarterly_data["svi_quartile"].unique():
            data = self.quarterly_data[self.quarterly_data["svi_quartile"] == quartile]
            ax.plot(
                data["date"],
                data["cumulative_sites"],
                marker="o",
                label=quartile,
                color=colors.get(quartile),
                linewidth=2.5,
                markersize=4,
            )

        ax.set_xlabel("Date", fontsize=12, fontweight="bold")
        ax.set_ylabel("Cumulative Monitoring Sites", fontsize=12, fontweight="bold")
        ax.set_title(
            "Wastewater Surveillance Expansion by State SVI Quartile\n"
            + "Cumulative Sites Over Time",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        ax.legend(title="State SVI Quartile", fontsize=10, title_fontsize=11)
        ax.grid(True, alpha=0.3)

        # Add vertical line for 2024
        ax.axvline(
            pd.Timestamp("2024-01-01"), color="gray", linestyle="--", alpha=0.5, label="2024 Start"
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved plot to {save_path}")

        return fig

    def plot_quarterly_new_sites(self, save_path: str = None):
        """Plot new sites per quarter by SVI quartile."""
        fig, ax = plt.subplots(figsize=(14, 8))

        # Pivot data for stacked bar chart
        pivot_data = self.quarterly_data.pivot(
            index="date", columns="svi_quartile", values="new_sites"
        ).fillna(0)

        # Define colors
        colors = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c"]

        pivot_data.plot(kind="bar", stacked=False, ax=ax, color=colors, width=0.8)

        ax.set_xlabel("Quarter", fontsize=12, fontweight="bold")
        ax.set_ylabel("New Monitoring Sites", fontsize=12, fontweight="bold")
        ax.set_title(
            "Quarterly Expansion of Wastewater Surveillance by State SVI Quartile\n"
            + "New Sites per Quarter",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        ax.legend(title="State SVI Quartile", fontsize=9, title_fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

        # Rotate x-axis labels
        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved plot to {save_path}")

        return fig

    def plot_gap_trajectory(self, save_path: str = None):
        """Plot the gap ratio trajectory over time."""
        gap_metrics = self.calculate_gap_metrics()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Plot 1: Cumulative gap ratio
        ax1.plot(
            gap_metrics["date"],
            gap_metrics["cumulative_gap_ratio"],
            marker="o",
            color="#e74c3c",
            linewidth=2.5,
            markersize=6,
        )
        ax1.axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="Equity (Ratio = 1)")
        ax1.set_ylabel("Gap Ratio\n(Q1 Sites / Q4 Sites)", fontsize=11, fontweight="bold")
        ax1.set_title(
            "Equity Gap Trajectory: Cumulative Sites Ratio (Q1 vs Q4)\n"
            + "Higher = Wider Gap Favoring Low-SVI States",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)

        # Plot 2: New sites gap ratio (smoothed)
        # Use rolling average to smooth quarterly volatility
        gap_metrics["new_sites_gap_smooth"] = (
            gap_metrics["new_sites_gap_ratio"].rolling(window=4, min_periods=1).mean()
        )

        ax2.plot(
            gap_metrics["date"],
            gap_metrics["new_sites_gap_smooth"],
            marker="o",
            color="#3498db",
            linewidth=2.5,
            markersize=6,
        )
        ax2.axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="Equity (Ratio = 1)")
        ax2.set_xlabel("Date", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Gap Ratio\n(Q1 New Sites / Q4 New Sites)", fontsize=11, fontweight="bold")
        ax2.set_title(
            "Equity Gap Trajectory: New Sites Ratio (4-Quarter Moving Average)\n"
            + "Higher = More New Sites in Low-SVI States",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved plot to {save_path}")

        return fig

    def plot_expansion_rates(self, save_path: str = None):
        """Plot expansion rates by SVI quartile over time."""
        fig, ax = plt.subplots(figsize=(14, 8))

        # Calculate expansion rate for pre-2024 and 2024+ periods
        sites_with_svi = self.sites[self.sites["svi_quartile"].notna()].copy()

        # Create period bins
        sites_with_svi["period"] = pd.cut(
            sites_with_svi["year"],
            bins=[2019, 2022, 2024, 2026],
            labels=["2020-2022", "2023", "2024-2025"],
        )

        expansion_rates = (
            sites_with_svi.groupby(["period", "svi_quartile"]).size().reset_index(name="sites")
        )

        # Pivot for grouped bar chart
        pivot_rates = expansion_rates.pivot(
            index="svi_quartile", columns="period", values="sites"
        ).fillna(0)

        # Define colors for periods
        colors = ["#95a5a6", "#3498db", "#2ecc71"]

        pivot_rates.plot(kind="bar", ax=ax, color=colors, width=0.7)

        ax.set_xlabel("State SVI Quartile", fontsize=12, fontweight="bold")
        ax.set_ylabel("Number of Sites Added", fontsize=12, fontweight="bold")
        ax.set_title(
            "Wastewater Surveillance Expansion by Time Period and SVI Quartile\n"
            + "Are High-SVI States Catching Up?",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        ax.legend(title="Time Period", fontsize=10, title_fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")
        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved plot to {save_path}")

        return fig

    def generate_summary_report(self):
        """Generate a comprehensive summary report of findings."""
        stats = self.calculate_expansion_stats()
        gap_metrics = self.calculate_gap_metrics()

        report = []
        report.append("=" * 80)
        report.append("WASTEWATER SURVEILLANCE EQUITY GAP TRAJECTORY ANALYSIS")
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d')}")
        report.append("=" * 80)

        report.append("\n### OVERALL STATISTICS ###\n")
        report.append(f"Total monitoring sites analyzed: {len(self.sites):,}")
        report.append(f"Sites with SVI data: {self.sites['svi_quartile'].notna().sum():,}")
        report.append(
            f"Time period: {self.sites['first_sample_date'].min().strftime('%Y-%m-%d')} to "
            + f"{self.sites['first_sample_date'].max().strftime('%Y-%m-%d')}"
        )

        report.append("\n### EXPANSION BY SVI QUARTILE ###\n")
        report.append(stats.to_string())

        report.append("\n\n### KEY FINDINGS ###\n")

        # Finding 1: Current gap
        q1_sites = stats.loc["Q1 (Lowest SVI)", "total_sites"]
        q4_sites = stats.loc["Q4 (Highest SVI)", "total_sites"]
        current_gap = q1_sites / q4_sites

        report.append(f"1. CURRENT EQUITY GAP")
        report.append(f"   - Q1 (Lowest SVI) states: {q1_sites} sites")
        report.append(f"   - Q4 (Highest SVI) states: {q4_sites} sites")
        report.append(
            f"   - Gap Ratio: {current_gap:.2f}x (Q1 has {current_gap:.2f}x more sites than Q4)"
        )

        # Finding 2: 2024+ expansion
        q1_new = stats.loc["Q1 (Lowest SVI)", "2024+"]
        q4_new = stats.loc["Q4 (Highest SVI)", "2024+"]
        new_gap = q1_new / q4_new if q4_new > 0 else np.inf

        report.append(f"\n2. 2024+ EXPANSION PATTERN")
        report.append(
            f"   - Q1 states added: {q1_new} sites ({stats.loc['Q1 (Lowest SVI)', 'pct_new_2024']:.1f}% of their total)"
        )
        report.append(
            f"   - Q4 states added: {q4_new} sites ({stats.loc['Q4 (Highest SVI)', 'pct_new_2024']:.1f}% of their total)"
        )
        report.append(f"   - 2024+ Gap Ratio: {new_gap:.2f}x")

        # Finding 3: Gap trajectory
        gap_start = gap_metrics["cumulative_gap_ratio"].iloc[0]
        gap_end = gap_metrics["cumulative_gap_ratio"].iloc[-1]
        gap_change = ((gap_end - gap_start) / gap_start * 100) if not np.isnan(gap_start) else 0

        report.append(f"\n3. GAP TRAJECTORY")
        report.append(f"   - Initial gap ratio (early period): {gap_start:.2f}x")
        report.append(f"   - Current gap ratio: {gap_end:.2f}x")
        report.append(f"   - Gap change: {gap_change:+.1f}%")

        if gap_change > 5:
            trajectory = "WIDENING"
            interpretation = "The gap is getting worse - low-SVI states are expanding faster."
        elif gap_change < -5:
            trajectory = "CLOSING"
            interpretation = "The gap is improving - high-SVI states are catching up."
        else:
            trajectory = "STABLE"
            interpretation = "The gap is relatively stable - expansion is proportional."

        report.append(f"   - Trajectory: {trajectory}")
        report.append(f"   - Interpretation: {interpretation}")

        # Finding 4: Per capita disparity
        q1_per_capita = stats.loc["Q1 (Lowest SVI)", "sites_per_100k"]
        q4_per_capita = stats.loc["Q4 (Highest SVI)", "sites_per_100k"]

        report.append(f"\n4. PER CAPITA DISPARITY")
        report.append(f"   - Q1 states: {q1_per_capita:.2f} sites per 100k population")
        report.append(f"   - Q4 states: {q4_per_capita:.2f} sites per 100k population")
        report.append(f"   - Disparity ratio: {(q1_per_capita/q4_per_capita):.2f}x")

        report.append("\n### ANSWER TO KEY QUESTION ###\n")
        report.append("IS THE EQUITY GAP CLOSING OR WIDENING?")
        report.append("-" * 80)

        if trajectory == "WIDENING":
            answer = (
                "The equity gap is WIDENING. Despite significant expansion in 2024-2025,\n"
                "low-SVI (wealthier) states are adding monitoring sites at a faster rate than\n"
                "high-SVI (vulnerable) states. This exacerbates existing disparities in public\n"
                "health surveillance infrastructure."
            )
        elif trajectory == "CLOSING":
            answer = (
                "The equity gap is CLOSING. High-SVI states are adding monitoring sites at\n"
                "a faster rate than low-SVI states, indicating progress toward equitable\n"
                "surveillance coverage. However, significant disparities remain."
            )
        else:
            answer = (
                "The equity gap is STABLE. While substantial expansion occurred in 2024-2025,\n"
                "the growth has been proportional across SVI quartiles, maintaining existing\n"
                "disparities rather than closing them."
            )

        report.append(answer)

        report.append("\n" + "=" * 80)

        return "\n".join(report)

    def run_full_analysis(self, output_dir: str = None):
        """
        Run the complete equity gap trajectory analysis.

        Args:
            output_dir: Directory to save outputs (plots and report)
        """
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent / "reports"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 80)
        print("STARTING EQUITY GAP TRAJECTORY ANALYSIS")
        print("=" * 80 + "\n")

        # Step 1: Load data
        self.load_data()

        # Step 2: Create state mapping
        print("\nMapping state names to abbreviations...")
        self.create_state_mapping()

        # Step 3: Assign SVI quartiles
        print("\nAssigning SVI quartiles to states...")
        self.assign_svi_quartiles()

        # Step 4: Merge SVI to sites
        print("\nMerging SVI data to monitoring sites...")
        self.merge_svi_to_sites()

        # Step 5: Calculate quarterly expansion
        print("\nCalculating quarterly expansion patterns...")
        self.calculate_quarterly_expansion()

        # Step 6: Generate visualizations
        print("\nGenerating visualizations...")

        self.plot_cumulative_expansion(save_path=output_dir / "gap_trajectory_cumulative.png")

        self.plot_quarterly_new_sites(save_path=output_dir / "gap_trajectory_quarterly_new.png")

        self.plot_gap_trajectory(save_path=output_dir / "gap_trajectory_ratio.png")

        self.plot_expansion_rates(save_path=output_dir / "gap_trajectory_expansion_rates.png")

        # Step 7: Generate and save report
        print("\nGenerating summary report...")
        report = self.generate_summary_report()

        report_path = output_dir / "gap_trajectory_report.txt"
        with open(report_path, "w") as f:
            f.write(report)

        print(f"\nReport saved to: {report_path}")

        # Print report to console
        print("\n" + report)

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80 + "\n")

        return report


def main():
    """Main execution function."""
    # Define paths
    base_dir = Path(__file__).parent.parent.parent
    wastewater_path = base_dir / "data" / "raw" / "nwss" / "nwss_metrics_20260111.parquet"
    svi_path = base_dir / "data" / "external" / "svi_2022_county.csv"
    output_dir = base_dir / "reports"

    # Run analysis
    analyzer = EquityGapTrajectory(wastewater_path=str(wastewater_path), svi_path=str(svi_path))

    analyzer.run_full_analysis(output_dir=str(output_dir))


if __name__ == "__main__":
    main()
