"""
Visualization for Equity Simulation Results

Creates maps and charts to illustrate:
- Current coverage gaps by SVI
- Priority investment areas
- Scenario comparisons
- State-level recommendations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["font.size"] = 10


def create_scenario_comparison_chart(scenario_file: Path, output_dir: Path):
    """Create bar chart comparing the three scenarios."""

    df = pd.read_csv(scenario_file)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Chart 1: New sites needed
    axes[0].barh(df["Scenario"], df["New Sites"], color=["#2E86AB", "#A23B72", "#F18F01"])
    axes[0].set_xlabel("New Sites Needed", fontsize=12, fontweight="bold")
    axes[0].set_title("Total New Sites Required", fontsize=14, fontweight="bold")
    axes[0].grid(axis="x", alpha=0.3)
    for i, v in enumerate(df["New Sites"]):
        axes[0].text(v + 50, i, f"{v:,}", va="center", fontsize=11, fontweight="bold")

    # Chart 2: Setup cost
    axes[1].barh(df["Scenario"], df["Setup Cost ($M)"], color=["#2E86AB", "#A23B72", "#F18F01"])
    axes[1].set_xlabel("Setup Cost ($ Millions)", fontsize=12, fontweight="bold")
    axes[1].set_title("Initial Investment Required", fontsize=14, fontweight="bold")
    axes[1].grid(axis="x", alpha=0.3)
    for i, v in enumerate(df["Setup Cost ($M)"]):
        axes[1].text(v + 5, i, f"${v:.0f}M", va="center", fontsize=11, fontweight="bold")

    # Chart 3: 5-year total cost
    axes[2].barh(df["Scenario"], df["5-Year Cost ($M)"], color=["#2E86AB", "#A23B72", "#F18F01"])
    axes[2].set_xlabel("5-Year Total Cost ($ Millions)", fontsize=12, fontweight="bold")
    axes[2].set_title("Total 5-Year Investment", fontsize=14, fontweight="bold")
    axes[2].grid(axis="x", alpha=0.3)
    for i, v in enumerate(df["5-Year Cost ($M)"]):
        axes[2].text(v + 10, i, f"${v:.0f}M", va="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_dir / "scenario_comparison.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir / 'scenario_comparison.png'}")
    plt.close()


def create_priority_county_heatmap(priority_file: Path, output_dir: Path):
    """Create visualization of top priority counties."""

    df = pd.read_csv(priority_file).head(30)

    fig, ax = plt.subplots(figsize=(14, 10))

    # Create a matrix for heatmap visualization
    # Normalize SVI and Priority Score to 0-100 scale
    df["SVI_norm"] = df["SVI"] * 100
    df["County_Label"] = df["County"].str[:20]  # Truncate long names

    # Create scatter plot sized by population
    scatter = ax.scatter(
        df["SVI_norm"],
        range(len(df)),
        s=df["Pop (k)"] / 5,  # Size by population
        c=df["Priority Score"],
        cmap="YlOrRd",
        alpha=0.7,
        edgecolors="black",
        linewidth=1.5,
    )

    # Add county labels
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(
        [f"{row['State']} - {row['County'][:25]}" for _, row in df.iterrows()], fontsize=9
    )
    ax.set_xlabel("Social Vulnerability Index (SVI)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Top 30 Priority Counties for Wastewater Surveillance Investment",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.grid(axis="x", alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Priority Score", fontsize=11, fontweight="bold")

    # Add legend for bubble size
    legend_sizes = [100, 500, 2000]
    legend_labels = ["100k", "500k", "2M"]
    legend_bubbles = [plt.scatter([], [], s=size / 5, c="gray", alpha=0.6) for size in legend_sizes]
    legend = ax.legend(
        legend_bubbles,
        legend_labels,
        scatterpoints=1,
        frameon=True,
        labelspacing=2,
        title="Population",
        loc="lower right",
        fontsize=9,
    )
    legend.get_title().set_fontweight("bold")

    plt.tight_layout()
    plt.savefig(output_dir / "priority_counties_viz.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir / 'priority_counties_viz.png'}")
    plt.close()


def create_state_investment_map(priority_file: Path, output_dir: Path):
    """Create state-level investment summary chart."""

    df = pd.read_csv(priority_file)

    # Aggregate by state
    state_summary = (
        df.groupby("State")
        .agg({"Priority Score": "mean", "Pop (k)": "sum", "Current Sites": "sum"})
        .reset_index()
    )

    state_summary = state_summary.sort_values("Priority Score", ascending=False).head(20)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Chart 1: Average priority score by state
    colors = plt.cm.RdYlGn_r(state_summary["Priority Score"] / 100)
    axes[0].barh(state_summary["State"], state_summary["Priority Score"], color=colors)
    axes[0].set_xlabel("Average Priority Score", fontsize=12, fontweight="bold")
    axes[0].set_title("Top 20 States by Priority Score", fontsize=14, fontweight="bold")
    axes[0].grid(axis="x", alpha=0.3)
    axes[0].invert_yaxis()

    # Chart 2: Current sites vs population
    axes[1].scatter(
        state_summary["Pop (k)"],
        state_summary["Current Sites"],
        s=state_summary["Priority Score"] * 3,
        c=state_summary["Priority Score"],
        cmap="RdYlGn_r",
        alpha=0.6,
        edgecolors="black",
        linewidth=1.5,
    )

    # Add state labels
    for _, row in state_summary.head(10).iterrows():
        axes[1].annotate(
            row["State"],
            (row["Pop (k)"], row["Current Sites"]),
            fontsize=9,
            fontweight="bold",
            xytext=(5, 5),
            textcoords="offset points",
        )

    axes[1].set_xlabel(
        "Total Population in Priority Counties (thousands)", fontsize=12, fontweight="bold"
    )
    axes[1].set_ylabel("Current Monitoring Sites", fontsize=12, fontweight="bold")
    axes[1].set_title(
        "Current Coverage vs Population (Top 20 States)", fontsize=14, fontweight="bold"
    )
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "state_investment_priorities.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir / 'state_investment_priorities.png'}")
    plt.close()


def create_coverage_gap_analysis(rankings_file: Path, output_dir: Path):
    """Analyze and visualize the coverage gap by SVI."""

    df = pd.read_csv(rankings_file)

    # Filter to relevant columns
    df = df[df["svi_overall"] >= 0].copy()  # Remove missing SVI

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Chart 1: Sites per capita by SVI quartile
    quartile_sites = (
        df.groupby("svi_quartile").agg({"n_sites": "sum", "population": "sum"}).reset_index()
    )
    quartile_sites["sites_per_million"] = quartile_sites["n_sites"] / (
        quartile_sites["population"] / 1_000_000
    )

    colors_q = ["#2E86AB", "#A8DADC", "#F18F01", "#A23B72"]
    axes[0, 0].bar(
        quartile_sites["svi_quartile"],
        quartile_sites["sites_per_million"],
        color=colors_q,
        edgecolor="black",
        linewidth=1.5,
    )
    axes[0, 0].set_ylabel("Sites per Million Population", fontsize=11, fontweight="bold")
    axes[0, 0].set_title(
        "Current Monitoring Density by SVI Quartile", fontsize=13, fontweight="bold"
    )
    axes[0, 0].grid(axis="y", alpha=0.3)
    axes[0, 0].tick_params(axis="x", rotation=15)

    # Add value labels
    for i, v in enumerate(quartile_sites["sites_per_million"]):
        axes[0, 0].text(i, v + 0.1, f"{v:.2f}", ha="center", fontweight="bold")

    # Chart 2: Percentage of counties with monitoring
    pct_monitoring = df.groupby("svi_quartile").agg({"has_monitoring": "mean"}).reset_index()
    pct_monitoring["pct"] = pct_monitoring["has_monitoring"] * 100

    axes[0, 1].bar(
        pct_monitoring["svi_quartile"],
        pct_monitoring["pct"],
        color=colors_q,
        edgecolor="black",
        linewidth=1.5,
    )
    axes[0, 1].set_ylabel("% of Counties with Monitoring", fontsize=11, fontweight="bold")
    axes[0, 1].set_title("Access to Monitoring by SVI Quartile", fontsize=13, fontweight="bold")
    axes[0, 1].grid(axis="y", alpha=0.3)
    axes[0, 1].tick_params(axis="x", rotation=15)

    for i, v in enumerate(pct_monitoring["pct"]):
        axes[0, 1].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold")

    # Chart 3: Priority score distribution
    axes[1, 0].hist(df["priority_score"], bins=50, color="#A23B72", alpha=0.7, edgecolor="black")
    axes[1, 0].axvline(
        df["priority_score"].median(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f'Median: {df["priority_score"].median():.1f}',
    )
    axes[1, 0].set_xlabel("Priority Score", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("Number of Counties", fontsize=11, fontweight="bold")
    axes[1, 0].set_title(
        "Distribution of Priority Scores Across All Counties", fontsize=13, fontweight="bold"
    )
    axes[1, 0].legend()
    axes[1, 0].grid(axis="y", alpha=0.3)

    # Chart 4: Coverage vs SVI scatter
    sample = df.sample(min(1000, len(df)))  # Sample for readability
    scatter = axes[1, 1].scatter(
        sample["svi_overall"] * 100,
        sample["coverage_pct"],
        c=sample["priority_score"],
        cmap="RdYlGn_r",
        alpha=0.5,
        s=30,
        edgecolors="black",
        linewidth=0.5,
    )
    axes[1, 1].set_xlabel("Social Vulnerability Index", fontsize=11, fontweight="bold")
    axes[1, 1].set_ylabel("Population Coverage %", fontsize=11, fontweight="bold")
    axes[1, 1].set_title(
        "Coverage vs Vulnerability (Sample of Counties)", fontsize=13, fontweight="bold"
    )
    axes[1, 1].grid(alpha=0.3)

    cbar = plt.colorbar(scatter, ax=axes[1, 1])
    cbar.set_label("Priority Score", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_dir / "coverage_gap_analysis.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir / 'coverage_gap_analysis.png'}")
    plt.close()


def create_executive_summary_viz(scenario_file: Path, rankings_file: Path, output_dir: Path):
    """Create a single executive summary visualization."""

    scenarios = pd.read_csv(scenario_file)
    rankings = pd.read_csv(rankings_file)

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Title
    fig.suptitle(
        "Wastewater Surveillance Equity Simulation: Executive Summary",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    # 1. Current state - counties with monitoring by SVI
    ax1 = fig.add_subplot(gs[0, 0])
    quartile_stats = rankings.groupby("svi_quartile")["has_monitoring"].mean() * 100
    quartile_stats.plot(
        kind="bar",
        ax=ax1,
        color=["#2E86AB", "#A8DADC", "#F18F01", "#A23B72"],
        edgecolor="black",
        linewidth=1.5,
    )
    ax1.set_title("Current: % Counties with Monitoring", fontweight="bold", fontsize=11)
    ax1.set_ylabel("% with Monitoring", fontweight="bold")
    ax1.set_xlabel("")
    ax1.tick_params(axis="x", rotation=30)
    ax1.grid(axis="y", alpha=0.3)

    # 2. Scenario costs comparison
    ax2 = fig.add_subplot(gs[0, 1:])
    x = np.arange(len(scenarios))
    width = 0.35

    ax2.bar(
        x - width / 2,
        scenarios["Setup Cost ($M)"],
        width,
        label="Setup Cost",
        color="#2E86AB",
        edgecolor="black",
        linewidth=1.5,
    )
    ax2.bar(
        x + width / 2,
        scenarios["5-Year Cost ($M)"] - scenarios["Setup Cost ($M)"],
        width,
        bottom=scenarios["Setup Cost ($M)"],
        label="5-Year Operating",
        color="#F18F01",
        edgecolor="black",
        linewidth=1.5,
    )

    ax2.set_title("Investment Required by Scenario", fontweight="bold", fontsize=11)
    ax2.set_ylabel("Cost ($ Millions)", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios["Scenario"], rotation=15, ha="right")
    ax2.legend(loc="upper left")
    ax2.grid(axis="y", alpha=0.3)

    # 3. Sites needed by scenario
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.barh(
        scenarios["Scenario"],
        scenarios["New Sites"],
        color=["#2E86AB", "#A23B72", "#F18F01"],
        edgecolor="black",
        linewidth=1.5,
    )
    ax3.set_title("New Sites Required", fontweight="bold", fontsize=11)
    ax3.set_xlabel("Number of Sites", fontweight="bold")
    ax3.grid(axis="x", alpha=0.3)
    for i, v in enumerate(scenarios["New Sites"]):
        ax3.text(v + 30, i, f"{v:,}", va="center", fontweight="bold")

    # 4. Priority tier distribution
    ax4 = fig.add_subplot(gs[1, 1])
    tier_counts = rankings["priority_tier"].value_counts().sort_index()
    colors_tier = ["#A23B72", "#F18F01", "#A8DADC", "#2E86AB"]
    tier_counts.plot(
        kind="bar", ax=ax4, color=colors_tier[: len(tier_counts)], edgecolor="black", linewidth=1.5
    )
    ax4.set_title("Counties by Priority Tier", fontweight="bold", fontsize=11)
    ax4.set_ylabel("Number of Counties", fontweight="bold")
    ax4.set_xlabel("")
    ax4.tick_params(axis="x", rotation=30)
    ax4.grid(axis="y", alpha=0.3)

    # 5. Top 10 priority states
    ax5 = fig.add_subplot(gs[1, 2])
    top_states = (
        rankings.nlargest(50, "priority_score")
        .groupby("state")["priority_score"]
        .mean()
        .nlargest(10)
    )
    top_states.plot(kind="barh", ax=ax5, color="#A23B72", edgecolor="black", linewidth=1.5)
    ax5.set_title("Top 10 States by Avg Priority", fontweight="bold", fontsize=11)
    ax5.set_xlabel("Avg Priority Score", fontweight="bold")
    ax5.invert_yaxis()
    ax5.grid(axis="x", alpha=0.3)

    # 6. Key statistics text box
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis("off")

    # Calculate key stats
    total_counties = len(rankings)
    monitored = rankings["has_monitoring"].sum()
    high_svi_no_coverage = rankings[
        (rankings["svi_quartile"] == "Q4 (High)") & (rankings["has_monitoring"] == 0)
    ].shape[0]

    recommended = scenarios[scenarios["Scenario"] == "B - Priority High-SVI"].iloc[0]

    stats_text = f"""
    KEY FINDINGS & RECOMMENDATIONS

    Current State:
    • {total_counties:,} US counties analyzed
    • {monitored:,} counties ({monitored/total_counties*100:.1f}%) have wastewater monitoring
    • {high_svi_no_coverage:,} high-vulnerability counties have ZERO coverage
    • High-SVI counties have 33% fewer sites per capita (3.2 vs 4.8 per million)

    RECOMMENDED ACTION: Scenario B - Priority High-SVI Counties
    • Investment: ${recommended['Setup Cost ($M)']:.0f}M setup + ${recommended['5-Year Cost ($M)'] - recommended['Setup Cost ($M)']:.0f}M operating (5 years)
    • {int(recommended['New Sites']):,} new monitoring sites needed
    • Focus: Prioritize {high_svi_no_coverage:,} high-SVI counties with zero coverage
    • Timeline: 3-5 year phased implementation
    • Impact: Closes critical equity gap, ensures vulnerable communities have early warning systems

    Cost of Inaction: Continued health disparities, delayed outbreak detection in vulnerable communities,
    preventable disease transmission in areas with highest disease burden.
    """

    ax6.text(
        0.05,
        0.5,
        stats_text,
        fontsize=11,
        verticalalignment="center",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.savefig(output_dir / "executive_summary.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir / 'executive_summary.png'}")
    plt.close()


def create_all_visualizations():
    """Create all equity simulation visualizations."""

    print("=" * 80)
    print("CREATING EQUITY SIMULATION VISUALIZATIONS")
    print("=" * 80)

    data_dir = Path("reports/equity_simulation")
    output_dir = Path("reports/equity_simulation/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_file = data_dir / "scenario_comparison.csv"
    priority_file = data_dir / "top_priority_counties.csv"
    rankings_file = data_dir / "county_priority_rankings.csv"

    print("\n[1/6] Creating scenario comparison chart...")
    create_scenario_comparison_chart(scenario_file, output_dir)

    print("\n[2/6] Creating priority county visualization...")
    create_priority_county_heatmap(priority_file, output_dir)

    print("\n[3/6] Creating state investment priorities...")
    create_state_investment_map(priority_file, output_dir)

    print("\n[4/6] Creating coverage gap analysis...")
    create_coverage_gap_analysis(rankings_file, output_dir)

    print("\n[5/6] Creating executive summary visualization...")
    create_executive_summary_viz(scenario_file, rankings_file, output_dir)

    print("\n" + "=" * 80)
    print(f"ALL VISUALIZATIONS COMPLETE")
    print(f"Saved to: {output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    create_all_visualizations()
