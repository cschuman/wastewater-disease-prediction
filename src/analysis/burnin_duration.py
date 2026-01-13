"""
Burn-in Duration Analysis for Wastewater Monitoring Sites

This script analyzes how long new wastewater monitoring sites need to "mature"
before providing stable, reliable data. Based on findings that states expanding
their monitoring networks saw increased signal variability (p=0.041), we investigate
the calibration period needed for new sites.

Key Questions:
1. How long does it take for sites to stabilize?
2. Does burn-in duration vary by population size, state, or other factors?
3. What are the implications for expansion planning?

Author: Epidemiology Analysis
Date: 2026-01-11
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import spearmanr, mannwhitneyu, kruskal
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class BurninAnalyzer:
    """Analyzer for wastewater site burn-in effects."""

    def __init__(self, data_path):
        """
        Initialize the analyzer.

        Parameters:
        -----------
        data_path : str
            Path to the parquet file with wastewater data
        """
        self.data_path = data_path
        self.df = None
        self.site_metrics = None
        self.maturity_curves = None

    def load_data(self):
        """Load and prepare the wastewater data."""
        print("Loading wastewater data...")
        self.df = pd.read_parquet(self.data_path)

        print(f"Loaded {len(self.df):,} records from {self.df['key_plot_id'].nunique()} sites")
        print(f"Date range: {self.df['date_start'].min()} to {self.df['date_end'].max()}")
        print(f"First sample dates: {self.df['first_sample_date'].min()} to {self.df['first_sample_date'].max()}")

        # Create a consistent observation date (use date_end as the observation date)
        self.df['obs_date'] = self.df['date_end']

        # Calculate site age in days and weeks
        self.df['site_age_days'] = (self.df['obs_date'] - self.df['first_sample_date']).dt.days
        self.df['site_age_weeks'] = self.df['site_age_days'] / 7

        # Filter out negative ages (data quality issue)
        initial_count = len(self.df)
        self.df = self.df[self.df['site_age_days'] >= 0].copy()
        removed = initial_count - len(self.df)
        if removed > 0:
            print(f"Removed {removed:,} records with negative site age (data quality issue)")

        print(f"\nSite age range: {self.df['site_age_days'].min():.0f} to {self.df['site_age_days'].max():.0f} days")
        print(f"                ({self.df['site_age_weeks'].min():.1f} to {self.df['site_age_weeks'].max():.1f} weeks)")

        return self

    def calculate_signal_metrics_by_age(self, age_bins_weeks=[4, 8, 12, 16, 20, 26, 39, 52, 78, 104]):
        """
        Calculate signal quality metrics for sites at different ages.

        Parameters:
        -----------
        age_bins_weeks : list
            Age breakpoints in weeks for binning sites
        """
        print("\n" + "="*80)
        print("CALCULATING SIGNAL METRICS BY SITE AGE")
        print("="*80)

        # Create age bins
        age_bins = [0] + age_bins_weeks + [self.df['site_age_weeks'].max() + 1]
        self.df['age_bin'] = pd.cut(self.df['site_age_weeks'], bins=age_bins,
                                     labels=[f"{age_bins[i]:.0f}-{age_bins[i+1]:.0f}w"
                                            for i in range(len(age_bins)-1)],
                                     include_lowest=True)

        results = []

        # For each age bin, calculate metrics
        for age_bin in self.df['age_bin'].unique():
            if pd.isna(age_bin):
                continue

            bin_data = self.df[self.df['age_bin'] == age_bin].copy()

            # Calculate metrics for sites in this age range
            metrics = {
                'age_bin': age_bin,
                'n_sites': bin_data['key_plot_id'].nunique(),
                'n_observations': len(bin_data),
                'avg_age_weeks': bin_data['site_age_weeks'].mean(),
            }

            # Signal variability (std of percentile)
            # Calculate per-site std, then average across sites
            site_stds = bin_data.groupby('key_plot_id')['percentile'].std()
            metrics['percentile_variability_mean'] = site_stds.mean()
            metrics['percentile_variability_median'] = site_stds.median()

            # Detection rate stability (std of detection proportion over time)
            detect_stds = bin_data.groupby('key_plot_id')['detect_prop_15d'].std()
            metrics['detection_stability_mean'] = detect_stds.mean()
            metrics['detection_stability_median'] = detect_stds.median()

            # Week-to-week volatility (absolute change in percentile)
            volatility_list = []
            for site in bin_data['key_plot_id'].unique():
                site_data = bin_data[bin_data['key_plot_id'] == site].sort_values('obs_date')
                if len(site_data) > 1:
                    site_data['percentile_change'] = site_data['percentile'].diff().abs()
                    volatility_list.append(site_data['percentile_change'].mean())

            if volatility_list:
                metrics['week_to_week_volatility'] = np.mean(volatility_list)
            else:
                metrics['week_to_week_volatility'] = np.nan

            # Missing data rate
            metrics['percentile_missing_rate'] = bin_data['percentile'].isna().mean()
            metrics['detect_missing_rate'] = bin_data['detect_prop_15d'].isna().mean()

            # Average percentile value
            metrics['avg_percentile'] = bin_data['percentile'].mean()
            metrics['median_percentile'] = bin_data['percentile'].median()

            results.append(metrics)

        self.maturity_curves = pd.DataFrame(results)
        self.maturity_curves = self.maturity_curves.sort_values('avg_age_weeks')

        print("\nMaturity Curves Summary:")
        print(self.maturity_curves.to_string(index=False))

        return self

    def calculate_per_site_metrics(self, window_weeks=4):
        """
        Calculate comprehensive metrics for each site across different time windows.

        Parameters:
        -----------
        window_weeks : int
            Size of rolling window for calculating early vs. late metrics
        """
        print("\n" + "="*80)
        print(f"CALCULATING PER-SITE METRICS (using {window_weeks}-week windows)")
        print("="*80)

        site_results = []

        for site_id in self.df['key_plot_id'].unique():
            site_data = self.df[self.df['key_plot_id'] == site_id].sort_values('obs_date').copy()

            if len(site_data) < 2:
                continue  # Need at least 2 observations

            # Basic site information
            site_info = {
                'key_plot_id': site_id,
                'state': site_data['wwtp_jurisdiction'].iloc[0],
                'population_served': site_data['population_served'].iloc[0],
                'first_sample_date': site_data['first_sample_date'].iloc[0],
                'n_observations': len(site_data),
                'monitoring_days': (site_data['obs_date'].max() - site_data['obs_date'].min()).days,
            }

            # Split into early and late periods
            max_age_weeks = site_data['site_age_weeks'].max()

            # Early period: first window_weeks
            early_data = site_data[site_data['site_age_weeks'] <= window_weeks]
            # Late period: last window_weeks (if site is old enough)
            late_data = site_data[site_data['site_age_weeks'] >= max(max_age_weeks - window_weeks, window_weeks)]

            # Metrics for early period
            if len(early_data) > 1:
                site_info['early_percentile_std'] = early_data['percentile'].std()
                site_info['early_percentile_mean'] = early_data['percentile'].mean()
                site_info['early_detect_std'] = early_data['detect_prop_15d'].std()
                site_info['early_missing_rate'] = early_data['percentile'].isna().mean()

                # Volatility
                early_data_sorted = early_data.sort_values('obs_date')
                early_data_sorted['pct_change'] = early_data_sorted['percentile'].diff().abs()
                site_info['early_volatility'] = early_data_sorted['pct_change'].mean()
            else:
                site_info['early_percentile_std'] = np.nan
                site_info['early_percentile_mean'] = np.nan
                site_info['early_detect_std'] = np.nan
                site_info['early_missing_rate'] = np.nan
                site_info['early_volatility'] = np.nan

            # Metrics for late period (if available)
            if len(late_data) > 1 and max_age_weeks > window_weeks * 2:
                site_info['late_percentile_std'] = late_data['percentile'].std()
                site_info['late_percentile_mean'] = late_data['percentile'].mean()
                site_info['late_detect_std'] = late_data['detect_prop_15d'].std()
                site_info['late_missing_rate'] = late_data['percentile'].isna().mean()

                # Volatility
                late_data_sorted = late_data.sort_values('obs_date')
                late_data_sorted['pct_change'] = late_data_sorted['percentile'].diff().abs()
                site_info['late_volatility'] = late_data_sorted['pct_change'].mean()

                # Calculate improvement metrics
                site_info['variability_improvement'] = site_info['early_percentile_std'] - site_info['late_percentile_std']
                site_info['volatility_improvement'] = site_info['early_volatility'] - site_info['late_volatility']
                site_info['stability_improved'] = site_info['variability_improvement'] > 0
            else:
                site_info['late_percentile_std'] = np.nan
                site_info['late_percentile_mean'] = np.nan
                site_info['late_detect_std'] = np.nan
                site_info['late_missing_rate'] = np.nan
                site_info['late_volatility'] = np.nan
                site_info['variability_improvement'] = np.nan
                site_info['volatility_improvement'] = np.nan
                site_info['stability_improved'] = np.nan

            # Overall metrics
            site_info['overall_percentile_std'] = site_data['percentile'].std()
            site_info['overall_volatility'] = site_data.sort_values('obs_date')['percentile'].diff().abs().mean()

            site_results.append(site_info)

        self.site_metrics = pd.DataFrame(site_results)

        print(f"\nCalculated metrics for {len(self.site_metrics)} sites")
        print(f"Sites with both early and late data: {self.site_metrics['late_percentile_std'].notna().sum()}")
        print(f"Sites showing improved stability: {self.site_metrics['stability_improved'].sum()}")

        return self

    def estimate_stabilization_time(self, stability_threshold=0.8):
        """
        Estimate when sites typically stabilize.

        We define stabilization as when variability drops to stability_threshold
        of the initial (early) variability.

        Parameters:
        -----------
        stability_threshold : float
            Threshold for considering a site stabilized (0.8 = 80% of early variability)
        """
        print("\n" + "="*80)
        print(f"ESTIMATING STABILIZATION TIME (threshold: {stability_threshold*100:.0f}% of early variability)")
        print("="*80)

        # For each site with sufficient data, find when it stabilized
        stabilization_times = []

        for site_id in self.df['key_plot_id'].unique():
            site_data = self.df[self.df['key_plot_id'] == site_id].sort_values('obs_date').copy()

            if len(site_data) < 10:  # Need sufficient observations
                continue

            # Get early variability (first 4 weeks)
            early_data = site_data[site_data['site_age_weeks'] <= 4]
            if len(early_data) < 2:
                continue

            early_std = early_data['percentile'].std()
            if pd.isna(early_std) or early_std == 0:
                continue

            target_std = early_std * stability_threshold

            # Calculate rolling std with 4-week window
            site_data_sorted = site_data.sort_values('site_age_weeks')
            site_data_sorted['rolling_std'] = site_data_sorted['percentile'].rolling(window=4, min_periods=2).std()

            # Find first point where rolling std drops below threshold
            stabilized = site_data_sorted[site_data_sorted['rolling_std'] <= target_std]

            if len(stabilized) > 0:
                stabilization_week = stabilized['site_age_weeks'].iloc[0]
                stabilization_times.append({
                    'site_id': site_id,
                    'state': site_data['wwtp_jurisdiction'].iloc[0],
                    'population_served': site_data['population_served'].iloc[0],
                    'stabilization_weeks': stabilization_week,
                    'early_std': early_std,
                    'stable_std': stabilized['rolling_std'].iloc[0],
                    'reduction_pct': (1 - stabilized['rolling_std'].iloc[0] / early_std) * 100
                })

        stabilization_df = pd.DataFrame(stabilization_times)

        if len(stabilization_df) > 0:
            print(f"\nAnalyzed {len(stabilization_df)} sites with detectable stabilization")
            print(f"\nStabilization time distribution:")
            print(stabilization_df['stabilization_weeks'].describe())
            print(f"\nMedian stabilization time: {stabilization_df['stabilization_weeks'].median():.1f} weeks")
            print(f"                            ({stabilization_df['stabilization_weeks'].median() / 4.33:.1f} months)")
            print(f"75th percentile: {stabilization_df['stabilization_weeks'].quantile(0.75):.1f} weeks")
            print(f"                 ({stabilization_df['stabilization_weeks'].quantile(0.75) / 4.33:.1f} months)")
            print(f"90th percentile: {stabilization_df['stabilization_weeks'].quantile(0.90):.1f} weeks")
            print(f"                 ({stabilization_df['stabilization_weeks'].quantile(0.90) / 4.33:.1f} months)")
        else:
            print("\nInsufficient data to estimate stabilization times")

        return stabilization_df

    def analyze_burnin_factors(self):
        """
        Test if burn-in duration varies by population, state, or other factors.
        """
        print("\n" + "="*80)
        print("ANALYZING FACTORS AFFECTING BURN-IN DURATION")
        print("="*80)

        # Filter to sites with both early and late data
        analysis_df = self.site_metrics[
            (self.site_metrics['early_percentile_std'].notna()) &
            (self.site_metrics['late_percentile_std'].notna())
        ].copy()

        print(f"\nAnalyzing {len(analysis_df)} sites with sufficient data")

        # 1. Population size effect
        print("\n" + "-"*80)
        print("1. POPULATION SIZE EFFECT")
        print("-"*80)

        # Create population categories
        analysis_df['population_category'] = pd.cut(
            analysis_df['population_served'],
            bins=[0, 10000, 50000, 100000, 500000, float('inf')],
            labels=['<10k', '10k-50k', '50k-100k', '100k-500k', '>500k']
        )

        pop_groups = analysis_df.groupby('population_category').agg({
            'variability_improvement': ['mean', 'median', 'count'],
            'early_percentile_std': 'mean',
            'late_percentile_std': 'mean',
            'stability_improved': 'mean'
        }).round(2)

        print("\nVariability improvement by population size:")
        print(pop_groups)

        # Statistical test
        groups = [group['variability_improvement'].dropna().values
                 for name, group in analysis_df.groupby('population_category')]
        groups = [g for g in groups if len(g) > 0]

        if len(groups) >= 2:
            h_stat, p_val = kruskal(*groups)
            print(f"\nKruskal-Wallis test: H={h_stat:.3f}, p={p_val:.4f}")
            if p_val < 0.05:
                print("SIGNIFICANT: Population size affects burn-in improvement")
            else:
                print("NOT SIGNIFICANT: No clear population size effect")

        # Correlation with population
        valid_data = analysis_df[['population_served', 'variability_improvement']].dropna()
        if len(valid_data) > 10:
            corr, p_val = spearmanr(valid_data['population_served'],
                                   valid_data['variability_improvement'])
            print(f"\nSpearman correlation with population: r={corr:.3f}, p={p_val:.4f}")

        # 2. State-level variation
        print("\n" + "-"*80)
        print("2. STATE-LEVEL VARIATION")
        print("-"*80)

        state_groups = analysis_df.groupby('state').agg({
            'variability_improvement': ['mean', 'median', 'count'],
            'stability_improved': 'mean'
        }).round(2)

        # Only show states with at least 5 sites
        state_groups = state_groups[state_groups[('variability_improvement', 'count')] >= 5]
        state_groups = state_groups.sort_values(('variability_improvement', 'mean'), ascending=False)

        print("\nTop 10 states by variability improvement (min 5 sites):")
        print(state_groups.head(10))

        print("\nBottom 10 states by variability improvement (min 5 sites):")
        print(state_groups.tail(10))

        # 3. Monitoring duration effect
        print("\n" + "-"*80)
        print("3. MONITORING DURATION EFFECT")
        print("-"*80)

        analysis_df['monitoring_months'] = analysis_df['monitoring_days'] / 30.44
        analysis_df['duration_category'] = pd.cut(
            analysis_df['monitoring_months'],
            bins=[0, 3, 6, 12, 24, float('inf')],
            labels=['<3mo', '3-6mo', '6-12mo', '12-24mo', '>24mo']
        )

        duration_groups = analysis_df.groupby('duration_category').agg({
            'variability_improvement': ['mean', 'median', 'count'],
            'stability_improved': 'mean'
        }).round(2)

        print("\nImprovement by monitoring duration:")
        print(duration_groups)

        # 4. Summary statistics
        print("\n" + "-"*80)
        print("4. OVERALL SUMMARY")
        print("-"*80)

        improved_pct = (analysis_df['stability_improved'].sum() / len(analysis_df)) * 100
        print(f"\nSites showing improved stability: {analysis_df['stability_improved'].sum()}/{len(analysis_df)} ({improved_pct:.1f}%)")
        print(f"\nAverage variability improvement: {analysis_df['variability_improvement'].mean():.2f}")
        print(f"Median variability improvement: {analysis_df['variability_improvement'].median():.2f}")
        print(f"\nEarly period avg std: {analysis_df['early_percentile_std'].mean():.2f}")
        print(f"Late period avg std: {analysis_df['late_percentile_std'].mean():.2f}")
        print(f"Overall reduction: {((analysis_df['early_percentile_std'].mean() - analysis_df['late_percentile_std'].mean()) / analysis_df['early_percentile_std'].mean() * 100):.1f}%")

        return analysis_df

    def generate_visualizations(self, output_dir="/Users/corey/Projects/the-playground/wastewater-disease-prediction/output"):
        """Generate visualizations of burn-in effects."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)

        # 1. Maturity curves
        if self.maturity_curves is not None:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # Variability over time
            ax = axes[0, 0]
            ax.plot(self.maturity_curves['avg_age_weeks'],
                   self.maturity_curves['percentile_variability_mean'],
                   marker='o', linewidth=2, markersize=8, label='Mean')
            ax.plot(self.maturity_curves['avg_age_weeks'],
                   self.maturity_curves['percentile_variability_median'],
                   marker='s', linewidth=2, markersize=8, label='Median', linestyle='--')
            ax.set_xlabel('Site Age (weeks)', fontsize=12)
            ax.set_ylabel('Percentile Variability (Std Dev)', fontsize=12)
            ax.set_title('Signal Variability vs Site Age', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Week-to-week volatility
            ax = axes[0, 1]
            ax.plot(self.maturity_curves['avg_age_weeks'],
                   self.maturity_curves['week_to_week_volatility'],
                   marker='o', linewidth=2, markersize=8, color='coral')
            ax.set_xlabel('Site Age (weeks)', fontsize=12)
            ax.set_ylabel('Week-to-Week Volatility', fontsize=12)
            ax.set_title('Signal Volatility vs Site Age', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Detection stability
            ax = axes[1, 0]
            ax.plot(self.maturity_curves['avg_age_weeks'],
                   self.maturity_curves['detection_stability_mean'],
                   marker='o', linewidth=2, markersize=8, color='green')
            ax.set_xlabel('Site Age (weeks)', fontsize=12)
            ax.set_ylabel('Detection Rate Std Dev', fontsize=12)
            ax.set_title('Detection Rate Stability vs Site Age', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Missing data rate
            ax = axes[1, 1]
            ax.plot(self.maturity_curves['avg_age_weeks'],
                   self.maturity_curves['percentile_missing_rate'] * 100,
                   marker='o', linewidth=2, markersize=8, color='red')
            ax.set_xlabel('Site Age (weeks)', fontsize=12)
            ax.set_ylabel('Missing Data Rate (%)', fontsize=12)
            ax.set_title('Data Completeness vs Site Age', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            maturity_path = f"{output_dir}/burnin_maturity_curves.png"
            plt.savefig(maturity_path, dpi=300, bbox_inches='tight')
            print(f"Saved maturity curves to: {maturity_path}")
            plt.close()

        # 2. Early vs Late comparison
        if self.site_metrics is not None:
            analysis_df = self.site_metrics[
                (self.site_metrics['early_percentile_std'].notna()) &
                (self.site_metrics['late_percentile_std'].notna())
            ].copy()

            if len(analysis_df) > 0:
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))

                # Scatter: Early vs Late variability
                ax = axes[0, 0]
                ax.scatter(analysis_df['early_percentile_std'],
                          analysis_df['late_percentile_std'],
                          alpha=0.5, s=50)
                max_val = max(analysis_df['early_percentile_std'].max(),
                             analysis_df['late_percentile_std'].max())
                ax.plot([0, max_val], [0, max_val], 'r--', label='No change', linewidth=2)
                ax.set_xlabel('Early Period Variability', fontsize=12)
                ax.set_ylabel('Late Period Variability', fontsize=12)
                ax.set_title('Early vs Late Signal Variability', fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Distribution of improvement
                ax = axes[0, 1]
                improvement = analysis_df['variability_improvement'].dropna()
                ax.hist(improvement, bins=30, edgecolor='black', alpha=0.7)
                ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No improvement')
                ax.axvline(improvement.median(), color='green', linestyle='--', linewidth=2,
                          label=f'Median: {improvement.median():.1f}')
                ax.set_xlabel('Variability Improvement', fontsize=12)
                ax.set_ylabel('Number of Sites', fontsize=12)
                ax.set_title('Distribution of Stability Improvement', fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Population effect
                ax = axes[1, 0]
                analysis_df['population_category'] = pd.cut(
                    analysis_df['population_served'],
                    bins=[0, 10000, 50000, 100000, 500000, float('inf')],
                    labels=['<10k', '10k-50k', '50k-100k', '100k-500k', '>500k']
                )
                pop_data = [analysis_df[analysis_df['population_category'] == cat]['variability_improvement'].dropna()
                           for cat in ['<10k', '10k-50k', '50k-100k', '100k-500k', '>500k']]
                ax.boxplot([d for d in pop_data if len(d) > 0],
                          labels=[cat for cat, d in zip(['<10k', '10k-50k', '50k-100k', '100k-500k', '>500k'], pop_data) if len(d) > 0])
                ax.axhline(0, color='red', linestyle='--', alpha=0.5)
                ax.set_xlabel('Population Served', fontsize=12)
                ax.set_ylabel('Variability Improvement', fontsize=12)
                ax.set_title('Burn-in Improvement by Population Size', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

                # Monitoring duration effect
                ax = axes[1, 1]
                ax.scatter(analysis_df['monitoring_days'] / 30.44,
                          analysis_df['variability_improvement'],
                          alpha=0.5, s=50)
                ax.axhline(0, color='red', linestyle='--', alpha=0.5)
                ax.set_xlabel('Monitoring Duration (months)', fontsize=12)
                ax.set_ylabel('Variability Improvement', fontsize=12)
                ax.set_title('Improvement vs Monitoring Duration', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)

                plt.tight_layout()
                comparison_path = f"{output_dir}/burnin_early_late_comparison.png"
                plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
                print(f"Saved early/late comparison to: {comparison_path}")
                plt.close()

        print("\nVisualization generation complete!")

    def run_full_analysis(self):
        """Run the complete burn-in analysis pipeline."""
        print("\n" + "="*80)
        print("WASTEWATER SITE BURN-IN DURATION ANALYSIS")
        print("="*80)
        print("\nThis analysis investigates how long new wastewater monitoring sites")
        print("need to 'mature' before providing stable, reliable data.")
        print("="*80)

        # Load data
        self.load_data()

        # Calculate metrics by age
        self.calculate_signal_metrics_by_age()

        # Calculate per-site metrics
        self.calculate_per_site_metrics()

        # Estimate stabilization time
        stabilization_df = self.estimate_stabilization_time()

        # Analyze factors
        analysis_df = self.analyze_burnin_factors()

        # Generate visualizations
        self.generate_visualizations()

        # Final summary
        print("\n" + "="*80)
        print("FINAL SUMMARY AND RECOMMENDATIONS")
        print("="*80)

        if self.maturity_curves is not None:
            early_curve = self.maturity_curves[self.maturity_curves['avg_age_weeks'] < 12]
            late_curve = self.maturity_curves[self.maturity_curves['avg_age_weeks'] > 26]

            if len(early_curve) > 0 and len(late_curve) > 0:
                early_var = early_curve['percentile_variability_mean'].mean()
                late_var = late_curve['percentile_variability_mean'].mean()
                reduction = ((early_var - late_var) / early_var) * 100

                print(f"\n1. Signal Quality Improvement:")
                print(f"   - Early sites (<12 weeks): avg variability = {early_var:.2f}")
                print(f"   - Mature sites (>26 weeks): avg variability = {late_var:.2f}")
                print(f"   - Reduction in variability: {reduction:.1f}%")

        if len(stabilization_df) > 0:
            median_weeks = stabilization_df['stabilization_weeks'].median()
            p75_weeks = stabilization_df['stabilization_weeks'].quantile(0.75)

            print(f"\n2. Stabilization Timeline:")
            print(f"   - Median time to stabilize: {median_weeks:.0f} weeks ({median_weeks/4.33:.1f} months)")
            print(f"   - 75% of sites stabilize by: {p75_weeks:.0f} weeks ({p75_weeks/4.33:.1f} months)")

        if analysis_df is not None and len(analysis_df) > 0:
            improved_pct = (analysis_df['stability_improved'].sum() / len(analysis_df)) * 100
            print(f"\n3. Overall Improvement Rate:")
            print(f"   - {improved_pct:.1f}% of sites show improved stability over time")

        print("\n4. Recommendations for Expansion Planning:")
        print("   - Budget for 3-6 month burn-in period when adding new sites")
        print("   - Avoid using data from sites <12 weeks old for critical decisions")
        print("   - Implement quality control procedures for new sites")
        print("   - Consider phased rollouts to maintain overall network stability")
        print("   - Provide additional technical support during first 6 months")

        print("\n" + "="*80)
        print("Analysis complete! Results saved to output directory.")
        print("="*80)

        return {
            'maturity_curves': self.maturity_curves,
            'site_metrics': self.site_metrics,
            'stabilization_times': stabilization_df,
            'analysis_summary': analysis_df
        }


def main():
    """Main execution function."""
    # Initialize analyzer
    data_path = "/Users/corey/Projects/the-playground/wastewater-disease-prediction/data/raw/nwss/nwss_metrics_20260111.parquet"
    analyzer = BurninAnalyzer(data_path)

    # Run full analysis
    results = analyzer.run_full_analysis()

    # Save results to CSV
    output_dir = "/Users/corey/Projects/the-playground/wastewater-disease-prediction/output"
    import os
    os.makedirs(output_dir, exist_ok=True)

    if results['maturity_curves'] is not None:
        results['maturity_curves'].to_csv(f"{output_dir}/burnin_maturity_curves.csv", index=False)
        print(f"\nSaved maturity curves to: {output_dir}/burnin_maturity_curves.csv")

    if results['site_metrics'] is not None:
        results['site_metrics'].to_csv(f"{output_dir}/burnin_site_metrics.csv", index=False)
        print(f"Saved site metrics to: {output_dir}/burnin_site_metrics.csv")

    if results['stabilization_times'] is not None and len(results['stabilization_times']) > 0:
        results['stabilization_times'].to_csv(f"{output_dir}/burnin_stabilization_times.csv", index=False)
        print(f"Saved stabilization times to: {output_dir}/burnin_stabilization_times.csv")

    return results


if __name__ == "__main__":
    results = main()
