export interface County {
	fips: string;
	state: string;
	county_name: string;
	population: number;
	svi_overall: number;
	svi_quartile: string;
	n_sites: number;
	coverage_pct: number;
	priority_score: number;
	priority_tier: string;
	// Optional fields from SVI/RUCC merge
	svi_socioeconomic?: number;
	svi_household?: number;
	svi_minority?: number;
	svi_housing?: number;
	rucc_2023?: number;
	is_metro?: boolean;
	is_rural?: boolean;
	metro_size?: string;
	urbanization?: string;
}

export interface TopPriorityCounty {
	state: string;
	county_name: string;
	population: number;
	svi_overall: number;
	svi_quartile: string;
	n_sites: number;
	coverage_pct: number;
	priority_score: number;
	priority_tier: string;
}

export interface StateAggregate {
	state: string;
	population: number;
	n_sites: number;
	svi_overall: number;
	priority_score: number;
	county_count: number;
	sites_per_100k: number;
}

export interface SviQuartileStats {
	svi_quartile: string;
	population: number;
	n_sites: number;
	coverage_pct: number;
	county_count: number;
	sites_per_million: number;
}

export interface SummaryStats {
	total_counties: number;
	total_population: number;
	total_sites: number;
	counties_with_monitoring: number;
	coverage_pct: number;
	high_svi_zero_coverage: number;
	q1_sites_per_million: number;
	q4_sites_per_million: number;
	disparity_pct: number;
	analysis_date: string;
}

export type SviQuartile = 'Q1' | 'Q2' | 'Q3' | 'Q4 (High)';
export type PriorityTier = 'Tier 1 (Highest)' | 'Tier 2' | 'Tier 3';
