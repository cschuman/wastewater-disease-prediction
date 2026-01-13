import { scaleSequential, scaleOrdinal } from 'd3-scale';
import { interpolateRdYlGn, interpolateBlues, interpolateReds } from 'd3-scale-chromatic';

// SVI color scale (0 = low vulnerability/green, 1 = high vulnerability/red)
export const sviColorScale = scaleSequential(interpolateRdYlGn).domain([1, 0]);

// Priority score scale (0-100, higher = more urgent = redder)
export const priorityColorScale = scaleSequential(interpolateReds).domain([0, 100]);

// Coverage percent scale (0-100%, higher = more blue)
export const coverageColorScale = scaleSequential(interpolateBlues).domain([0, 100]);

// Quartile colors
export const quartileColors: Record<string, string> = {
	Q1: '#2563eb', // Blue - low vulnerability
	Q2: '#16a34a', // Green
	Q3: '#f59e0b', // Amber
	'Q4 (High)': '#dc2626' // Red - high vulnerability
};

// Priority tier colors
export const tierColors: Record<string, string> = {
	'Tier 1 (Highest)': '#dc2626',
	'Tier 2': '#f59e0b',
	'Tier 3': '#2563eb'
};

export function getQuartileColor(quartile: string): string {
	return quartileColors[quartile] || '#9ca3af';
}

export function getTierColor(tier: string): string {
	return tierColors[tier] || '#9ca3af';
}

export function getSviColor(svi: number): string {
	return sviColorScale(svi);
}

export function getPriorityColor(score: number): string {
	return priorityColorScale(score);
}
