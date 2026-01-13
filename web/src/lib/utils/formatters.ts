import { format } from 'd3-format';

export const formatNumber = format(',');
export const formatPercent = format('.1%');
export const formatDecimal = format('.2f');
export const formatCompact = format('.2s');

export function formatPopulation(value: number): string {
	if (value >= 1_000_000) {
		return `${(value / 1_000_000).toFixed(1)}M`;
	}
	if (value >= 1_000) {
		return `${(value / 1_000).toFixed(0)}k`;
	}
	return value.toString();
}

export function formatCurrency(millions: number): string {
	if (millions >= 1000) {
		return `$${(millions / 1000).toFixed(1)}B`;
	}
	return `$${millions.toFixed(0)}M`;
}

export function formatSvi(value: number): string {
	return value.toFixed(2);
}
