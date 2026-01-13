import { error } from '@sveltejs/kit';
import type { LayoutLoad } from './$types';
import type { County, Scenario, SummaryStats, SviQuartileStats, StateAggregate } from '$lib/types';

export const prerender = true;

/**
 * Fetch JSON with proper error handling.
 */
async function fetchJson<T>(fetchFn: typeof fetch, url: string): Promise<T> {
	const response = await fetchFn(url);
	if (!response.ok) {
		throw error(response.status, `Failed to fetch ${url}: ${response.statusText}`);
	}
	return response.json() as Promise<T>;
}

export const load: LayoutLoad = async ({ fetch }) => {
	try {
		const [counties, scenarios, summary, quartileStats, stateAggregates] = await Promise.all([
			fetchJson<County[]>(fetch, '/data/counties.json'),
			fetchJson<Scenario[]>(fetch, '/data/scenarios.json'),
			fetchJson<SummaryStats>(fetch, '/data/summary-stats.json'),
			fetchJson<SviQuartileStats[]>(fetch, '/data/svi-quartile-stats.json'),
			fetchJson<StateAggregate[]>(fetch, '/data/state-aggregates.json')
		]);

		return {
			counties,
			scenarios,
			summary,
			quartileStats,
			stateAggregates
		};
	} catch (e) {
		// Re-throw SvelteKit errors
		if (e && typeof e === 'object' && 'status' in e) {
			throw e;
		}
		// Handle unexpected errors
		console.error('Failed to load data:', e);
		throw error(500, 'Failed to load application data. Please try again later.');
	}
};
