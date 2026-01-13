<script lang="ts">
	import type { PageData } from './$types';
	import { formatPopulation, formatNumber } from '$lib/utils/formatters';
	import { getSviColor } from '$lib/utils/colors';

	let { data }: { data: PageData } = $props();

	interface StateData {
		state: string;
		population: number;
		n_sites: number;
		svi_overall: number;
		priority_score: number;
		county_count: number;
		sites_per_100k: number;
	}

	let sortColumn = $state<keyof StateData>('priority_score');
	let sortDirection = $state<'asc' | 'desc'>('desc');

	// State name mapping
	const stateNames: Record<string, string> = {
		AL: 'Alabama', AK: 'Alaska', AZ: 'Arizona', AR: 'Arkansas', CA: 'California',
		CO: 'Colorado', CT: 'Connecticut', DE: 'Delaware', DC: 'District of Columbia',
		FL: 'Florida', GA: 'Georgia', HI: 'Hawaii', ID: 'Idaho', IL: 'Illinois',
		IN: 'Indiana', IA: 'Iowa', KS: 'Kansas', KY: 'Kentucky', LA: 'Louisiana',
		ME: 'Maine', MD: 'Maryland', MA: 'Massachusetts', MI: 'Michigan', MN: 'Minnesota',
		MS: 'Mississippi', MO: 'Missouri', MT: 'Montana', NE: 'Nebraska', NV: 'Nevada',
		NH: 'New Hampshire', NJ: 'New Jersey', NM: 'New Mexico', NY: 'New York',
		NC: 'North Carolina', ND: 'North Dakota', OH: 'Ohio', OK: 'Oklahoma', OR: 'Oregon',
		PA: 'Pennsylvania', RI: 'Rhode Island', SC: 'South Carolina', SD: 'South Dakota',
		TN: 'Tennessee', TX: 'Texas', UT: 'Utah', VT: 'Vermont', VA: 'Virginia',
		WA: 'Washington', WV: 'West Virginia', WI: 'Wisconsin', WY: 'Wyoming'
	};

	const sortedStates = $derived.by(() => {
		return (data.stateAggregates as StateData[]).slice().sort((a, b) => {
			const aVal = a[sortColumn];
			const bVal = b[sortColumn];
			if (typeof aVal === 'number' && typeof bVal === 'number') {
				return sortDirection === 'asc' ? aVal - bVal : bVal - aVal;
			}
			return sortDirection === 'asc'
				? String(aVal).localeCompare(String(bVal))
				: String(bVal).localeCompare(String(aVal));
		});
	});

	function setSort(column: keyof StateData) {
		if (sortColumn === column) {
			sortDirection = sortDirection === 'asc' ? 'desc' : 'asc';
		} else {
			sortColumn = column;
			sortDirection = 'desc';
		}
	}

	function getSortIndicator(column: keyof StateData) {
		if (sortColumn !== column) return '';
		return sortDirection === 'asc' ? ' ↑' : ' ↓';
	}

	// Summary stats
	const totalPop = $derived(data.stateAggregates.reduce((sum: number, s: StateData) => sum + s.population, 0));
	const totalSites = $derived(data.stateAggregates.reduce((sum: number, s: StateData) => sum + s.n_sites, 0));
	const avgSvi = $derived(data.stateAggregates.reduce((sum: number, s: StateData) => sum + s.svi_overall, 0) / data.stateAggregates.length);
</script>

<svelte:head>
	<title>States | Wastewater Surveillance Equity</title>
	<meta name="description" content="State-level wastewater surveillance coverage data across all 50 US states and DC." />
</svelte:head>

<div class="space-y-6">
	<div>
		<h1 class="text-2xl font-bold">State Overview</h1>
		<p class="text-gray-600">Aggregated wastewater surveillance data by state</p>
	</div>

	<!-- Summary cards -->
	<div class="grid grid-cols-2 md:grid-cols-4 gap-4">
		<div class="bg-white rounded-lg border p-4">
			<p class="text-sm text-gray-500">States & Territories</p>
			<p class="text-2xl font-bold">{data.stateAggregates.length}</p>
		</div>
		<div class="bg-white rounded-lg border p-4">
			<p class="text-sm text-gray-500">Total Population</p>
			<p class="text-2xl font-bold">{formatPopulation(totalPop)}</p>
		</div>
		<div class="bg-white rounded-lg border p-4">
			<p class="text-sm text-gray-500">Total Sites</p>
			<p class="text-2xl font-bold">{formatNumber(totalSites)}</p>
		</div>
		<div class="bg-white rounded-lg border p-4">
			<p class="text-sm text-gray-500">Avg SVI</p>
			<p class="text-2xl font-bold">{avgSvi.toFixed(2)}</p>
		</div>
	</div>

	<!-- Table -->
	<div class="bg-white rounded-lg border overflow-hidden">
		<div class="overflow-x-auto">
			<table class="w-full text-sm">
				<thead class="bg-gray-50 border-b">
					<tr>
						<th class="px-4 py-3 text-left font-medium text-gray-500">
							<button class="hover:text-gray-700" onclick={() => setSort('state')}>
								State{getSortIndicator('state')}
							</button>
						</th>
						<th class="px-4 py-3 text-right font-medium text-gray-500">
							<button class="hover:text-gray-700" onclick={() => setSort('population')}>
								Population{getSortIndicator('population')}
							</button>
						</th>
						<th class="px-4 py-3 text-right font-medium text-gray-500">
							<button class="hover:text-gray-700" onclick={() => setSort('county_count')}>
								Counties{getSortIndicator('county_count')}
							</button>
						</th>
						<th class="px-4 py-3 text-right font-medium text-gray-500">
							<button class="hover:text-gray-700" onclick={() => setSort('n_sites')}>
								Sites{getSortIndicator('n_sites')}
							</button>
						</th>
						<th class="px-4 py-3 text-right font-medium text-gray-500">
							<button class="hover:text-gray-700" onclick={() => setSort('sites_per_100k')}>
								Sites/100k{getSortIndicator('sites_per_100k')}
							</button>
						</th>
						<th class="px-4 py-3 text-right font-medium text-gray-500">
							<button class="hover:text-gray-700" onclick={() => setSort('svi_overall')}>
								Avg SVI{getSortIndicator('svi_overall')}
							</button>
						</th>
						<th class="px-4 py-3 text-right font-medium text-gray-500">
							<button class="hover:text-gray-700" onclick={() => setSort('priority_score')}>
								Avg Priority{getSortIndicator('priority_score')}
							</button>
						</th>
					</tr>
				</thead>
				<tbody class="divide-y divide-gray-100">
					{#each sortedStates as state}
						<tr class="hover:bg-gray-50">
							<td class="px-4 py-3">
								<a href="/counties?state={stateNames[state.state] || state.state}" class="text-blue-600 hover:text-blue-800 font-medium">
									{stateNames[state.state] || state.state}
								</a>
							</td>
							<td class="px-4 py-3 text-right text-gray-600">{formatPopulation(state.population)}</td>
							<td class="px-4 py-3 text-right text-gray-600">{state.county_count}</td>
							<td class="px-4 py-3 text-right font-medium">{state.n_sites}</td>
							<td class="px-4 py-3 text-right font-mono">{state.sites_per_100k.toFixed(2)}</td>
							<td class="px-4 py-3 text-right">
								<span
									class="inline-block w-3 h-3 rounded-full mr-2"
									style="background-color: {getSviColor(state.svi_overall)}"
								></span>
								<span class="font-mono">{state.svi_overall.toFixed(2)}</span>
							</td>
							<td class="px-4 py-3 text-right font-mono">{state.priority_score.toFixed(1)}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	</div>
</div>
