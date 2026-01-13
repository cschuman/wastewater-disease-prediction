<script lang="ts">
	import type { PageData } from './$types';
	import { Badge } from '$lib/components/ui/index.js';
	import { formatPopulation, formatNumber } from '$lib/utils/formatters';
	import type { County } from '$lib/types';

	let { data }: { data: PageData } = $props();

	// Filter state
	let search = $state('');
	let selectedStates = $state<string[]>([]);
	let selectedQuartiles = $state<string[]>([]);
	let selectedTiers = $state<string[]>([]);
	let sortColumn = $state<keyof County>('priority_score');
	let sortDirection = $state<'asc' | 'desc'>('desc');
	let currentPage = $state(1);
	const pageSize = 50;

	// Get unique values for filters
	const states = [...new Set(data.counties.map((c) => c.state))].sort();
	const quartiles = [...new Set(data.counties.map((c) => c.svi_quartile))].sort();
	const tiers = [...new Set(data.counties.map((c) => c.priority_tier))];

	// Filtered and sorted counties - use $derived.by for complex logic
	const filteredCounties = $derived.by(() => {
		let result = data.counties;

		// Search filter
		if (search) {
			const s = search.toLowerCase();
			result = result.filter(
				(c) =>
					c.county_name.toLowerCase().includes(s) ||
					c.state.toLowerCase().includes(s) ||
					c.fips.includes(s)
			);
		}

		// State filter
		if (selectedStates.length > 0) {
			result = result.filter((c) => selectedStates.includes(c.state));
		}

		// Quartile filter
		if (selectedQuartiles.length > 0) {
			result = result.filter((c) => selectedQuartiles.includes(c.svi_quartile));
		}

		// Tier filter
		if (selectedTiers.length > 0) {
			result = result.filter((c) => selectedTiers.includes(c.priority_tier));
		}

		// Sort
		result = result.slice().sort((a, b) => {
			const aVal = a[sortColumn];
			const bVal = b[sortColumn];
			if (typeof aVal === 'number' && typeof bVal === 'number') {
				return sortDirection === 'asc' ? aVal - bVal : bVal - aVal;
			}
			return sortDirection === 'asc'
				? String(aVal).localeCompare(String(bVal))
				: String(bVal).localeCompare(String(aVal));
		});

		return result;
	});

	// Pagination - now filteredCounties is a value, not a function
	const totalPages = $derived(Math.ceil(filteredCounties.length / pageSize));
	const paginatedCounties = $derived.by(() => {
		const start = (currentPage - 1) * pageSize;
		return filteredCounties.slice(start, start + pageSize);
	});

	function setSort(column: keyof County) {
		if (sortColumn === column) {
			sortDirection = sortDirection === 'asc' ? 'desc' : 'asc';
		} else {
			sortColumn = column;
			sortDirection = 'desc';
		}
		currentPage = 1;
	}

	function toggleFilter<T>(arr: T[], value: T): T[] {
		const idx = arr.indexOf(value);
		if (idx >= 0) {
			return [...arr.slice(0, idx), ...arr.slice(idx + 1)];
		}
		return [...arr, value];
	}

	function clearFilters() {
		search = '';
		selectedStates = [];
		selectedQuartiles = [];
		selectedTiers = [];
		currentPage = 1;
	}

	function exportCSV() {
		const counties = filteredCounties;
		const headers = [
			'FIPS',
			'County',
			'State',
			'Population',
			'SVI Overall',
			'SVI Quartile',
			'Sites',
			'Coverage %',
			'Priority Score',
			'Priority Tier'
		];

		const rows = counties.map((c) => [
			c.fips,
			c.county_name,
			c.state,
			c.population,
			c.svi_overall.toFixed(4),
			c.svi_quartile,
			c.n_sites,
			c.coverage_pct.toFixed(2),
			c.priority_score.toFixed(2),
			c.priority_tier
		]);

		const csv = [headers.join(','), ...rows.map((r) => r.join(','))].join('\n');

		const blob = new Blob([csv], { type: 'text/csv' });
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = `counties-export-${new Date().toISOString().split('T')[0]}.csv`;
		a.click();
		URL.revokeObjectURL(url);
	}

	// Reset page when filters change
	$effect(() => {
		search;
		selectedStates;
		selectedQuartiles;
		selectedTiers;
		currentPage = 1;
	});
</script>

<div class="space-y-6">
	<div class="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
		<h1 class="text-2xl font-bold">County Explorer</h1>
		<div class="flex items-center gap-4">
			<span class="text-sm text-gray-500">
				{formatNumber(filteredCounties.length)} of {formatNumber(data.counties.length)} counties
			</span>
			<button
				onclick={exportCSV}
				class="inline-flex items-center gap-2 px-3 py-1.5 text-sm bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
			>
				<svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
				</svg>
				Export CSV
			</button>
		</div>
	</div>

	<!-- Filters -->
	<div class="bg-white rounded-lg border p-4 space-y-4">
		<!-- Search -->
		<div>
			<input
				type="text"
				bind:value={search}
				placeholder="Search counties, states, or FIPS codes..."
				class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
			/>
		</div>

		<!-- Filter chips -->
		<div class="flex flex-wrap gap-4">
			<!-- State filter -->
			<div class="relative">
				<select
					class="appearance-none px-3 py-1.5 pr-8 border rounded-lg text-sm bg-white"
					onchange={(e) => {
						const value = (e.target as HTMLSelectElement).value;
						if (value) {
							selectedStates = toggleFilter(selectedStates, value);
							(e.target as HTMLSelectElement).value = '';
						}
					}}
				>
					<option value="">+ State</option>
					{#each states as state}
						<option value={state} disabled={selectedStates.includes(state)}>{state}</option>
					{/each}
				</select>
			</div>

			<!-- Quartile filter -->
			<div class="flex gap-1">
				{#each quartiles as q}
					<button
						class="px-2 py-1 text-xs rounded border transition-colors {selectedQuartiles.includes(q)
							? 'bg-blue-100 border-blue-300 text-blue-700'
							: 'bg-white border-gray-200 text-gray-600 hover:bg-gray-50'}"
						onclick={() => (selectedQuartiles = toggleFilter(selectedQuartiles, q))}
					>
						{q}
					</button>
				{/each}
			</div>

			<!-- Tier filter -->
			<div class="flex gap-1">
				{#each tiers as tier}
					<button
						class="px-2 py-1 text-xs rounded border transition-colors {selectedTiers.includes(tier)
							? 'bg-blue-100 border-blue-300 text-blue-700'
							: 'bg-white border-gray-200 text-gray-600 hover:bg-gray-50'}"
						onclick={() => (selectedTiers = toggleFilter(selectedTiers, tier))}
					>
						{tier.replace(' (Highest)', '')}
					</button>
				{/each}
			</div>

			{#if selectedStates.length > 0 || selectedQuartiles.length > 0 || selectedTiers.length > 0 || search}
				<button class="px-2 py-1 text-xs text-red-600 hover:text-red-800" onclick={clearFilters}>
					Clear all
				</button>
			{/if}
		</div>

		<!-- Active state filters -->
		{#if selectedStates.length > 0}
			<div class="flex flex-wrap gap-1">
				{#each selectedStates as state}
					<button
						class="inline-flex items-center gap-1 px-2 py-0.5 bg-gray-100 rounded text-xs"
						onclick={() => (selectedStates = toggleFilter(selectedStates, state))}
					>
						{state}
						<span class="text-gray-400">&times;</span>
					</button>
				{/each}
			</div>
		{/if}
	</div>

	<!-- Table -->
	<div class="bg-white rounded-lg border overflow-hidden">
		<div class="overflow-x-auto">
			<table class="w-full text-sm">
				<thead class="bg-gray-50 border-b">
					<tr>
						<th class="px-4 py-3 text-left font-medium text-gray-500">
							<button class="hover:text-gray-700" onclick={() => setSort('county_name')}>
								County {sortColumn === 'county_name' ? (sortDirection === 'asc' ? '↑' : '↓') : ''}
							</button>
						</th>
						<th class="px-4 py-3 text-left font-medium text-gray-500">
							<button class="hover:text-gray-700" onclick={() => setSort('state')}>
								State {sortColumn === 'state' ? (sortDirection === 'asc' ? '↑' : '↓') : ''}
							</button>
						</th>
						<th class="px-4 py-3 text-right font-medium text-gray-500">
							<button class="hover:text-gray-700" onclick={() => setSort('population')}>
								Population {sortColumn === 'population' ? (sortDirection === 'asc' ? '↑' : '↓') : ''}
							</button>
						</th>
						<th class="px-4 py-3 text-right font-medium text-gray-500">
							<button class="hover:text-gray-700" onclick={() => setSort('svi_overall')}>
								SVI {sortColumn === 'svi_overall' ? (sortDirection === 'asc' ? '↑' : '↓') : ''}
							</button>
						</th>
						<th class="px-4 py-3 text-center font-medium text-gray-500">Quartile</th>
						<th class="px-4 py-3 text-right font-medium text-gray-500">
							<button class="hover:text-gray-700" onclick={() => setSort('n_sites')}>
								Sites {sortColumn === 'n_sites' ? (sortDirection === 'asc' ? '↑' : '↓') : ''}
							</button>
						</th>
						<th class="px-4 py-3 text-right font-medium text-gray-500">
							<button class="hover:text-gray-700" onclick={() => setSort('priority_score')}>
								Priority {sortColumn === 'priority_score' ? (sortDirection === 'asc' ? '↑' : '↓') : ''}
							</button>
						</th>
						<th class="px-4 py-3 text-center font-medium text-gray-500">Tier</th>
					</tr>
				</thead>
				<tbody class="divide-y divide-gray-100">
					{#each paginatedCounties as county}
						<tr class="hover:bg-gray-50">
							<td class="px-4 py-3">
								<a href="/counties/{county.fips}" class="text-blue-600 hover:text-blue-800 font-medium">
									{county.county_name}
								</a>
							</td>
							<td class="px-4 py-3 text-gray-600">{county.state}</td>
							<td class="px-4 py-3 text-right text-gray-600">{formatPopulation(county.population)}</td>
							<td class="px-4 py-3 text-right font-mono">{county.svi_overall.toFixed(2)}</td>
							<td class="px-4 py-3 text-center">
								<Badge text={county.svi_quartile} type="quartile" />
							</td>
							<td class="px-4 py-3 text-right">{county.n_sites}</td>
							<td class="px-4 py-3 text-right font-mono">{county.priority_score.toFixed(1)}</td>
							<td class="px-4 py-3 text-center">
								<Badge text={county.priority_tier} type="tier" />
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>

		<!-- Pagination -->
		<div class="flex items-center justify-between px-4 py-3 border-t bg-gray-50">
			<div class="text-sm text-gray-500">
				Showing {(currentPage - 1) * pageSize + 1} to {Math.min(currentPage * pageSize, filteredCounties.length)} of {filteredCounties.length}
			</div>
			<div class="flex gap-1">
				<button
					class="px-3 py-1 rounded border text-sm disabled:opacity-50 disabled:cursor-not-allowed hover:bg-white"
					disabled={currentPage === 1}
					onclick={() => (currentPage = 1)}
				>
					First
				</button>
				<button
					class="px-3 py-1 rounded border text-sm disabled:opacity-50 disabled:cursor-not-allowed hover:bg-white"
					disabled={currentPage === 1}
					onclick={() => currentPage--}
				>
					Prev
				</button>
				<span class="px-3 py-1 text-sm">
					Page {currentPage} of {totalPages}
				</span>
				<button
					class="px-3 py-1 rounded border text-sm disabled:opacity-50 disabled:cursor-not-allowed hover:bg-white"
					disabled={currentPage === totalPages}
					onclick={() => currentPage++}
				>
					Next
				</button>
				<button
					class="px-3 py-1 rounded border text-sm disabled:opacity-50 disabled:cursor-not-allowed hover:bg-white"
					disabled={currentPage === totalPages}
					onclick={() => (currentPage = totalPages)}
				>
					Last
				</button>
			</div>
		</div>
	</div>
</div>
