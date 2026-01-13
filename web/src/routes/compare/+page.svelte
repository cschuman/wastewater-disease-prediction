<script lang="ts">
	import type { PageData } from './$types';
	import type { County } from '$lib/types/county';
	import { formatPopulation, formatNumber } from '$lib/utils/formatters';
	import { getSviColor, getTierColor } from '$lib/utils/colors';
	import { page } from '$app/stores';
	import { goto } from '$app/navigation';

	let { data }: { data: PageData } = $props();

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

	// State for selected county FIPS codes
	let county1Fips = $state('');
	let county2Fips = $state('');

	// Initialize from URL params on mount (client-side only)
	$effect(() => {
		const urlParams = $page.url.searchParams;
		const c1 = urlParams.get('county1');
		const c2 = urlParams.get('county2');
		if (c1 && county1Fips !== c1) county1Fips = c1;
		if (c2 && county2Fips !== c2) county2Fips = c2;
	});

	// Search state
	let search1 = $state('');
	let search2 = $state('');
	let showDropdown1 = $state(false);
	let showDropdown2 = $state(false);

	// Get selected counties
	const county1 = $derived((data.counties as County[]).find(c => c.fips === county1Fips));
	const county2 = $derived((data.counties as County[]).find(c => c.fips === county2Fips));

	// Filter counties for search
	const filteredCounties1 = $derived(() => {
		if (!search1) return [];
		const term = search1.toLowerCase();
		return (data.counties as County[])
			.filter(c =>
				c.county_name.toLowerCase().includes(term) ||
				(stateNames[c.state] || c.state).toLowerCase().includes(term) ||
				c.fips.includes(term)
			)
			.slice(0, 10);
	});

	const filteredCounties2 = $derived(() => {
		if (!search2) return [];
		const term = search2.toLowerCase();
		return (data.counties as County[])
			.filter(c =>
				c.county_name.toLowerCase().includes(term) ||
				(stateNames[c.state] || c.state).toLowerCase().includes(term) ||
				c.fips.includes(term)
			)
			.slice(0, 10);
	});

	function selectCounty1(county: County) {
		county1Fips = county.fips;
		search1 = '';
		showDropdown1 = false;
		updateUrl();
	}

	function selectCounty2(county: County) {
		county2Fips = county.fips;
		search2 = '';
		showDropdown2 = false;
		updateUrl();
	}

	function updateUrl() {
		const params = new URLSearchParams();
		if (county1Fips) params.set('county1', county1Fips);
		if (county2Fips) params.set('county2', county2Fips);
		goto(`/compare?${params.toString()}`, { replaceState: true });
	}

	function swapCounties() {
		const temp = county1Fips;
		county1Fips = county2Fips;
		county2Fips = temp;
		updateUrl();
	}

	function clearCounty1() {
		county1Fips = '';
		updateUrl();
	}

	function clearCounty2() {
		county2Fips = '';
		updateUrl();
	}

	// Helper for comparison coloring
	function getComparisonClass(val1: number | undefined, val2: number | undefined, higherIsBetter: boolean): { class1: string; class2: string } {
		if (val1 === undefined || val2 === undefined) return { class1: '', class2: '' };
		if (val1 === val2) return { class1: '', class2: '' };

		const val1Better = higherIsBetter ? val1 > val2 : val1 < val2;
		return {
			class1: val1Better ? 'text-green-600 font-semibold' : 'text-red-600',
			class2: val1Better ? 'text-red-600' : 'text-green-600 font-semibold'
		};
	}

	// Calculate sites per 100k for both counties
	const sitesPer100k1 = $derived(county1 ? (county1.n_sites / county1.population) * 100000 : 0);
	const sitesPer100k2 = $derived(county2 ? (county2.n_sites / county2.population) * 100000 : 0);

	// Pre-compute all comparison classes
	const popComp = $derived(getComparisonClass(county1?.population, county2?.population, true));
	const sitesComp = $derived(getComparisonClass(county1?.n_sites, county2?.n_sites, true));
	const per100kComp = $derived(getComparisonClass(sitesPer100k1, sitesPer100k2, true));
	const covComp = $derived(getComparisonClass(county1?.coverage_pct, county2?.coverage_pct, true));
	const sviComp = $derived(getComparisonClass(county1?.svi_overall, county2?.svi_overall, false));
	const socComp = $derived(getComparisonClass(county1?.svi_socioeconomic, county2?.svi_socioeconomic, false));
	const hhComp = $derived(getComparisonClass(county1?.svi_household, county2?.svi_household, false));
	const minComp = $derived(getComparisonClass(county1?.svi_minority, county2?.svi_minority, false));
	const housComp = $derived(getComparisonClass(county1?.svi_housing, county2?.svi_housing, false));
	const prioComp = $derived(getComparisonClass(county1?.priority_score, county2?.priority_score, true));
</script>

<svelte:head>
	<title>Compare Counties | Wastewater Surveillance Equity</title>
	<meta name="description" content="Compare wastewater surveillance coverage between two US counties side-by-side." />
</svelte:head>

<div class="space-y-6">
	<div>
		<h1 class="text-2xl font-bold">Compare Counties</h1>
		<p class="text-gray-600">Select two counties to compare their wastewater surveillance metrics side-by-side</p>
	</div>

	<!-- County Selectors -->
	<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
		<!-- County 1 Selector -->
		<div class="bg-white rounded-lg border p-4">
			<label class="block text-sm font-medium text-gray-700 mb-2">County 1</label>
			{#if county1}
				<div class="flex items-center justify-between bg-blue-50 rounded-lg p-3">
					<div>
						<span class="font-medium">{county1.county_name}</span>
						<span class="text-gray-500">, {stateNames[county1.state] || county1.state}</span>
					</div>
					<button onclick={clearCounty1} class="text-gray-400 hover:text-red-500 text-xl">&times;</button>
				</div>
			{:else}
				<div class="relative">
					<input
						type="text"
						bind:value={search1}
						onfocus={() => showDropdown1 = true}
						onblur={() => setTimeout(() => showDropdown1 = false, 200)}
						placeholder="Search by county name, state, or FIPS..."
						class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
					/>
					{#if showDropdown1 && filteredCounties1().length > 0}
						<div class="absolute z-10 w-full mt-1 bg-white border rounded-lg shadow-lg max-h-60 overflow-y-auto">
							{#each filteredCounties1() as county}
								<button
									onmousedown={() => selectCounty1(county)}
									class="w-full px-4 py-2 text-left hover:bg-gray-100 flex justify-between items-center"
								>
									<span>{county.county_name}, {stateNames[county.state] || county.state}</span>
									<span class="text-gray-400 text-sm">{county.fips}</span>
								</button>
							{/each}
						</div>
					{/if}
				</div>
			{/if}
		</div>

		<!-- County 2 Selector -->
		<div class="bg-white rounded-lg border p-4">
			<label class="block text-sm font-medium text-gray-700 mb-2">County 2</label>
			{#if county2}
				<div class="flex items-center justify-between bg-blue-50 rounded-lg p-3">
					<div>
						<span class="font-medium">{county2.county_name}</span>
						<span class="text-gray-500">, {stateNames[county2.state] || county2.state}</span>
					</div>
					<button onclick={clearCounty2} class="text-gray-400 hover:text-red-500 text-xl">&times;</button>
				</div>
			{:else}
				<div class="relative">
					<input
						type="text"
						bind:value={search2}
						onfocus={() => showDropdown2 = true}
						onblur={() => setTimeout(() => showDropdown2 = false, 200)}
						placeholder="Search by county name, state, or FIPS..."
						class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
					/>
					{#if showDropdown2 && filteredCounties2().length > 0}
						<div class="absolute z-10 w-full mt-1 bg-white border rounded-lg shadow-lg max-h-60 overflow-y-auto">
							{#each filteredCounties2() as county}
								<button
									onmousedown={() => selectCounty2(county)}
									class="w-full px-4 py-2 text-left hover:bg-gray-100 flex justify-between items-center"
								>
									<span>{county.county_name}, {stateNames[county.state] || county.state}</span>
									<span class="text-gray-400 text-sm">{county.fips}</span>
								</button>
							{/each}
						</div>
					{/if}
				</div>
			{/if}
		</div>
	</div>

	<!-- Swap button -->
	{#if county1 && county2}
		<div class="flex justify-center">
			<button
				onclick={swapCounties}
				class="px-4 py-2 text-sm bg-gray-100 hover:bg-gray-200 rounded-lg flex items-center gap-2"
			>
				<span>⇄</span> Swap Counties
			</button>
		</div>
	{/if}

	<!-- Comparison Table -->
	{#if county1 && county2}
		<div class="bg-white rounded-lg border overflow-hidden">
			<table class="w-full">
				<thead class="bg-gray-50 border-b">
					<tr>
						<th class="px-4 py-3 text-left font-medium text-gray-500 w-1/3">Metric</th>
						<th class="px-4 py-3 text-center font-medium text-gray-700 w-1/3">{county1.county_name}</th>
						<th class="px-4 py-3 text-center font-medium text-gray-700 w-1/3">{county2.county_name}</th>
					</tr>
				</thead>
				<tbody class="divide-y divide-gray-100">
					<!-- State -->
					<tr class="hover:bg-gray-50">
						<td class="px-4 py-3 text-gray-500">State</td>
						<td class="px-4 py-3 text-center">{stateNames[county1.state] || county1.state}</td>
						<td class="px-4 py-3 text-center">{stateNames[county2.state] || county2.state}</td>
					</tr>

					<!-- FIPS -->
					<tr class="hover:bg-gray-50">
						<td class="px-4 py-3 text-gray-500">FIPS Code</td>
						<td class="px-4 py-3 text-center font-mono">{county1.fips}</td>
						<td class="px-4 py-3 text-center font-mono">{county2.fips}</td>
					</tr>

					<!-- Population -->
					<tr class="hover:bg-gray-50">
						<td class="px-4 py-3 text-gray-500">Population</td>
						<td class="px-4 py-3 text-center {popComp.class1}">{formatPopulation(county1.population)}</td>
						<td class="px-4 py-3 text-center {popComp.class2}">{formatPopulation(county2.population)}</td>
					</tr>

					<!-- Monitoring Sites -->
					<tr class="hover:bg-gray-50">
						<td class="px-4 py-3 text-gray-500">Monitoring Sites</td>
						<td class="px-4 py-3 text-center {sitesComp.class1}">{county1.n_sites}</td>
						<td class="px-4 py-3 text-center {sitesComp.class2}">{county2.n_sites}</td>
					</tr>

					<!-- Sites per 100k -->
					<tr class="hover:bg-gray-50">
						<td class="px-4 py-3 text-gray-500">Sites per 100k</td>
						<td class="px-4 py-3 text-center font-mono {per100kComp.class1}">{sitesPer100k1.toFixed(2)}</td>
						<td class="px-4 py-3 text-center font-mono {per100kComp.class2}">{sitesPer100k2.toFixed(2)}</td>
					</tr>

					<!-- Coverage -->
					<tr class="hover:bg-gray-50">
						<td class="px-4 py-3 text-gray-500">Coverage %</td>
						<td class="px-4 py-3 text-center {covComp.class1}">{(county1.coverage_pct * 100).toFixed(1)}%</td>
						<td class="px-4 py-3 text-center {covComp.class2}">{(county2.coverage_pct * 100).toFixed(1)}%</td>
					</tr>

					<!-- SVI Overall -->
					<tr class="hover:bg-gray-50">
						<td class="px-4 py-3 text-gray-500">SVI Overall</td>
						<td class="px-4 py-3 text-center">
							<span class="inline-block w-3 h-3 rounded-full mr-2" style="background-color: {getSviColor(county1.svi_overall)}"></span>
							<span class="font-mono {sviComp.class1}">{county1.svi_overall.toFixed(3)}</span>
						</td>
						<td class="px-4 py-3 text-center">
							<span class="inline-block w-3 h-3 rounded-full mr-2" style="background-color: {getSviColor(county2.svi_overall)}"></span>
							<span class="font-mono {sviComp.class2}">{county2.svi_overall.toFixed(3)}</span>
						</td>
					</tr>

					<!-- SVI Quartile -->
					<tr class="hover:bg-gray-50">
						<td class="px-4 py-3 text-gray-500">SVI Quartile</td>
						<td class="px-4 py-3 text-center">
							<span class="px-2 py-1 rounded text-sm" style="background-color: {getSviColor(county1.svi_overall)}20; color: {getSviColor(county1.svi_overall)}">
								{county1.svi_quartile}
							</span>
						</td>
						<td class="px-4 py-3 text-center">
							<span class="px-2 py-1 rounded text-sm" style="background-color: {getSviColor(county2.svi_overall)}20; color: {getSviColor(county2.svi_overall)}">
								{county2.svi_quartile}
							</span>
						</td>
					</tr>

					<!-- SVI Theme: Socioeconomic -->
					{#if county1.svi_socioeconomic !== undefined && county2.svi_socioeconomic !== undefined}
						<tr class="hover:bg-gray-50">
							<td class="px-4 py-3 text-gray-500 pl-8">↳ Socioeconomic</td>
							<td class="px-4 py-3 text-center font-mono {socComp.class1}">{county1.svi_socioeconomic.toFixed(3)}</td>
							<td class="px-4 py-3 text-center font-mono {socComp.class2}">{county2.svi_socioeconomic.toFixed(3)}</td>
						</tr>
					{/if}

					<!-- SVI Theme: Household/Disability -->
					{#if county1.svi_household !== undefined && county2.svi_household !== undefined}
						<tr class="hover:bg-gray-50">
							<td class="px-4 py-3 text-gray-500 pl-8">↳ Household/Disability</td>
							<td class="px-4 py-3 text-center font-mono {hhComp.class1}">{county1.svi_household.toFixed(3)}</td>
							<td class="px-4 py-3 text-center font-mono {hhComp.class2}">{county2.svi_household.toFixed(3)}</td>
						</tr>
					{/if}

					<!-- SVI Theme: Minority/Language -->
					{#if county1.svi_minority !== undefined && county2.svi_minority !== undefined}
						<tr class="hover:bg-gray-50">
							<td class="px-4 py-3 text-gray-500 pl-8">↳ Minority/Language</td>
							<td class="px-4 py-3 text-center font-mono {minComp.class1}">{county1.svi_minority.toFixed(3)}</td>
							<td class="px-4 py-3 text-center font-mono {minComp.class2}">{county2.svi_minority.toFixed(3)}</td>
						</tr>
					{/if}

					<!-- SVI Theme: Housing/Transportation -->
					{#if county1.svi_housing !== undefined && county2.svi_housing !== undefined}
						<tr class="hover:bg-gray-50">
							<td class="px-4 py-3 text-gray-500 pl-8">↳ Housing/Transportation</td>
							<td class="px-4 py-3 text-center font-mono {housComp.class1}">{county1.svi_housing.toFixed(3)}</td>
							<td class="px-4 py-3 text-center font-mono {housComp.class2}">{county2.svi_housing.toFixed(3)}</td>
						</tr>
					{/if}

					<!-- Priority Score -->
					<tr class="hover:bg-gray-50">
						<td class="px-4 py-3 text-gray-500">Priority Score</td>
						<td class="px-4 py-3 text-center font-mono {prioComp.class1}">{county1.priority_score.toFixed(2)}</td>
						<td class="px-4 py-3 text-center font-mono {prioComp.class2}">{county2.priority_score.toFixed(2)}</td>
					</tr>

					<!-- Priority Tier -->
					<tr class="hover:bg-gray-50">
						<td class="px-4 py-3 text-gray-500">Priority Tier</td>
						<td class="px-4 py-3 text-center">
							<span class="px-2 py-1 rounded text-sm" style="background-color: {getTierColor(county1.priority_tier)}20; color: {getTierColor(county1.priority_tier)}">
								{county1.priority_tier}
							</span>
						</td>
						<td class="px-4 py-3 text-center">
							<span class="px-2 py-1 rounded text-sm" style="background-color: {getTierColor(county2.priority_tier)}20; color: {getTierColor(county2.priority_tier)}">
								{county2.priority_tier}
							</span>
						</td>
					</tr>

					<!-- Urbanization -->
					{#if county1.urbanization !== undefined && county2.urbanization !== undefined}
						<tr class="hover:bg-gray-50">
							<td class="px-4 py-3 text-gray-500">Urbanization</td>
							<td class="px-4 py-3 text-center">{county1.urbanization}</td>
							<td class="px-4 py-3 text-center">{county2.urbanization}</td>
						</tr>
					{/if}

					<!-- RUCC Code -->
					{#if county1.rucc_2023 !== undefined && county2.rucc_2023 !== undefined}
						<tr class="hover:bg-gray-50">
							<td class="px-4 py-3 text-gray-500">RUCC Code (2023)</td>
							<td class="px-4 py-3 text-center font-mono">{county1.rucc_2023}</td>
							<td class="px-4 py-3 text-center font-mono">{county2.rucc_2023}</td>
						</tr>
					{/if}
				</tbody>
			</table>
		</div>

		<!-- Links to individual county pages -->
		<div class="flex flex-col sm:flex-row gap-4 justify-center">
			<a
				href="/counties/{county1.fips}"
				class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-center"
			>
				View {county1.county_name} Details
			</a>
			<a
				href="/counties/{county2.fips}"
				class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-center"
			>
				View {county2.county_name} Details
			</a>
		</div>
	{:else if county1 || county2}
		<div class="bg-yellow-50 border border-yellow-200 rounded-lg p-6 text-center">
			<p class="text-yellow-800">Select a second county to compare</p>
		</div>
	{:else}
		<div class="bg-gray-50 border border-gray-200 rounded-lg p-8 text-center">
			<p class="text-gray-500 mb-4">Select two counties above to see a side-by-side comparison</p>
			<p class="text-sm text-gray-400">You can search by county name, state name, or FIPS code</p>
		</div>
	{/if}

	<!-- Legend -->
	{#if county1 && county2}
		<div class="bg-gray-50 rounded-lg p-4 text-sm text-gray-600">
			<p class="font-medium mb-2">Reading the comparison:</p>
			<ul class="list-disc list-inside space-y-1">
				<li><span class="text-green-600 font-semibold">Green</span> = Better value for this metric</li>
				<li><span class="text-red-600">Red</span> = Worse value for this metric</li>
				<li>Higher is better for: Population, Sites, Sites per 100k, Coverage, Priority Score</li>
				<li>Lower is better for: SVI scores (lower = less vulnerable)</li>
			</ul>
		</div>
	{/if}
</div>
