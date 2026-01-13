<script lang="ts">
	import type { PageData } from './$types';
	import { Badge, MetricCard } from '$lib/components/ui/index';
	import { formatPopulation, formatNumber } from '$lib/utils/formatters';

	let { data }: { data: PageData } = $props();
	const county = data.county;
</script>

<svelte:head>
	<title>{county.county_name}, {county.state} | WW Equity</title>
</svelte:head>

<div class="space-y-6">
	<!-- Back link -->
	<a href="/counties" class="inline-flex items-center text-sm text-gray-600 hover:text-gray-900">
		&larr; Back to County Explorer
	</a>

	<!-- Header -->
	<div class="bg-white rounded-lg border p-6">
		<div class="flex flex-col md:flex-row justify-between gap-4">
			<div>
				<h1 class="text-2xl font-bold">{county.county_name}</h1>
				<p class="text-gray-600">{county.state} &middot; FIPS: {county.fips}</p>
			</div>
			<div class="flex gap-2">
				<Badge text={county.svi_quartile} type="quartile" />
				<Badge text={county.priority_tier} type="tier" />
			</div>
		</div>
	</div>

	<!-- Key metrics -->
	<div class="grid grid-cols-2 md:grid-cols-4 gap-4">
		<MetricCard label="Population" value={formatPopulation(county.population)} />
		<MetricCard
			label="SVI Score"
			value={county.svi_overall.toFixed(2)}
			subtext="Social Vulnerability Index"
		/>
		<MetricCard
			label="Monitoring Sites"
			value={county.n_sites}
			variant={county.n_sites === 0 ? 'danger' : 'default'}
		/>
		<MetricCard
			label="Priority Score"
			value={county.priority_score.toFixed(1)}
			subtext="out of 100"
			variant={county.priority_score > 50 ? 'danger' : county.priority_score > 30 ? 'warning' : 'default'}
		/>
	</div>

	<!-- Details -->
	<div class="grid md:grid-cols-2 gap-6">
		<!-- Coverage details -->
		<div class="bg-white rounded-lg border p-6">
			<h2 class="text-lg font-semibold mb-4">Surveillance Coverage</h2>
			<dl class="space-y-3">
				<div class="flex justify-between">
					<dt class="text-gray-600">Coverage Percentage</dt>
					<dd class="font-medium">{county.coverage_pct.toFixed(1)}%</dd>
				</div>
				<div class="flex justify-between">
					<dt class="text-gray-600">Number of Sites</dt>
					<dd class="font-medium">{county.n_sites}</dd>
				</div>
				<div class="flex justify-between">
					<dt class="text-gray-600">Priority Tier</dt>
					<dd><Badge text={county.priority_tier} type="tier" /></dd>
				</div>
			</dl>

			{#if county.n_sites === 0}
				<div class="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
					<p class="text-sm text-red-700">
						This county currently has <strong>no wastewater monitoring coverage</strong>.
						It is a priority candidate for surveillance expansion.
					</p>
				</div>
			{/if}
		</div>

		<!-- Vulnerability details -->
		<div class="bg-white rounded-lg border p-6">
			<h2 class="text-lg font-semibold mb-4">Social Vulnerability</h2>
			<dl class="space-y-3">
				<div class="flex justify-between">
					<dt class="text-gray-600">Overall SVI</dt>
					<dd class="font-medium">{county.svi_overall.toFixed(3)}</dd>
				</div>
				<div class="flex justify-between">
					<dt class="text-gray-600">SVI Quartile</dt>
					<dd><Badge text={county.svi_quartile} type="quartile" /></dd>
				</div>
				{#if county.svi_socioeconomic !== undefined}
					<div class="flex justify-between">
						<dt class="text-gray-600">Socioeconomic</dt>
						<dd class="font-medium">{county.svi_socioeconomic.toFixed(3)}</dd>
					</div>
				{/if}
				{#if county.svi_household !== undefined}
					<div class="flex justify-between">
						<dt class="text-gray-600">Household Composition</dt>
						<dd class="font-medium">{county.svi_household.toFixed(3)}</dd>
					</div>
				{/if}
				{#if county.svi_minority !== undefined}
					<div class="flex justify-between">
						<dt class="text-gray-600">Minority Status</dt>
						<dd class="font-medium">{county.svi_minority.toFixed(3)}</dd>
					</div>
				{/if}
				{#if county.svi_housing !== undefined}
					<div class="flex justify-between">
						<dt class="text-gray-600">Housing/Transportation</dt>
						<dd class="font-medium">{county.svi_housing.toFixed(3)}</dd>
					</div>
				{/if}
			</dl>
		</div>
	</div>

	<!-- Geographic details -->
	{#if county.metro_size || county.urbanization}
		<div class="bg-white rounded-lg border p-6">
			<h2 class="text-lg font-semibold mb-4">Geography</h2>
			<dl class="grid md:grid-cols-3 gap-4">
				{#if county.metro_size}
					<div>
						<dt class="text-gray-600 text-sm">Metro Size</dt>
						<dd class="font-medium">{county.metro_size}</dd>
					</div>
				{/if}
				{#if county.urbanization}
					<div>
						<dt class="text-gray-600 text-sm">Classification</dt>
						<dd class="font-medium">{county.urbanization}</dd>
					</div>
				{/if}
				{#if county.rucc_2023}
					<div>
						<dt class="text-gray-600 text-sm">RUCC Code</dt>
						<dd class="font-medium">{county.rucc_2023}</dd>
					</div>
				{/if}
			</dl>
		</div>
	{/if}
</div>
