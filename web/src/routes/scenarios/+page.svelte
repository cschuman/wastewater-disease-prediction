<script lang="ts">
	import type { PageData } from './$types';
	import { formatCurrency, formatNumber } from '$lib/utils/formatters';

	let { data }: { data: PageData } = $props();
</script>

<div class="space-y-8">
	<div>
		<h1 class="text-2xl font-bold">Investment Scenarios</h1>
		<p class="text-gray-600 mt-1">
			Three policy scenarios to close the wastewater surveillance equity gap
		</p>
	</div>

	<!-- Scenario cards -->
	<div class="grid md:grid-cols-3 gap-6">
		{#each data.scenarios as scenario}
			<div
				class="bg-white rounded-lg border-2 p-6 {scenario.recommended
					? 'border-blue-500 ring-2 ring-blue-100'
					: 'border-gray-200'}"
			>
				{#if scenario.recommended}
					<span class="inline-block px-2 py-0.5 bg-blue-100 text-blue-700 text-xs font-medium rounded mb-3">
						Recommended
					</span>
				{/if}
				<h2 class="text-xl font-bold">{scenario.name}</h2>
				<p class="text-gray-600 text-sm mt-1">{scenario.description}</p>

				<div class="mt-6 space-y-4">
					<div>
						<p class="text-sm text-gray-500">New Sites</p>
						<p class="text-2xl font-bold">{formatNumber(scenario.new_sites)}</p>
					</div>
					<div>
						<p class="text-sm text-gray-500">Setup Cost</p>
						<p class="text-2xl font-bold">{formatCurrency(scenario.setup_cost_millions)}</p>
					</div>
					<div>
						<p class="text-sm text-gray-500">5-Year Total Cost</p>
						<p class="text-2xl font-bold">{formatCurrency(scenario.five_year_cost_millions)}</p>
					</div>
					<div>
						<p class="text-sm text-gray-500">Timeline</p>
						<p class="text-lg font-medium">{scenario.timeline}</p>
					</div>
				</div>
			</div>
		{/each}
	</div>

	<!-- Cost comparison chart -->
	<div class="bg-white rounded-lg border p-6">
		<h2 class="text-lg font-semibold mb-4">Cost Comparison</h2>
		<div class="space-y-4">
			{#each data.scenarios as scenario}
				<div>
					<div class="flex justify-between text-sm mb-1">
						<span class="font-medium">{scenario.name}</span>
						<span>{formatCurrency(scenario.five_year_cost_millions)}</span>
					</div>
					<div class="h-4 bg-gray-100 rounded-full overflow-hidden">
						<div
							class="h-full rounded-full transition-all {scenario.recommended ? 'bg-blue-500' : 'bg-gray-400'}"
							style="width: {(scenario.five_year_cost_millions / 1000) * 100}%"
						></div>
					</div>
				</div>
			{/each}
		</div>
	</div>

	<!-- Implementation roadmap -->
	<div class="bg-white rounded-lg border p-6">
		<h2 class="text-lg font-semibold mb-4">Implementation Roadmap (Scenario B)</h2>
		<div class="space-y-6">
			<div class="flex gap-4">
				<div class="flex-shrink-0 w-8 h-8 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center font-bold">
					1
				</div>
				<div>
					<h3 class="font-semibold">Phase 1: Emergency Priority (Year 1)</h3>
					<p class="text-gray-600 text-sm">Target 1,259 high-SVI counties with zero monitoring. Deploy ~602 new sites.</p>
					<p class="text-sm text-blue-600 mt-1">Budget: $60M setup</p>
				</div>
			</div>
			<div class="flex gap-4">
				<div class="flex-shrink-0 w-8 h-8 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center font-bold">
					2
				</div>
				<div>
					<h3 class="font-semibold">Phase 2: Expand Coverage (Years 2-3)</h3>
					<p class="text-gray-600 text-sm">Target Q3/Q4 counties with below-target coverage. Deploy ~688 new sites.</p>
					<p class="text-sm text-blue-600 mt-1">Budget: $69M setup</p>
				</div>
			</div>
			<div class="flex gap-4">
				<div class="flex-shrink-0 w-8 h-8 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center font-bold">
					3
				</div>
				<div>
					<h3 class="font-semibold">Phase 3: Close Remaining Gaps (Years 4-5)</h3>
					<p class="text-gray-600 text-sm">Target all remaining counties below national average. Deploy ~430 new sites.</p>
					<p class="text-sm text-blue-600 mt-1">Budget: $43M setup</p>
				</div>
			</div>
		</div>
	</div>

	<!-- ROI callout -->
	<div class="bg-green-50 border border-green-200 rounded-lg p-6">
		<h2 class="text-lg font-semibold text-green-900">Return on Investment</h2>
		<p class="text-green-700 mt-2">
			For every <strong>$1 invested</strong>, save <strong>$5-15</strong> in outbreak response and disease burden costs.
		</p>
		<ul class="mt-4 space-y-2 text-sm text-green-700">
			<li>Early detection for 141M people in underserved areas</li>
			<li>Reduced outbreak response costs ($50-500M per major outbreak)</li>
			<li>Wastewater surveillance is 10-100x more cost-effective than clinical testing at scale</li>
		</ul>
	</div>

	<!-- Top priority states -->
	<div class="bg-white rounded-lg border p-6">
		<h2 class="text-lg font-semibold mb-4">Top Priority States for Investment</h2>
		<div class="overflow-x-auto">
			<table class="w-full text-sm">
				<thead class="bg-gray-50">
					<tr>
						<th class="px-4 py-2 text-left">Rank</th>
						<th class="px-4 py-2 text-left">State</th>
						<th class="px-4 py-2 text-right">New Sites</th>
						<th class="px-4 py-2 text-right">Setup Cost</th>
						<th class="px-4 py-2 text-right">Counties Needing Sites</th>
					</tr>
				</thead>
				<tbody class="divide-y">
					{#each [
						{ rank: 1, state: 'Texas', sites: 225, cost: 22.5, counties: 193 },
						{ rank: 2, state: 'California', sites: 157, cost: 15.7, counties: 45 },
						{ rank: 3, state: 'Georgia', sites: 126, cost: 12.6, counties: 117 },
						{ rank: 4, state: 'Florida', sites: 97, cost: 9.7, counties: 55 },
						{ rank: 5, state: 'Mississippi', sites: 76, cost: 7.6, counties: 75 },
						{ rank: 6, state: 'Kentucky', sites: 73, cost: 7.3, counties: 71 },
						{ rank: 7, state: 'North Carolina', sites: 71, cost: 7.1, counties: 60 },
						{ rank: 8, state: 'Louisiana', sites: 65, cost: 6.5, counties: 55 },
						{ rank: 9, state: 'Oklahoma', sites: 63, cost: 6.3, counties: 57 },
						{ rank: 10, state: 'Arkansas', sites: 62, cost: 6.2, counties: 58 }
					] as item}
						<tr class="hover:bg-gray-50">
							<td class="px-4 py-2 text-gray-500">{item.rank}</td>
							<td class="px-4 py-2 font-medium">{item.state}</td>
							<td class="px-4 py-2 text-right">{item.sites}</td>
							<td class="px-4 py-2 text-right">${item.cost}M</td>
							<td class="px-4 py-2 text-right">{item.counties}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	</div>
</div>
