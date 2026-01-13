<script lang="ts">
	import type { PageData } from './$types';
	import { MetricCard, Badge } from '$lib/components/ui';
	import { QuartileBarChart } from '$lib/components/charts';
	import { MiniMap } from '$lib/components/map';
	import { formatPopulation, formatNumber } from '$lib/utils/formatters';

	let { data }: { data: PageData } = $props();

	const { summary, quartileStats, counties } = data;

	// Get top 10 priority counties
	const topCounties = counties
		.slice()
		.sort((a, b) => b.priority_score - a.priority_score)
		.slice(0, 10);
</script>

<div class="space-y-8">
	<!-- Hero section -->
	<div class="bg-gradient-to-r from-red-600 to-amber-600 rounded-2xl p-8 text-white">
		<h1 class="text-3xl font-bold mb-2">Wastewater Surveillance Equity Gap</h1>
		<p class="text-lg text-white/90 max-w-2xl">
			High-vulnerability counties have <strong>{summary.disparity_pct}% fewer</strong> monitoring sites per capita
			than low-vulnerability counties. This dashboard explores the gap and paths to equity.
		</p>
	</div>

	<!-- Key metrics -->
	<div class="grid grid-cols-2 md:grid-cols-4 gap-4">
		<MetricCard
			label="US Counties"
			value={formatNumber(summary.total_counties)}
			subtext="{summary.coverage_pct}% have monitoring"
		/>
		<MetricCard
			label="Monitoring Sites"
			value={formatNumber(summary.total_sites)}
			subtext="Active wastewater sites"
		/>
		<MetricCard
			label="High-SVI Zero Coverage"
			value={formatNumber(summary.high_svi_zero_coverage)}
			subtext="Vulnerable counties with no monitoring"
			variant="danger"
		/>
		<MetricCard
			label="Per-Capita Gap"
			value="{(summary.q1_sites_per_million / summary.q4_sites_per_million).toFixed(1)}x"
			subtext="Q1 vs Q4 disparity ratio"
			variant="warning"
		/>
	</div>

	<!-- Two column layout -->
	<div class="grid md:grid-cols-2 gap-8">
		<!-- Coverage by SVI Quartile -->
		<div class="bg-white rounded-lg border p-6">
			<h2 class="text-lg font-semibold mb-2">Sites per Million by Vulnerability</h2>
			<p class="text-sm text-gray-500 mb-4">Higher SVI quartiles have fewer monitoring sites per capita</p>
			<QuartileBarChart data={quartileStats} />
		</div>

		<!-- Top Priority Counties -->
		<div class="bg-white rounded-lg border p-6">
			<div class="flex justify-between items-center mb-4">
				<h2 class="text-lg font-semibold">Top Priority Counties</h2>
				<a href="/counties" class="text-sm text-blue-600 hover:text-blue-800">View all &rarr;</a>
			</div>
			<div class="space-y-2">
				{#each topCounties as county, i}
					<div class="flex items-center justify-between py-2 border-b border-gray-100 last:border-0">
						<div class="flex items-center gap-3">
							<span class="text-sm text-gray-400 w-5">{i + 1}</span>
							<div>
								<p class="font-medium text-sm">{county.county_name}</p>
								<p class="text-xs text-gray-500">{county.state} &middot; {formatPopulation(county.population)}</p>
							</div>
						</div>
						<div class="flex items-center gap-2">
							<Badge text={county.svi_quartile} type="quartile" />
							<span class="text-sm font-medium">{county.priority_score.toFixed(0)}</span>
						</div>
					</div>
				{/each}
			</div>
		</div>
	</div>

	<!-- Mini Map Preview -->
	<div>
		<h2 class="text-lg font-semibold mb-4">National Overview</h2>
		<MiniMap {counties} />
	</div>

	<!-- Investment call to action -->
	<div class="bg-blue-50 border border-blue-200 rounded-lg p-6">
		<div class="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
			<div>
				<h2 class="text-lg font-semibold text-blue-900">Closing the Gap</h2>
				<p class="text-blue-700">
					An investment of <strong>$172M</strong> over 5 years could add 1,720 new sites to high-SVI counties.
				</p>
			</div>
			<a
				href="/scenarios"
				class="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
			>
				View Investment Scenarios
			</a>
		</div>
	</div>

	<!-- Quick links -->
	<div class="grid md:grid-cols-3 gap-4">
		<a href="/counties" class="group bg-white rounded-lg border p-6 hover:border-blue-300 transition-colors">
			<h3 class="font-semibold group-hover:text-blue-600">County Explorer</h3>
			<p class="text-sm text-gray-600 mt-1">Search and filter all 3,144 US counties</p>
		</a>
		<a href="/map" class="group bg-white rounded-lg border p-6 hover:border-blue-300 transition-colors">
			<h3 class="font-semibold group-hover:text-blue-600">Interactive Map</h3>
			<p class="text-sm text-gray-600 mt-1">Visualize equity gaps by geography</p>
		</a>
		<a href="/about" class="group bg-white rounded-lg border p-6 hover:border-blue-300 transition-colors">
			<h3 class="font-semibold group-hover:text-blue-600">Methodology</h3>
			<p class="text-sm text-gray-600 mt-1">Learn about the analysis approach</p>
		</a>
	</div>
</div>
