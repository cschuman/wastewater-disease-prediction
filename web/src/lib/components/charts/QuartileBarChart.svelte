<script lang="ts">
	import { scaleLinear, scaleBand } from 'd3-scale';

	interface QuartileStat {
		svi_quartile: string;
		sites_per_million: number;
		population: number;
		county_count: number;
	}

	let { data }: { data: QuartileStat[] } = $props();

	const colors: Record<string, string> = {
		'Q1 (Low)': '#2563eb',
		'Q2': '#16a34a',
		'Q3': '#f59e0b',
		'Q4 (High)': '#dc2626'
	};

	// Chart dimensions
	const width = 400;
	const height = 200;
	const margin = { top: 10, right: 50, bottom: 25, left: 75 };
	const innerWidth = width - margin.left - margin.right;
	const innerHeight = height - margin.top - margin.bottom;

	// Guard against empty data (Math.max on empty array returns -Infinity)
	const maxValue = data.length > 0 ? Math.max(...data.map((d) => d.sites_per_million)) : 0;

	const xScale = scaleLinear()
		.domain([0, maxValue * 1.15])
		.range([0, innerWidth]);

	const yScale = scaleBand<string>()
		.domain(data.map((d) => d.svi_quartile))
		.range([0, innerHeight])
		.padding(0.3);

	const ticks = xScale.ticks(5);
</script>

<div class="w-full">
	<svg viewBox="0 0 {width} {height}" class="w-full h-auto">
		<g transform="translate({margin.left}, {margin.top})">
			<!-- Grid lines -->
			{#each ticks as tick}
				<line
					x1={xScale(tick)}
					x2={xScale(tick)}
					y1={0}
					y2={innerHeight}
					stroke="#e5e7eb"
					stroke-dasharray="4,4"
				/>
			{/each}

			<!-- Bars -->
			{#each data as d}
				<rect
					x={0}
					y={yScale(d.svi_quartile)}
					width={xScale(d.sites_per_million)}
					height={yScale.bandwidth()}
					fill={colors[d.svi_quartile] || '#6b7280'}
					rx={4}
				/>
				<!-- Value label -->
				<text
					x={xScale(d.sites_per_million) + 6}
					y={(yScale(d.svi_quartile) ?? 0) + yScale.bandwidth() / 2}
					dominant-baseline="middle"
					font-size="11"
					font-weight="600"
					fill="#374151"
				>
					{d.sites_per_million.toFixed(2)}
				</text>
			{/each}

			<!-- Y axis labels -->
			{#each data as d}
				<text
					x={-8}
					y={(yScale(d.svi_quartile) ?? 0) + yScale.bandwidth() / 2}
					text-anchor="end"
					dominant-baseline="middle"
					font-size="11"
					fill="#4b5563"
				>
					{d.svi_quartile}
				</text>
			{/each}

			<!-- X axis -->
			<line x1={0} x2={innerWidth} y1={innerHeight} y2={innerHeight} stroke="#d1d5db" />
			{#each ticks as tick}
				<text
					x={xScale(tick)}
					y={innerHeight + 16}
					text-anchor="middle"
					font-size="10"
					fill="#6b7280"
				>
					{tick}
				</text>
			{/each}
		</g>
	</svg>
	<p class="text-xs text-gray-500 text-center mt-1">Sites per million population</p>
</div>
