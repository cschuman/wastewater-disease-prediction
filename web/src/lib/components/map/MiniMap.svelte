<script lang="ts">
	import * as d3 from 'd3';
	import { feature } from 'topojson-client';
	import { getSviColor } from '$lib/utils/colors';
	import type { County } from '$lib/types';

	let { counties }: { counties: County[] } = $props();

	let svgNode: SVGSVGElement | null = null;
	let loading = $state(true);

	const countyMap = new Map(counties.map((c) => [c.fips, c]));

	function mapAction(node: SVGSVGElement) {
		svgNode = node;

		(async () => {
			try {
				const response = await fetch('/data/us-counties.json');
				const us = await response.json();

				const svg = d3.select(node);
				const path = d3.geoPath();

				const countiesGeo = feature(us, us.objects.counties) as any;

				svg
					.append('g')
					.selectAll('path')
					.data(countiesGeo.features)
					.join('path')
					.attr('d', path as any)
					.attr('fill', (d: any) => {
						const county = countyMap.get(d.id);
						return county ? getSviColor(county.svi_overall) : '#e5e7eb';
					})
					.attr('stroke', '#fff')
					.attr('stroke-width', 0.1);

				loading = false;
			} catch (e) {
				console.error('Failed to load mini map', e);
				loading = false;
			}
		})();
	}
</script>

<a href="/map" class="block relative group">
	<div class="bg-white rounded-lg border overflow-hidden">
		{#if loading}
			<div class="h-48 flex items-center justify-center text-gray-400">
				Loading map...
			</div>
		{/if}
		<svg use:mapAction viewBox="0 0 975 610" class="w-full h-auto"></svg>
		<div class="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors flex items-center justify-center">
			<span class="opacity-0 group-hover:opacity-100 transition-opacity bg-blue-600 text-white px-4 py-2 rounded-lg font-medium shadow-lg">
				Open Interactive Map
			</span>
		</div>
	</div>
	<div class="flex items-center justify-between mt-2">
		<span class="text-sm text-gray-600">Colored by Social Vulnerability Index</span>
		<span class="text-sm text-blue-600 group-hover:text-blue-800">View full map &rarr;</span>
	</div>
</a>
