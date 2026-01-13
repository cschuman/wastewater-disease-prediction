<script lang="ts">
	import type { PageData } from './$types';
	import * as d3 from 'd3';
	import { feature, mesh } from 'topojson-client';
	import { getSviColor, getPriorityColor } from '$lib/utils/colors';
	import { formatPopulation } from '$lib/utils/formatters';
	import type { County } from '$lib/types';

	let { data }: { data: PageData } = $props();

	type ColorBy = 'svi' | 'priority' | 'sites';
	let colorBy = $state<ColorBy>('svi');

	let svgNode: SVGSVGElement | null = null;
	let tooltip = $state({ visible: false, x: 0, y: 0, county: null as County | null });
	let loading = $state(true);
	let error = $state<string | null>(null);

	// State and county search
	let selectedState = $state<string>('');
	let countySearch = $state('');
	let showCountyDropdown = $state(false);
	let geoData: any = null; // Store geo data for zoom calculations

	// Build county lookup map from data
	const countyMap = new Map(data.counties.map((c) => [c.fips, c]));

	// Get unique states sorted alphabetically
	const states = [...new Set(data.counties.map((c) => c.state))].sort();

	// Filter counties for search
	const filteredCounties = $derived(
		countySearch.length >= 2
			? data.counties
					.filter(
						(c) =>
							c.county_name.toLowerCase().includes(countySearch.toLowerCase()) ||
							c.state.toLowerCase().includes(countySearch.toLowerCase())
					)
					.slice(0, 10)
			: []
	);

	function getColor(county: County | undefined): string {
		if (!county) return '#e5e7eb';
		switch (colorBy) {
			case 'svi':
				return getSviColor(county.svi_overall);
			case 'priority':
				return getPriorityColor(county.priority_score);
			case 'sites':
				return county.n_sites > 0 ? '#2563eb' : '#fee2e2';
			default:
				return '#e5e7eb';
		}
	}

	let zoomBehavior: d3.ZoomBehavior<SVGSVGElement, unknown> | null = null;

	function zoomIn() {
		if (svgNode && zoomBehavior) {
			d3.select(svgNode).transition().duration(300).call(zoomBehavior.scaleBy, 1.5);
		}
	}

	function zoomOut() {
		if (svgNode && zoomBehavior) {
			d3.select(svgNode).transition().duration(300).call(zoomBehavior.scaleBy, 0.67);
		}
	}

	function resetZoom() {
		if (svgNode && zoomBehavior) {
			d3.select(svgNode).transition().duration(300).call(zoomBehavior.transform, d3.zoomIdentity);
		}
	}

	// State FIPS code mapping (first 2 digits of county FIPS)
	const stateFipsMap: Record<string, string> = {
		'Alabama': '01', 'Alaska': '02', 'Arizona': '04', 'Arkansas': '05', 'California': '06',
		'Colorado': '08', 'Connecticut': '09', 'Delaware': '10', 'District of Columbia': '11',
		'Florida': '12', 'Georgia': '13', 'Hawaii': '15', 'Idaho': '16', 'Illinois': '17',
		'Indiana': '18', 'Iowa': '19', 'Kansas': '20', 'Kentucky': '21', 'Louisiana': '22',
		'Maine': '23', 'Maryland': '24', 'Massachusetts': '25', 'Michigan': '26', 'Minnesota': '27',
		'Mississippi': '28', 'Missouri': '29', 'Montana': '30', 'Nebraska': '31', 'Nevada': '32',
		'New Hampshire': '33', 'New Jersey': '34', 'New Mexico': '35', 'New York': '36',
		'North Carolina': '37', 'North Dakota': '38', 'Ohio': '39', 'Oklahoma': '40', 'Oregon': '41',
		'Pennsylvania': '42', 'Puerto Rico': '72', 'Rhode Island': '44', 'South Carolina': '45',
		'South Dakota': '46', 'Tennessee': '47', 'Texas': '48', 'Utah': '49', 'Vermont': '50',
		'Virginia': '51', 'Washington': '53', 'West Virginia': '54', 'Wisconsin': '55', 'Wyoming': '56'
	};

	function zoomToState(stateName: string) {
		if (!svgNode || !zoomBehavior || !geoData) return;

		const stateFips = stateFipsMap[stateName];
		if (!stateFips) return;

		// Get all county features for this state
		const stateCounties = geoData.features.filter((f: any) => f.id?.startsWith(stateFips));
		if (stateCounties.length === 0) return;

		// Calculate bounding box
		const path = d3.geoPath();
		let bounds = path.bounds({ type: 'FeatureCollection', features: stateCounties });

		const [[x0, y0], [x1, y1]] = bounds;
		const dx = x1 - x0;
		const dy = y1 - y0;
		const x = (x0 + x1) / 2;
		const y = (y0 + y1) / 2;

		// Calculate scale with padding
		const scale = Math.min(8, 0.9 / Math.max(dx / 975, dy / 610));
		const translate = [975 / 2 - scale * x, 610 / 2 - scale * y];

		d3.select(svgNode)
			.transition()
			.duration(750)
			.call(
				zoomBehavior.transform,
				d3.zoomIdentity.translate(translate[0], translate[1]).scale(scale)
			);
	}

	function zoomToCounty(county: County) {
		if (!svgNode || !zoomBehavior || !geoData) return;

		// Find the county feature
		const countyFeature = geoData.features.find((f: any) => f.id === county.fips);
		if (!countyFeature) return;

		const path = d3.geoPath();
		const bounds = path.bounds(countyFeature);
		const [[x0, y0], [x1, y1]] = bounds;
		const dx = x1 - x0;
		const dy = y1 - y0;
		const x = (x0 + x1) / 2;
		const y = (y0 + y1) / 2;

		// Zoom to county with more zoom
		const scale = Math.min(8, 0.5 / Math.max(dx / 975, dy / 610));
		const translate = [975 / 2 - scale * x, 610 / 2 - scale * y];

		d3.select(svgNode)
			.transition()
			.duration(750)
			.call(
				zoomBehavior.transform,
				d3.zoomIdentity.translate(translate[0], translate[1]).scale(scale)
			);

		// Clear search
		countySearch = '';
		showCountyDropdown = false;
	}

	function handleStateChange(e: Event) {
		const value = (e.target as HTMLSelectElement).value;
		selectedState = value;
		if (value) {
			zoomToState(value);
		} else {
			resetZoom();
		}
	}

	function selectCounty(county: County) {
		zoomToCounty(county);
	}

	// Svelte action to render the map when SVG is mounted
	function mapAction(node: SVGSVGElement) {
		svgNode = node;
		let svg: d3.Selection<SVGSVGElement, unknown, null, undefined>;

		(async () => {
			try {
				const response = await fetch('/data/us-counties.json');
				if (!response.ok) {
					throw new Error(`Failed to fetch map: ${response.status}`);
				}
				const us = await response.json();

				svg = d3.select(node);
				const path = d3.geoPath();

				// Create a group for all map content (for zooming)
				const g = svg.append('g').attr('class', 'map-content');

				// Draw counties
				const counties = feature(us, us.objects.counties) as any;
				geoData = counties; // Store for zoom calculations

				g.append('g')
					.attr('class', 'counties')
					.selectAll('path')
					.data(counties.features)
					.join('path')
					.attr('d', path as any)
					.attr('fill', (d: any) => getColor(countyMap.get(d.id)))
					.attr('stroke', '#fff')
					.attr('stroke-width', 0.2)
					.style('cursor', 'pointer')
					.on('mouseenter', function (event: MouseEvent, d: any) {
						const county = countyMap.get(d.id);
						d3.select(this).attr('stroke', '#000').attr('stroke-width', 1);
						tooltip = {
							visible: true,
							x: event.pageX,
							y: event.pageY,
							county: county || null
						};
					})
					.on('mousemove', (event: MouseEvent) => {
						tooltip.x = event.pageX;
						tooltip.y = event.pageY;
					})
					.on('mouseleave', function () {
						d3.select(this).attr('stroke', '#fff').attr('stroke-width', 0.2);
						tooltip.visible = false;
					})
					.on('click', (event: MouseEvent, d: any) => {
						const county = countyMap.get(d.id);
						if (county) {
							window.location.href = `/counties/${county.fips}`;
						}
					});

				// Draw state borders
				const states = mesh(us, us.objects.states, (a: any, b: any) => a !== b);
				g.append('path')
					.attr('class', 'state-borders')
					.datum(states)
					.attr('fill', 'none')
					.attr('stroke', '#374151')
					.attr('stroke-width', 0.5)
					.attr('d', path as any);

				// Use vector-effect for non-scaling strokes (keeps stroke width consistent)
				g.selectAll('.counties path').attr('vector-effect', 'non-scaling-stroke');
				g.selectAll('.state-borders').attr('vector-effect', 'non-scaling-stroke');

				// Setup zoom behavior with optimized performance
				zoomBehavior = d3.zoom<SVGSVGElement, unknown>()
					.scaleExtent([1, 8])
					.filter((event) => {
						// Allow all touch events (pinch, pan)
						if (event.type.startsWith('touch')) return true;
						// Allow double-click to zoom
						if (event.type === 'dblclick') return true;
						// Allow drag for panning
						if (event.type === 'mousedown') return true;
						// For wheel events, require ctrl/cmd key on desktop
						if (event.type === 'wheel') {
							return event.ctrlKey || event.metaKey;
						}
						return false;
					})
					.on('zoom', (event) => {
						g.attr('transform', event.transform);
					});

				svg.call(zoomBehavior);

				// Disable double-tap-to-zoom default behavior on mobile
				svg.on('touchstart', (event) => {
					if (event.touches.length > 1) {
						event.preventDefault();
					}
				}, { passive: false });

				loading = false;
			} catch (e) {
				error = 'Failed to load map data';
				loading = false;
				console.error(e);
			}
		})();

		// Return cleanup function to prevent memory leaks
		return {
			destroy() {
				if (svg) {
					// Remove all event listeners
					svg.selectAll('.counties path')
						.on('mouseenter', null)
						.on('mousemove', null)
						.on('mouseleave', null)
						.on('click', null);
					svg.on('touchstart', null);

					// Remove zoom behavior
					if (zoomBehavior) {
						svg.on('.zoom', null);
					}
				}
				svgNode = null;
				geoData = null;
				zoomBehavior = null;
			}
		};
	}

	// Update colors when colorBy changes
	$effect(() => {
		colorBy; // Track dependency
		if (svgNode) {
			d3.select(svgNode)
				.select('.map-content')
				.selectAll('.counties path')
				.attr('fill', (d: any) => getColor(countyMap.get(d.id)));
		}
	});
</script>

<div class="space-y-4">
	<div class="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4">
		<div>
			<h1 class="text-2xl font-bold">Interactive Map</h1>
			<p class="text-gray-600">Click on a county to view details</p>
		</div>

		<div class="flex flex-wrap items-center gap-4">
			<!-- State filter -->
			<div class="flex items-center gap-2">
				<label for="state-select" class="text-sm text-gray-600">State:</label>
				<select
					id="state-select"
					bind:value={selectedState}
					onchange={handleStateChange}
					class="px-3 py-1.5 text-sm border rounded-lg bg-white"
				>
					<option value="">All States</option>
					{#each states as state}
						<option value={state}>{state}</option>
					{/each}
				</select>
			</div>

			<!-- County search -->
			<div class="relative">
				<input
					type="text"
					placeholder="Search county..."
					bind:value={countySearch}
					onfocus={() => (showCountyDropdown = true)}
					onblur={() => setTimeout(() => (showCountyDropdown = false), 200)}
					class="px-3 py-1.5 text-sm border rounded-lg w-48"
				/>
				{#if showCountyDropdown && filteredCounties.length > 0}
					<div class="absolute top-full left-0 right-0 mt-1 bg-white border rounded-lg shadow-lg z-50 max-h-64 overflow-y-auto">
						{#each filteredCounties as county}
							<button
								type="button"
								class="w-full px-3 py-2 text-left hover:bg-gray-50 text-sm"
								onmousedown={() => selectCounty(county)}
							>
								<span class="font-medium">{county.county_name}</span>
								<span class="text-gray-500">, {county.state}</span>
							</button>
						{/each}
					</div>
				{/if}
			</div>

			<!-- Color by selector -->
			<div class="flex items-center gap-2">
				<span class="text-sm text-gray-600">Color by:</span>
				<div class="flex rounded-lg border overflow-hidden">
					<button
						class="px-3 py-1.5 text-sm {colorBy === 'svi'
							? 'bg-blue-600 text-white'
							: 'bg-white text-gray-600 hover:bg-gray-50'}"
						onclick={() => (colorBy = 'svi')}
					>
						SVI
					</button>
					<button
						class="px-3 py-1.5 text-sm border-l {colorBy === 'priority'
							? 'bg-blue-600 text-white'
							: 'bg-white text-gray-600 hover:bg-gray-50'}"
						onclick={() => (colorBy = 'priority')}
					>
						Priority
					</button>
					<button
						class="px-3 py-1.5 text-sm border-l {colorBy === 'sites'
						? 'bg-blue-600 text-white'
						: 'bg-white text-gray-600 hover:bg-gray-50'}"
					onclick={() => (colorBy = 'sites')}
				>
						Has Sites
				</button>
				</div>
			</div>
		</div>
	</div>

	<!-- Legend -->
	<div class="bg-white rounded-lg border p-4">
		{#if colorBy === 'svi'}
			<div class="flex items-center gap-4">
				<span class="text-sm text-gray-600">Social Vulnerability:</span>
				<div class="flex items-center gap-1">
					<div class="w-6 h-4 rounded" style="background: {getSviColor(0)}"></div>
					<span class="text-xs">Low (0)</span>
				</div>
				<div class="flex-1 h-4 rounded" style="background: linear-gradient(to right, {getSviColor(0)}, {getSviColor(0.5)}, {getSviColor(1)})"></div>
				<div class="flex items-center gap-1">
					<div class="w-6 h-4 rounded" style="background: {getSviColor(1)}"></div>
					<span class="text-xs">High (1)</span>
				</div>
			</div>
		{:else if colorBy === 'priority'}
			<div class="flex items-center gap-4">
				<span class="text-sm text-gray-600">Priority Score:</span>
				<div class="flex items-center gap-1">
					<div class="w-6 h-4 rounded" style="background: {getPriorityColor(0)}"></div>
					<span class="text-xs">Low (0)</span>
				</div>
				<div class="flex-1 h-4 rounded" style="background: linear-gradient(to right, {getPriorityColor(0)}, {getPriorityColor(50)}, {getPriorityColor(100)})"></div>
				<div class="flex items-center gap-1">
					<div class="w-6 h-4 rounded" style="background: {getPriorityColor(100)}"></div>
					<span class="text-xs">High (100)</span>
				</div>
			</div>
		{:else}
			<div class="flex items-center gap-4">
				<span class="text-sm text-gray-600">Monitoring Status:</span>
				<div class="flex items-center gap-2">
					<div class="w-6 h-4 rounded bg-blue-600"></div>
					<span class="text-xs">Has Sites</span>
				</div>
				<div class="flex items-center gap-2">
					<div class="w-6 h-4 rounded bg-red-100"></div>
					<span class="text-xs">No Sites</span>
				</div>
			</div>
		{/if}
	</div>

	<!-- Map container -->
	<div class="bg-white rounded-lg border overflow-hidden relative">
		{#if loading}
			<div class="absolute inset-0 flex items-center justify-center h-[600px] text-gray-500 bg-white z-10">
				Loading map...
			</div>
		{/if}
		{#if error}
			<div class="flex items-center justify-center h-[600px] text-red-500">
				{error}
			</div>
		{:else}
			<svg use:mapAction viewBox="0 0 975 610" class="w-full h-auto" style="will-change: transform;"></svg>
			<!-- Zoom controls -->
			<div class="absolute bottom-4 right-4 flex flex-col gap-1" role="group" aria-label="Map zoom controls">
				<button
					onclick={zoomIn}
					class="w-8 h-8 bg-white border rounded shadow hover:bg-gray-50 flex items-center justify-center text-lg font-bold"
					title="Zoom in"
					aria-label="Zoom in"
				>+</button>
				<button
					onclick={zoomOut}
					class="w-8 h-8 bg-white border rounded shadow hover:bg-gray-50 flex items-center justify-center text-lg font-bold"
					title="Zoom out"
					aria-label="Zoom out"
				>−</button>
				<button
					onclick={resetZoom}
					class="w-8 h-8 bg-white border rounded shadow hover:bg-gray-50 flex items-center justify-center text-xs"
					title="Reset zoom"
					aria-label="Reset zoom to default view"
				>Reset</button>
			</div>
			<p class="absolute bottom-4 left-4 text-xs text-gray-500 hidden md:block">Ctrl+scroll to zoom, drag to pan</p>
			<p class="absolute bottom-4 left-4 text-xs text-gray-500 md:hidden">Pinch to zoom, drag to pan</p>
		{/if}
	</div>

	<!-- Tooltip -->
	{#if tooltip.visible && tooltip.county}
		<div
			class="fixed z-50 bg-white rounded-lg shadow-lg border p-3 pointer-events-none"
			style="left: {tooltip.x + 10}px; top: {tooltip.y + 10}px;"
		>
			<p class="font-semibold">{tooltip.county.county_name}</p>
			<p class="text-sm text-gray-600">{tooltip.county.state}</p>
			<div class="mt-2 text-sm space-y-1">
				<p>Population: {formatPopulation(tooltip.county.population)}</p>
				<p>SVI: {tooltip.county.svi_overall.toFixed(2)} ({tooltip.county.svi_quartile})</p>
				<p>Sites: {tooltip.county.n_sites}</p>
				<p>Priority: {tooltip.county.priority_score.toFixed(1)}</p>
			</div>
		</div>
	{/if}
</div>
