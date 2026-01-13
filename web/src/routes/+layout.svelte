<script lang="ts">
	import '../app.css';
	import type { LayoutData } from './$types';

	let { data, children }: { data: LayoutData; children: any } = $props();

	const navItems = [
		{ href: '/', label: 'Dashboard' },
		{ href: '/states', label: 'States' },
		{ href: '/counties', label: 'Counties' },
		{ href: '/compare', label: 'Compare' },
		{ href: '/map', label: 'Map' },
		{ href: '/scenarios', label: 'Scenarios' },
		{ href: '/about', label: 'About' }
	];
</script>

<svelte:head>
	<title>Wastewater Surveillance Equity Dashboard</title>
	<meta name="description" content="Interactive analysis of health equity gaps in CDC wastewater surveillance coverage" />
</svelte:head>

<div class="min-h-screen bg-gray-50 flex flex-col">
	<!-- Header -->
	<header class="bg-white border-b border-gray-200 sticky top-0 z-50">
		<div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
			<div class="flex justify-between items-center h-16">
				<div class="flex items-center gap-8">
					<a href="/" class="flex items-center gap-2">
						<span class="text-xl font-bold text-gray-900">WW Equity</span>
					</a>
					<nav class="hidden md:flex items-center gap-1">
						{#each navItems as item}
							<a
								href={item.href}
								class="px-3 py-2 rounded-md text-sm font-medium text-gray-600 hover:text-gray-900 hover:bg-gray-100 transition-colors"
							>
								{item.label}
							</a>
						{/each}
					</nav>
				</div>
				<div class="text-sm text-gray-500">
					Data: {data.summary?.analysis_date || 'Jan 2026'}
				</div>
			</div>
		</div>
	</header>

	<!-- Mobile nav -->
	<nav class="md:hidden bg-white border-b border-gray-200 px-4 py-2 flex gap-2 overflow-x-auto">
		{#each navItems as item}
			<a
				href={item.href}
				class="px-3 py-1.5 rounded-full text-sm font-medium text-gray-600 hover:text-gray-900 hover:bg-gray-100 whitespace-nowrap"
			>
				{item.label}
			</a>
		{/each}
	</nav>

	<!-- Main content -->
	<main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex-1 w-full">
		{@render children()}
	</main>

	<!-- Footer -->
	<footer class="bg-white border-t border-gray-200 mt-auto">
		<div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
			<div class="flex flex-col md:flex-row justify-between items-center gap-4 text-sm text-gray-500">
				<p>Wastewater Surveillance Health Equity Analysis</p>
				<p>Data sources: CDC NWSS, CDC/ATSDR SVI 2022</p>
			</div>
		</div>
	</footer>
</div>
