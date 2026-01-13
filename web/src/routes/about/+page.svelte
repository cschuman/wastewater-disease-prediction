<script lang="ts">
	import type { PageData } from './$types';

	let { data }: { data: PageData } = $props();
</script>

<div class="max-w-3xl space-y-8">
	<div>
		<h1 class="text-2xl font-bold">About This Analysis</h1>
		<p class="text-gray-600 mt-1">Methodology and data sources</p>
	</div>

	<div class="prose prose-gray max-w-none">
		<section class="bg-white rounded-lg border p-6 space-y-4">
			<h2 class="text-lg font-semibold">Executive Summary</h2>
			<p>
				This dashboard presents findings from an analysis of health equity gaps in CDC wastewater
				surveillance coverage. The key finding: <strong>high-vulnerability counties have 33% fewer
				monitoring sites per capita</strong> than low-vulnerability counties.
			</p>
			<p>
				This disparity means vulnerable communities—those most likely to experience severe disease
				burden—have the least access to early warning systems.
			</p>
		</section>

		<section class="bg-white rounded-lg border p-6 space-y-4">
			<h2 class="text-lg font-semibold">Methodology</h2>
			<p>
				Based on methodology from: <em>"Expansion of wastewater-based disease surveillance to improve
				health equity in California's Central Valley"</em> (Frontiers in Public Health, 2023)
			</p>
			<h3 class="font-medium">Approach</h3>
			<ol class="list-decimal list-inside space-y-1 text-sm">
				<li>Calculate hospitalization-to-wastewater ratios by state</li>
				<li>Measure signal quality via R² (wastewater-hospitalization correlation)</li>
				<li>Measure signal stability via coefficient of variation (CV) of weekly ratios</li>
				<li>Correlate metrics with CDC Social Vulnerability Index (SVI)</li>
			</ol>
		</section>

		<section class="bg-white rounded-lg border p-6 space-y-4">
			<h2 class="text-lg font-semibold">Key Findings</h2>
			<ul class="space-y-3">
				<li class="flex gap-2">
					<span class="text-blue-600 font-bold">1.</span>
					<span>High-vulnerability states have <strong>44% fewer</strong> wastewater monitoring sites per capita.</span>
				</li>
				<li class="flex gap-2">
					<span class="text-blue-600 font-bold">2.</span>
					<span>County-level analysis (n=3,144) confirms the gap is statistically significant (p=0.0001).</span>
				</li>
				<li class="flex gap-2">
					<span class="text-blue-600 font-bold">3.</span>
					<span>Natural experiment (DiD) shows rapidly expanding monitoring temporarily increases signal noise (p=0.041).</span>
				</li>
				<li class="flex gap-2">
					<span class="text-blue-600 font-bold">4.</span>
					<span>Urban/rural confounding analysis confirms the SVI effect persists after controlling for urbanization (p=0.007).</span>
				</li>
				<li class="flex gap-2">
					<span class="text-blue-600 font-bold">5.</span>
					<span>Gap trajectory analysis shows the disparity is STABLE—despite 52% expansion in 2024+.</span>
				</li>
				<li class="flex gap-2">
					<span class="text-blue-600 font-bold">6.</span>
					<span>Equity simulation estimates closing the gap requires $172M setup + $602M over 5 years.</span>
				</li>
			</ul>
		</section>

		<section class="bg-white rounded-lg border p-6 space-y-4">
			<h2 class="text-lg font-semibold">Data Sources</h2>
			<ul class="space-y-2 text-sm">
				<li>
					<strong>Wastewater:</strong> CDC National Wastewater Surveillance System (NWSS), March 2024 - September 2025
				</li>
				<li>
					<strong>Hospitalizations:</strong> CDC NHSN state-level data, same period
				</li>
				<li>
					<strong>Vulnerability:</strong> CDC/ATSDR Social Vulnerability Index (SVI) 2022
				</li>
				<li>
					<strong>Urban/Rural:</strong> USDA Rural-Urban Continuum Codes (RUCC) 2023
				</li>
			</ul>
		</section>

		<section class="bg-white rounded-lg border p-6 space-y-4">
			<h2 class="text-lg font-semibold">Limitations</h2>
			<ul class="list-disc list-inside space-y-1 text-sm text-gray-600">
				<li>Using CDC-computed percentiles rather than raw concentrations</li>
				<li>Data covers post-emergency phase only (2024-2025)</li>
				<li>State-level SVI averages mask within-state variation</li>
				<li>Cannot distinguish calibration issues from inherent variance in new sites</li>
			</ul>
		</section>

		<section class="bg-white rounded-lg border p-6 space-y-4">
			<h2 class="text-lg font-semibold">Conclusion</h2>
			<p>
				This analysis reveals that <strong>wastewater surveillance infrastructure itself has an equity gap</strong>.
				Rather than demonstrating that wastewater works better in vulnerable populations, we found that
				current NWSS coverage is biased toward lower-vulnerability states.
			</p>
			<p class="font-medium">
				The path to health equity in wastewater surveillance is infrastructure investment, not algorithmic adjustment.
			</p>
		</section>
	</div>

	<div class="text-sm text-gray-500">
		<p>Analysis completed: January 2026</p>
		<p>Data sources: CDC NWSS, CDC NHSN, CDC/ATSDR SVI 2022, USDA ERS RUCC 2023</p>
	</div>
</div>
