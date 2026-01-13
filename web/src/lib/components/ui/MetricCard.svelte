<script lang="ts">
	interface Props {
		label: string;
		value: string | number;
		subtext?: string;
		variant?: 'default' | 'danger' | 'warning' | 'success';
	}

	let { label, value, subtext, variant = 'default' }: Props = $props();

	const variantClasses: Record<string, string> = {
		default: 'bg-white',
		danger: 'bg-red-50 border-red-200',
		warning: 'bg-amber-50 border-amber-200',
		success: 'bg-green-50 border-green-200'
	};

	const valueClasses: Record<string, string> = {
		default: 'text-gray-900',
		danger: 'text-red-700',
		warning: 'text-amber-700',
		success: 'text-green-700'
	};
</script>

<div class="rounded-lg border p-4 {variantClasses[variant]}">
	<p class="text-sm font-medium text-gray-500">{label}</p>
	<p class="mt-1 text-2xl font-semibold {valueClasses[variant]}">{value}</p>
	{#if subtext}
		<p class="mt-1 text-sm text-gray-500">{subtext}</p>
	{/if}
</div>
