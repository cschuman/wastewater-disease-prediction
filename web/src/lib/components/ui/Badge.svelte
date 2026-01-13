<script lang="ts">
	import { getQuartileColor, getTierColor } from '$lib/utils/colors';

	interface Props {
		text: string;
		type?: 'quartile' | 'tier' | 'default';
	}

	let { text, type = 'default' }: Props = $props();

	function getColor(text: string, type: string): string {
		if (type === 'quartile') return getQuartileColor(text);
		if (type === 'tier') return getTierColor(text);
		return '#6b7280';
	}

	const color = getColor(text, type);
</script>

<span
	class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium"
	style="background-color: {color}20; color: {color};"
>
	{text}
</span>
