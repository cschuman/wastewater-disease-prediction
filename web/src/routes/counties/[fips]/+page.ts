import { error } from '@sveltejs/kit';
import type { PageLoad } from './$types';

export const load: PageLoad = async ({ params, parent }) => {
	const { counties } = await parent();
	const county = counties.find((c) => c.fips === params.fips);

	if (!county) {
		error(404, 'County not found');
	}

	return { county };
};
