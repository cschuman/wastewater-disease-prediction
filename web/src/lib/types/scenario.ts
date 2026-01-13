export interface Scenario {
	id: string;
	name: string;
	description: string;
	new_sites: number;
	setup_cost_millions: number;
	five_year_cost_millions: number;
	timeline: string;
	recommended?: boolean;
}
