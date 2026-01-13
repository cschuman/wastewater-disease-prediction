export interface FilterState {
	search: string;
	states: string[];
	sviQuartiles: string[];
	priorityTiers: string[];
	hasMonitoring: boolean | null;
}

export const DEFAULT_FILTERS: FilterState = {
	search: '',
	states: [],
	sviQuartiles: [],
	priorityTiers: [],
	hasMonitoring: null
};
