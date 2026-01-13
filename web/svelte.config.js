import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	preprocess: vitePreprocess(),

	compilerOptions: {
		// Disable warnings for static site where data doesn't change after load
		warningFilter: (warning) => {
			// Ignore state_referenced_locally for static prerendered pages
			if (warning.code === 'state_referenced_locally') return false;
			// Ignore a11y warnings during build (handled separately)
			if (warning.code === 'a11y_label_has_associated_control') return false;
			return true;
		}
	},

	kit: {
		adapter: adapter({
			pages: 'build',
			assets: 'build',
			fallback: 'index.html',
			precompress: true,
			strict: false
		}),
		prerender: {
			handleMissingId: 'warn'
		}
	}
};

export default config;
