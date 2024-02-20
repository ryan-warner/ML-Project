/** @type {import('tailwindcss').Config} */
export default {
	content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
	theme: {
		extend: {
			colors: {
				'dark-primary': '#1A1A1A',
				'light-primary': '#F5F5F5',
			},
		},
	},
	plugins: [],
}
