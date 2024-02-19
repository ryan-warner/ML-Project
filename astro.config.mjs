import { defineConfig } from 'astro/config';

import tailwind from "@astrojs/tailwind";

// https://astro.build/config
export default defineConfig({
  outDir: './docs',
  integrations: [tailwind()],
  output: 'static',
  base: "/pages/rwarner31/ML-Project/",
  site: "https://github.gatech.edu/pages/rwarner31/ML-Project/",
  relative: true,
});