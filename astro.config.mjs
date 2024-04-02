import { defineConfig } from 'astro/config';
import tailwind from "@astrojs/tailwind";

// https://astro.build/config
export default defineConfig({
  outDir: './docs',
  integrations: [tailwind()],
  site: 'https://github.gatech.edu',
  base: '/pages/rwarner31/ML-ProjectV2/',
  output: 'static',
  build: {
    inlineStylesheets: 'never'
  }
});