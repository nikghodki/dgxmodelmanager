/// <reference types="vitest" />
import path from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import pkg from "./package.json" with { type: "json" };

export default defineConfig({
  base: "/dgxmm/",
  plugins: [react()],
  resolve: { alias: { "@": path.resolve(__dirname, "src") } },
  define: { __APP_VERSION__: JSON.stringify(pkg.version) },
  server: { proxy: { "/dgxmm/api": { target: "http://127.0.0.1:8090" } } },
  test: {
    environment: "jsdom",
    include: ["src/**/*.test.{ts,tsx}"],
    setupFiles: ["src/test/setup.ts"],
    css: false,
  },
});
