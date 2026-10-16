/// <reference types="vitest" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "/dgxmm/",
  plugins: [react()],
  server: { proxy: { "/dgxmm/api": { target: "http://127.0.0.1:8090" } } },
  test: { environment: "node", include: ["src/**/*.test.ts"] },
});
