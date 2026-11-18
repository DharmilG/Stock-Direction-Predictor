import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

/**
 * The dashboard talks to the FastAPI service on port 8000. Proxying keeps the
 * browser same-origin in development, so no CORS pre-flight is involved and the
 * client can run with an empty VITE_API_BASE_URL.
 */
const API_TARGET = process.env.DASHBOARD_API_URL || 'http://127.0.0.1:8000';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/dashboard': { target: API_TARGET, changeOrigin: true },
      '/health': { target: API_TARGET, changeOrigin: true },
      '/cv': { target: API_TARGET, changeOrigin: true },
    },
  },
  preview: {
    port: 4173,
    proxy: {
      '/dashboard': { target: API_TARGET, changeOrigin: true },
      '/health': { target: API_TARGET, changeOrigin: true },
      '/cv': { target: API_TARGET, changeOrigin: true },
    },
  },
});
