import { defineConfig } from "vite";
import { fileURLToPath } from "url";
import react from "@vitejs/plugin-react";
import path from "path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const BACKEND_PORT = 50001;

const react_config = {
  babel: {
    plugins: [
      [
        "babel-plugin-react-compiler",
        {
          /* Compiler config here, for now empty */
        },
      ],
    ],
  },
};

// https://vitejs.dev/config/
export default defineConfig(() => {
  return {
    plugins: [react(react_config)],
    resolve: {
      alias: { "@": path.resolve(__dirname, "./src") },
    },
    build: {
      sourcemap: true,
      outDir: path.resolve(__dirname, "../py_neuromodulation/gui/frontend"),
      emptyOutDir: true,
      rollupOptions: {
        output: {
          manualChunks: {
            plotly: ["plotly.js-basic-dist-min"],
          },
        },
        onLog(level, log, handler) {
          if (
            log.cause &&
            log.cause.message === `Can't resolve original location of error.`
          ) {
            return;
          }
          handler(level, log);
        },
      },
    },
    server: {
      proxy: {
        "/api": {
          target: `http://localhost:${BACKEND_PORT}`,
          changeOrigin: true,
        },
        "/ws": {
          target: `ws://localhost:${BACKEND_PORT}`,
          ws: true,
          changeOrigin: true,
        },
      },
    },
  };
});
