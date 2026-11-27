import http from 'node:http';
import { defineConfig } from 'vite';

/**
 * Serves GET /__cad_health from Vite dev server so frontend health checks
 * never trigger Chrome net::ERR_CONNECTION_REFUSED logs when Python bridge is stopped.
 */
function cadKernelHealthPlugin() {
  return {
    name: 'cad-kernel-health-proxy',
    configureServer(server) {
      server.middlewares.use('/__cad_health', (_req, res) => {
        const probe = http.get('http://127.0.0.1:5174/health', { timeout: 2000 }, (upstream) => {
          let body = '';
          upstream.setEncoding('utf8');
          upstream.on('data', (chunk) => { body += chunk; });
          upstream.on('end', () => {
            res.statusCode = 200;
            res.setHeader('Content-Type', 'application/json; charset=utf-8');
            res.end(upstream.statusCode === 200 && body ? body : '{"ok":false}');
          });
        });
        probe.on('timeout', () => {
          probe.destroy();
          if (!res.writableEnded) {
            res.statusCode = 200;
            res.setHeader('Content-Type', 'application/json; charset=utf-8');
            res.end('{"ok":false}');
          }
        });
        probe.on('error', () => {
          if (!res.writableEnded) {
            res.statusCode = 200;
            res.setHeader('Content-Type', 'application/json; charset=utf-8');
            res.end('{"ok":false}');
          }
        });
      });
    },
  };
}

export default defineConfig({
  plugins: [cadKernelHealthPlugin()],
});
