/**
 * CrediPulse AI - Dynamic API Configuration & Health Monitoring
 * Uses secure relative paths so the backend URL is completely hidden behind Vercel Reverse Proxy.
 */

const API_CONFIG = {
    // Returns relative API path (avoids exposing backend URL in browser / network tabs)
    getBaseUrl() {
        return "";
    },

    // Build endpoint URL (relative)
    endpoint(path) {
        const cleanPath = path.startsWith("/") ? path : `/${path}`;
        return cleanPath;
    },

    // Test connection to backend and database
    async checkHealth() {
        const url = this.endpoint("/health");
        const t0 = performance.now();
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 6000);
            
            const res = await fetch(url, {
                method: "GET",
                signal: controller.signal,
                headers: { "Accept": "application/json" }
            });
            clearTimeout(timeoutId);
            const latency = Math.round(performance.now() - t0);
            
            if (res.ok) {
                const data = await res.json();
                return {
                    online: true,
                    latency: latency,
                    data: data,
                    error: null
                };
            } else {
                return {
                    online: false,
                    latency: latency,
                    data: null,
                    error: `HTTP ${res.status}: ${res.statusText}`
                };
            }
        } catch (e) {
            return {
                online: false,
                latency: Math.round(performance.now() - t0),
                data: null,
                error: e.name === "AbortError" ? "Connection timeout" : e.message
            };
        }
    }
};

window.API_CONFIG = API_CONFIG;
