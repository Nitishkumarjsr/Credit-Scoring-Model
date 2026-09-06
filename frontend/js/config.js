/**
 * CrediPulse AI - Dynamic API Configuration & Health Monitoring
 * Supports local development, Render backend, Vercel frontend, and Supabase integration.
 */

const API_CONFIG = {
    // Default Render production backend URL (can be customized via UI or localStorage)
    DEFAULT_RENDER_BACKEND: "https://credit-scoring-backend.onrender.com",
    
    // Get currently active Backend API URL
    getBaseUrl() {
        const stored = localStorage.getItem("credipulse_api_url");
        if (stored && stored.trim()) {
            return stored.trim().replace(/\/+$/, "");
        }
        
        // Auto-detect local vs production environment
        const hostname = window.location.hostname;
        if (hostname === "localhost" || hostname === "127.0.0.1" || hostname === "") {
            // If served via Flask on port 5001, use relative path, otherwise port 5001
            if (window.location.port === "5001") {
                return "";
            }
            return "http://127.0.0.1:5001";
        }
        
        // When deployed on Vercel or other domains
        return this.DEFAULT_RENDER_BACKEND;
    },

    // Save custom backend API URL
    setBaseUrl(url) {
        if (!url || !url.trim()) {
            localStorage.removeItem("credipulse_api_url");
        } else {
            localStorage.setItem("credipulse_api_url", url.trim().replace(/\/+$/, ""));
        }
    },

    // Build endpoint URL
    endpoint(path) {
        const base = this.getBaseUrl();
        const cleanPath = path.startsWith("/") ? path : `/${path}`;
        return base ? `${base}${cleanPath}` : cleanPath;
    },

    // Test connection to backend and database
    async checkHealth() {
        const url = this.endpoint("/api/health");
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
                error: e.name === "AbortError" ? "Connection timeout (Render server might be waking up)" : e.message
            };
        }
    }
};

window.API_CONFIG = API_CONFIG;
