#!/usr/bin/env python3
"""
Adaptive-K Monitoring Dashboard

Exports metrics to Prometheus/Grafana for production monitoring.
Essential for tracking savings and detecting quality regressions.

Usage:
    python monitoring_dashboard.py --port 8000
"""

import time
import json
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import deque
from http.server import HTTPServer, BaseHTTPRequestHandler
import argparse


@dataclass
class MetricsWindow:
    """Sliding window for metrics aggregation."""
    window_size: int = 1000
    values: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def add(self, value: float):
        self.values.append(value)
    
    def mean(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0
    
    def percentile(self, p: float) -> float:
        if not self.values:
            return 0
        sorted_vals = sorted(self.values)
        idx = int(len(sorted_vals) * p / 100)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]


class AdaptiveKMetrics:
    """
    Metrics collector for Adaptive-K routing.
    
    Tracks:
    - K selection distribution
    - Entropy values
    - Compute savings
    - Quality proxies
    """
    
    def __init__(self, k_values: List[int] = [1, 2]):
        self.k_values = k_values
        self.start_time = time.time()
        
        # Counters
        self.total_tokens = 0
        self.k_counts = {k: 0 for k in k_values}
        
        # Windows for recent stats
        self.entropy_window = MetricsWindow(1000)
        self.k_window = MetricsWindow(1000)
        self.latency_window = MetricsWindow(1000)
        
        # Histogram buckets for entropy
        self.entropy_buckets = {
            "0.0-0.5": 0,
            "0.5-1.0": 0,
            "1.0-1.5": 0,
            "1.5-2.0": 0,
            "2.0+": 0,
        }
        
        self._lock = threading.Lock()
    
    def record_routing(
        self,
        k_selected: int,
        entropy: float,
        latency_ms: Optional[float] = None
    ):
        """Record a routing decision."""
        with self._lock:
            self.total_tokens += 1
            self.k_counts[k_selected] = self.k_counts.get(k_selected, 0) + 1
            
            self.entropy_window.add(entropy)
            self.k_window.add(k_selected)
            
            if latency_ms:
                self.latency_window.add(latency_ms)
            
            # Update histogram
            if entropy < 0.5:
                self.entropy_buckets["0.0-0.5"] += 1
            elif entropy < 1.0:
                self.entropy_buckets["0.5-1.0"] += 1
            elif entropy < 1.5:
                self.entropy_buckets["1.0-1.5"] += 1
            elif entropy < 2.0:
                self.entropy_buckets["1.5-2.0"] += 1
            else:
                self.entropy_buckets["2.0+"] += 1
    
    def get_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        with self._lock:
            lines = []
            
            # Help and type declarations
            lines.append("# HELP adaptive_k_tokens_total Total tokens processed")
            lines.append("# TYPE adaptive_k_tokens_total counter")
            lines.append(f"adaptive_k_tokens_total {self.total_tokens}")
            
            lines.append("# HELP adaptive_k_selected K value selected for routing")
            lines.append("# TYPE adaptive_k_selected counter")
            for k, count in self.k_counts.items():
                lines.append(f'adaptive_k_selected{{k="{k}"}} {count}')
            
            lines.append("# HELP adaptive_k_entropy_mean Mean routing entropy")
            lines.append("# TYPE adaptive_k_entropy_mean gauge")
            lines.append(f"adaptive_k_entropy_mean {self.entropy_window.mean():.4f}")
            
            lines.append("# HELP adaptive_k_avg_k Average K value")
            lines.append("# TYPE adaptive_k_avg_k gauge")
            avg_k = self.k_window.mean() if self.k_window.values else 2.0
            lines.append(f"adaptive_k_avg_k {avg_k:.4f}")
            
            lines.append("# HELP adaptive_k_savings_ratio Compute savings ratio")
            lines.append("# TYPE adaptive_k_savings_ratio gauge")
            baseline_k = max(self.k_values)
            savings = (baseline_k - avg_k) / baseline_k if baseline_k > 0 else 0
            lines.append(f"adaptive_k_savings_ratio {savings:.4f}")
            
            lines.append("# HELP adaptive_k_latency_ms Inference latency in milliseconds")
            lines.append("# TYPE adaptive_k_latency_ms summary")
            if self.latency_window.values:
                lines.append(f'adaptive_k_latency_ms{{quantile="0.5"}} {self.latency_window.percentile(50):.2f}')
                lines.append(f'adaptive_k_latency_ms{{quantile="0.9"}} {self.latency_window.percentile(90):.2f}')
                lines.append(f'adaptive_k_latency_ms{{quantile="0.99"}} {self.latency_window.percentile(99):.2f}')
            
            lines.append("# HELP adaptive_k_entropy_bucket Entropy distribution histogram")
            lines.append("# TYPE adaptive_k_entropy_bucket counter")
            for bucket, count in self.entropy_buckets.items():
                lines.append(f'adaptive_k_entropy_bucket{{le="{bucket}"}} {count}')
            
            lines.append("# HELP adaptive_k_uptime_seconds Service uptime")
            lines.append("# TYPE adaptive_k_uptime_seconds gauge")
            lines.append(f"adaptive_k_uptime_seconds {time.time() - self.start_time:.0f}")
            
            return "\n".join(lines) + "\n"
    
    def get_json_metrics(self) -> dict:
        """Export metrics as JSON."""
        with self._lock:
            avg_k = self.k_window.mean() if self.k_window.values else 2.0
            baseline_k = max(self.k_values)
            savings = (baseline_k - avg_k) / baseline_k if baseline_k > 0 else 0
            
            return {
                "total_tokens": self.total_tokens,
                "k_distribution": self.k_counts,
                "entropy": {
                    "mean": self.entropy_window.mean(),
                    "p50": self.entropy_window.percentile(50),
                    "p90": self.entropy_window.percentile(90),
                },
                "performance": {
                    "avg_k": avg_k,
                    "baseline_k": baseline_k,
                    "savings_ratio": savings,
                    "savings_percent": f"{savings*100:.1f}%",
                },
                "latency_ms": {
                    "p50": self.latency_window.percentile(50),
                    "p90": self.latency_window.percentile(90),
                    "p99": self.latency_window.percentile(99),
                },
                "uptime_seconds": time.time() - self.start_time,
            }


# Global metrics instance
_metrics = AdaptiveKMetrics()


def get_metrics() -> AdaptiveKMetrics:
    """Get global metrics instance."""
    return _metrics


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for metrics endpoint."""
    
    def do_GET(self):
        if self.path == "/metrics":
            # Prometheus format
            content = _metrics.get_prometheus_metrics()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(content.encode())
        
        elif self.path == "/metrics/json":
            # JSON format
            content = json.dumps(_metrics.get_json_metrics(), indent=2)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(content.encode())
        
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "healthy"}')
        
        else:
            # Dashboard HTML
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(self._dashboard_html().encode())
    
    def _dashboard_html(self) -> str:
        m = _metrics.get_json_metrics()
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Adaptive-K Dashboard</title>
    <meta http-equiv="refresh" content="5">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        h1 {{ color: #333; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; margin: 10px 0; 
                 box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric {{ display: inline-block; margin: 10px 20px; }}
        .metric-value {{ font-size: 32px; font-weight: bold; color: #059669; }}
        .metric-label {{ font-size: 14px; color: #666; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f9f9f9; }}
        .savings {{ color: #059669; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>📊 Adaptive-K Monitoring Dashboard</h1>
    
    <div class="grid">
        <div class="card">
            <div class="metric">
                <div class="metric-value">{m['total_tokens']:,}</div>
                <div class="metric-label">Total Tokens</div>
            </div>
        </div>
        <div class="card">
            <div class="metric">
                <div class="metric-value">{m['performance']['avg_k']:.2f}</div>
                <div class="metric-label">Average K (baseline: {m['performance']['baseline_k']})</div>
            </div>
        </div>
        <div class="card">
            <div class="metric">
                <div class="metric-value savings">{m['performance']['savings_percent']}</div>
                <div class="metric-label">Compute Savings</div>
            </div>
        </div>
        <div class="card">
            <div class="metric">
                <div class="metric-value">{m['entropy']['mean']:.3f}</div>
                <div class="metric-label">Mean Entropy</div>
            </div>
        </div>
    </div>
    
    <div class="card">
        <h2>K Distribution</h2>
        <table>
            <tr><th>K Value</th><th>Count</th><th>Percentage</th></tr>
            {''.join(f"<tr><td>K={k}</td><td>{count:,}</td><td>{count/max(m['total_tokens'],1)*100:.1f}%</td></tr>" 
                     for k, count in m['k_distribution'].items())}
        </table>
    </div>
    
    <div class="card">
        <h2>Latency (ms)</h2>
        <table>
            <tr><th>Percentile</th><th>Value</th></tr>
            <tr><td>P50</td><td>{m['latency_ms']['p50']:.2f} ms</td></tr>
            <tr><td>P90</td><td>{m['latency_ms']['p90']:.2f} ms</td></tr>
            <tr><td>P99</td><td>{m['latency_ms']['p99']:.2f} ms</td></tr>
        </table>
    </div>
    
    <div class="card">
        <h2>Endpoints</h2>
        <ul>
            <li><a href="/metrics">Prometheus metrics</a> - /metrics</li>
            <li><a href="/metrics/json">JSON metrics</a> - /metrics/json</li>
            <li><a href="/health">Health check</a> - /health</li>
        </ul>
    </div>
    
    <p style="color: #666; font-size: 12px;">
        Uptime: {m['uptime_seconds']:.0f}s | Auto-refresh every 5s
    </p>
</body>
</html>
"""
    
    def log_message(self, format, *args):
        pass  # Suppress logging


def simulate_traffic(metrics: AdaptiveKMetrics, rate: float = 100):
    """Simulate traffic for demo."""
    import random
    
    while True:
        # Simulate routing decision
        entropy = random.gauss(1.0, 0.5)
        entropy = max(0, min(entropy, 2.5))
        
        k = 1 if entropy < 1.275 else 2
        latency = random.gauss(50, 10) if k == 1 else random.gauss(80, 15)
        
        metrics.record_routing(k, entropy, latency)
        
        time.sleep(1.0 / rate)


def main():
    parser = argparse.ArgumentParser(description="Adaptive-K Monitoring Dashboard")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on")
    parser.add_argument("--demo", action="store_true", help="Run with simulated traffic")
    
    args = parser.parse_args()
    
    print(f"\n📊 Starting Adaptive-K Monitoring Dashboard")
    print(f"   Dashboard: http://localhost:{args.port}/")
    print(f"   Prometheus: http://localhost:{args.port}/metrics")
    print(f"   JSON: http://localhost:{args.port}/metrics/json")
    
    if args.demo:
        print("\n🔄 Running with simulated traffic...")
        traffic_thread = threading.Thread(
            target=simulate_traffic, 
            args=(_metrics, 100),
            daemon=True
        )
        traffic_thread.start()
    
    print("\nPress Ctrl+C to stop\n")
    
    server = HTTPServer(("", args.port), MetricsHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
