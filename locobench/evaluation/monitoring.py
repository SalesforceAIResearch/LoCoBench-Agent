"""
Real-time Evaluation Monitoring Dashboard

A simple web interface to monitor ongoing evaluations in real-time.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

class MonitoringDashboard:
    """Simple web dashboard for monitoring evaluation progress"""
    
    def __init__(self, port: int = 8080):
        self.port = port
        self.server = None
        self.server_thread = None
        # Default checkpoint files in intermediate_results directory
        intermediate_dir = Path("intermediate_results")
        intermediate_dir.mkdir(exist_ok=True)
        self.checkpoint_file = intermediate_dir / "evaluation_checkpoint.json"
        self.incremental_file = intermediate_dir / "evaluation_incremental_results.json"
        
    def start(self):
        """Start the monitoring dashboard web server"""
        handler = self._create_handler()
        self.server = HTTPServer(('localhost', self.port), handler)
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()
        print(f"📊 Monitoring dashboard started at http://localhost:{self.port}")
        
    def stop(self):
        """Stop the monitoring dashboard"""
        if self.server:
            self.server.shutdown()
            self.server_thread.join()
            print("📊 Monitoring dashboard stopped")
    
    def _create_handler(self):
        """Create the HTTP request handler"""
        dashboard = self
        
        class MonitoringHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/':
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(dashboard._get_dashboard_html().encode())
                elif self.path == '/api/status':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    status = dashboard._get_status_data()
                    self.wfile.write(json.dumps(status).encode())
                else:
                    self.send_response(404)
                    self.end_headers()
                    
            def log_message(self, format, *args):
                # Suppress server logs
                pass
        
        return MonitoringHandler
    
    def _get_status_data(self) -> Dict[str, Any]:
        """Get current evaluation status data"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'checkpoint_exists': self.checkpoint_file.exists(),
            'incremental_exists': self.incremental_file.exists(),
            'checkpoint_data': None,
            'recent_results': [],
            'summary': {}
        }
        
        # Load checkpoint data
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    status['checkpoint_data'] = json.load(f)
            except Exception as e:
                status['checkpoint_error'] = str(e)
        
        # Load recent incremental results
        if self.incremental_file.exists():
            try:
                with open(self.incremental_file, 'r') as f:
                    all_results = json.load(f)
                    # Get last 10 results
                    status['recent_results'] = all_results[-10:] if len(all_results) > 10 else all_results
                    
                    # Generate summary
                    if all_results:
                        models = set()
                        total_score_sum = 0
                        for result in all_results:
                            models.add(result.get('model_name', 'unknown'))
                            total_score_sum += result.get('total_score', 0)
                        
                        status['summary'] = {
                            'total_completed': len(all_results),
                            'models_evaluated': list(models),
                            'average_score': total_score_sum / len(all_results) if all_results else 0
                        }
            except Exception as e:
                status['incremental_error'] = str(e)
        
        return status
    
    def _get_dashboard_html(self) -> str:
        """Generate the HTML dashboard"""
        return '''<!DOCTYPE html>
<html>
<head>
    <title>LoCoBench Monitoring</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0; padding: 20px; background: #f5f5f5; 
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .card { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }
        .metric { text-align: center; padding: 15px; background: #f8f9fa; border-radius: 6px; }
        .metric-value { font-size: 2em; font-weight: bold; color: #2563eb; }
        .metric-label { color: #6b7280; margin-top: 5px; }
        .progress-bar { width: 100%; height: 20px; background: #e5e7eb; border-radius: 10px; overflow: hidden; }
        .progress-fill { height: 100%; background: linear-gradient(90deg, #10b981, #059669); transition: width 0.3s ease; }
        .result-item { padding: 10px; border-left: 4px solid #10b981; margin: 10px 0; background: #f0fdf4; }
        .error { color: #dc2626; }
        .loading { text-align: center; padding: 40px; color: #6b7280; }
        .refresh-btn { background: #2563eb; color: white; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; }
        .refresh-btn:hover { background: #1d4ed8; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧪 LoCoBench Monitoring Dashboard</h1>
            <p>Real-time evaluation progress and results</p>
            <button class="refresh-btn" onclick="loadStatus()">🔄 Refresh</button>
        </div>
        
        <div class="status-grid">
            <div class="card">
                <div class="metric">
                    <div class="metric-value" id="completed-count">-</div>
                    <div class="metric-label">Completed Evaluations</div>
                </div>
            </div>
            <div class="card">
                <div class="metric">
                    <div class="metric-value" id="progress-percent">-</div>
                    <div class="metric-label">Progress</div>
                </div>
            </div>
            <div class="card">
                <div class="metric">
                    <div class="metric-value" id="avg-score">-</div>
                    <div class="metric-label">Average Score</div>
                </div>
            </div>
            <div class="card">
                <div class="metric">
                    <div class="metric-value" id="models-count">-</div>
                    <div class="metric-label">Models</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h3>📊 Overall Progress</h3>
            <div class="progress-bar">
                <div class="progress-fill" id="progress-bar" style="width: 0%"></div>
            </div>
            <p id="progress-text">Loading...</p>
        </div>
        
        <div class="card">
            <h3>📝 Recent Results</h3>
            <div id="recent-results">
                <div class="loading">Loading recent results...</div>
            </div>
        </div>
    </div>

    <script>
        function loadStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => updateDashboard(data))
                .catch(error => {
                    console.error('Error:', error);
                    document.getElementById('recent-results').innerHTML = 
                        '<div class="error">Failed to load status data</div>';
                });
        }
        
        function updateDashboard(data) {
            // Update metrics
            const summary = data.summary || {};
            const checkpoint = data.checkpoint_data || {};
            
            document.getElementById('completed-count').textContent = summary.total_completed || 0;
            document.getElementById('avg-score').textContent = summary.average_score ? 
                summary.average_score.toFixed(3) : '-';
            document.getElementById('models-count').textContent = summary.models_evaluated ? 
                summary.models_evaluated.length : '-';
            
            // Update progress
            if (checkpoint.total_scenarios && checkpoint.completed_count !== undefined) {
                const percent = Math.round((checkpoint.completed_count / checkpoint.total_scenarios) * 100);
                document.getElementById('progress-percent').textContent = percent + '%';
                document.getElementById('progress-bar').style.width = percent + '%';
                document.getElementById('progress-text').textContent = 
                    `${checkpoint.completed_count} / ${checkpoint.total_scenarios} scenarios completed`;
            } else {
                document.getElementById('progress-percent').textContent = '-';
                document.getElementById('progress-text').textContent = 'No active evaluation';
            }
            
            // Update recent results
            const resultsContainer = document.getElementById('recent-results');
            if (data.recent_results && data.recent_results.length > 0) {
                resultsContainer.innerHTML = data.recent_results.map(result => 
                    `<div class="result-item">
                        <strong>${result.model_name}</strong> on ${result.scenario_title || result.scenario_id}
                        <br>Score: ${result.total_score ? result.total_score.toFixed(3) : 'N/A'} 
                        (${result.difficulty || 'unknown'} difficulty)
                    </div>`
                ).join('');
            } else {
                resultsContainer.innerHTML = '<div class="loading">No recent results found</div>';
            }
        }
        
        // Auto-refresh every 10 seconds
        setInterval(loadStatus, 10000);
        
        // Initial load
        loadStatus();
    </script>
</body>
</html>''' 