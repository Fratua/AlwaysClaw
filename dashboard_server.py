"""
OpenClaw Monitoring Dashboard
Web-based dashboard for monitoring status and change visualization
"""

import asyncio
import json
import logging
import os
import time as _time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import asdict

logger = logging.getLogger(__name__)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore[assignment]
    HAS_PSUTIL = False

try:
    from monitor_engine import MonitorEngine
except ImportError:
    MonitorEngine = None  # type: ignore[assignment,misc]

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
    from fastapi.middleware.cors import CORSMiddleware
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    logger.warning(
        "FastAPI not installed. Dashboard server will not be available. "
        "Install with: pip install fastapi uvicorn"
    )


class DashboardAPI:
    """Dashboard API and WebSocket handlers"""
    
    def __init__(self, engine, scheduler, alert_manager, config: Dict = None):
        self.engine = engine
        self.scheduler = scheduler
        self.alert_manager = alert_manager
        self.config = config or {}
        self.refresh_interval = self.config.get('refresh_interval', 5)
        self.connected_clients: List[WebSocket] = []
        self._start_time = _time.monotonic()

        # Try to initialise a MonitorEngine for extra system data
        self._monitor_engine: Optional[Any] = None
        if MonitorEngine is not None:
            try:
                self._monitor_engine = engine if isinstance(engine, MonitorEngine) else MonitorEngine({})
            except Exception as exc:
                logger.warning("Could not initialize MonitorEngine: %s", exc)

        if HAS_FASTAPI:
            self.app = FastAPI(
                title="OpenClaw Web Monitor",
                description="Real-time web monitoring dashboard",
                version="1.0.0"
            )
            self._setup_middleware()
            self._setup_routes()
    
    def _setup_middleware(self):
        """Setup CORS middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard():
            return self._get_dashboard_html()
        
        @self.app.get("/api/overview")
        async def get_overview():
            return self._get_overview_data()
        
        @self.app.get("/api/sites")
        async def get_sites():
            return self._get_sites_data()
        
        @self.app.post("/api/sites/{site_id}/check")
        async def check_site(site_id: str):
            change = await self.engine.run_check(site_id)
            return {
                'change_detected': change is not None,
                'change': change.to_dict() if change else None
            }
        
        @self.app.get("/api/sites/{site_id}/status")
        async def get_site_status(site_id: str):
            status = self.engine.get_monitor_status(site_id)
            if not status:
                raise HTTPException(status_code=404, detail="Site not found")
            return status
        
        @self.app.get("/api/sites/{site_id}/history")
        async def get_site_history(
            site_id: str,
            limit: int = Query(50, ge=1, le=1000),
            days: int = Query(7, ge=1, le=365)
        ):
            return self._get_site_history(site_id, limit, days)
        
        @self.app.get("/api/changes")
        async def get_changes(
            site_id: Optional[str] = None,
            severity: Optional[str] = None,
            category: Optional[str] = None,
            limit: int = Query(50, ge=1, le=1000)
        ):
            return self._get_changes_data(site_id, severity, category, limit)
        
        @self.app.get("/api/changes/{change_id}")
        async def get_change(change_id: str):
            return self._get_change_data(change_id)
        
        @self.app.get("/api/alerts")
        async def get_alerts(
            site_id: Optional[str] = None,
            severity: Optional[str] = None,
            limit: int = Query(50, ge=1, le=1000)
        ):
            return self._get_alerts_data(site_id, severity, limit)
        
        @self.app.post("/api/alerts/{alert_id}/acknowledge")
        async def acknowledge_alert(alert_id: str):
            return {'acknowledged': True}
        
        @self.app.get("/api/scheduler")
        async def get_scheduler_status():
            return self.scheduler.get_statistics() if self.scheduler else {}
        
        @self.app.get("/api/statistics")
        async def get_statistics():
            return self._get_statistics()
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self._handle_websocket(websocket)
    
    def _get_dashboard_html(self) -> str:
        """Get dashboard HTML"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OpenClaw Web Monitor Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            min-height: 100vh;
        }
        
        .header {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            padding: 20px 30px;
            border-bottom: 1px solid #334155;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .header h1 {
            font-size: 24px;
            font-weight: 600;
            background: linear-gradient(90deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .status-badge {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            background: #1e293b;
            border-radius: 20px;
            font-size: 14px;
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #22c55e;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .container {
            padding: 30px;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .stat-card {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 24px;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .stat-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }
        
        .stat-card.critical { border-left: 4px solid #ef4444; }
        .stat-card.high { border-left: 4px solid #f97316; }
        .stat-card.medium { border-left: 4px solid #eab308; }
        .stat-card.normal { border-left: 4px solid #22c55e; }
        
        .stat-label {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #94a3b8;
            margin-bottom: 8px;
        }
        
        .stat-value {
            font-size: 36px;
            font-weight: 700;
            color: #f8fafc;
        }
        
        .panel {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            margin-bottom: 20px;
            overflow: hidden;
        }
        
        .panel-header {
            padding: 20px 24px;
            border-bottom: 1px solid #334155;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .panel-title {
            font-size: 18px;
            font-weight: 600;
        }
        
        .panel-actions {
            display: flex;
            gap: 10px;
        }
        
        .btn {
            padding: 8px 16px;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .btn-primary {
            background: #3b82f6;
            color: white;
        }
        
        .btn-primary:hover {
            background: #2563eb;
        }
        
        .btn-secondary {
            background: #334155;
            color: #e2e8f0;
        }
        
        .btn-secondary:hover {
            background: #475569;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
        }
        
        th, td {
            padding: 16px 24px;
            text-align: left;
            border-bottom: 1px solid #334155;
        }
        
        th {
            font-weight: 600;
            color: #94a3b8;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        tr:hover {
            background: #0f172a;
        }
        
        .site-name {
            font-weight: 500;
            color: #f8fafc;
        }
        
        .site-url {
            font-size: 12px;
            color: #64748b;
        }
        
        .badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 500;
        }
        
        .badge-online {
            background: rgba(34, 197, 94, 0.2);
            color: #22c55e;
        }
        
        .badge-offline {
            background: rgba(239, 68, 68, 0.2);
            color: #ef4444;
        }
        
        .badge-critical {
            background: rgba(239, 68, 68, 0.2);
            color: #ef4444;
        }
        
        .badge-high {
            background: rgba(249, 115, 22, 0.2);
            color: #f97316;
        }
        
        .badge-medium {
            background: rgba(234, 179, 8, 0.2);
            color: #eab308;
        }
        
        .alert-item {
            padding: 16px 24px;
            border-bottom: 1px solid #334155;
            display: flex;
            align-items: flex-start;
            gap: 16px;
        }
        
        .alert-item:last-child {
            border-bottom: none;
        }
        
        .alert-icon {
            width: 40px;
            height: 40px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
        }
        
        .alert-icon.critical { background: rgba(239, 68, 68, 0.2); }
        .alert-icon.high { background: rgba(249, 115, 22, 0.2); }
        .alert-icon.medium { background: rgba(234, 179, 8, 0.2); }
        
        .alert-content {
            flex: 1;
        }
        
        .alert-title {
            font-weight: 500;
            margin-bottom: 4px;
        }
        
        .alert-meta {
            font-size: 12px;
            color: #64748b;
        }
        
        .loading {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 40px;
            color: #64748b;
        }
        
        .spinner {
            width: 20px;
            height: 20px;
            border: 2px solid #334155;
            border-top-color: #3b82f6;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 10px;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #64748b;
        }
        
        .empty-state-icon {
            font-size: 48px;
            margin-bottom: 16px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 OpenClaw Web Monitor</h1>
        <div class="status-badge">
            <span class="status-dot"></span>
            <span id="system-status">System Active</span>
        </div>
    </div>
    
    <div class="container">
        <!-- Stats Grid -->
        <div class="stats-grid">
            <div class="stat-card normal">
                <div class="stat-label">Active Monitors</div>
                <div class="stat-value" id="monitor-count">-</div>
            </div>
            <div class="stat-card critical">
                <div class="stat-label">Critical Alerts</div>
                <div class="stat-value" id="critical-count">-</div>
            </div>
            <div class="stat-card high">
                <div class="stat-label">High Alerts</div>
                <div class="stat-value" id="high-count">-</div>
            </div>
            <div class="stat-card medium">
                <div class="stat-label">Changes Today</div>
                <div class="stat-value" id="changes-count">-</div>
            </div>
        </div>
        
        <!-- Sites Panel -->
        <div class="panel">
            <div class="panel-header">
                <h2 class="panel-title">📡 Monitored Sites</h2>
                <div class="panel-actions">
                    <button class="btn btn-primary" onclick="checkAll()">Check All</button>
                    <button class="btn btn-secondary" onclick="refreshSites()">Refresh</button>
                </div>
            </div>
            <div id="sites-content">
                <div class="loading">
                    <div class="spinner"></div>
                    Loading sites...
                </div>
            </div>
        </div>
        
        <!-- Alerts Panel -->
        <div class="panel">
            <div class="panel-header">
                <h2 class="panel-title">🔔 Recent Alerts</h2>
                <div class="panel-actions">
                    <button class="btn btn-secondary" onclick="refreshAlerts()">Refresh</button>
                </div>
            </div>
            <div id="alerts-content">
                <div class="loading">
                    <div class="spinner"></div>
                    Loading alerts...
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const ws = new WebSocket(`ws://${window.location.host}/ws`);
        
        ws.onopen = function() {
            console.log('WebSocket connected');
        };
        
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            updateDashboard(data);
        };
        
        ws.onerror = function(error) {
            console.error('WebSocket error:', error);
        };
        
        ws.onclose = function() {
            console.log('WebSocket disconnected');
            setTimeout(() => location.reload(), 5000);
        };
        
        // Initial load
        loadOverview();
        loadSites();
        loadAlerts();
        
        function updateDashboard(data) {
            document.getElementById('monitor-count').textContent = data.monitor_count || 0;
            document.getElementById('critical-count').textContent = data.critical_alerts || 0;
            document.getElementById('high-count').textContent = data.high_alerts || 0;
            document.getElementById('changes-count').textContent = data.changes_today || 0;
        }
        
        async function loadOverview() {
            try {
                const response = await fetch('/api/overview');
                const data = await response.json();
                updateDashboard(data);
            } catch (error) {
                console.error('Failed to load overview:', error);
            }
        }
        
        async function loadSites() {
            try {
                const response = await fetch('/api/sites');
                const sites = await response.json();
                renderSites(sites);
            } catch (error) {
                console.error('Failed to load sites:', error);
            }
        }
        
        function renderSites(sites) {
            const container = document.getElementById('sites-content');
            
            if (!sites || sites.length === 0) {
                container.innerHTML = `
                    <div class="empty-state">
                        <div class="empty-state-icon">📭</div>
                        <p>No monitored sites configured</p>
                    </div>
                `;
                return;
            }
            
            container.innerHTML = `
                <table>
                    <thead>
                        <tr>
                            <th>Site</th>
                            <th>Status</th>
                            <th>Last Check</th>
                            <th>Last Change</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${sites.map(site => `
                            <tr>
                                <td>
                                    <div class="site-name">${site.name}</div>
                                    <div class="site-url">${site.url}</div>
                                </td>
                                <td>
                                    <span class="badge badge-${site.status}">
                                        ${site.status === 'active' ? '🟢' : '🔴'} ${site.status}
                                    </span>
                                </td>
                                <td>${site.last_check ? new Date(site.last_check).toLocaleString() : 'Never'}</td>
                                <td>${site.last_change ? new Date(site.last_change).toLocaleString() : 'Never'}</td>
                                <td>
                                    <button class="btn btn-secondary" onclick="checkSite('${site.id}')">Check</button>
                                    <button class="btn btn-secondary" onclick="viewHistory('${site.id}')">History</button>
                                </td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            `;
        }
        
        async function loadAlerts() {
            try {
                const response = await fetch('/api/alerts?limit=10');
                const alerts = await response.json();
                renderAlerts(alerts);
            } catch (error) {
                console.error('Failed to load alerts:', error);
            }
        }
        
        function renderAlerts(alerts) {
            const container = document.getElementById('alerts-content');
            
            if (!alerts || alerts.length === 0) {
                container.innerHTML = `
                    <div class="empty-state">
                        <div class="empty-state-icon">✅</div>
                        <p>No alerts in the last 24 hours</p>
                    </div>
                `;
                return;
            }
            
            container.innerHTML = alerts.map(alert => `
                <div class="alert-item">
                    <div class="alert-icon ${alert.severity.toLowerCase()}">
                        ${alert.severity === 'CRITICAL' ? '🚨' : alert.severity === 'HIGH' ? '⚠️' : '🔔'}
                    </div>
                    <div class="alert-content">
                        <div class="alert-title">${alert.site_name}: ${alert.category} Change</div>
                        <div class="alert-meta">
                            ${alert.severity} • ${alert.change_type} • ${new Date(alert.detected_at).toLocaleString()}
                        </div>
                        <div style="margin-top: 8px; color: #94a3b8;">${alert.description}</div>
                    </div>
                </div>
            `).join('');
        }
        
        async function checkSite(siteId) {
            try {
                await fetch(`/api/sites/${siteId}/check`, {method: 'POST'});
                setTimeout(loadSites, 2000);
            } catch (error) {
                console.error('Failed to check site:', error);
            }
        }
        
        async function checkAll() {
            const sites = await fetch('/api/sites').then(r => r.json());
            for (const site of sites) {
                await checkSite(site.id);
            }
        }
        
        function viewHistory(siteId) {
            window.location.href = `/history/${siteId}`;
        }
        
        function refreshSites() {
            loadSites();
        }
        
        function refreshAlerts() {
            loadAlerts();
        }
        
        // Auto-refresh every 30 seconds
        setInterval(() => {
            loadOverview();
            loadSites();
            loadAlerts();
        }, 30000);
    </script>
</body>
</html>
"""
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Collect real system metrics via psutil."""
        metrics: Dict[str, Any] = {}
        if not HAS_PSUTIL:
            return metrics
        try:
            metrics['cpu_percent'] = psutil.cpu_percent(interval=0)
            mem = psutil.virtual_memory()
            metrics['memory'] = {
                'total': mem.total,
                'available': mem.available,
                'percent': mem.percent,
            }
            disk = psutil.disk_usage(os.path.abspath(os.sep))
            metrics['disk'] = {
                'total': disk.total,
                'used': disk.used,
                'free': disk.free,
                'percent': disk.percent,
            }
            metrics['uptime_seconds'] = round(_time.monotonic() - self._start_time, 1)
        except Exception as exc:
            logger.debug("psutil metrics error: %s", exc)
        return metrics

    def _get_overview_data(self) -> Dict:
        """Get overview statistics with real system metrics"""
        # Count alerts by severity
        alert_stats = self.alert_manager.get_statistics() if self.alert_manager else {}

        # Count changes today
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        changes_today = sum(
            1 for change in self.engine.change_history
            if change.detected_at >= today
        )

        overview = {
            'monitor_count': len(self.engine.monitors),
            'checks_run': self.engine.check_count,
            'changes_today': changes_today,
            'critical_alerts': alert_stats.get('by_severity', {}).get('CRITICAL', 0),
            'high_alerts': alert_stats.get('by_severity', {}).get('HIGH', 0),
            'medium_alerts': alert_stats.get('by_severity', {}).get('MEDIUM', 0),
            'low_alerts': alert_stats.get('by_severity', {}).get('LOW', 0),
            'system_status': 'healthy' if self.engine.running else 'stopped',
            'timestamp': datetime.now().isoformat(),
        }

        # Merge real system metrics
        overview['system_metrics'] = self._get_system_metrics()

        return overview
    
    def _get_sites_data(self) -> List[Dict]:
        """Get all monitored sites"""
        sites = []
        
        for monitor_id, config in self.engine.monitors.items():
            snapshot = self.engine.snapshots.get(monitor_id)
            
            # Get last change for this site
            last_change = None
            for change in reversed(self.engine.change_history):
                if change.site_id == monitor_id:
                    last_change = change.detected_at
                    break
            
            sites.append({
                'id': monitor_id,
                'name': config.name,
                'url': config.url,
                'status': 'active',
                'last_check': snapshot.timestamp.isoformat() if snapshot else None,
                'last_change': last_change.isoformat() if last_change else None,
                'config': {
                    'dom_monitoring': config.dom_monitoring,
                    'visual_monitoring': config.visual_monitoring,
                    'content_monitoring': config.content_monitoring,
                    'check_interval': config.check_interval
                }
            })
        
        return sites
    
    def _get_site_history(self, site_id: str, limit: int, days: int) -> List[Dict]:
        """Get change history for a site"""
        cutoff = datetime.now() - timedelta(days=days)
        
        changes = [
            change.to_dict()
            for change in self.engine.change_history
            if change.site_id == site_id and change.detected_at >= cutoff
        ]
        
        return sorted(changes, key=lambda c: c['detected_at'], reverse=True)[:limit]
    
    def _get_changes_data(self, site_id: Optional[str], severity: Optional[str],
                          category: Optional[str], limit: int) -> List[Dict]:
        """Get changes with filtering"""
        changes = self.engine.change_history
        
        if site_id:
            changes = [c for c in changes if c.site_id == site_id]
        
        if severity:
            changes = [c for c in changes if c.severity.name == severity.upper()]
        
        if category:
            changes = [c for c in changes if c.category == category]
        
        return [
            change.to_dict()
            for change in sorted(changes, key=lambda c: c.detected_at, reverse=True)[:limit]
        ]
    
    def _get_change_data(self, change_id: str) -> Optional[Dict]:
        """Get detailed change data"""
        for change in self.engine.change_history:
            # Note: This assumes change_id matches some identifier
            # You may need to add an ID field to the Change class
            if hasattr(change, 'id') and change.id == change_id:
                return change.to_dict()
        
        return None
    
    def _get_alerts_data(self, site_id: Optional[str], severity: Optional[str],
                         limit: int) -> List[Dict]:
        """Get alerts with filtering"""
        if not self.alert_manager:
            return []
        
        from alert_manager import SeverityLevel
        
        sev = None
        if severity:
            try:
                sev = SeverityLevel[severity.upper()]
            except KeyError:
                pass
        
        alerts = self.alert_manager.get_alert_history(site_id, sev, limit)
        return [alert.to_dict() for alert in alerts]
    
    def _get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        engine_stats = self.engine.get_status()
        scheduler_stats = self.scheduler.get_statistics() if self.scheduler else {}
        alert_stats = self.alert_manager.get_statistics() if self.alert_manager else {}
        
        # Calculate change trends
        now = datetime.now()
        day_ago = now - timedelta(days=1)
        week_ago = now - timedelta(days=7)
        
        changes_24h = sum(1 for c in self.engine.change_history if c.detected_at >= day_ago)
        changes_7d = sum(1 for c in self.engine.change_history if c.detected_at >= week_ago)
        
        return {
            'engine': engine_stats,
            'scheduler': scheduler_stats,
            'alerts': alert_stats,
            'changes': {
                'total': len(self.engine.change_history),
                'last_24h': changes_24h,
                'last_7d': changes_7d
            },
            'timestamp': now.isoformat()
        }
    
    async def _handle_websocket(self, websocket: WebSocket):
        """Handle WebSocket connections with background metric pushes."""
        await websocket.accept()
        self.connected_clients.append(websocket)

        async def _push_metrics():
            """Background task: push system metrics every 5 seconds."""
            try:
                while True:
                    metrics = self._get_system_metrics()
                    if metrics:
                        await websocket.send_json({
                            'type': 'system_metrics',
                            'data': metrics,
                            'timestamp': datetime.now().isoformat(),
                        })
                    await asyncio.sleep(5)
            except (WebSocketDisconnect, OSError, RuntimeError):
                pass

        metrics_task = asyncio.create_task(_push_metrics())

        try:
            while True:
                # Send periodic overview updates
                data = self._get_overview_data()
                data['type'] = 'overview'
                await websocket.send_json(data)

                # Wait before next update
                await asyncio.sleep(self.refresh_interval)

        except WebSocketDisconnect:
            pass
        except (OSError, RuntimeError, ValueError) as e:
            logger.warning("WebSocket error: %s", e)
        finally:
            metrics_task.cancel()
            if websocket in self.connected_clients:
                self.connected_clients.remove(websocket)
    
    async def broadcast_update(self):
        """Broadcast update to all connected clients"""
        if not self.connected_clients:
            return
        
        data = self._get_overview_data()
        
        disconnected = []
        for client in self.connected_clients:
            try:
                await client.send_json(data)
            except (OSError, RuntimeError, ValueError) as e:
                logger.warning(f"Dashboard error: {e}")
                disconnected.append(client)
        
        # Remove disconnected clients
        for client in disconnected:
            self.connected_clients.remove(client)
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the dashboard server"""
        if not HAS_FASTAPI:
            print("FastAPI not available. Cannot start dashboard server.")
            return
        
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)


# Example usage
def main():
    """Example usage of the dashboard"""
    from monitor_engine import MonitorEngine, MonitorConfig
    from scheduler import CronScheduler
    from alert_manager import AlertManager
    
    # Create components
    engine = MonitorEngine({
        'dom': {},
        'visual': {},
        'content': {},
        'storage': {'snapshots_dir': './data/snapshots'}
    })
    
    scheduler = CronScheduler(engine)
    alert_manager = AlertManager({'channels': []})
    
    # Add a monitor
    monitor_id = engine.add_monitor(MonitorConfig(
        url='https://example.com',
        name='Example Site',
        dom_monitoring=True,
        visual_monitoring=True,
        content_monitoring=True
    ))
    
    # Add schedule
    scheduler.add_job(monitor_id, 'frequent')
    
    # Create dashboard
    dashboard = DashboardAPI(engine, scheduler, alert_manager, {
        'refresh_interval': 5
    })
    
    dashboard_url = os.environ.get('DASHBOARD_URL', 'http://localhost:8000')
    print(f"Dashboard starting on {dashboard_url}")
    dashboard.run()


if __name__ == '__main__':
    main()
