"""
OpenClaw Web Monitoring Agent Loop
Integration of web monitoring into OpenClaw's 15 agentic loops
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

# Import monitoring components
from monitor_engine import MonitorEngine, MonitorConfig
from scheduler import CronScheduler, SchedulePresets
from alert_manager import AlertManager
from dashboard_server import DashboardAPI


class WebMonitoringAgentLoop:
    """
    Agent Loop: Web Monitoring & Change Detection
    
    This is one of the 15 hardcoded agentic loops in the OpenClaw system.
    It provides continuous web monitoring with intelligent change detection
    and multi-channel alerting.
    
    Loop ID: web_monitor
    Priority: 8 (High)
    """
    
    LOOP_ID = "web_monitor"
    LOOP_NAME = "Web Monitoring & Change Detection"
    PRIORITY = 8
    
    def __init__(self, agent_core):
        """
        Initialize the web monitoring agent loop
        
        Args:
            agent_core: Reference to the main OpenClaw agent core
        """
        self.agent = agent_core
        self.monitor_engine: Optional[MonitorEngine] = None
        self.scheduler: Optional[CronScheduler] = None
        self.alert_manager: Optional[AlertManager] = None
        self.dashboard: Optional[DashboardAPI] = None
        
        # State
        self.initialized = False
        self.running = False
        self._monitor_task = None
        self._dashboard_task = None
        
        # Configuration
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """Load configuration from file or use defaults"""
        config_paths = [
            './config/web_monitoring.yaml',
            './config.yaml',
            os.path.expanduser('~/.openclaw/web_monitoring.yaml'),
        ]
        
        for path in config_paths:
            if os.path.exists(path):
                try:
                    import yaml
                    with open(path, 'r') as f:
                        config = yaml.safe_load(f)
                        return config.get('web_monitoring', {})
                except Exception as e:
                    print(f"Failed to load config from {path}: {e}")
        
        # Default configuration
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'system': {
                'max_concurrent_checks': 5,
                'request_timeout': 30,
                'user_agent': 'OpenClaw-WebMonitor/1.0'
            },
            'dom': {
                'enabled': True,
                'ignore_elements': ['script', 'style', 'noscript']
            },
            'visual': {
                'enabled': True,
                'threshold': 0.1
            },
            'content': {
                'enabled': True,
                'selectors': ['article', 'main', '.content']
            },
            'alerts': {
                'enabled': True,
                'channels': ['email', 'tts'],
                'throttling': {
                    'cooldown_periods': {
                        'CRITICAL': 0,
                        'HIGH': 300,
                        'MEDIUM': 900,
                        'LOW': 3600
                    }
                }
            },
            'scheduling': {
                'default_interval': 3600
            },
            'storage': {
                'snapshots_dir': './data/snapshots'
            },
            'dashboard': {
                'enabled': True,
                'host': '0.0.0.0',
                'port': 8000
            }
        }
    
    async def initialize(self):
        """Initialize the monitoring loop"""
        await self.agent.log(
            f"Initializing {self.LOOP_NAME}",
            level="INFO",
            component=self.LOOP_ID
        )
        
        try:
            # Create monitoring engine
            self.monitor_engine = MonitorEngine(self.config)
            
            # Create scheduler
            self.scheduler = CronScheduler(self.monitor_engine, self.config.get('scheduling', {}))
            
            # Create alert manager
            self.alert_manager = AlertManager(self.config.get('alerts', {}))
            
            # Create dashboard
            if self.config.get('dashboard', {}).get('enabled', True):
                self.dashboard = DashboardAPI(
                    self.monitor_engine,
                    self.scheduler,
                    self.alert_manager,
                    self.config.get('dashboard', {})
                )
            
            # Load monitored sites from agent memory
            await self._load_sites_from_memory()
            
            # Register with agent's heartbeat
            self.agent.heartbeat.register_component(self.LOOP_ID, self.get_status)
            
            self.initialized = True
            
            await self.agent.log(
                f"{self.LOOP_NAME} initialized successfully",
                level="INFO",
                component=self.LOOP_ID
            )
            
        except Exception as e:
            await self.agent.log(
                f"Failed to initialize {self.LOOP_NAME}: {e}",
                level="ERROR",
                component=self.LOOP_ID
            )
            raise
    
    async def _load_sites_from_memory(self):
        """Load monitored sites from agent memory"""
        sites = await self.agent.memory.get('monitored_sites', [])
        
        for site_data in sites:
            try:
                monitor_id = self.add_site(
                    url=site_data['url'],
                    name=site_data['name'],
                    schedule=site_data.get('schedule', 'regular'),
                    config=site_data.get('config', {})
                )
                
                await self.agent.log(
                    f"Loaded site: {site_data['name']} ({monitor_id})",
                    level="DEBUG",
                    component=self.LOOP_ID
                )
                
            except Exception as e:
                await self.agent.log(
                    f"Failed to load site {site_data.get('name')}: {e}",
                    level="WARNING",
                    component=self.LOOP_ID
                )
    
    async def run(self):
        """Main loop execution"""
        if not self.initialized:
            await self.initialize()
        
        await self.agent.log(
            f"Starting {self.LOOP_NAME}",
            level="INFO",
            component=self.LOOP_ID
        )
        
        self.running = True
        
        # Start scheduler
        self._monitor_task = asyncio.create_task(self._run_scheduler())
        
        # Start dashboard if enabled
        if self.dashboard:
            self._dashboard_task = asyncio.create_task(self._run_dashboard())
        
        # Main loop
        while self.running:
            try:
                # Periodic status update
                await self.agent.log(
                    f"Status: {self.get_status()}",
                    level="DEBUG",
                    component=self.LOOP_ID
                )
                
                # Wait before next iteration
                await asyncio.sleep(60)
                
            except Exception as e:
                await self.agent.log(
                    f"Error in {self.LOOP_NAME} loop: {e}",
                    level="ERROR",
                    component=self.LOOP_ID
                )
                await asyncio.sleep(5)
    
    async def _run_scheduler(self):
        """Run the monitoring scheduler"""
        try:
            await self.scheduler.run()
        except Exception as e:
            await self.agent.log(
                f"Scheduler error: {e}",
                level="ERROR",
                component=self.LOOP_ID
            )
    
    async def _run_dashboard(self):
        """Run the dashboard server"""
        try:
            config = self.config.get('dashboard', {})
            host = config.get('host', '0.0.0.0')
            port = config.get('port', 8000)
            
            await self.agent.log(
                f"Dashboard starting on http://{host}:{port}",
                level="INFO",
                component=self.LOOP_ID
            )
            
            # Run in executor since uvicorn is blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.dashboard.run,
                host,
                port
            )
            
        except Exception as e:
            await self.agent.log(
                f"Dashboard error: {e}",
                level="ERROR",
                component=self.LOOP_ID
            )
    
    async def on_change_detected(self, change):
        """Handle detected change"""
        # Log to agent
        severity_level = "WARNING" if change.severity.value >= 4 else "INFO"
        
        await self.agent.log(
            f"Change detected on {change.site_name}: {change.description}",
            level=severity_level,
            component=self.LOOP_ID,
            data={
                'site_id': change.site_id,
                'change_type': change.change_type.value,
                'severity': change.severity.name,
                'category': change.category
            }
        )
        
        # Store in agent memory
        await self.agent.memory.store(
            'recent_changes',
            change.to_dict(),
            ttl=86400  # 24 hours
        )
        
        # Process alert
        alert = await self.alert_manager.process_change(change.to_dict())
        
        if alert:
            # Notify through agent's notification system
            await self.agent.notify({
                'type': 'web_change',
                'severity': change.severity.name,
                'site': change.site_name,
                'message': change.description,
                'category': change.category,
                'channels': alert.channels_sent
            })
            
            # Use GPT-5.2 for intelligent analysis
            if change.severity.value >= 3:  # MEDIUM or higher
                await self._analyze_change_with_ai(change)
    
    async def _analyze_change_with_ai(self, change):
        """Use GPT-5.2 to analyze the change"""
        try:
            prompt = f"""
            Analyze this website change detected by OpenClaw monitoring:
            
            Site: {change.site_name}
            URL: {change.site_url}
            Change Type: {change.change_type.value}
            Category: {change.category}
            Severity: {change.severity.name}
            Description: {change.description}
            
            Change Details:
            {json.dumps(change.diff_data, indent=2)[:2000]}
            
            Please provide:
            1. A concise summary of what changed
            2. The potential impact of this change
            3. Recommended action for the user
            4. Any patterns or anomalies you notice
            
            Keep your response brief and actionable.
            """
            
            analysis = await self.agent.gpt52.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=500
            )
            
            await self.agent.log(
                f"AI Analysis for {change.site_name}: {analysis}",
                level="INFO",
                component=self.LOOP_ID
            )
            
            # Store analysis
            await self.agent.memory.store(
                f'change_analysis:{change.site_id}:{int(change.detected_at.timestamp())}',
                {
                    'change': change.to_dict(),
                    'analysis': analysis
                },
                ttl=604800  # 7 days
            )
            
        except Exception as e:
            await self.agent.log(
                f"AI analysis failed: {e}",
                level="WARNING",
                component=self.LOOP_ID
            )
    
    def add_site(self, url: str, name: str, schedule: str = 'regular',
                 config: Dict = None) -> str:
        """
        Add a new site to monitor
        
        Args:
            url: URL to monitor
            name: Display name for the site
            schedule: Schedule preset or cron expression
            config: Additional monitoring configuration
        
        Returns:
            monitor_id: Unique identifier for the monitor
        """
        config = config or {}
        
        # Create monitor config
        monitor_config = MonitorConfig(
            url=url,
            name=name,
            check_interval=config.get('check_interval', 3600),
            dom_monitoring=config.get('dom', True),
            visual_monitoring=config.get('visual', True),
            content_monitoring=config.get('content', True),
            selectors=config.get('selectors', []),
            alert_threshold=config.get('alert_threshold', 0.1),
            notification_channels=config.get('notification_channels', ['email'])
        )
        
        # Add to engine
        monitor_id = self.monitor_engine.add_monitor(monitor_config)
        
        # Add to scheduler
        self.scheduler.add_job(monitor_id, schedule)
        
        return monitor_id
    
    def remove_site(self, monitor_id: str) -> bool:
        """Remove a monitored site"""
        self.scheduler.remove_job(monitor_id)
        return self.monitor_engine.remove_monitor(monitor_id)
    
    def update_site_schedule(self, monitor_id: str, schedule: str) -> bool:
        """Update a site's monitoring schedule"""
        return self.scheduler.update_job(monitor_id, schedule) is not None
    
    async def check_site_now(self, monitor_id: str) -> Optional[Dict]:
        """Immediately check a specific site"""
        change = await self.monitor_engine.run_check(monitor_id)
        return change.to_dict() if change else None
    
    def get_site_status(self, monitor_id: str) -> Optional[Dict]:
        """Get status for a specific site"""
        return self.monitor_engine.get_monitor_status(monitor_id)
    
    def list_sites(self) -> List[Dict]:
        """List all monitored sites"""
        sites = []
        for monitor_id, config in self.monitor_engine.monitors.items():
            sites.append({
                'id': monitor_id,
                'name': config.name,
                'url': config.url,
                'schedule': self.scheduler.get_job(monitor_id).to_dict() if self.scheduler.get_job(monitor_id) else None,
                'status': self.get_site_status(monitor_id)
            })
        return sites
    
    def get_change_history(self, site_id: str = None, limit: int = 50) -> List[Dict]:
        """Get change history"""
        changes = self.monitor_engine.change_history
        
        if site_id:
            changes = [c for c in changes if c.site_id == site_id]
        
        return [
            change.to_dict()
            for change in sorted(changes, key=lambda c: c.detected_at, reverse=True)[:limit]
        ]
    
    def get_alert_history(self, site_id: str = None, limit: int = 50) -> List[Dict]:
        """Get alert history"""
        alerts = self.alert_manager.get_alert_history(site_id, limit=limit)
        return [alert.to_dict() for alert in alerts]
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        return {
            'loop_id': self.LOOP_ID,
            'loop_name': self.LOOP_NAME,
            'initialized': self.initialized,
            'running': self.running,
            'monitors': len(self.monitor_engine.monitors) if self.monitor_engine else 0,
            'checks_run': self.monitor_engine.check_count if self.monitor_engine else 0,
            'changes_detected': self.monitor_engine.change_count if self.monitor_engine else 0,
            'alerts_sent': len(self.alert_manager.alert_history) if self.alert_manager else 0,
            'scheduler_jobs': len(self.scheduler.schedules) if self.scheduler else 0,
            'dashboard_enabled': self.dashboard is not None
        }
    
    def get_status(self) -> Dict:
        """Return current status for heartbeat"""
        return {
            'status': 'active' if self.running else 'inactive',
            'initialized': self.initialized,
            'monitors': len(self.monitor_engine.monitors) if self.monitor_engine else 0,
            'last_check': None,  # Could be enhanced to track last check time
            'health': 'healthy' if self.running else 'stopped'
        }
    
    async def stop(self):
        """Stop the monitoring loop"""
        await self.agent.log(
            f"Stopping {self.LOOP_NAME}",
            level="INFO",
            component=self.LOOP_ID
        )
        
        self.running = False
        
        if self.scheduler:
            self.scheduler.stop()
        
        if self.monitor_engine:
            self.monitor_engine.stop()
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        if self._dashboard_task:
            self._dashboard_task.cancel()
            try:
                await self._dashboard_task
            except asyncio.CancelledError:
                pass
        
        await self.agent.log(
            f"{self.LOOP_NAME} stopped",
            level="INFO",
            component=self.LOOP_ID
        )
    
    # Agent Commands
    
    async def cmd_add_site(self, url: str, name: str, schedule: str = 'regular') -> Dict:
        """Command: Add a new site to monitor"""
        monitor_id = self.add_site(url, name, schedule)
        
        # Save to agent memory
        sites = await self.agent.memory.get('monitored_sites', [])
        sites.append({
            'id': monitor_id,
            'url': url,
            'name': name,
            'schedule': schedule
        })
        await self.agent.memory.store('monitored_sites', sites)
        
        return {
            'success': True,
            'monitor_id': monitor_id,
            'message': f"Added {name} to monitoring"
        }
    
    async def cmd_remove_site(self, monitor_id: str) -> Dict:
        """Command: Remove a site from monitoring"""
        success = self.remove_site(monitor_id)
        
        if success:
            # Update agent memory
            sites = await self.agent.memory.get('monitored_sites', [])
            sites = [s for s in sites if s.get('id') != monitor_id]
            await self.agent.memory.store('monitored_sites', sites)
        
        return {
            'success': success,
            'message': f"Removed {monitor_id}" if success else f"Site {monitor_id} not found"
        }
    
    async def cmd_list_sites(self) -> Dict:
        """Command: List all monitored sites"""
        return {
            'success': True,
            'sites': self.list_sites()
        }
    
    async def cmd_check_now(self, monitor_id: str = None) -> Dict:
        """Command: Run a check immediately"""
        if monitor_id:
            change = await self.check_site_now(monitor_id)
            return {
                'success': True,
                'change_detected': change is not None,
                'change': change
            }
        else:
            changes = await self.monitor_engine.run_all_checks()
            return {
                'success': True,
                'changes_detected': len(changes),
                'changes': [c.to_dict() for c in changes]
            }
    
    async def cmd_get_history(self, site_id: str = None, limit: int = 50) -> Dict:
        """Command: Get change history"""
        return {
            'success': True,
            'history': self.get_change_history(site_id, limit)
        }
    
    async def cmd_get_statistics(self) -> Dict:
        """Command: Get monitoring statistics"""
        return {
            'success': True,
            'statistics': self.get_statistics()
        }


# Integration with OpenClaw Agent Core
class OpenClawAgentCore:
    """
    Mock agent core for demonstration.
    In production, this would be the actual OpenClaw agent core.
    """
    
    def __init__(self):
        self.heartbeat = HeartbeatManager()
        self.memory = MemoryManager()
        self.gpt52 = GPT52Mock()
        
    async def log(self, message: str, level: str = "INFO", component: str = None, data: Dict = None):
        """Log a message"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        comp = f"[{component}] " if component else ""
        print(f"[{timestamp}] [{level}] {comp}{message}")
        
    async def notify(self, notification: Dict):
        """Send a notification"""
        print(f"[NOTIFICATION] {notification}")


class HeartbeatManager:
    """Mock heartbeat manager"""
    
    def __init__(self):
        self.components = {}
        
    def register_component(self, name: str, status_fn):
        """Register a component for heartbeat"""
        self.components[name] = status_fn


class MemoryManager:
    """Mock memory manager"""
    
    def __init__(self):
        self._data = {}
        
    async def get(self, key: str, default=None):
        """Get value from memory"""
        return self._data.get(key, default)
        
    async def store(self, key: str, value, ttl: int = None):
        """Store value in memory"""
        self._data[key] = value


class GPT52Mock:
    """Mock GPT-5.2 for demonstration"""
    
    async def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 500):
        """Generate text"""
        return f"[AI Analysis] This is a mock analysis of the change. The change appears to be significant and requires attention."


# Example usage
async def main():
    """Example usage of the web monitoring agent loop"""
    
    # Create mock agent core
    agent = OpenClawAgentCore()
    
    # Create and initialize the loop
    loop = WebMonitoringAgentLoop(agent)
    
    # Initialize
    await loop.initialize()
    
    # Add a test site
    result = await loop.cmd_add_site(
        url='https://example.com',
        name='Example Site',
        schedule='frequent'
    )
    print(f"Add site result: {result}")
    
    # List sites
    sites = await loop.cmd_list_sites()
    print(f"Sites: {sites}")
    
    # Get statistics
    stats = await loop.cmd_get_statistics()
    print(f"Statistics: {stats}")
    
    # Run for a short time
    print("\nRunning for 30 seconds...")
    
    async def stop_after_delay():
        await asyncio.sleep(30)
        await loop.stop()
    
    await asyncio.gather(
        loop.run(),
        stop_after_delay()
    )
    
    print("\nFinal statistics:")
    print(await loop.cmd_get_statistics())


if __name__ == '__main__':
    asyncio.run(main())
