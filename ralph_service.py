"""
Ralph Loop Windows Service
Multi-Layered Background Processing with Priority Queuing
OpenClaw Windows 10 AI Agent System

This module provides a Windows service wrapper for the Ralph Loop.
"""

import asyncio
import sys
import os
import logging
import yaml
import signal
from pathlib import Path
from datetime import datetime

# Windows service imports
import win32service
import win32serviceutil
import win32event
import servicemanager

# Add Ralph Loop to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ralph_loop_implementation import RalphLoop, Task, Job, PriorityLevel

logger = logging.getLogger("RalphService")


class RalphLoopService(win32serviceutil.ServiceFramework):
    """
    Windows Service for Ralph Loop
    """
    
    # Service metadata
    _svc_name_ = "RalphLoop"
    _svc_display_name_ = "OpenClaw Ralph Loop Service"
    _svc_description_ = "Multi-layered background processing with priority queuing for OpenClaw AI Agent"
    
    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        
        # Create stop event
        self.stop_event = win32event.CreateEvent(None, 0, 0, None)
        
        # Service state
        self.running = False
        self.ralph_loop: Optional[RalphLoop] = None
        self.config: Optional[dict] = None
        
    def SvcStop(self):
        """Handle service stop request"""
        logger.info("Service stop requested")
        self.running = False
        win32event.SetEvent(self.stop_event)
        
        # Stop Ralph Loop
        if self.ralph_loop:
            asyncio.create_task(self.ralph_loop.stop())
        
        # Report service status
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        
    def SvcDoRun(self):
        """Main service execution"""
        try:
            # Log service start
            servicemanager.LogMsg(
                servicemanager.EVENTLOG_INFORMATION_TYPE,
                servicemanager.PYS_SERVICE_STARTED,
                (self._svc_name_, '')
            )
            
            logger.info("=" * 60)
            logger.info("Ralph Loop Service Starting")
            logger.info(f"Timestamp: {datetime.now().isoformat()}")
            logger.info("=" * 60)
            
            # Load configuration
            self._load_config()
            
            # Initialize and run Ralph Loop
            self.running = True
            asyncio.run(self._run_service())
            
        except (OSError, RuntimeError, ValueError) as e:
            logger.exception("Service error")
            servicemanager.LogMsg(
                servicemanager.EVENTLOG_ERROR_TYPE,
                servicemanager.PYS_SERVICE_STOPPED,
                (self._svc_name_, str(e))
            )
            raise
            
    def _load_config(self):
        """Load configuration from YAML file"""
        config_paths = [
            'config/ralph_loop_config.yaml',
            'ralph_loop_config.yaml',
            'C:/OpenClaw/Config/ralph_loop_config.yaml',
            os.path.expandvars('${RALPH_CONFIG_PATH}'),
        ]
        
        for path in config_paths:
            if path and os.path.exists(path):
                logger.info(f"Loading configuration from: {path}")
                with open(path, 'r') as f:
                    self.config = yaml.safe_load(f)
                    if 'ralph_loop' in self.config:
                        self.config = self.config['ralph_loop']
                return
        
        # Use default configuration
        logger.warning("No configuration file found, using defaults")
        self.config = self._default_config()
        
    def _default_config(self) -> dict:
        """Return default configuration"""
        return {
            'layers': {
                'critical': {'max_concurrent': 10},
                'real_time': {'max_concurrent': 20},
                'standard': {'max_concurrent': 100}
            },
            'recovery': {'auto_recover': True},
            'storage': {'db_path': 'data/ralph_queue.db'},
            'monitoring': {'interval_seconds': 10}
        }
        
    async def _run_service(self):
        """Run the Ralph Loop service"""
        try:
            # Create Ralph Loop instance
            self.ralph_loop = RalphLoop(self.config)
            
            # Initialize
            await self.ralph_loop.initialize()
            
            # Start
            await self.ralph_loop.start()
            
            logger.info("Ralph Loop service is running")
            
            # Keep running until stopped
            while self.running:
                await asyncio.sleep(1)
                
                # Check Windows service stop event
                if win32event.WaitForSingleObject(self.stop_event, 0) == win32event.WAIT_OBJECT_0:
                    logger.info("Stop event received")
                    break
                    
        except (OSError, RuntimeError, ValueError) as e:
            logger.exception("Error in service loop")
            raise
        finally:
            # Cleanup
            if self.ralph_loop:
                await self.ralph_loop.stop()
                
            logger.info("Ralph Loop service stopped")
            
            # Log service stop
            servicemanager.LogMsg(
                servicemanager.EVENTLOG_INFORMATION_TYPE,
                servicemanager.PYS_SERVICE_STOPPED,
                (self._svc_name_, '')
            )


class RalphLoopController:
    """
    Controller for managing Ralph Loop outside of Windows service context
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config: Optional[dict] = None
        self.ralph_loop: Optional[RalphLoop] = None
        self.running = False
        
    def load_config(self, config_path: Optional[str] = None):
        """Load configuration"""
        path = config_path or self.config_path
        
        if path and os.path.exists(path):
            logger.info(f"Loading configuration from: {path}")
            with open(path, 'r') as f:
                self.config = yaml.safe_load(f)
                if 'ralph_loop' in self.config:
                    self.config = self.config['ralph_loop']
        else:
            logger.warning("Using default configuration")
            self.config = {
                'layers': {
                    'critical': {'max_concurrent': 10},
                    'real_time': {'max_concurrent': 20},
                    'standard': {'max_concurrent': 100}
                },
                'recovery': {'auto_recover': True},
                'storage': {'db_path': 'data/ralph_queue.db'},
                'monitoring': {'interval_seconds': 10}
            }
            
    async def start(self):
        """Start Ralph Loop"""
        if self.running:
            logger.warning("Ralph Loop is already running")
            return
            
        if not self.config:
            self.load_config()
            
        self.ralph_loop = RalphLoop(self.config)
        await self.ralph_loop.initialize()
        await self.ralph_loop.start()
        self.running = True
        
        logger.info("Ralph Loop controller started")
        
    async def stop(self):
        """Stop Ralph Loop"""
        if not self.running:
            logger.warning("Ralph Loop is not running")
            return
            
        if self.ralph_loop:
            await self.ralph_loop.stop()
            
        self.running = False
        logger.info("Ralph Loop controller stopped")
        
    async def submit_task(self, task_type: str, handler, data=None, 
                          priority: Optional[int] = None) -> str:
        """Submit a task"""
        if not self.running or not self.ralph_loop:
            raise RuntimeError("Ralph Loop is not running")
            
        task = Task(
            task_type=task_type,
            handler=handler,
            data=data,
            priority=priority or PriorityLevel.P64_AGENT_LOOP
        )
        
        return await self.ralph_loop.submit_task(task)
        
    async def get_stats(self) -> dict:
        """Get system statistics"""
        if not self.running or not self.ralph_loop:
            return {"error": "Ralph Loop is not running"}
            
        return await self.ralph_loop.get_stats()
        
    async def get_health(self) -> dict:
        """Get system health"""
        if not self.running or not self.ralph_loop:
            return {"error": "Ralph Loop is not running"}
            
        return await self.ralph_loop.get_health()


def install_service():
    """Install Ralph Loop as Windows service"""
    try:
        win32serviceutil.InstallService(
            RalphLoopService._svc_class_path_,
            RalphLoopService._svc_name_,
            RalphLoopService._svc_display_name_,
            startType=win32service.SERVICE_AUTO_START
        )
        print(f"Service '{RalphLoopService._svc_name_}' installed successfully")
        print(f"Display Name: {RalphLoopService._svc_display_name_}")
        print("\nTo start the service, run:")
        print(f"  python ralph_service.py start")
        print("\nOr use Windows Services console:")
        print("  services.msc")
    except (OSError, RuntimeError) as e:
        print(f"Failed to install service: {e}")
        raise


def remove_service():
    """Remove Ralph Loop Windows service"""
    try:
        win32serviceutil.RemoveService(RalphLoopService._svc_name_)
        print(f"Service '{RalphLoopService._svc_name_}' removed successfully")
    except (OSError, RuntimeError) as e:
        print(f"Failed to remove service: {e}")
        raise


def start_service():
    """Start Ralph Loop Windows service"""
    try:
        win32serviceutil.StartService(RalphLoopService._svc_name_)
        print(f"Service '{RalphLoopService._svc_name_}' started")
    except (OSError, RuntimeError) as e:
        print(f"Failed to start service: {e}")
        raise


def stop_service():
    """Stop Ralph Loop Windows service"""
    try:
        win32serviceutil.StopService(RalphLoopService._svc_name_)
        print(f"Service '{RalphLoopService._svc_name_}' stopped")
    except (OSError, RuntimeError) as e:
        print(f"Failed to stop service: {e}")
        raise


def restart_service():
    """Restart Ralph Loop Windows service"""
    stop_service()
    start_service()


def run_console():
    """Run Ralph Loop in console mode (not as service)"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ralph Loop Controller')
    parser.add_argument('--config', '-c', help='Path to configuration file')
    parser.add_argument('--command', '-cmd', 
                       choices=['start', 'stop', 'stats', 'health'],
                       default='start',
                       help='Command to execute')
    
    args = parser.parse_args()
    
    controller = RalphLoopController(args.config)
    
    if args.command == 'start':
        try:
            asyncio.run(controller.start())
            
            # Keep running
            print("Ralph Loop is running. Press Ctrl+C to stop.")
            while True:
                try:
                    asyncio.run(asyncio.sleep(1))
                except KeyboardInterrupt:
                    break
                    
        finally:
            asyncio.run(controller.stop())
            
    elif args.command == 'stop':
        asyncio.run(controller.stop())
        
    elif args.command == 'stats':
        stats = asyncio.run(controller.get_stats())
        import json
        print(json.dumps(stats, indent=2, default=str))
        
    elif args.command == 'health':
        health = asyncio.run(controller.get_health())
        print(f"Overall Health: {health.overall}")
        for name, result in health.checks.items():
            print(f"  {name}: {result.status} - {result.message}")


def main():
    """Main entry point"""
    if len(sys.argv) == 1:
        # No arguments - run as console application
        run_console()
    elif sys.argv[1] in ['install', 'remove', 'start', 'stop', 'restart']:
        # Service management commands
        command = sys.argv[1]
        if command == 'install':
            install_service()
        elif command == 'remove':
            remove_service()
        elif command == 'start':
            start_service()
        elif command == 'stop':
            stop_service()
        elif command == 'restart':
            restart_service()
    else:
        # Run as Windows service
        win32serviceutil.HandleCommandLine(RalphLoopService)


if __name__ == '__main__':
    main()
