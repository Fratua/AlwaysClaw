"""
OpenClaw Monitoring Scheduler
Cron-based and adaptive scheduling for web monitoring
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import hashlib


class ScheduleType(Enum):
    """Types of scheduling"""
    CRON = "cron"
    INTERVAL = "interval"
    ADAPTIVE = "adaptive"
    EVENT = "event"


@dataclass
class ScheduleConfig:
    """Configuration for a scheduled job"""
    monitor_id: str
    schedule_type: ScheduleType
    # For cron schedules
    cron_expression: Optional[str] = None
    # For interval schedules
    interval_seconds: Optional[int] = None
    # For adaptive schedules
    min_interval: int = 60
    max_interval: int = 3600
    # Common
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'monitor_id': self.monitor_id,
            'schedule_type': self.schedule_type.value,
            'cron_expression': self.cron_expression,
            'interval_seconds': self.interval_seconds,
            'min_interval': self.min_interval,
            'max_interval': self.max_interval,
            'enabled': self.enabled,
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'next_run': self.next_run.isoformat() if self.next_run else None,
            'run_count': self.run_count
        }


class CronParser:
    """Simple cron expression parser"""
    
    FIELD_NAMES = ['minute', 'hour', 'day_of_month', 'month', 'day_of_week']
    FIELD_RANGES = {
        'minute': (0, 59),
        'hour': (0, 23),
        'day_of_month': (1, 31),
        'month': (1, 12),
        'day_of_week': (0, 6)  # 0 = Sunday
    }
    
    def __init__(self, expression: str):
        self.expression = expression
        self.fields = self._parse(expression)
    
    def _parse(self, expression: str) -> Dict:
        """Parse cron expression into fields"""
        parts = expression.split()
        
        if len(parts) != 5:
            raise ValueError(f"Invalid cron expression: {expression}")
        
        fields = {}
        for name, part in zip(self.FIELD_NAMES, parts):
            fields[name] = self._parse_field(part, name)
        
        return fields
    
    def _parse_field(self, part: str, field_name: str) -> List[int]:
        """Parse a single cron field"""
        min_val, max_val = self.FIELD_RANGES[field_name]
        values = set()
        
        # Handle comma-separated values
        for segment in part.split(','):
            # Handle ranges
            if '/' in segment:
                range_part, step = segment.split('/')
                step = int(step)
            else:
                range_part = segment
                step = 1
            
            if range_part == '*':
                start, end = min_val, max_val
            elif '-' in range_part:
                start, end = map(int, range_part.split('-'))
            else:
                start = end = int(range_part)
            
            for val in range(start, end + 1, step):
                if min_val <= val <= max_val:
                    values.add(val)
        
        return sorted(values)
    
    def get_next(self, from_time: datetime = None) -> datetime:
        """Get next execution time"""
        if from_time is None:
            from_time = datetime.now()
        
        # Start from next minute
        next_time = from_time.replace(second=0, microsecond=0) + timedelta(minutes=1)
        
        # Find next matching time
        for _ in range(366 * 24 * 60):  # Max 1 year
            if (next_time.minute in self.fields['minute'] and
                next_time.hour in self.fields['hour'] and
                next_time.day in self.fields['day_of_month'] and
                next_time.month in self.fields['month'] and
                next_time.weekday() in self.fields['day_of_week']):
                return next_time
            
            next_time += timedelta(minutes=1)
        
        raise ValueError("Could not find next execution time")


class AdaptiveFrequency:
    """
    Dynamically adjusts monitoring frequency based on:
    - Historical change patterns
    - Time-of-day patterns
    - Recent activity levels
    - User-defined importance
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.change_history: Dict[str, List[datetime]] = {}
        self.base_intervals: Dict[str, int] = {}
    
    def register_monitor(self, monitor_id: str, base_interval: int):
        """Register a monitor for adaptive scheduling"""
        self.base_intervals[monitor_id] = base_interval
        self.change_history[monitor_id] = []
    
    def record_change(self, monitor_id: str):
        """Record a detected change"""
        if monitor_id not in self.change_history:
            self.change_history[monitor_id] = []
        
        self.change_history[monitor_id].append(datetime.now())
        
        # Keep only last 30 days
        cutoff = datetime.now() - timedelta(days=30)
        self.change_history[monitor_id] = [
            t for t in self.change_history[monitor_id] if t > cutoff
        ]
    
    def calculate_next_interval(self, monitor_id: str) -> int:
        """Calculate adaptive interval for a monitor"""
        base_interval = self.base_intervals.get(monitor_id, 3600)
        history = self.change_history.get(monitor_id, [])
        
        if not history:
            return base_interval
        
        # Calculate change frequency (changes per day)
        day_ago = datetime.now() - timedelta(days=1)
        recent_changes = [t for t in history if t > day_ago]
        change_frequency = len(recent_changes)
        
        # Adjust based on frequency
        if change_frequency > 10:  # Very frequent changes
            multiplier = 0.3  # Check more often
        elif change_frequency > 5:
            multiplier = 0.5
        elif change_frequency > 2:
            multiplier = 0.8
        elif change_frequency == 0:
            multiplier = 2.0  # Check less often
        else:
            multiplier = 1.0
        
        # Adjust for business hours
        now = datetime.now()
        if 9 <= now.hour <= 17 and now.weekday() < 5:
            multiplier *= 0.8  # More frequent during business hours
        
        # Calculate new interval
        new_interval = int(base_interval * multiplier)
        
        # Apply min/max constraints
        min_interval = self.config.get('min_interval', 60)
        max_interval = self.config.get('max_interval', 86400)
        
        return max(min_interval, min(new_interval, max_interval))
    
    def get_statistics(self, monitor_id: str) -> Dict:
        """Get adaptive scheduling statistics"""
        history = self.change_history.get(monitor_id, [])
        
        if not history:
            return {
                'total_changes': 0,
                'changes_24h': 0,
                'changes_7d': 0,
                'avg_interval': None,
                'current_interval': self.base_intervals.get(monitor_id, 3600)
            }
        
        now = datetime.now()
        day_ago = now - timedelta(days=1)
        week_ago = now - timedelta(days=7)
        
        changes_24h = sum(1 for t in history if t > day_ago)
        changes_7d = sum(1 for t in history if t > week_ago)
        
        # Calculate average interval between changes
        if len(history) > 1:
            sorted_history = sorted(history)
            intervals = [
                (sorted_history[i+1] - sorted_history[i]).total_seconds()
                for i in range(len(sorted_history) - 1)
            ]
            avg_interval = sum(intervals) / len(intervals)
        else:
            avg_interval = None
        
        return {
            'total_changes': len(history),
            'changes_24h': changes_24h,
            'changes_7d': changes_7d,
            'avg_interval': avg_interval,
            'current_interval': self.calculate_next_interval(monitor_id)
        }


class SchedulePresets:
    """Predefined schedule presets"""
    
    PRESETS = {
        'realtime': {
            'type': ScheduleType.INTERVAL,
            'interval': 30,
            'description': 'Every 30 seconds'
        },
        'frequent': {
            'type': ScheduleType.INTERVAL,
            'interval': 300,
            'description': 'Every 5 minutes'
        },
        'regular': {
            'type': ScheduleType.INTERVAL,
            'interval': 3600,
            'description': 'Every hour'
        },
        'daily': {
            'type': ScheduleType.CRON,
            'cron': '0 9 * * *',
            'description': 'Daily at 9 AM'
        },
        'weekly': {
            'type': ScheduleType.CRON,
            'cron': '0 9 * * 1',
            'description': 'Weekly on Monday 9 AM'
        },
        'business_hours': {
            'type': ScheduleType.CRON,
            'cron': '0 9-17 * * 1-5',
            'description': 'Every hour during business hours'
        },
        'adaptive': {
            'type': ScheduleType.ADAPTIVE,
            'min_interval': 60,
            'max_interval': 3600,
            'description': 'AI-optimized frequency'
        }
    }
    
    @classmethod
    def get_preset(cls, name: str) -> Optional[Dict]:
        """Get a preset by name"""
        return cls.PRESETS.get(name)
    
    @classmethod
    def list_presets(cls) -> Dict:
        """List all available presets"""
        return {
            name: preset['description']
            for name, preset in cls.PRESETS.items()
        }


class CronScheduler:
    """Cron-based monitoring scheduler"""
    
    def __init__(self, engine, config: Dict = None):
        self.engine = engine
        self.config = config or {}
        self.schedules: Dict[str, ScheduleConfig] = {}
        self.cron_parsers: Dict[str, CronParser] = {}
        self.adaptive = AdaptiveFrequency(self.config.get('adaptive', {}))
        self.running = False
        self._task = None
    
    def add_job(self, monitor_id: str, schedule: str or Dict) -> ScheduleConfig:
        """Add a scheduled monitoring job"""
        # Check if it's a preset
        if isinstance(schedule, str):
            preset = SchedulePresets.get_preset(schedule)
            if preset:
                schedule = preset
            else:
                # Try to parse as cron
                try:
                    CronParser(schedule)
                    schedule = {'type': ScheduleType.CRON, 'cron': schedule}
                except ValueError:
                    raise ValueError(f"Invalid schedule: {schedule}")
        
        # Create schedule config
        schedule_type = schedule.get('type', ScheduleType.INTERVAL)
        
        if isinstance(schedule_type, str):
            schedule_type = ScheduleType(schedule_type)
        
        config = ScheduleConfig(
            monitor_id=monitor_id,
            schedule_type=schedule_type,
            cron_expression=schedule.get('cron'),
            interval_seconds=schedule.get('interval'),
            min_interval=schedule.get('min_interval', 60),
            max_interval=schedule.get('max_interval', 3600)
        )
        
        self.schedules[monitor_id] = config
        
        # Setup cron parser if needed
        if config.schedule_type == ScheduleType.CRON and config.cron_expression:
            self.cron_parsers[monitor_id] = CronParser(config.cron_expression)
            config.next_run = self.cron_parsers[monitor_id].get_next()
        
        # Setup adaptive if needed
        if config.schedule_type == ScheduleType.ADAPTIVE:
            base_interval = schedule.get('base_interval', 3600)
            self.adaptive.register_monitor(monitor_id, base_interval)
            config.next_run = datetime.now() + timedelta(seconds=base_interval)
        
        # Setup interval if needed
        if config.schedule_type == ScheduleType.INTERVAL and config.interval_seconds:
            config.next_run = datetime.now() + timedelta(seconds=config.interval_seconds)
        
        return config
    
    def remove_job(self, monitor_id: str) -> bool:
        """Remove a scheduled job"""
        if monitor_id in self.schedules:
            del self.schedules[monitor_id]
            if monitor_id in self.cron_parsers:
                del self.cron_parsers[monitor_id]
            return True
        return False
    
    def update_job(self, monitor_id: str, schedule: Dict) -> Optional[ScheduleConfig]:
        """Update an existing job"""
        if monitor_id not in self.schedules:
            return None
        
        self.remove_job(monitor_id)
        return self.add_job(monitor_id, schedule)
    
    def get_job(self, monitor_id: str) -> Optional[ScheduleConfig]:
        """Get job configuration"""
        return self.schedules.get(monitor_id)
    
    def list_jobs(self) -> List[Dict]:
        """List all scheduled jobs"""
        return [s.to_dict() for s in self.schedules.values()]
    
    async def run(self):
        """Run scheduler loop"""
        self.running = True
        
        while self.running:
            now = datetime.now()
            
            # Find jobs to run
            jobs_to_run = []
            for monitor_id, config in self.schedules.items():
                if not config.enabled:
                    continue
                
                if config.next_run and config.next_run <= now:
                    jobs_to_run.append(monitor_id)
            
            # Run jobs
            for monitor_id in jobs_to_run:
                config = self.schedules[monitor_id]
                
                # Run the check
                asyncio.create_task(self._run_check(monitor_id))
                
                # Update schedule
                config.last_run = now
                config.run_count += 1
                
                # Calculate next run
                if config.schedule_type == ScheduleType.CRON:
                    parser = self.cron_parsers[monitor_id]
                    config.next_run = parser.get_next(now)
                
                elif config.schedule_type == ScheduleType.INTERVAL:
                    config.next_run = now + timedelta(seconds=config.interval_seconds)
                
                elif config.schedule_type == ScheduleType.ADAPTIVE:
                    interval = self.adaptive.calculate_next_interval(monitor_id)
                    config.next_run = now + timedelta(seconds=interval)
            
            # Wait before next check
            await asyncio.sleep(1)
    
    async def _run_check(self, monitor_id: str):
        """Run a check and handle results"""
        try:
            change = await self.engine.run_check(monitor_id)
            
            if change:
                # Record change for adaptive scheduling
                if monitor_id in self.schedules:
                    config = self.schedules[monitor_id]
                    if config.schedule_type == ScheduleType.ADAPTIVE:
                        self.adaptive.record_change(monitor_id)
                        
        except (RuntimeError, ValueError, TypeError) as e:
            print(f"Error running check for {monitor_id}: {e}")
    
    def stop(self):
        """Stop scheduler"""
        self.running = False
    
    def pause_job(self, monitor_id: str) -> bool:
        """Pause a scheduled job"""
        if monitor_id in self.schedules:
            self.schedules[monitor_id].enabled = False
            return True
        return False
    
    def resume_job(self, monitor_id: str) -> bool:
        """Resume a paused job"""
        if monitor_id in self.schedules:
            self.schedules[monitor_id].enabled = True
            return True
        return False
    
    def get_next_runs(self) -> Dict[str, datetime]:
        """Get next scheduled run times"""
        return {
            monitor_id: config.next_run
            for monitor_id, config in self.schedules.items()
            if config.enabled and config.next_run
        }
    
    def get_statistics(self) -> Dict:
        """Get scheduler statistics"""
        total_jobs = len(self.schedules)
        enabled_jobs = sum(1 for s in self.schedules.values() if s.enabled)
        total_runs = sum(s.run_count for s in self.schedules.values())
        
        return {
            'total_jobs': total_jobs,
            'enabled_jobs': enabled_jobs,
            'total_runs': total_runs,
            'running': self.running,
            'next_runs': {
                mid: config.next_run.isoformat() if config.next_run else None
                for mid, config in self.schedules.items()
            }
        }


class EventTrigger:
    """Event-triggered monitoring"""
    
    def __init__(self, engine, config: Dict = None):
        self.engine = engine
        self.config = config or {}
        self.triggers: Dict[str, Dict] = {}
        self.handlers: Dict[str, Callable] = {}
    
    def register_trigger(self, event_name: str, monitor_ids: List[str], 
                        handler: Callable = None):
        """Register an event trigger"""
        self.triggers[event_name] = {
            'monitor_ids': monitor_ids,
            'handler': handler
        }
    
    async def trigger_event(self, event_name: str, event_data: Dict = None):
        """Trigger an event"""
        if event_name not in self.triggers:
            return
        
        trigger = self.triggers[event_name]
        
        # Call custom handler if provided
        if trigger.get('handler'):
            await trigger['handler'](event_name, event_data)
        
        # Run monitors
        for monitor_id in trigger['monitor_ids']:
            asyncio.create_task(self.engine.run_check(monitor_id))
    
    def list_triggers(self) -> Dict:
        """List all registered triggers"""
        return {
            name: {
                'monitor_ids': trigger['monitor_ids']
            }
            for name, trigger in self.triggers.items()
        }


# Example usage
async def main():
    """Example usage of the scheduler"""
    from monitor_engine import MonitorEngine
    
    # Create engine
    engine = MonitorEngine({
        'dom': {},
        'visual': {},
        'content': {},
        'storage': {'snapshots_dir': './data/snapshots'}
    })
    
    # Add a monitor
    from monitor_engine import MonitorConfig
    monitor_id = engine.add_monitor(MonitorConfig(
        url='https://example.com',
        name='Example Site',
        dom_monitoring=True,
        visual_monitoring=True,
        content_monitoring=True
    ))
    
    # Create scheduler
    scheduler = CronScheduler(engine, {
        'adaptive': {
            'min_interval': 60,
            'max_interval': 3600
        }
    })
    
    # Add jobs with different schedules
    scheduler.add_job(monitor_id, 'frequent')  # Every 5 minutes
    
    # Or use cron expression
    # scheduler.add_job(monitor_id, '0 */6 * * *')  # Every 6 hours
    
    # Or use adaptive
    # scheduler.add_job(monitor_id, {
    #     'type': 'adaptive',
    #     'base_interval': 1800
    # })
    
    # Print schedule info
    print("Scheduled jobs:")
    for job in scheduler.list_jobs():
        print(f"  {job['monitor_id']}: {job['schedule_type']}")
        print(f"    Next run: {job['next_run']}")
    
    # Print presets
    print("\nAvailable presets:")
    for name, desc in SchedulePresets.list_presets().items():
        print(f"  {name}: {desc}")
    
    # Run scheduler for a short time
    print("\nRunning scheduler for 10 seconds...")
    
    async def stop_after_delay():
        await asyncio.sleep(10)
        scheduler.stop()
    
    await asyncio.gather(
        scheduler.run(),
        stop_after_delay()
    )
    
    print(f"\nStatistics: {scheduler.get_statistics()}")


if __name__ == '__main__':
    asyncio.run(main())
