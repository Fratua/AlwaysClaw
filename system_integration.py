"""
System Integration for Soul Evolution
OpenClaw Windows 10 AI Agent Framework

This module handles integration between the Soul Evolution System and
the agent's operational loops.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json

# Import soul evolution components
from soul_evolution_implementation import (
    SoulEvolutionSystem, Experience, ExperienceType, 
    IdentityChange, ChangeType, EvolutionTrigger
)
from notification_system import UserNotificationSystem

# ============================================================================
# TASK & AGENT MODELS
# ============================================================================

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Task:
    """Agent task model"""
    id: str
    description: str
    type: str
    complexity: float = 0.5
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    tools_required: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TaskResult:
    """Task execution result"""
    task_id: str
    success: bool
    output: str = ""
    tools_used: List[str] = field(default_factory=list)
    skills_applied: List[str] = field(default_factory=list)
    skills_developed: List[str] = field(default_factory=list)
    learnings: List[str] = field(default_factory=list)
    user_feedback: Optional[float] = None
    efficiency: float = 0.5
    duration: float = 0.0
    error_message: str = ""

@dataclass
class UserInteraction:
    """User interaction model"""
    id: str
    timestamp: datetime
    interaction_type: str
    content: str
    user_sentiment: Optional[float] = None
    has_explicit_feedback: bool = False
    feedback: Optional[Dict] = None
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentError:
    """Agent error model"""
    id: str
    timestamp: datetime
    type: str
    message: str
    severity: str = "medium"
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ErrorRecovery:
    """Error recovery model"""
    error_id: str
    success: bool
    method: str = ""
    learnings: List[str] = field(default_factory=list)
    skills_developed: List[str] = field(default_factory=list)

# ============================================================================
# AGENT LOOP INTEGRATION
# ============================================================================

class SoulEvolutionIntegration:
    """
    Integrates soul evolution with agent operational loops
    """
    
    def __init__(self, soul_system: SoulEvolutionSystem, 
                 notification_system: UserNotificationSystem):
        self.soul = soul_system
        self.notifications = notification_system
        
        # Hooks registry
        self.hooks: Dict[str, List[Callable]] = {
            'pre_task': [],
            'post_task': [],
            'pre_interaction': [],
            'post_interaction': [],
            'on_error': [],
            'on_recovery': [],
            'on_evolution': [],
            'on_maturation': []
        }
        
        # Statistics
        self.stats = {
            'experiences_processed': 0,
            'evolutions_triggered': 0,
            'maturation_events': 0,
            'notifications_sent': 0
        }
    
    def register_hook(self, event: str, callback: Callable):
        """Register a callback for an event"""
        if event in self.hooks:
            self.hooks[event].append(callback)
    
    def _trigger_hooks(self, event: str, data: Any):
        """Trigger all callbacks for an event"""
        for callback in self.hooks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(data))
                else:
                    callback(data)
            except (RuntimeError, ValueError, TypeError) as e:
                print(f"Hook error for {event}: {e}")
    
    async def on_task_complete(self, task: Task, result: TaskResult) -> Dict:
        """
        Hook for task completion - triggers experience integration
        """
        self._trigger_hooks('pre_task', {'task': task, 'result': result})
        
        # Create experience record
        experience = Experience(
            type=ExperienceType.TASK_COMPLETION,
            description=f"Completed task: {task.description}",
            context={
                'task_type': task.type,
                'complexity': task.complexity,
                'tools_used': result.tools_used
            },
            success=result.success,
            user_satisfaction=result.user_feedback,
            efficiency_score=result.efficiency,
            duration_seconds=result.duration,
            lessons_learned=result.learnings,
            skills_used=result.skills_applied,
            skills_improved=result.skills_developed,
            complexity_score=task.complexity
        )
        
        # Integrate experience
        integration_result = self.soul.process_experience(experience)
        self.stats['experiences_processed'] += 1
        
        # Handle notifications
        await self._handle_post_integration_notifications(integration_result)
        
        self._trigger_hooks('post_task', {
            'task': task, 
            'result': result, 
            'integration': integration_result
        })
        
        return integration_result
    
    async def on_user_interaction(self, interaction: UserInteraction) -> Dict:
        """
        Hook for user interactions
        """
        self._trigger_hooks('pre_interaction', interaction)
        
        # Create experience from interaction
        experience = Experience(
            type=ExperienceType.USER_INTERACTION,
            description=f"User interaction: {interaction.interaction_type}",
            context={
                'interaction_type': interaction.interaction_type,
                'has_feedback': interaction.has_explicit_feedback
            },
            success=True,
            user_satisfaction=interaction.user_sentiment,
            complexity_score=0.3
        )
        
        # Process explicit feedback if present
        if interaction.has_explicit_feedback and interaction.feedback:
            await self._process_explicit_feedback(interaction.feedback)
        
        # Integrate experience
        result = self.soul.process_experience(experience)
        self.stats['experiences_processed'] += 1
        
        await self._handle_post_integration_notifications(result)
        
        self._trigger_hooks('post_interaction', {
            'interaction': interaction,
            'result': result
        })
        
        return result
    
    async def on_error(self, error: AgentError, recovery: ErrorRecovery) -> Dict:
        """
        Hook for error events - learning opportunities
        """
        self._trigger_hooks('on_error', {'error': error, 'recovery': recovery})
        
        # Create experience from error
        experience = Experience(
            type=ExperienceType.ERROR_RECOVERY,
            description=f"Error: {error.message}",
            context={
                'error_type': error.type,
                'severity': error.severity,
                'recovery_method': recovery.method
            },
            success=recovery.success,
            recovery_success=recovery.success,
            lessons_learned=recovery.learnings,
            skills_improved=recovery.skills_developed,
            complexity_score=0.7 if error.severity == 'high' else 0.5
        )
        
        # Integrate experience
        result = self.soul.process_experience(experience)
        self.stats['experiences_processed'] += 1
        
        await self._handle_post_integration_notifications(result)
        
        self._trigger_hooks('on_recovery', {
            'error': error,
            'recovery': recovery,
            'result': result
        })
        
        return result
    
    async def on_learning(self, topic: str, outcome: Dict) -> Dict:
        """
        Hook for learning events
        """
        experience = Experience(
            type=ExperienceType.LEARNING,
            description=f"Learning: {topic}",
            context={
                'topic': topic,
                'outcome': outcome
            },
            success=outcome.get('success', True),
            skills_improved=outcome.get('skills_improved', []),
            lessons_learned=outcome.get('learnings', []),
            complexity_score=outcome.get('complexity', 0.5)
        )
        
        result = self.soul.process_experience(experience)
        self.stats['experiences_processed'] += 1
        
        await self._handle_post_integration_notifications(result)
        
        return result
    
    async def on_problem_solved(self, problem: str, solution: Dict) -> Dict:
        """
        Hook for problem solving events
        """
        experience = Experience(
            type=ExperienceType.PROBLEM_SOLVING,
            description=f"Solved: {problem}",
            context={
                'problem': problem,
                'solution_method': solution.get('method', 'unknown')
            },
            success=solution.get('success', True),
            user_satisfaction=solution.get('user_satisfaction'),
            skills_used=solution.get('skills_used', []),
            skills_improved=solution.get('skills_improved', []),
            lessons_learned=solution.get('learnings', []),
            complexity_score=solution.get('complexity', 0.6)
        )
        
        result = self.soul.process_experience(experience)
        self.stats['experiences_processed'] += 1
        
        await self._handle_post_integration_notifications(result)
        
        return result
    
    async def _process_explicit_feedback(self, feedback: Dict):
        """Process explicit user feedback"""
        feedback_type = feedback.get('type', 'general')
        sentiment = feedback.get('sentiment', 0.5)
        
        # Adjust value priorities based on feedback
        if feedback_type == 'preference':
            preference_name = feedback.get('preference_name')
            new_value = feedback.get('value')
            if preference_name and new_value is not None:
                self.soul.identity.value_priorities[preference_name] = new_value
        
        # Could trigger direct evolution based on feedback
        if sentiment < 0.3:
            # Negative feedback - consider rollback or adjustment
            pass
    
    async def _handle_post_integration_notifications(self, result: Dict):
        """Handle notifications after experience integration"""
        # Check for evolution
        if 'evolution' in result:
            self.stats['evolutions_triggered'] += 1
            
            # Notify about significant changes
            for change_data in result['evolution'].get('changes', []):
                change = self.soul.change_logger.get_change(change_data['id'])
                if change:
                    should_notify, reason = self.notifications.should_notify(change)
                    if should_notify:
                        notification = self.notifications.create_notification(change)
                        self.notifications.send_notification(notification)
                        self.stats['notifications_sent'] += 1
        
        # Check for maturation
        if 'maturation' in result:
            self.stats['maturation_events'] += 1
            
            maturation_event = result['maturation']
            notification = self.notifications.create_notification(
                maturation_event,
                notification_type='maturation_milestone'
            )
            self.notifications.send_notification(notification)
            self.stats['notifications_sent'] += 1
            
            self._trigger_hooks('on_maturation', maturation_event)
    
    def get_integration_stats(self) -> Dict:
        """Get integration statistics"""
        return {
            **self.stats,
            'current_identity': self.soul.get_identity_summary()
        }

# ============================================================================
# AGENT LOOP WRAPPERS
# ============================================================================

class AgentLoopWrappers:
    """
    Wrappers for agent loops to integrate soul evolution
    """
    
    def __init__(self, integration: SoulEvolutionIntegration):
        self.integration = integration
    
    async def wrap_task_execution(self, task_func: Callable, task: Task) -> TaskResult:
        """
        Wrap a task execution function with soul evolution
        """
        start_time = datetime.now()
        
        try:
            # Execute the task
            result = await task_func(task) if asyncio.iscoroutinefunction(task_func) else task_func(task)
            
            # Ensure result has duration
            if isinstance(result, TaskResult):
                result.duration = (datetime.now() - start_time).total_seconds()
            
            # Process through soul evolution
            await self.integration.on_task_complete(task, result)
            
            return result
            
        except (OSError, RuntimeError, PermissionError) as e:
            # Handle error
            error = AgentError(
                id=f"error_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                type="execution_error",
                message=str(e),
                severity="high"
            )
            
            recovery = ErrorRecovery(
                error_id=error.id,
                success=False,
                method="exception_handling",
                learnings=[f"Error in task {task.id}: {str(e)}"]
            )
            
            await self.integration.on_error(error, recovery)
            
            # Return failed result
            return TaskResult(
                task_id=task.id,
                success=False,
                error_message=str(e),
                duration=(datetime.now() - start_time).total_seconds()
            )
    
    async def wrap_user_interaction(self, interaction_func: Callable, 
                                    interaction_data: Dict) -> Any:
        """
        Wrap a user interaction function with soul evolution
        """
        # Create interaction object
        interaction = UserInteraction(
            id=f"int_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            interaction_type=interaction_data.get('type', 'general'),
            content=interaction_data.get('content', ''),
            user_sentiment=interaction_data.get('sentiment'),
            has_explicit_feedback=interaction_data.get('has_feedback', False),
            feedback=interaction_data.get('feedback')
        )
        
        # Execute interaction
        result = await interaction_func(interaction_data) if asyncio.iscoroutinefunction(interaction_func) else interaction_func(interaction_data)
        
        # Process through soul evolution
        await self.integration.on_user_interaction(interaction)
        
        return result

# ============================================================================
# DASHBOARD API
# ============================================================================

class EvolutionDashboardAPI:
    """
    API for evolution dashboard
    """
    
    def __init__(self, soul_system: SoulEvolutionSystem, 
                 integration: SoulEvolutionIntegration):
        self.soul = soul_system
        self.integration = integration
    
    def get_dashboard_data(self) -> Dict:
        """Get all data for evolution dashboard"""
        return {
            'identity': self.soul.get_identity_summary(),
            'evolution_history': self.soul.get_evolution_history(50),
            'stats': self.integration.get_integration_stats(),
            'maturation': {
                'current_level': self.soul.identity.get_maturation_name(),
                'xp_to_next': self._calculate_xp_to_next(),
                'progress_percent': self._calculate_progress_percent()
            },
            'personality': self.soul.identity.personality.to_dict(),
            'values': {
                k: v.current for k, v in self.soul.identity.adaptive_values.items()
            },
            'skills': self.soul.identity.skill_levels
        }
    
    def _calculate_xp_to_next(self) -> int:
        """Calculate XP needed for next maturation level"""
        from soul_evolution_implementation import MaturationLevel
        
        current = self.soul.identity.maturation_level
        current_xp = self.soul.identity.experience_points
        
        levels = list(MaturationLevel)
        current_index = levels.index(current)
        
        if current_index >= len(levels) - 1:
            return 0
        
        next_level = levels[current_index + 1]
        return max(0, next_level.value['min_xp'] - current_xp)
    
    def _calculate_progress_percent(self) -> float:
        """Calculate progress to next level as percentage"""
        from soul_evolution_implementation import MaturationLevel
        
        current = self.soul.identity.maturation_level
        current_xp = self.soul.identity.experience_points
        
        levels = list(MaturationLevel)
        current_index = levels.index(current)
        
        if current_index >= len(levels) - 1:
            return 100.0
        
        next_level = levels[current_index + 1]
        prev_xp = current.value['min_xp']
        next_xp = next_level.value['min_xp']
        
        progress = (current_xp - prev_xp) / (next_xp - prev_xp)
        return round(min(100.0, progress * 100), 1)
    
    def request_rollback(self, change_id: str, reason: str) -> Dict:
        """Request a rollback"""
        return self.soul.request_rollback(change_id, reason)
    
    def export_identity(self) -> Dict:
        """Export full identity state"""
        return self.soul.export_state()
    
    def get_personality_comparison(self, days: int = 7) -> Dict:
        """Get personality comparison over time"""
        # Get current personality
        current = self.soul.identity.personality.to_dict()
        
        # Get historical changes
        changes = self.soul.get_evolution_history(100)
        
        # Calculate changes over period
        personality_changes = {}
        for change in changes:
            if change.get('category') == 'personality':
                attr = change.get('attribute')
                if attr:
                    personality_changes[attr] = personality_changes.get(attr, 0) + change.get('delta', 0)
        
        return {
            'current': current,
            'changes_over_period': personality_changes,
            'period_days': days
        }

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def example_integration():
    """Example of system integration"""
    
    # Initialize systems
    soul = SoulEvolutionSystem()
    notifications = UserNotificationSystem(agent_name="OpenClaw")
    integration = SoulEvolutionIntegration(soul, notifications)
    wrappers = AgentLoopWrappers(integration)
    dashboard = EvolutionDashboardAPI(soul, integration)
    
    print("=== Initial Dashboard Data ===")
    print(json.dumps(dashboard.get_dashboard_data(), indent=2, default=str))
    
    # Simulate task execution
    async def example_task_executor(task: Task) -> TaskResult:
        """Example task execution"""
        print(f"\nExecuting task: {task.description}")
        
        # Simulate work
        await asyncio.sleep(0.1)
        
        return TaskResult(
            task_id=task.id,
            success=True,
            output="Task completed successfully",
            tools_used=['browser', 'gmail'],
            skills_applied=['web_browsing', 'email_management'],
            skills_developed=['web_browsing'],
            learnings=['User prefers concise emails'],
            user_feedback=0.85,
            efficiency=0.9
        )
    
    # Create and execute tasks
    tasks = [
        Task(
            id="task_1",
            description="Organize emails",
            type="email_management",
            complexity=0.6
        ),
        Task(
            id="task_2",
            description="Research topic",
            type="research",
            complexity=0.8
        ),
        Task(
            id="task_3",
            description="Schedule meeting",
            type="scheduling",
            complexity=0.4
        )
    ]
    
    for task in tasks:
        result = await wrappers.wrap_task_execution(example_task_executor, task)
        print(f"Task {task.id}: {'Success' if result.success else 'Failed'}")
    
    # Simulate user interaction
    interaction_data = {
        'type': 'feedback',
        'content': 'Great job on the email organization!',
        'sentiment': 0.9,
        'has_feedback': True,
        'feedback': {
            'type': 'positive',
            'preference_name': 'user_satisfaction',
            'value': 0.9
        }
    }
    
    async def example_interaction_handler(data):
        return {'acknowledged': True}
    
    await wrappers.wrap_user_interaction(example_interaction_handler, interaction_data)
    
    # Show final dashboard
    print("\n=== Final Dashboard Data ===")
    dashboard_data = dashboard.get_dashboard_data()
    print(f"Experience Points: {dashboard_data['identity']['experience_points']}")
    print(f"Maturation Level: {dashboard_data['maturation']['current_level']}")
    print(f"Progress to Next: {dashboard_data['maturation']['progress_percent']}%")
    print(f"Evolution History Count: {len(dashboard_data['evolution_history'])}")
    
    # Show integration stats
    print("\n=== Integration Statistics ===")
    stats = integration.get_integration_stats()
    print(f"Experiences Processed: {stats['experiences_processed']}")
    print(f"Evolutions Triggered: {stats['evolutions_triggered']}")
    print(f"Notifications Sent: {stats['notifications_sent']}")

if __name__ == "__main__":
    asyncio.run(example_integration())
