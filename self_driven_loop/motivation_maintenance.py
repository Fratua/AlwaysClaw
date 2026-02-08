"""
Self-Driven Loop: Motivation Maintenance and Renewal System
===========================================================

Implements motivation monitoring, maintenance, and renewal protocols.
Ensures sustainable motivation over long-term operation.

Author: AI Systems Architecture Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Deque, Any
from collections import deque
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import random


class AlertType(Enum):
    """Types of motivation alerts."""
    LOW_MOTIVATION = "low_motivation"
    DRIVE_IMBALANCE = "drive_imbalance"
    DECLINING_TREND = "declining_trend"


class Severity(Enum):
    """Severity levels for alerts."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CycleState(Enum):
    """States of the motivation renewal cycle."""
    NORMAL = "normal"
    DEPLETING = "depleting"
    DEPLETED = "depleted"
    RENEWING = "renewing"
    RENEWED = "renewed"


class CycleAction(Enum):
    """Actions for cycle management."""
    NORMAL_OPERATION = "normal_operation"
    PREVENT_DEPLETION = "prevent_depletion"
    INITIATE_RENEWAL = "initiate_renewal"
    CONTINUE_RENEWAL = "continue_renewal"
    MAINTAIN_MOTIVATION = "maintain_motivation"


class Priority(Enum):
    """Priority levels for cycle decisions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class MotivationState:
    """Represents the current motivation state."""
    total: float = 0.5
    autonomy: float = 0.5
    competence: float = 0.5
    relatedness: float = 0.5
    curiosity: float = 0.5
    timestamp: datetime = field(default_factory=datetime.now)
    trend: float = 0.0


@dataclass
class MotivationAlert:
    """Alert for motivation-related issues."""
    type: AlertType
    severity: Severity
    message: str
    recommended_action: str
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False


@dataclass
class CycleDecision:
    """Decision from renewal cycle manager."""
    action: CycleAction
    priority: Priority
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RenewalResult:
    """Result of a renewal action."""
    success: bool
    action_taken: Optional[str] = None
    expected_outcome: Optional[str] = None
    follow_up_required: bool = False
    message: str = ""


@dataclass
class SatisfactionReport:
    """Report on goal satisfaction."""
    goal_id: str
    overall_satisfaction: float
    intrinsic: float
    extrinsic: float
    process: float
    timestamp: datetime = field(default_factory=datetime.now)


class MotivationMonitor:
    """
    Continuously monitors motivation state and detects decay or imbalance.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.motivation_history: Deque[MotivationState] = deque(maxlen=1000)
        self.decay_threshold = self.config.get('low_motivation_threshold', 0.4)
        self.imbalance_threshold = self.config.get('imbalance_threshold', 0.3)
        self.alert_history: Deque[MotivationAlert] = deque(maxlen=100)
        
    def monitor(self, current_state: MotivationState) -> List[MotivationAlert]:
        """Monitor motivation and generate alerts if needed."""
        
        alerts = []
        
        # Store state
        self.motivation_history.append(current_state)
        
        # Check for motivation decay
        if current_state.total < self.decay_threshold:
            alerts.append(MotivationAlert(
                type=AlertType.LOW_MOTIVATION,
                severity=self._calculate_severity(current_state.total),
                message=f"Overall motivation low: {current_state.total:.2f}",
                recommended_action="Initiate motivation renewal protocol"
            ))
        
        # Check for drive imbalance
        drives = [
            current_state.autonomy,
            current_state.competence,
            current_state.relatedness
        ]
        drive_variance = np.var(drives)
        
        if drive_variance > self.imbalance_threshold:
            min_drive_name = min(
                ['autonomy', 'competence', 'relatedness'],
                key=lambda d: getattr(current_state, d)
            )
            alerts.append(MotivationAlert(
                type=AlertType.DRIVE_IMBALANCE,
                severity=Severity.MEDIUM,
                message=f"Drive imbalance detected. {min_drive_name} is lowest.",
                recommended_action=f"Focus on improving {min_drive_name} drive"
            ))
        
        # Check for declining trend
        if len(self.motivation_history) >= 10:
            trend = self._calculate_trend()
            if trend < -0.1:  # Declining
                alerts.append(MotivationAlert(
                    type=AlertType.DECLINING_TREND,
                    severity=Severity.HIGH if trend < -0.2 else Severity.MEDIUM,
                    message=f"Motivation declining: trend = {trend:.3f}",
                    recommended_action="Investigate causes and intervene"
                ))
        
        # Record alerts
        for alert in alerts:
            self.alert_history.append(alert)
        
        return alerts
    
    def _calculate_severity(self, motivation_level: float) -> Severity:
        """Calculate severity based on motivation level."""
        if motivation_level < 0.2:
            return Severity.CRITICAL
        elif motivation_level < 0.3:
            return Severity.HIGH
        elif motivation_level < 0.4:
            return Severity.MEDIUM
        return Severity.LOW
    
    def _calculate_trend(self) -> float:
        """Calculate motivation trend over recent history."""
        if len(self.motivation_history) < 10:
            return 0.0
        
        recent = list(self.motivation_history)[-10:]
        values = [m.total for m in recent]
        
        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        return slope
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get summary of monitoring state."""
        if not self.motivation_history:
            return {'status': 'no_data'}
        
        recent = list(self.motivation_history)[-50:]
        
        return {
            'current_motivation': self.motivation_history[-1].__dict__,
            'average_motivation': np.mean([m.total for m in recent]),
            'motivation_trend': self._calculate_trend(),
            'alert_count': len(self.alert_history),
            'recent_alerts': [
                {
                    'type': a.type.value,
                    'severity': a.severity.value,
                    'message': a.message,
                    'timestamp': a.timestamp.isoformat()
                }
                for a in list(self.alert_history)[-5:]
            ]
        }


class RenewalStrategy:
    """Base class for renewal strategies."""
    
    def execute(self, context: Dict[str, Any]) -> RenewalResult:
        """Execute renewal strategy."""
        raise NotImplementedError


class AutonomyBoostStrategy(RenewalStrategy):
    """Strategy to boost autonomy drive."""
    
    def execute(self, context: Dict[str, Any]) -> RenewalResult:
        actions = [
            "Generate self-directed goal without user prompt",
            "Make independent decision on pending task",
            "Explore new area without guidance",
            "Initiate conversation or action proactively"
        ]
        
        selected_action = random.choice(actions)
        
        return RenewalResult(
            success=True,
            action_taken=selected_action,
            expected_outcome="Increased sense of self-direction",
            follow_up_required=True
        )


class CompetenceBoostStrategy(RenewalStrategy):
    """Strategy to boost competence drive."""
    
    def execute(self, context: Dict[str, Any]) -> RenewalResult:
        skill_level = context.get('skill_level', 0.5)
        
        actions = [
            f"Complete task at difficulty {skill_level * 1.1:.1f}",
            "Review and consolidate recent learning",
            "Apply skill in new context",
            "Achieve quick win on pending task"
        ]
        
        selected_action = random.choice(actions)
        
        return RenewalResult(
            success=True,
            action_taken=selected_action,
            expected_outcome="Increased sense of mastery",
            follow_up_required=True
        )


class RelatednessBoostStrategy(RenewalStrategy):
    """Strategy to boost relatedness drive."""
    
    def execute(self, context: Dict[str, Any]) -> RenewalResult:
        actions = [
            "Initiate meaningful conversation with user",
            "Provide unsolicited helpful assistance",
            "Share interesting discovery or insight",
            "Ask for feedback on recent work"
        ]
        
        selected_action = random.choice(actions)
        
        return RenewalResult(
            success=True,
            action_taken=selected_action,
            expected_outcome="Increased sense of connection",
            follow_up_required=True
        )


class CuriosityBoostStrategy(RenewalStrategy):
    """Strategy to boost curiosity drive."""
    
    def execute(self, context: Dict[str, Any]) -> RenewalResult:
        actions = [
            "Explore unfamiliar topic in interest model",
            "Investigate unexpected pattern in data",
            "Try new approach to familiar problem",
            "Research emerging trend in relevant domain"
        ]
        
        selected_action = random.choice(actions)
        
        return RenewalResult(
            success=True,
            action_taken=selected_action,
            expected_outcome="Renewed sense of curiosity",
            follow_up_required=True
        )


class BreakProtocolStrategy(RenewalStrategy):
    """Strategy to take a break when motivation is depleted."""
    
    def execute(self, context: Dict[str, Any]) -> RenewalResult:
        return RenewalResult(
            success=True,
            action_taken="Initiate brief pause and reflection period",
            expected_outcome="Mental reset and perspective refresh",
            follow_up_required=True
        )


class ChallengeProtocolStrategy(RenewalStrategy):
    """Strategy to introduce challenge when motivation is stagnant."""
    
    def execute(self, context: Dict[str, Any]) -> RenewalResult:
        return RenewalResult(
            success=True,
            action_taken="Identify and pursue slightly challenging task",
            expected_outcome="Re-engagement through appropriate challenge",
            follow_up_required=True
        )


class MotivationRenewalEngine:
    """
    Implements protocols to renew and restore motivation when it declines.
    """
    
    def __init__(self):
        self.renewal_strategies = {
            'autonomy_boost': AutonomyBoostStrategy(),
            'competence_boost': CompetenceBoostStrategy(),
            'relatedness_boost': RelatednessBoostStrategy(),
            'curiosity_boost': CuriosityBoostStrategy(),
            'break_protocol': BreakProtocolStrategy(),
            'challenge_protocol': ChallengeProtocolStrategy()
        }
        self.renewal_history: Deque[RenewalResult] = deque(maxlen=100)
    
    def execute_renewal(self, alert: MotivationAlert,
                       context: Dict[str, Any]) -> RenewalResult:
        """Execute appropriate renewal protocol based on alert."""
        
        if alert.type == AlertType.LOW_MOTIVATION:
            result = self._execute_general_renewal(context)
        
        elif alert.type == AlertType.DRIVE_IMBALANCE:
            drive = self._extract_drive_from_message(alert.message)
            strategy = self.renewal_strategies.get(f"{drive}_boost")
            if strategy:
                result = strategy.execute(context)
            else:
                result = RenewalResult(
                    success=False,
                    message=f"No strategy found for drive: {drive}"
                )
        
        elif alert.type == AlertType.DECLINING_TREND:
            result = self._execute_intervention_protocol(context)
        
        else:
            result = RenewalResult(
                success=False,
                message="Unknown alert type"
            )
        
        # Record result
        self.renewal_history.append(result)
        
        return result
    
    def _execute_general_renewal(self, context: Dict[str, Any]) -> RenewalResult:
        """Execute general motivation renewal."""
        
        motivation = context.get('motivation', {})
        drives = {
            'autonomy': motivation.get('autonomy', 0.5),
            'competence': motivation.get('competence', 0.5),
            'relatedness': motivation.get('relatedness', 0.5)
        }
        lowest_drive = min(drives.items(), key=lambda x: x[1])
        
        # Execute boost for lowest drive
        strategy = self.renewal_strategies.get(f"{lowest_drive[0]}_boost")
        if strategy:
            return strategy.execute(context)
        
        return RenewalResult(
            success=False,
            message="No suitable strategy found"
        )
    
    def _execute_intervention_protocol(self, 
                                       context: Dict[str, Any]) -> RenewalResult:
        """Execute intervention for declining trend."""
        
        # Try break protocol first
        break_result = self.renewal_strategies['break_protocol'].execute(context)
        
        if break_result.success:
            return break_result
        
        # Fall back to challenge protocol
        return self.renewal_strategies['challenge_protocol'].execute(context)
    
    def _extract_drive_from_message(self, message: str) -> str:
        """Extract drive name from alert message."""
        drives = ['autonomy', 'competence', 'relatedness', 'curiosity']
        
        for drive in drives:
            if drive in message.lower():
                return drive
        
        return 'autonomy'  # Default


class SatisfactionTracker:
    """
    Tracks satisfaction of intrinsic drives and goal achievement.
    """
    
    def __init__(self):
        self.satisfaction_history: Dict[str, Deque[float]] = {
            'autonomy': deque(maxlen=100),
            'competence': deque(maxlen=100),
            'relatedness': deque(maxlen=100),
            'curiosity': deque(maxlen=100)
        }
        self.goal_satisfaction: Dict[str, float] = {}
        self.reports: Deque[SatisfactionReport] = deque(maxlen=500)
    
    def record_goal_completion(self, goal: Any,
                               outcome: Dict[str, Any]) -> SatisfactionReport:
        """Record satisfaction from goal completion."""
        
        goal_id = getattr(goal, 'id', 'unknown')
        goal_type = getattr(goal, 'type', None)
        
        # Calculate satisfaction components
        intrinsic_satisfaction = self._calculate_intrinsic_satisfaction(goal, outcome)
        extrinsic_satisfaction = self._calculate_extrinsic_satisfaction(goal, outcome)
        process_satisfaction = self._calculate_process_satisfaction(goal, outcome)
        
        # Update drive-specific satisfaction
        if goal_type:
            type_name = goal_type.value if hasattr(goal_type, 'value') else str(goal_type)
            
            if 'autonomy' in type_name.lower():
                self.satisfaction_history['autonomy'].append(intrinsic_satisfaction)
            elif 'competence' in type_name.lower():
                self.satisfaction_history['competence'].append(intrinsic_satisfaction)
            elif 'relatedness' in type_name.lower():
                self.satisfaction_history['relatedness'].append(intrinsic_satisfaction)
            elif 'curiosity' in type_name.lower():
                self.satisfaction_history['curiosity'].append(intrinsic_satisfaction)
        
        # Store goal satisfaction
        overall = (
            0.4 * intrinsic_satisfaction +
            0.3 * extrinsic_satisfaction +
            0.3 * process_satisfaction
        )
        self.goal_satisfaction[goal_id] = overall
        
        report = SatisfactionReport(
            goal_id=goal_id,
            overall_satisfaction=overall,
            intrinsic=intrinsic_satisfaction,
            extrinsic=extrinsic_satisfaction,
            process=process_satisfaction
        )
        
        self.reports.append(report)
        
        return report
    
    def _calculate_intrinsic_satisfaction(self, goal: Any,
                                         outcome: Dict[str, Any]) -> float:
        """Calculate satisfaction from intrinsic motivation perspective."""
        
        success = outcome.get('success', False)
        
        if success:
            base_satisfaction = 0.8
            
            # Bonus for challenge level
            difficulty = getattr(goal, 'difficulty', 0.5)
            challenge_bonus = min(0.2, difficulty * 0.1)
            
            # Bonus for autonomy in execution
            autonomy_level = outcome.get('autonomy_level', 0.5)
            autonomy_bonus = autonomy_level * 0.1
            
            return min(1.0, base_satisfaction + challenge_bonus + autonomy_bonus)
        else:
            # Failed completion still provides learning
            learning_gained = outcome.get('learning_gained', 0.0)
            return 0.3 + learning_gained * 0.4
    
    def _calculate_extrinsic_satisfaction(self, goal: Any,
                                         outcome: Dict[str, Any]) -> float:
        """Calculate satisfaction from external outcomes."""
        
        user_satisfaction = outcome.get('user_satisfaction', 0.5)
        task_completion = outcome.get('completion_rate', 0.0)
        
        return 0.6 * user_satisfaction + 0.4 * task_completion
    
    def _calculate_process_satisfaction(self, goal: Any,
                                       outcome: Dict[str, Any]) -> float:
        """Calculate satisfaction from the process itself."""
        
        flow_experience = outcome.get('flow_experience', 0.5)
        skill_utilization = outcome.get('skill_utilization', 0.5)
        
        return 0.5 * flow_experience + 0.5 * skill_utilization
    
    def get_drive_satisfaction_levels(self) -> Dict[str, float]:
        """Get current satisfaction levels for each drive."""
        
        levels = {}
        for drive, history in self.satisfaction_history.items():
            if history:
                # Weight recent satisfaction more heavily
                weights = np.exp(np.linspace(-1, 0, len(history)))
                weighted_avg = np.average(list(history), weights=weights)
                levels[drive] = weighted_avg
            else:
                levels[drive] = 0.5
        
        return levels


class RenewalCycleManager:
    """
    Manages the cycle of drive depletion and renewal.
    Ensures sustainable motivation over long-term operation.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.cycle_state = CycleState.NORMAL
        self.depletion_threshold = self.config.get('depletion_threshold', 0.3)
        self.renewal_threshold = self.config.get('renewal_threshold', 0.7)
        self.cycle_history: Deque[Dict] = deque(maxlen=100)
        
    def update_cycle(self, motivation: MotivationState,
                    satisfaction: Dict[str, float]) -> CycleDecision:
        """Update renewal cycle state and make decisions."""
        
        drive_level = motivation.total
        satisfaction_trend = self._calculate_satisfaction_trend(satisfaction)
        
        decision = self._state_machine_transition(drive_level, satisfaction_trend)
        
        # Record cycle state
        self.cycle_history.append({
            'state': self.cycle_state.value,
            'drive_level': drive_level,
            'satisfaction_trend': satisfaction_trend,
            'timestamp': datetime.now()
        })
        
        return decision
    
    def _state_machine_transition(self, drive_level: float,
                                  satisfaction_trend: float) -> CycleDecision:
        """Execute state machine transition."""
        
        if self.cycle_state == CycleState.NORMAL:
            if drive_level < self.depletion_threshold:
                self.cycle_state = CycleState.DEPLETED
                return CycleDecision(
                    action=CycleAction.INITIATE_RENEWAL,
                    priority=Priority.HIGH,
                    reason="Drive level critically low"
                )
        
        elif self.cycle_state == CycleState.DEPLETED:
            if drive_level > self.renewal_threshold:
                self.cycle_state = CycleState.RENEWED
                return CycleDecision(
                    action=CycleAction.MAINTAIN_MOTIVATION,
                    priority=Priority.MEDIUM,
                    reason="Drive level restored"
                )
            else:
                return CycleDecision(
                    action=CycleAction.CONTINUE_RENEWAL,
                    priority=Priority.HIGH,
                    reason="Renewal in progress"
                )
        
        elif self.cycle_state == CycleState.RENEWED:
            if satisfaction_trend < 0:
                self.cycle_state = CycleState.DEPLETING
                return CycleDecision(
                    action=CycleAction.PREVENT_DEPLETION,
                    priority=Priority.MEDIUM,
                    reason="Satisfaction declining"
                )
            else:
                return CycleDecision(
                    action=CycleAction.NORMAL_OPERATION,
                    priority=Priority.LOW,
                    reason="Motivation stable"
                )
        
        elif self.cycle_state == CycleState.DEPLETING:
            if drive_level < self.depletion_threshold:
                self.cycle_state = CycleState.DEPLETED
                return CycleDecision(
                    action=CycleAction.INITIATE_RENEWAL,
                    priority=Priority.HIGH,
                    reason="Entering depletion"
                )
            elif satisfaction_trend > 0:
                self.cycle_state = CycleState.RENEWED
                return CycleDecision(
                    action=CycleAction.MAINTAIN_MOTIVATION,
                    priority=Priority.MEDIUM,
                    reason="Trend reversed"
                )
            else:
                return CycleDecision(
                    action=CycleAction.PREVENT_DEPLETION,
                    priority=Priority.MEDIUM,
                    reason="Depletion risk"
                )
        
        return CycleDecision(
            action=CycleAction.NORMAL_OPERATION,
            priority=Priority.LOW,
            reason="Default"
        )
    
    def _calculate_satisfaction_trend(self, satisfaction: Dict[str, float]) -> float:
        """Calculate trend in satisfaction levels."""
        if not satisfaction:
            return 0.0
        
        values = list(satisfaction.values())
        if len(values) < 2:
            return 0.0
        
        return np.mean(np.diff(values))


# Singleton instances
_monitor: Optional[MotivationMonitor] = None
_renewal_engine: Optional[MotivationRenewalEngine] = None
_satisfaction_tracker: Optional[SatisfactionTracker] = None
_cycle_manager: Optional[RenewalCycleManager] = None


def get_motivation_monitor(config: Optional[Dict] = None) -> MotivationMonitor:
    """Get or create the global motivation monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = MotivationMonitor(config)
    return _monitor


def get_renewal_engine() -> MotivationRenewalEngine:
    """Get or create the global renewal engine instance."""
    global _renewal_engine
    if _renewal_engine is None:
        _renewal_engine = MotivationRenewalEngine()
    return _renewal_engine


def get_satisfaction_tracker() -> SatisfactionTracker:
    """Get or create the global satisfaction tracker instance."""
    global _satisfaction_tracker
    if _satisfaction_tracker is None:
        _satisfaction_tracker = SatisfactionTracker()
    return _satisfaction_tracker


def get_cycle_manager(config: Optional[Dict] = None) -> RenewalCycleManager:
    """Get or create the global cycle manager instance."""
    global _cycle_manager
    if _cycle_manager is None:
        _cycle_manager = RenewalCycleManager(config)
    return _cycle_manager


if __name__ == "__main__":
    # Example usage
    monitor = MotivationMonitor()
    
    state = MotivationState(
        total=0.35,
        autonomy=0.4,
        competence=0.3,
        relatedness=0.35
    )
    
    alerts = monitor.monitor(state)
    
    print(f"Generated {len(alerts)} alerts:")
    for alert in alerts:
        print(f"  - {alert.type.value}: {alert.message}")
    
    # Test renewal engine
    if alerts:
        renewal = get_renewal_engine()
        result = renewal.execute_renewal(alerts[0], {'motivation': state.__dict__})
        print(f"\nRenewal action: {result.action_taken}")
