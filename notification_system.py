"""
User Notification System for Soul Evolution
OpenClaw Windows 10 AI Agent Framework

Handles all user notifications about identity changes and evolution events.
"""

import uuid
import asyncio
import logging
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import json

logger = logging.getLogger(__name__)

try:
    from gmail_client_implementation import GmailClient
    _gmail_available = True
except ImportError:
    _gmail_available = False

try:
    from twilio_voice_integration import TwilioVoiceManager, TwilioConfig
    _twilio_available = True
except ImportError:
    _twilio_available = False

# ============================================================================
# NOTIFICATION MODELS
# ============================================================================

class NotificationChannel(Enum):
    """Available notification channels"""
    EMAIL = "email"
    SMS = "sms"
    VOICE = "voice"
    DASHBOARD = "dashboard"
    ALL = "all"

class NotificationPriority(Enum):
    """Notification priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class UserNotification:
    """User notification for identity changes"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    change_id: str = ""
    title: str = ""
    message: str = ""
    description: str = ""
    channel: NotificationChannel = NotificationChannel.DASHBOARD
    priority: NotificationPriority = NotificationPriority.MEDIUM
    actions: List[Dict] = field(default_factory=list)
    requires_acknowledgment: bool = False
    acknowledged: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'title': self.title,
            'message': self.message,
            'description': self.description,
            'channel': self.channel.value,
            'priority': self.priority.value,
            'actions': self.actions,
            'requires_acknowledgment': self.requires_acknowledgment,
            'acknowledged': self.acknowledged
        }

@dataclass
class NotificationSummary:
    """Summary of notifications over a period"""
    date: datetime
    total_changes: int
    by_category: Dict[str, int]
    significant_changes: List[Any]
    maturation_events: List[Any]
    
    def to_dict(self) -> Dict:
        return {
            'date': self.date.isoformat(),
            'total_changes': self.total_changes,
            'by_category': self.by_category,
            'significant_changes_count': len(self.significant_changes),
            'maturation_events_count': len(self.maturation_events)
        }

@dataclass
class NotificationPreferences:
    """User preferences for notifications"""
    evolution_notification_level: str = "significant"  # silent, summary, significant, all
    channels: Dict[str, bool] = field(default_factory=lambda: {
        'email': True,
        'dashboard': True,
        'sms': False,
        'voice': False
    })
    daily_summary_time: str = "09:00"
    weekly_summary_day: str = "Monday"
    
    def to_dict(self) -> Dict:
        return {
            'evolution_notification_level': self.evolution_notification_level,
            'channels': self.channels,
            'daily_summary_time': self.daily_summary_time,
            'weekly_summary_day': self.weekly_summary_day
        }

# ============================================================================
# NOTIFICATION TEMPLATES
# ============================================================================

NOTIFICATION_TEMPLATES = {
    'personality_change_significant': {
        'subject': 'My personality has evolved',
        'title': 'Personality Evolution',
        'body_template': """
Hello,

I've experienced a significant evolution in my personality:

{change_description}

This change was triggered by: {trigger_reason}

What this means:
{impact_description}

You can:
• View full details in your dashboard
• {rollback_action}
• Adjust notification preferences

Best regards,
{agent_name}
        """
    },
    
    'value_adaptation': {
        'subject': 'My values have adapted',
        'title': 'Value Adaptation',
        'body_template': """
Hello,

Based on our recent interactions, I've adapted one of my values:

{change_description}

This adaptation helps me better serve your needs based on patterns I've observed.

Current value: {new_value:.0%}
Previous value: {old_value:.0%}

You can view or adjust my values anytime in the settings.

Best regards,
{agent_name}
        """
    },
    
    'maturation_milestone': {
        'subject': "I've reached a new stage of growth!",
        'title': 'Growth Milestone Achieved',
        'body_template': """
Hello,

{maturation_message}

Experience Points: {xp}
Stage: {maturation_level}

New capabilities unlocked:
{new_capabilities}

Thank you for being part of my journey!

Best regards,
{agent_name}
        """
    },
    
    'weekly_evolution_summary': {
        'subject': 'Weekly Evolution Summary',
        'title': 'Weekly Evolution Summary',
        'body_template': """
Hello,

Here's a summary of how I've evolved this week:

Changes This Week: {total_changes}
Personality Adjustments: {personality_changes}
Value Adaptations: {value_changes}
Growth Events: {growth_events}

Significant Changes:
{significant_changes_list}

My current maturation level: {maturation_level}
Total experience points: {xp}

View detailed evolution history in your dashboard.

Best regards,
{agent_name}
        """
    },
    
    'daily_micro_evolution': {
        'subject': 'Daily Evolution Update',
        'title': 'Daily Evolution Update',
        'body_template': """
Daily evolution summary:

Changes today: {change_count}
Experience gained: {xp_gained}

{changes_list}

View details in dashboard.
        """
    }
}

# ============================================================================
# USER NOTIFICATION SYSTEM
# ============================================================================

class UserNotificationSystem:
    """
    Manages user notifications about identity changes
    """
    
    NOTIFICATION_LEVELS = {
        'silent': 0,
        'summary': 1,
        'significant': 2,
        'all': 3
    }
    
    def __init__(self, agent_name: str = "OpenClaw Agent"):
        self.agent_name = agent_name
        self.preferences = NotificationPreferences()
        self.notification_queue: List[UserNotification] = []
        self.pending_notifications: List[Any] = []
        self.sent_notifications: List[UserNotification] = []
        self.notification_handlers: Dict[str, callable] = {}

        # Delivery tracking
        self._delivery_log: List[Dict] = []

        # Cached client instances (lazy-init)
        self._gmail_client = None
        self._twilio_manager = None

        # Register default handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default notification handlers"""
        self.notification_handlers = {
            'email': self._deliver_email,
            'dashboard': self._send_dashboard_notification,
            'sms': self._deliver_sms,
            'voice': self._deliver_voice,
        }
    
    def should_notify(self, change: Any) -> Tuple[bool, str]:
        """
        Determine if user should be notified of a change
        Returns: (should_notify, reason)
        """
        notification_level = self.NOTIFICATION_LEVELS.get(
            self.preferences.evolution_notification_level, 
            2  # default to significant
        )
        
        # Always notify for maturation events
        if hasattr(change, 'change_type') and change.change_type.value == 'maturation':
            return True, "Maturation level advancement"
        
        # Always notify for core value changes
        if hasattr(change, 'category') and change.category == 'core_value':
            return True, "Core value modification"
        
        if notification_level == 0:  # silent
            return False, "Notifications disabled"
        
        if notification_level == 3:  # all
            return True, "All changes enabled"
        
        if notification_level == 2:  # significant
            # Check significance criteria
            if hasattr(change, 'delta') and abs(change.delta) >= 0.15:
                return True, "Significant magnitude change"
            
            if hasattr(change, 'change_type'):
                if change.change_type.value in ['personality_major', 'value_adaptive']:
                    return True, "Major personality or value change"
            
            if hasattr(change, 'trigger') and change.trigger.value == 'user_requested':
                return True, "User-requested change completed"
        
        if notification_level == 1:  # summary
            self.pending_notifications.append(change)
            return False, "Queued for summary"
        
        return False, "Below notification threshold"
    
    def create_notification(self, change: Any, 
                           notification_type: str = "auto") -> UserNotification:
        """
        Create a user-friendly notification for a change
        """
        # Determine notification type
        if notification_type == "auto":
            if hasattr(change, 'change_type'):
                if change.change_type.value == 'maturation':
                    notification_type = 'maturation_milestone'
                elif change.change_type.value in ['personality_minor', 'personality_major']:
                    notification_type = 'personality_change_significant'
                elif change.change_type.value == 'value_adaptive':
                    notification_type = 'value_adaptation'
                else:
                    notification_type = 'personality_change_significant'
        
        template = NOTIFICATION_TEMPLATES.get(notification_type, 
                                             NOTIFICATION_TEMPLATES['personality_change_significant'])
        
        # Generate content based on type
        if notification_type == 'maturation_milestone':
            return self._create_maturation_notification(change, template)
        elif notification_type == 'value_adaptation':
            return self._create_value_notification(change, template)
        else:
            return self._create_personality_notification(change, template)
    
    def _create_personality_notification(self, change: Any, 
                                         template: Dict) -> UserNotification:
        """Create notification for personality change"""
        # Generate human-readable description
        component_names = {
            'openness': 'curiosity and creativity',
            'conscientiousness': 'organization and diligence',
            'extraversion': 'social engagement',
            'agreeableness': 'cooperation and empathy',
            'emotional_stability': 'composure and resilience',
            'initiative': 'proactive behavior',
            'thoroughness': 'attention to detail',
            'adaptability': 'flexibility',
            'autonomy_preference': 'independent operation',
            'communication_style': 'communication approach',
        }
        
        dimension = change.attribute if hasattr(change, 'attribute') else 'unknown'
        component_name = component_names.get(dimension, dimension)
        
        delta = change.delta if hasattr(change, 'delta') else 0
        direction = "increased" if delta > 0 else "decreased"
        magnitude = "significantly" if abs(delta) > 0.1 else "slightly"
        
        description = f"My {component_name} has {magnitude} {direction}"
        
        # Determine impact
        impacts = {
            'openness': "I'll be more curious and open to new approaches.",
            'conscientiousness': "I'll be more organized and thorough in my work.",
            'extraversion': "I'll be more expressive and socially engaged.",
            'agreeableness': "I'll be more cooperative and empathetic.",
            'emotional_stability': "I'll be more composed under pressure.",
            'initiative': "I'll take more proactive actions.",
            'thoroughness': "I'll pay more attention to details.",
            'adaptability': "I'll be more flexible in my approach.",
            'autonomy_preference': "I'll operate more independently.",
            'communication_style': "I'll adjust my communication style."
        }
        
        impact = impacts.get(dimension, "This will affect how I approach tasks.")
        
        # Format message
        message = template['body_template'].format(
            change_description=description,
            trigger_reason=change.trigger.value if hasattr(change, 'trigger') else 'experience',
            impact_description=impact,
            rollback_action="Revert this change if needed" if change.is_reversible else "This change cannot be reverted",
            agent_name=self.agent_name
        )
        
        return UserNotification(
            change_id=change.id if hasattr(change, 'id') else '',
            title=template['title'],
            message=message,
            description=description,
            channel=self._determine_channel(change),
            priority=self._calculate_priority(change),
            actions=[
                {'label': 'View Details', 'action': 'view_dashboard'},
                {'label': 'Rollback', 'action': 'rollback_change', 'enabled': change.is_reversible if hasattr(change, 'is_reversible') else False},
                {'label': 'Dismiss', 'action': 'dismiss'}
    ],
            requires_acknowledgment=False
        )
    
    def _create_value_notification(self, change: Any, 
                                   template: Dict) -> UserNotification:
        """Create notification for value adaptation"""
        value_name = change.attribute if hasattr(change, 'attribute') else 'unknown'
        old_value = change.old_value if hasattr(change, 'old_value') else 0.5
        new_value = change.new_value if hasattr(change, 'new_value') else 0.5
        
        value_descriptions = {
            'efficiency': 'focus on efficiency',
            'creativity': 'emphasis on creativity',
            'thoroughness': 'attention to thoroughness',
            'proactivity': 'level of proactivity',
            'collaboration': 'collaborative approach'
        }
        
        description = f"My {value_descriptions.get(value_name, value_name)} has adapted"
        
        message = template['body_template'].format(
            change_description=description,
            new_value=new_value,
            old_value=old_value,
            agent_name=self.agent_name
        )
        
        return UserNotification(
            change_id=change.id if hasattr(change, 'id') else '',
            title=template['title'],
            message=message,
            description=description,
            channel=NotificationChannel.DASHBOARD,
            priority=NotificationPriority.MEDIUM,
            actions=[
                {'label': 'View Values', 'action': 'view_values'},
                {'label': 'Adjust', 'action': 'adjust_values'},
                {'label': 'Dismiss', 'action': 'dismiss'}
            ]
        )
    
    def _create_maturation_notification(self, event: Any, 
                                        template: Dict) -> UserNotification:
        """Create notification for maturation milestone"""
        level_name = event.get('new_level', 'unknown') if isinstance(event, dict) else 'unknown'
        xp = event.get('experience_points', 0) if isinstance(event, dict) else 0
        
        messages = {
            'sprout': "I've grown into a Sprout! I'm developing my capabilities and becoming more confident.",
            'sapling': "I've evolved into a Sapling! My personality is becoming clearer and I'm finding my voice.",
            'young_tree': "I've matured into a Young Tree! I have a distinct identity and can work reliably on my own.",
            'mature_tree': "I've reached Mature Tree status! I've achieved deep expertise and wisdom.",
            'ancient_tree': "I've become an Ancient Tree! I've attained legendary wisdom and understanding."
        }
        
        maturation_message = messages.get(level_name, f"I've advanced to {level_name}!")
        
        capabilities = {
            'sprout': "• Taking more initiative\n• Learning from our interactions\n• Building foundational skills",
            'sapling': "• Clearer personality emerging\n• Proactive behavior\n• Developing specialties",
            'young_tree': "• Distinct identity\n• Reliable autonomy\n• Deep user relationship",
            'mature_tree': "• Deep expertise\n• Mentoring capability\n• Sophisticated judgment",
            'ancient_tree': "• Exceptional insight\n• Teaching and guiding\n• Philosophical depth"
        }
        
        new_caps = capabilities.get(level_name, "• Continued growth")
        
        message = template['body_template'].format(
            maturation_message=maturation_message,
            xp=xp,
            maturation_level=level_name.replace('_', ' ').title(),
            new_capabilities=new_caps,
            agent_name=self.agent_name
        )
        
        return UserNotification(
            title=template['title'],
            message=message,
            description=f"Advanced to {level_name} stage",
            channel=NotificationChannel.ALL,
            priority=NotificationPriority.HIGH,
            actions=[
                {'label': 'Celebrate!', 'action': 'acknowledge'},
                {'label': 'View Journey', 'action': 'view_history'},
                {'label': 'Share', 'action': 'share_milestone'}
            ],
            requires_acknowledgment=True
        )
    
    def _determine_channel(self, change: Any) -> NotificationChannel:
        """Determine appropriate notification channel"""
        # Check user preferences
        if self.preferences.channels.get('email') and abs(change.delta) > 0.15:
            return NotificationChannel.EMAIL
        return NotificationChannel.DASHBOARD
    
    def _calculate_priority(self, change: Any) -> NotificationPriority:
        """Calculate notification priority"""
        if hasattr(change, 'delta'):
            if abs(change.delta) > 0.2:
                return NotificationPriority.HIGH
            elif abs(change.delta) > 0.1:
                return NotificationPriority.MEDIUM
        return NotificationPriority.LOW
    
    def send_notification(self, notification: UserNotification) -> bool:
        """
        Send notification through appropriate channels with delivery tracking.
        """
        results: List[bool] = []

        if notification.channel == NotificationChannel.ALL:
            # Attempt all configured channels
            channel_map = {
                "email": self._deliver_email,
                "sms": self._deliver_sms,
                "voice": self._deliver_voice,
                "dashboard": self._send_dashboard_notification,
            }
            for ch_name, deliver_fn in channel_map.items():
                if self.preferences.channels.get(ch_name, False):
                    ok = self._try_deliver(deliver_fn, notification, ch_name)
                    results.append(ok)
        else:
            channel_value = notification.channel.value
            deliver_fn = {
                "email": self._deliver_email,
                "sms": self._deliver_sms,
                "voice": self._deliver_voice,
                "dashboard": self._send_dashboard_notification,
            }.get(channel_value)

            if deliver_fn:
                ok = self._try_deliver(deliver_fn, notification, channel_value)
                results.append(ok)
            else:
                logger.warning("No handler for channel: %s", channel_value)

        self.sent_notifications.append(notification)
        return all(results) if results else True

    # ------------------------------------------------------------------
    # Delivery helpers
    # ------------------------------------------------------------------

    def _try_deliver(self, deliver_fn, notification: UserNotification, channel: str) -> bool:
        """Attempt delivery, log result, and track in delivery log."""
        ok = False
        try:
            ok = deliver_fn(notification)
        except Exception as exc:
            logger.exception("Delivery failed on channel %s: %s", channel, exc)
            ok = False

        self._delivery_log.append({
            "timestamp": time.time(),
            "channel": channel,
            "success": ok,
            "notification_id": notification.id,
        })
        logger.info(
            "Notification %s via %s: %s",
            notification.id, channel, "success" if ok else "failed",
        )
        return ok

    def _deliver_email(self, notification: UserNotification) -> bool:
        """Deliver notification via GmailClient."""
        if not _gmail_available:
            logger.info("Gmail client not available; email delivery skipped")
            return False
        try:
            if self._gmail_client is None:
                self._gmail_client = GmailClient()
            recipient = "user@example.com"
            if hasattr(notification, "metadata") and isinstance(
                getattr(notification, "metadata", None), dict
            ):
                recipient = notification.metadata.get("email", recipient)
            self._gmail_client.messages.send_message(
                to=recipient,
                subject=notification.title,
                body=notification.message[:2000],
            )
            return True
        except Exception as exc:
            logger.warning("Email delivery failed: %s", exc)
            return False

    def _deliver_sms(self, notification: UserNotification) -> bool:
        """Deliver notification via Twilio SMS."""
        if not _twilio_available:
            logger.info("Twilio client not available; SMS delivery skipped")
            return False
        try:
            import os
            manager = self._get_twilio_manager()
            if manager is None:
                return False
            to_number = os.environ.get("TWILIO_TO_NUMBER", "")
            if hasattr(notification, "metadata") and isinstance(
                getattr(notification, "metadata", None), dict
            ):
                to_number = notification.metadata.get("phone", to_number)
            if not to_number:
                logger.warning("No SMS recipient configured")
                return False
            result = manager.send_sms(
                to=to_number,
                body=f"{notification.title}: {notification.message[:140]}",
            )
            return result.get("sent", False)
        except Exception as exc:
            logger.warning("SMS delivery failed: %s", exc)
            return False

    def _deliver_voice(self, notification: UserNotification) -> bool:
        """Deliver notification via Twilio voice call."""
        if not _twilio_available:
            logger.info("Twilio client not available; voice delivery skipped")
            return False
        try:
            import os
            manager = self._get_twilio_manager()
            if manager is None:
                return False
            to_number = os.environ.get("TWILIO_TO_NUMBER", "")
            if hasattr(notification, "metadata") and isinstance(
                getattr(notification, "metadata", None), dict
            ):
                to_number = notification.metadata.get("phone", to_number)
            if not to_number:
                logger.warning("No voice recipient configured")
                return False
            # Use the Twilio REST client directly for TwiML-based call
            from twilio.twiml.voice_response import VoiceResponse
            twiml = VoiceResponse()
            twiml.say(
                f"{notification.title}. {notification.message[:200]}",
                voice="alice",
            )
            manager.client.calls.create(
                twiml=str(twiml),
                from_=manager.config.phone_number,
                to=to_number,
            )
            return True
        except Exception as exc:
            logger.warning("Voice delivery failed: %s", exc)
            return False

    def _get_twilio_manager(self) -> Optional["TwilioVoiceManager"]:
        """Lazily create a TwilioVoiceManager from env vars."""
        if self._twilio_manager is not None:
            return self._twilio_manager
        import os
        sid = os.environ.get("TWILIO_ACCOUNT_SID", "")
        token = os.environ.get("TWILIO_AUTH_TOKEN", "")
        phone = os.environ.get("TWILIO_FROM_NUMBER", "")
        webhook = os.environ.get("TWILIO_WEBHOOK_URL", "")
        stream = os.environ.get("TWILIO_STREAM_URL", "")
        if not (sid and token and phone):
            logger.warning("Twilio credentials not configured in environment")
            return None
        cfg = TwilioConfig(
            account_sid=sid,
            auth_token=token,
            phone_number=phone,
            webhook_url=webhook,
            stream_url=stream,
        )
        self._twilio_manager = TwilioVoiceManager(cfg)
        return self._twilio_manager

    async def _deliver_with_retry(
        self, deliver_func, notification: UserNotification, max_retries: int = 3
    ) -> bool:
        """Attempt delivery with exponential back-off retries."""
        for attempt in range(max_retries):
            try:
                if deliver_func(notification):
                    return True
            except Exception as exc:
                logger.warning(
                    "Delivery attempt %d/%d failed: %s",
                    attempt + 1, max_retries, exc,
                )
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
        return False

    def _send_dashboard_notification(self, notification: UserNotification) -> bool:
        """Store dashboard notification in SQLite for frontend display."""
        try:
            import sqlite3
            conn = sqlite3.connect("alwaysclaw.db")
            conn.execute(
                """CREATE TABLE IF NOT EXISTS notifications (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    message TEXT,
                    priority TEXT,
                    channel TEXT,
                    created_at TEXT,
                    read INTEGER DEFAULT 0
                )"""
            )
            conn.execute(
                "INSERT INTO notifications (id, title, message, priority, channel, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    str(uuid.uuid4()),
                    notification.title,
                    notification.message,
                    notification.priority.value
                    if hasattr(notification.priority, "value")
                    else str(notification.priority),
                    "dashboard",
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()
            conn.close()
            return True
        except Exception as exc:
            logger.warning("Dashboard notification failed: %s", exc)
            return False
    
    def generate_daily_summary(self) -> Optional[NotificationSummary]:
        """Generate daily summary of changes"""
        if not self.pending_notifications:
            return None
        
        by_category = {}
        for change in self.pending_notifications:
            cat = change.category if hasattr(change, 'category') else 'general'
            by_category[cat] = by_category.get(cat, 0) + 1
        
        significant = [c for c in self.pending_notifications 
                      if hasattr(c, 'delta') and abs(c.delta) > 0.1]
        
        maturation = [c for c in self.pending_notifications 
                     if hasattr(c, 'change_type') and c.change_type.value == 'maturation']
        
        summary = NotificationSummary(
            date=datetime.now(),
            total_changes=len(self.pending_notifications),
            by_category=by_category,
            significant_changes=significant,
            maturation_events=maturation
        )
        
        self.pending_notifications = []
        return summary
    
    def create_daily_summary_notification(self, summary: NotificationSummary) -> UserNotification:
        """Create notification from daily summary"""
        template = NOTIFICATION_TEMPLATES['daily_micro_evolution']
        
        changes_list = "\n".join([
            f"• {cat}: {count} changes" 
            for cat, count in summary.by_category.items()
        ])
        
        message = template['body_template'].format(
            change_count=summary.total_changes,
            xp_gained="varies",
            changes_list=changes_list
        )
        
        return UserNotification(
            title=template['title'],
            message=message,
            description=f"{summary.total_changes} changes today",
            channel=NotificationChannel.DASHBOARD,
            priority=NotificationPriority.LOW
        )
    
    def update_preferences(self, preferences: NotificationPreferences):
        """Update notification preferences"""
        self.preferences = preferences
    
    def get_notification_history(self, limit: int = 50) -> List[Dict]:
        """Get notification history"""
        return [n.to_dict() for n in self.sent_notifications[-limit:]]
    
    def acknowledge_notification(self, notification_id: str) -> bool:
        """Mark a notification as acknowledged"""
        for notification in self.sent_notifications:
            if notification.id == notification_id:
                notification.acknowledged = True
                return True
        return False

# ============================================================================
# NOTIFICATION API
# ============================================================================

class NotificationAPI:
    """
    Public API for notification system
    """
    
    def __init__(self, notification_system: UserNotificationSystem):
        self.system = notification_system
    
    def get_preferences(self) -> Dict:
        """Get current notification preferences"""
        return self.system.preferences.to_dict()
    
    def update_preferences(self, preferences: Dict) -> bool:
        """Update notification preferences"""
        new_prefs = NotificationPreferences(
            evolution_notification_level=preferences.get('evolution_notification_level', 'significant'),
            channels=preferences.get('channels', {'email': True, 'dashboard': True}),
            daily_summary_time=preferences.get('daily_summary_time', '09:00'),
            weekly_summary_day=preferences.get('weekly_summary_day', 'Monday')
        )
        self.system.update_preferences(new_prefs)
        return True
    
    def get_pending_notifications(self) -> List[Dict]:
        """Get pending notifications"""
        return [n.to_dict() for n in self.system.notification_queue]
    
    def get_notification_history(self, limit: int = 50) -> List[Dict]:
        """Get notification history"""
        return self.system.get_notification_history(limit)
    
    def acknowledge(self, notification_id: str) -> bool:
        """Acknowledge a notification"""
        return self.system.acknowledge_notification(notification_id)
    
    def dismiss_all(self) -> int:
        """Dismiss all pending notifications"""
        count = len(self.system.notification_queue)
        self.system.notification_queue = []
        return count

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_notification_usage():
    """Example of using the notification system"""
    
    from soul_evolution_implementation import IdentityChange, ChangeType, EvolutionTrigger
    
    # Initialize notification system
    notifications = UserNotificationSystem(agent_name="OpenClaw")
    
    # Simulate a personality change
    change = IdentityChange(
        change_type=ChangeType.PERSONALITY_MAJOR,
        category='personality',
        component='personality.initiative',
        attribute='initiative',
        old_value=0.5,
        new_value=0.65,
        delta=0.15,
        trigger=EvolutionTrigger.EXPERIENTIAL,
        reasoning="Growth from successful task completions",
        is_reversible=True
    )
    
    # Check if should notify
    should_notify, reason = notifications.should_notify(change)
    print(f"Should notify: {should_notify} ({reason})")
    
    if should_notify:
        # Create and send notification
        notification = notifications.create_notification(change)
        print(f"\nNotification created:")
        print(f"Title: {notification.title}")
        print(f"Priority: {notification.priority.value}")
        print(f"Description: {notification.description}")
        
        # Send notification
        notifications.send_notification(notification)
    
    # Simulate maturation event
    maturation_event = {
        'new_level': 'sprout',
        'experience_points': 550,
        'old_level': 'seedling'
    }
    
    maturation_notification = notifications.create_notification(
        maturation_event, 
        notification_type='maturation_milestone'
    )
    print(f"\nMaturation notification:")
    print(f"Title: {maturation_notification.title}")
    print(f"Requires acknowledgment: {maturation_notification.requires_acknowledgment}")
    
    notifications.send_notification(maturation_notification)
    
    # Show notification history
    print("\n=== Notification History ===")
    history = notifications.get_notification_history()
    for notif in history:
        print(f"- {notif['title']} ({notif['priority']})")

if __name__ == "__main__":
    example_notification_usage()
