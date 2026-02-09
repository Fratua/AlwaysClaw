"""
OpenClaw Alert Manager
Multi-channel notification system for web monitoring alerts
"""

import asyncio
import json
import logging
import smtplib
import os
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from enum import Enum
import aiohttp

logger = logging.getLogger(__name__)


class SeverityLevel(Enum):
    """Alert severity levels"""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    INFO = 1


@dataclass
class Alert:
    """Represents an alert to be sent"""
    id: str
    change_id: str
    site_id: str
    site_name: str
    site_url: str
    severity: SeverityLevel
    change_type: str
    category: str
    description: str
    diff_data: Dict
    detected_at: datetime
    channels_sent: List[str] = field(default_factory=list)
    status: str = "pending"
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'change_id': self.change_id,
            'site_id': self.site_id,
            'site_name': self.site_name,
            'site_url': self.site_url,
            'severity': self.severity.name,
            'change_type': self.change_type,
            'category': self.category,
            'description': self.description,
            'diff_data': self.diff_data,
            'detected_at': self.detected_at.isoformat(),
            'channels_sent': self.channels_sent,
            'status': self.status
        }


class BaseNotifier(ABC):
    """Base class for all notifiers"""

    def __init__(self, config: Dict):
        self.config = config
        self.enabled = config.get('enabled', True)

    @abstractmethod
    async def send(self, alert: Alert) -> bool:
        """Send alert - to be implemented by subclasses"""
        ...
    
    def format_message(self, alert: Alert) -> str:
        """Format alert message"""
        return f"""
[{alert.severity.name}] Change Detected

Site: {alert.site_name}
URL: {alert.site_url}
Type: {alert.change_type}
Category: {alert.category}
Time: {alert.detected_at.strftime('%Y-%m-%d %H:%M:%S')}

Description:
{alert.description}
"""


class LogNotifier(BaseNotifier):
    """Simple logging-based notifier that writes alerts to the application log.
    Used as a fallback when no other notification channels are configured."""

    def __init__(self, config: Dict):
        super().__init__(config)
        self._logger = logging.getLogger('alert_manager.log_notifier')

    async def send(self, alert: Alert) -> bool:
        if not self.enabled:
            return False
        self._logger.warning(
            f"ALERT [{alert.severity.name}] {alert.site_name}: "
            f"{alert.category} - {alert.description[:200]}"
        )
        alert.channels_sent.append('log')
        return True


class EmailNotifier(BaseNotifier):
    """Gmail/SMTP email notification handler"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.smtp_server = config.get('smtp_server', 'smtp.gmail.com')
        self.smtp_port = config.get('smtp_port', 587)
        self.use_tls = config.get('use_tls', True)
        self.username = os.environ.get('SMTP_USERNAME') or config.get('username')
        self.password = os.environ.get('SMTP_PASSWORD') or config.get('password')
        self.recipients = config.get('recipients', [])
        self.sender_name = config.get('sender_name', 'OpenClaw Monitor')
        self.templates_dir = config.get('templates_dir', './templates')
    
    async def send(self, alert: Alert) -> bool:
        """Send email notification"""
        if not self.enabled or not self.recipients:
            return False
        
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = self._get_subject(alert)
            msg['From'] = f"{self.sender_name} <{self.username}>"
            msg['To'] = ', '.join(self.recipients)
            
            # Plain text version
            text_body = self.format_message(alert)
            msg.attach(MIMEText(text_body, 'plain'))
            
            # HTML version
            html_body = self._get_html_body(alert)
            msg.attach(MIMEText(html_body, 'html'))
            
            # Send email
            await asyncio.to_thread(self._send_smtp, msg)
            
            alert.channels_sent.append('email')
            return True
            
        except (smtplib.SMTPException, ConnectionError, TimeoutError) as e:
            logger.error(f"Email notification failed: {e}")
            return False
    
    def _send_smtp(self, msg: MIMEMultipart):
        """Send email via SMTP"""
        with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            if self.use_tls:
                server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)
    
    def _get_subject(self, alert: Alert) -> str:
        """Generate email subject"""
        emoji_map = {
            SeverityLevel.CRITICAL: '🔴',
            SeverityLevel.HIGH: '🟠',
            SeverityLevel.MEDIUM: '🟡',
            SeverityLevel.LOW: '🔵',
            SeverityLevel.INFO: '⚪'
        }
        emoji = emoji_map.get(alert.severity, '⚪')
        return f"{emoji} [{alert.severity.name}] {alert.site_name}: {alert.category.title()} Change"
    
    def _get_html_body(self, alert: Alert) -> str:
        """Generate HTML email body"""
        severity_colors = {
            SeverityLevel.CRITICAL: '#dc3545',
            SeverityLevel.HIGH: '#fd7e14',
            SeverityLevel.MEDIUM: '#ffc107',
            SeverityLevel.LOW: '#17a2b8',
            SeverityLevel.INFO: '#6c757d'
        }
        
        color = severity_colors.get(alert.severity, '#6c757d')
        
        # Format diff data for display
        diff_html = self._format_diff_html(alert.diff_data)
        
        return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 0; background: #f5f5f5; }}
        .container {{ max-width: 600px; margin: 0 auto; background: white; }}
        .header {{ background: {color}; color: white; padding: 30px; }}
        .header h1 {{ margin: 0; font-size: 24px; }}
        .header p {{ margin: 10px 0 0 0; opacity: 0.9; }}
        .content {{ padding: 30px; }}
        .section {{ margin-bottom: 25px; }}
        .section h2 {{ font-size: 16px; color: #333; margin-bottom: 10px; border-bottom: 2px solid #eee; padding-bottom: 5px; }}
        .info-row {{ display: flex; margin-bottom: 8px; }}
        .info-label {{ font-weight: bold; width: 100px; color: #666; }}
        .info-value {{ flex: 1; color: #333; }}
        .badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; text-transform: uppercase; }}
        .badge-critical {{ background: #dc3545; color: white; }}
        .badge-high {{ background: #fd7e14; color: white; }}
        .badge-medium {{ background: #ffc107; color: black; }}
        .badge-low {{ background: #17a2b8; color: white; }}
        .diff-section {{ background: #f8f9fa; padding: 15px; border-radius: 4px; font-family: monospace; font-size: 12px; overflow-x: auto; }}
        .footer {{ background: #f8f9fa; padding: 20px 30px; text-align: center; color: #666; font-size: 12px; }}
        .btn {{ display: inline-block; padding: 12px 24px; background: {color}; color: white; text-decoration: none; border-radius: 4px; margin-top: 15px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{alert.severity.name} Alert</h1>
            <p>Change detected on {alert.site_name}</p>
        </div>
        
        <div class="content">
            <div class="section">
                <h2>Alert Details</h2>
                <div class="info-row">
                    <span class="info-label">Severity:</span>
                    <span class="info-value">
                        <span class="badge badge-{alert.severity.name.lower()}">{alert.severity.name}</span>
                    </span>
                </div>
                <div class="info-row">
                    <span class="info-label">Site:</span>
                    <span class="info-value">{alert.site_name}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">URL:</span>
                    <span class="info-value"><a href="{alert.site_url}">{alert.site_url}</a></span>
                </div>
                <div class="info-row">
                    <span class="info-label">Category:</span>
                    <span class="info-value">{alert.category.title()}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Type:</span>
                    <span class="info-value">{alert.change_type}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Detected:</span>
                    <span class="info-value">{alert.detected_at.strftime('%Y-%m-%d %H:%M:%S')}</span>
                </div>
            </div>
            
            <div class="section">
                <h2>Description</h2>
                <p>{alert.description}</p>
            </div>
            
            <div class="section">
                <h2>Change Details</h2>
                <div class="diff-section">
                    {diff_html}
                </div>
            </div>
            
            <div style="text-align: center;">
                <a href="{os.environ.get('DASHBOARD_URL', 'http://localhost:8000/alerts/')}{alert.id}" class="btn">View in Dashboard</a>
            </div>
        </div>
        
        <div class="footer">
            <p>OpenClaw Web Monitor &copy; 2025</p>
            <p>This is an automated alert. Please do not reply to this email.</p>
        </div>
    </div>
</body>
</html>
"""
    
    def _format_diff_html(self, diff_data: Dict) -> str:
        """Format diff data as HTML"""
        html_parts = []
        
        if 'dom_changes' in diff_data:
            html_parts.append("<strong>DOM Changes:</strong>")
            for change in diff_data['dom_changes'][:5]:  # Limit to 5
                html_parts.append(f"<div>• {change.get('type')}: {change.get('element', {}).get('tag', 'unknown')}</div>")
        
        if 'content_changes' in diff_data:
            html_parts.append("<br><strong>Content Changes:</strong>")
            for change in diff_data['content_changes'][:5]:
                html_parts.append(f"<div>• {change.get('type')}: {change.get('selector', 'unknown')}</div>")
        
        if 'visual_diff' in diff_data:
            visual = diff_data['visual_diff']
            html_parts.append(f"<br><strong>Visual Changes:</strong>")
            html_parts.append(f"<div>• Difference: {visual.get('diff_percentage', 0):.2f}%</div>")
            html_parts.append(f"<div>• Similarity: {visual.get('similarity_score', 0):.2f}%</div>")
        
        return '\n'.join(html_parts) if html_parts else "<em>No detailed diff available</em>"


class TwilioSMSNotifier(BaseNotifier):
    """Twilio SMS notification handler"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.account_sid = os.environ.get('TWILIO_ACCOUNT_SID') or config.get('account_sid')
        self.auth_token = os.environ.get('TWILIO_AUTH_TOKEN') or config.get('auth_token')
        self.from_number = config.get('from_number')
        self.to_numbers = config.get('to_numbers', [])
        self.max_length = config.get('max_length', 1600)
        self.include_link = config.get('include_link', True)
        self.dashboard_url = os.environ.get('DASHBOARD_URL', 'http://localhost:8000/alerts/')

    async def send(self, alert: Alert) -> bool:
        """Send SMS notification"""
        if not self.enabled or not self.to_numbers:
            return False

        try:
            from twilio.rest import Client
        except ImportError:
            logger.error("Twilio package not installed (pip install twilio)")
            return False

        try:
            client = Client(self.account_sid, self.auth_token)

            # Build message
            message = self._build_message(alert)

            # Send to all recipients
            for number in self.to_numbers:
                await asyncio.to_thread(
                    client.messages.create,
                    body=message,
                    from_=self.from_number,
                    to=number
                )

            alert.channels_sent.append('sms')
            return True

        except (ConnectionError, TimeoutError) as e:
            logger.error(f"SMS notification failed: {e}")
            return False

    def _build_message(self, alert: Alert) -> str:
        """Build SMS message"""
        emoji_map = {
            SeverityLevel.CRITICAL: '🚨',
            SeverityLevel.HIGH: '⚠️',
            SeverityLevel.MEDIUM: '🔔',
            SeverityLevel.LOW: 'ℹ️',
            SeverityLevel.INFO: '📝'
        }

        emoji = emoji_map.get(alert.severity, '🔔')

        message = f"""{emoji} [{alert.severity.name}] {alert.site_name}

{alert.description[:100]}{'...' if len(alert.description) > 100 else ''}

Category: {alert.category}
Time: {alert.detected_at.strftime('%H:%M')}"""

        if self.include_link:
            message += f"\n\nView: {self.dashboard_url}{alert.id}"

        return message[:self.max_length]


class TwilioVoiceNotifier(BaseNotifier):
    """Twilio voice call notification handler"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.account_sid = os.environ.get('TWILIO_ACCOUNT_SID') or config.get('account_sid')
        self.auth_token = os.environ.get('TWILIO_AUTH_TOKEN') or config.get('auth_token')
        self.from_number = config.get('from_number')
        self.to_numbers = config.get('to_numbers', [])
        self.voice = config.get('voice', 'Polly.Joanna')
        self.max_duration = config.get('max_duration', 60)
        self.retry_count = config.get('retry_count', 3)
        self.enabled_for_severity = config.get('enabled_for_severity', ['CRITICAL'])
    
    async def send(self, alert: Alert) -> bool:
        """Make voice call"""
        if not self.enabled or not self.to_numbers:
            return False
        
        # Only call for configured severity levels
        if alert.severity.name not in self.enabled_for_severity:
            return False
        
        try:
            from twilio.rest import Client
        except ImportError:
            logger.error("Twilio package not installed (pip install twilio)")
            return False

        try:
            client = Client(self.account_sid, self.auth_token)

            # Build TwiML
            twiml = self._build_twiml(alert)

            # Make calls
            for number in self.to_numbers:
                await asyncio.to_thread(
                    client.calls.create,
                    twiml=twiml,
                    to=number,
                    from_=self.from_number
                )

            alert.channels_sent.append('voice')
            return True

        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Voice notification failed: {e}")
            return False
    
    def _build_twiml(self, alert: Alert) -> str:
        """Build TwiML for voice call"""
        message = f"""This is an alert from OpenClaw web monitoring. 
        A {alert.severity.name.lower()} priority change has been detected on {alert.site_name}. 
        The change category is {alert.category}. 
        Please check your dashboard for details. 
        Goodbye."""
        
        # Clean up message for TTS
        message = ' '.join(message.split())
        
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="{self.voice}">{message}</Say>
</Response>"""


class TTSNotifier(BaseNotifier):
    """Text-to-speech notification handler for local agent"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.voice_id = config.get('voice_id', 'default')
        self.speak_on_severity = config.get('speak_on_severity', ['CRITICAL', 'HIGH'])
        self.rate = config.get('rate', 150)
        self.volume = config.get('volume', 1.0)
    
    async def send(self, alert: Alert) -> bool:
        """Generate and play TTS notification"""
        if not self.enabled:
            return False
        
        # Only speak for configured severity levels
        if alert.severity.name not in self.speak_on_severity:
            return False
        
        try:
            # Build message
            message = self._build_message(alert)
            
            # Use pyttsx3 for local TTS
            await asyncio.to_thread(self._speak, message)
            
            alert.channels_sent.append('tts')
            return True
            
        except (ImportError, OSError, RuntimeError) as e:
            logger.error(f"TTS notification failed: {e}")
            return False
    
    def _build_message(self, alert: Alert) -> str:
        """Build TTS message"""
        severity_words = {
            SeverityLevel.CRITICAL: 'critical',
            SeverityLevel.HIGH: 'high priority',
            SeverityLevel.MEDIUM: 'medium priority',
            SeverityLevel.LOW: 'low priority',
            SeverityLevel.INFO: 'information'
        }
        
        severity_word = severity_words.get(alert.severity, 'priority')
        
        return f"Attention. A {severity_word} change has been detected on {alert.site_name}. Category: {alert.category}."
    
    def _speak(self, message: str):
        """Speak message using TTS"""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', self.rate)
            engine.setProperty('volume', self.volume)
            engine.say(message)
            engine.runAndWait()
        except ImportError:
            logger.warning(f"[TTS] pyttsx3 not available, cannot speak: {message[:80]}")


class WebhookNotifier(BaseNotifier):
    """Webhook notification handler"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.webhook_url = config.get('webhook_url')
        self.headers = config.get('headers', {})
        self.timeout = config.get('timeout', 30)
        self.retry_count = config.get('retry_count', 3)
    
    async def send(self, alert: Alert) -> bool:
        """Send webhook notification"""
        if not self.enabled or not self.webhook_url:
            return False
        
        try:
            payload = {
                'alert': alert.to_dict(),
                'timestamp': datetime.now().isoformat(),
                'source': 'openclaw-web-monitor'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 200:
                        alert.channels_sent.append('webhook')
                        return True
                    else:
                        logger.warning(f"Webhook returned status {response.status}")
                        return False

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.error(f"Webhook notification failed: {e}")
            return False


class AlertThrottler:
    """Prevents alert fatigue through intelligent throttling"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.recent_alerts: Dict[str, List[datetime]] = {}
        self.cooldown_periods = config.get('cooldown_periods', {
            'CRITICAL': 0,      # No cooldown
            'HIGH': 300,        # 5 minutes
            'MEDIUM': 900,      # 15 minutes
            'LOW': 3600,        # 1 hour
            'INFO': 86400       # 24 hours
        })
        self.rate_limits = config.get('rate_limits', {
            'per_site_per_hour': 10,
            'per_site_per_day': 50
        })
    
    def should_send_alert(self, alert: Alert) -> bool:
        """Check if alert should be sent based on throttling rules"""
        site_key = f"{alert.site_id}:{alert.category}"
        now = datetime.now()
        
        # Initialize history for site
        if site_key not in self.recent_alerts:
            self.recent_alerts[site_key] = []
        
        # Clean old alerts
        self.recent_alerts[site_key] = [
            t for t in self.recent_alerts[site_key]
            if (now - t).total_seconds() < 86400  # Keep 24 hours
        ]
        
        # Check cooldown period
        cooldown = self.cooldown_periods.get(alert.severity.name, 3600)
        recent_for_severity = [
            t for t in self.recent_alerts[site_key]
            if (now - t).total_seconds() < cooldown
        ]
        
        if recent_for_severity:
            return False
        
        # Check rate limits
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)
        
        alerts_last_hour = sum(1 for t in self.recent_alerts[site_key] if t > hour_ago)
        alerts_last_day = sum(1 for t in self.recent_alerts[site_key] if t > day_ago)
        
        if alerts_last_hour >= self.rate_limits['per_site_per_hour']:
            return False
        
        if alerts_last_day >= self.rate_limits['per_site_per_day']:
            return False
        
        # Record this alert
        self.recent_alerts[site_key].append(now)
        
        return True


class AlertManager:
    """Main alert management system"""
    
    # Channel severity mapping
    CHANNEL_SEVERITY_MAP = {
        SeverityLevel.CRITICAL: ['email', 'sms', 'voice', 'tts', 'webhook'],
        SeverityLevel.HIGH: ['email', 'sms', 'tts', 'webhook'],
        SeverityLevel.MEDIUM: ['email', 'webhook'],
        SeverityLevel.LOW: ['email'],
        SeverityLevel.INFO: ['webhook']
    }
    
    def __init__(self, config: Dict):
        self.config = config
        self.channels: Dict[str, BaseNotifier] = {}
        self.throttler = AlertThrottler(config.get('throttling', {}))
        self.alert_history: List[Alert] = []
        self._setup_channels()
    
    def _setup_channels(self):
        """Initialize notification channels"""
        channels_config = self.config.get('channels', [])
        
        if 'email' in channels_config:
            self.channels['email'] = EmailNotifier(self.config.get('email', {}))
        
        if 'sms' in channels_config:
            self.channels['sms'] = TwilioSMSNotifier(self.config.get('sms', {}))
        
        if 'voice' in channels_config:
            self.channels['voice'] = TwilioVoiceNotifier(self.config.get('voice', {}))
        
        if 'tts' in channels_config:
            self.channels['tts'] = TTSNotifier(self.config.get('tts', {}))
        
        if 'webhook' in channels_config:
            self.channels['webhook'] = WebhookNotifier(self.config.get('webhook', {}))

        if 'log' in channels_config:
            self.channels['log'] = LogNotifier(self.config.get('log', {}))

        # Always register a log notifier as fallback if no channels configured
        if not self.channels:
            self.channels['log'] = LogNotifier({'enabled': True})

    def determine_severity(self, change_data: Dict) -> SeverityLevel:
        """Determine alert severity based on change data"""
        score = 0
        
        # Score based on change category
        category_scores = {
            'price': 40,
            'availability': 35,
            'security': 50,
            'status': 30,
            'content': 20,
            'general': 10
        }
        
        category = change_data.get('category', 'general')
        score += category_scores.get(category, 10)
        
        # Score based on change magnitude
        diff_data = change_data.get('diff_data', {})
        
        if 'visual_diff' in diff_data:
            diff_pct = diff_data['visual_diff'].get('diff_percentage', 0)
            score += min(diff_pct * 0.5, 30)
        
        if 'dom_changes' in diff_data:
            dom_changes = diff_data['dom_changes']
            score += min(len(dom_changes) * 5, 25)
        
        # Map score to severity
        if score >= 70:
            return SeverityLevel.CRITICAL
        elif score >= 50:
            return SeverityLevel.HIGH
        elif score >= 30:
            return SeverityLevel.MEDIUM
        elif score >= 10:
            return SeverityLevel.LOW
        else:
            return SeverityLevel.INFO
    
    async def process_change(self, change_data: Dict) -> Optional[Alert]:
        """Process a detected change and send alerts"""
        import uuid
        
        # Create alert
        severity = self.determine_severity(change_data)
        
        alert = Alert(
            id=str(uuid.uuid4())[:12],
            change_id=change_data.get('id', str(uuid.uuid4())[:12]),
            site_id=change_data.get('site_id', ''),
            site_name=change_data.get('site_name', 'Unknown'),
            site_url=change_data.get('site_url', ''),
            severity=severity,
            change_type=change_data.get('change_type', 'unknown'),
            category=change_data.get('category', 'general'),
            description=change_data.get('description', ''),
            diff_data=change_data.get('diff_data', {}),
            detected_at=datetime.now()
        )
        
        # Check throttling
        if not self.throttler.should_send_alert(alert):
            alert.status = "throttled"
            self.alert_history.append(alert)
            return alert
        
        # Determine channels to use
        channels_to_use = self.CHANNEL_SEVERITY_MAP.get(severity, [])
        
        # Send through each channel
        for channel_name in channels_to_use:
            if channel_name in self.channels:
                success = await self.channels[channel_name].send(alert)
                if not success:
                    logger.warning(f"Failed to send alert through {channel_name}")
        
        alert.status = "sent"
        self.alert_history.append(alert)
        
        return alert
    
    def get_alert_history(self, site_id: Optional[str] = None, 
                         severity: Optional[SeverityLevel] = None,
                         limit: int = 100) -> List[Alert]:
        """Get alert history with optional filtering"""
        alerts = self.alert_history
        
        if site_id:
            alerts = [a for a in alerts if a.site_id == site_id]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return sorted(alerts, key=lambda a: a.detected_at, reverse=True)[:limit]
    
    def get_statistics(self) -> Dict:
        """Get alert statistics"""
        total = len(self.alert_history)
        
        by_severity = {}
        by_channel = {}
        by_site = {}
        
        for alert in self.alert_history:
            # By severity
            sev = alert.severity.name
            by_severity[sev] = by_severity.get(sev, 0) + 1
            
            # By channel
            for ch in alert.channels_sent:
                by_channel[ch] = by_channel.get(ch, 0) + 1
            
            # By site
            by_site[alert.site_name] = by_site.get(alert.site_name, 0) + 1
        
        return {
            'total_alerts': total,
            'by_severity': by_severity,
            'by_channel': by_channel,
            'by_site': by_site,
            'throttled_count': sum(1 for a in self.alert_history if a.status == 'throttled')
        }
