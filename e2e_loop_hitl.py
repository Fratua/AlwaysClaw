"""
Human-in-the-Loop (HITL) System for E2E Loop
OpenClaw-Inspired AI Agent System for Windows 10

This module provides comprehensive human-in-the-loop functionality:
- Approval requests via multiple channels (Email, SMS, Voice)
- Web-based approval interface
- Timeout and escalation handling
- Audit logging
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Coroutine
from abc import ABC, abstractmethod
import uuid

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class ApprovalStatus(Enum):
    """Status of an approval request."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMED_OUT = "timed_out"
    ESCALATED = "escalated"


class NotificationChannel(Enum):
    """Available notification channels."""
    EMAIL = "email"
    SMS = "sms"
    VOICE = "voice"
    SLACK = "slack"
    DISCORD = "discord"
    WEB = "web"


class EscalationAction(Enum):
    """Escalation actions for timed-out approvals."""
    AUTO_APPROVE = "auto_approve"
    AUTO_REJECT = "auto_reject"
    ESCALATE = "escalate"
    NOTIFY = "notify"


@dataclass
class ApprovalContext:
    """Context for approval request."""
    title: str
    description: str
    details: Dict[str, Any] = field(default_factory=dict)
    notification_channels: List[NotificationChannel] = field(default_factory=lambda: [NotificationChannel.EMAIL])
    priority: str = "normal"  # low, normal, high, critical
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EscalationPolicy:
    """Policy for handling approval timeouts."""
    action: EscalationAction
    timeout: timedelta = field(default_factory=lambda: timedelta(hours=24))
    next_approvers: List[str] = field(default_factory=list)
    notification_message: Optional[str] = None


@dataclass
class ApprovalConfig:
    """Configuration for approval requirements."""
    enabled: bool = False
    approvers: List[str] = field(default_factory=list)
    timeout: timedelta = field(default_factory=lambda: timedelta(hours=24))
    escalation_policy: Optional[EscalationPolicy] = None
    allow_delegation: bool = False
    require_all_approvers: bool = False


@dataclass
class ApprovalResponse:
    """Response to an approval request."""
    approved: bool
    approver_id: str
    comments: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)
    responded_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ApprovalRequest:
    """Human approval request."""
    id: str
    workflow_id: str
    task_id: str
    approvers: List[str]
    context: ApprovalContext
    config: ApprovalConfig
    status: ApprovalStatus = ApprovalStatus.PENDING
    requested_at: datetime = field(default_factory=datetime.utcnow)
    timeout_at: Optional[datetime] = None
    responses: List[ApprovalResponse] = field(default_factory=list)
    escalated_to: Optional[str] = None
    escalation_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        if self.timeout_at is None and self.config.timeout:
            self.timeout_at = self.requested_at + self.config.timeout
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'workflow_id': self.workflow_id,
            'task_id': self.task_id,
            'approvers': self.approvers,
            'context': {
                'title': self.context.title,
                'description': self.context.description,
                'details': self.context.details,
                'priority': self.context.priority
            },
            'status': self.status.value,
            'requested_at': self.requested_at.isoformat(),
            'timeout_at': self.timeout_at.isoformat() if self.timeout_at else None,
            'responses': [
                {
                    'approved': r.approved,
                    'approver_id': r.approver_id,
                    'comments': r.comments,
                    'responded_at': r.responded_at.isoformat()
                }
                for r in self.responses
            ],
            'escalated_to': self.escalated_to
        }
    
    def is_approved(self) -> bool:
        """Check if request is approved based on configuration."""
        if not self.responses:
            return False
        
        if self.config.require_all_approvers:
            # All approvers must approve
            approver_ids = {r.approver_id for r in self.responses if r.approved}
            return len(approver_ids) == len(self.approvers)
        else:
            # Any approver can approve
            return any(r.approved for r in self.responses)
    
    def is_timed_out(self) -> bool:
        """Check if request has timed out."""
        if self.timeout_at is None:
            return False
        return datetime.utcnow() > self.timeout_at


@dataclass
class NotificationMessage:
    """Notification message content."""
    subject: str
    body: str
    html_body: Optional[str] = None
    sms_body: Optional[str] = None
    voice_message: Optional[str] = None
    actions: List[Dict[str, str]] = field(default_factory=list)


# ============================================================================
# NOTIFICATION SERVICE INTERFACES
# ============================================================================

class NotificationService(ABC):
    """Abstract base class for notification services."""
    
    @abstractmethod
    async def send_notification(
        self,
        recipient: str,
        message: NotificationMessage,
        context: Dict[str, Any]
    ) -> bool:
        """Send notification to recipient."""
        pass


class GmailNotificationService(NotificationService):
    """Gmail-based email notification service."""
    
    def __init__(self, credentials_path: str, sender_email: str):
        self.credentials_path = credentials_path
        self.sender_email = sender_email
        self._client = None
    
    async def send_notification(
        self,
        recipient: str,
        message: NotificationMessage,
        context: Dict[str, Any]
    ) -> bool:
        """Send email notification via Gmail."""
        try:
            # Import here to avoid dependency if not used
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build
            from email.mime.text import MIMEText
            import base64
            
            # Build Gmail service
            creds = Credentials.from_authorized_user_file(self.credentials_path)
            service = build('gmail', 'v1', credentials=creds)
            
            # Create message
            if message.html_body:
                mime_message = MIMEText(message.html_body, 'html')
            else:
                mime_message = MIMEText(message.body, 'plain')
            
            mime_message['to'] = recipient
            mime_message['from'] = self.sender_email
            mime_message['subject'] = message.subject
            
            # Encode and send
            encoded_message = base64.urlsafe_b64encode(mime_message.as_bytes()).decode()
            
            await asyncio.to_thread(
                service.users().messages().send(
                    userId='me',
                    body={'raw': encoded_message}
                ).execute
            )
            
            logger.info(f"Email sent to {recipient}")
            return True
            
        except (OSError, ValueError) as e:
            logger.error(f"Failed to send email to {recipient}: {e}")
            return False


class TwilioSMSService(NotificationService):
    """Twilio-based SMS notification service."""
    
    def __init__(self, account_sid: str, auth_token: str, from_number: str):
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number
    
    async def send_notification(
        self,
        recipient: str,
        message: NotificationMessage,
        context: Dict[str, Any]
    ) -> bool:
        """Send SMS notification via Twilio."""
        try:
            from twilio.rest import Client
            
            client = Client(self.account_sid, self.auth_token)
            
            sms_body = message.sms_body or message.body[:1600]  # Twilio limit
            
            await asyncio.to_thread(
                client.messages.create,
                body=sms_body,
                from_=self.from_number,
                to=recipient
            )
            
            logger.info(f"SMS sent to {recipient}")
            return True
            
        except (ImportError, OSError, ValueError) as e:
            logger.error(f"Failed to send SMS to {recipient}: {e}")
            return False


class TwilioVoiceService(NotificationService):
    """Twilio-based voice call notification service."""
    
    def __init__(self, account_sid: str, auth_token: str, from_number: str, twiml_url: str):
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number
        self.twiml_url = twiml_url
    
    async def send_notification(
        self,
        recipient: str,
        message: NotificationMessage,
        context: Dict[str, Any]
    ) -> bool:
        """Make voice call via Twilio."""
        try:
            from twilio.rest import Client
            
            client = Client(self.account_sid, self.auth_token)
            
            # Generate TwiML for the call
            voice_message = message.voice_message or message.body
            twiml = self._generate_twiml(voice_message, context)
            
            await asyncio.to_thread(
                client.calls.create,
                twiml=twiml,
                to=recipient,
                from_=self.from_number
            )
            
            logger.info(f"Voice call initiated to {recipient}")
            return True
            
        except (ImportError, OSError, ValueError) as e:
            logger.error(f"Failed to make voice call to {recipient}: {e}")
            return False
    
    def _generate_twiml(self, message: str, context: Dict[str, Any]) -> str:
        """Generate TwiML for voice call."""
        approval_url = context.get('approval_url', '')
        
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="alice">{message}</Say>
    <Pause length="2"/>
    <Say>To approve, press 1. To reject, press 2.</Say>
    <Gather numDigits="1" action="{approval_url}" method="POST">
        <Say>Please make your selection now.</Say>
    </Gather>
</Response>"""


class SlackNotificationService(NotificationService):
    """Slack notification service."""
    
    def __init__(self, bot_token: str, default_channel: Optional[str] = None):
        self.bot_token = bot_token
        self.default_channel = default_channel
    
    async def send_notification(
        self,
        recipient: str,
        message: NotificationMessage,
        context: Dict[str, Any]
    ) -> bool:
        """Send Slack notification."""
        try:
            import aiohttp
            
            channel = recipient or self.default_channel
            if not channel:
                raise ValueError("No Slack channel specified")
            
            blocks = self._build_blocks(message, context)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    'https://slack.com/api/chat.postMessage',
                    headers={'Authorization': f'Bearer {self.bot_token}'},
                    json={
                        'channel': channel,
                        'text': message.subject,
                        'blocks': blocks
                    }
                ) as response:
                    result = await response.json()
                    if result.get('ok'):
                        logger.info(f"Slack message sent to {channel}")
                        return True
                    else:
                        logger.error(f"Slack API error: {result.get('error')}")
                        return False
                        
        except (ImportError, OSError, ValueError) as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False
    
    def _build_blocks(self, message: NotificationMessage, context: Dict[str, Any]) -> List[Dict]:
        """Build Slack message blocks."""
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": message.subject
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": message.body
                }
            }
        ]
        
        # Add action buttons
        if message.actions:
            actions = []
            for action in message.actions:
                actions.append({
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": action['label']
                    },
                    "url": action.get('url', ''),
                    "style": "primary" if action.get('primary') else None
                })
            
            blocks.append({
                "type": "actions",
                "elements": actions
            })
        
        return blocks


# ============================================================================
# APPROVAL STORE
# ============================================================================

class ApprovalStore:
    """Store for approval requests."""
    
    def __init__(self):
        self._approvals: Dict[str, ApprovalRequest] = {}
        self._workflow_approvals: Dict[str, List[str]] = {}
        self._callbacks: Dict[str, List[Callable]] = {}
    
    async def save(self, request: ApprovalRequest) -> None:
        """Save approval request."""
        self._approvals[request.id] = request
        
        # Track by workflow
        if request.workflow_id not in self._workflow_approvals:
            self._workflow_approvals[request.workflow_id] = []
        if request.id not in self._workflow_approvals[request.workflow_id]:
            self._workflow_approvals[request.workflow_id].append(request.id)
        
        # Notify callbacks
        await self._notify_callbacks(request)
    
    async def get(self, request_id: str) -> Optional[ApprovalRequest]:
        """Get approval request by ID."""
        return self._approvals.get(request_id)
    
    async def get_by_workflow(self, workflow_id: str) -> List[ApprovalRequest]:
        """Get all approval requests for a workflow."""
        approval_ids = self._workflow_approvals.get(workflow_id, [])
        return [self._approvals[aid] for aid in approval_ids if aid in self._approvals]
    
    async def update(self, request: ApprovalRequest) -> None:
        """Update approval request."""
        self._approvals[request.id] = request
        await self._notify_callbacks(request)
    
    async def delete(self, request_id: str) -> None:
        """Delete approval request."""
        if request_id in self._approvals:
            request = self._approvals[request_id]
            del self._approvals[request_id]
            
            # Remove from workflow tracking
            if request.workflow_id in self._workflow_approvals:
                if request_id in self._workflow_approvals[request.workflow_id]:
                    self._workflow_approvals[request.workflow_id].remove(request_id)
    
    def register_callback(
        self,
        request_id: str,
        callback: Callable[[ApprovalRequest], Coroutine]
    ) -> None:
        """Register callback for approval updates."""
        if request_id not in self._callbacks:
            self._callbacks[request_id] = []
        self._callbacks[request_id].append(callback)
    
    async def _notify_callbacks(self, request: ApprovalRequest) -> None:
        """Notify registered callbacks."""
        callbacks = self._callbacks.get(request.id, [])
        for callback in callbacks:
            try:
                await callback(request)
            except (OSError, ConnectionError, TimeoutError, ValueError) as e:
                logger.error(f"Callback error: {e}")


# ============================================================================
# HITL MANAGER
# ============================================================================

class HITLManager:
    """
    Human-in-the-Loop Manager.
    
    Manages human approval workflows with:
    - Multi-channel notifications
    - Timeout handling
    - Escalation policies
    - Audit logging
    """
    
    def __init__(
        self,
        approval_store: Optional[ApprovalStore] = None,
        notification_services: Optional[Dict[NotificationChannel, NotificationService]] = None
    ):
        self.approval_store = approval_store or ApprovalStore()
        self.notification_services = notification_services or {}
        self._timeout_tasks: Dict[str, asyncio.Task] = {}
        self._approval_events: Dict[str, asyncio.Event] = {}
        self._audit_callbacks: List[Callable] = []
    
    async def request_approval(
        self,
        workflow_id: str,
        task_id: str,
        config: ApprovalConfig,
        context: ApprovalContext
    ) -> ApprovalRequest:
        """
        Request human approval for a task.
        
        Args:
            workflow_id: Workflow ID
            task_id: Task ID
            config: Approval configuration
            context: Approval context
        
        Returns:
            Approval request object
        """
        # Create approval request
        request = ApprovalRequest(
            id=str(uuid.uuid4()),
            workflow_id=workflow_id,
            task_id=task_id,
            approvers=config.approvers,
            context=context,
            config=config
        )
        
        # Save request
        await self.approval_store.save(request)
        
        # Send notifications
        await self._send_notifications(request)
        
        # Set up timeout handler
        if config.timeout:
            timeout_task = asyncio.create_task(
                self._handle_timeout(request.id)
            )
            self._timeout_tasks[request.id] = timeout_task
        
        # Log audit event
        await self._log_audit_event('approval_requested', request)
        
        logger.info(f"Approval requested: {request.id} for workflow {workflow_id}, task {task_id}")
        
        return request
    
    async def wait_for_approval(
        self,
        request_id: str,
        timeout: Optional[float] = None
    ) -> Optional[ApprovalResponse]:
        """
        Wait for approval response.
        
        Args:
            request_id: Approval request ID
            timeout: Maximum wait time in seconds
        
        Returns:
            Approval response or None if timed out
        """
        # Create event for this request
        event = asyncio.Event()
        self._approval_events[request_id] = event
        
        # Register callback
        async def on_update(request: ApprovalRequest):
            if request.status in [ApprovalStatus.APPROVED, ApprovalStatus.REJECTED]:
                event.set()
        
        self.approval_store.register_callback(request_id, on_update)
        
        try:
            # Wait for response
            await asyncio.wait_for(event.wait(), timeout=timeout)
            
            # Get final request state
            request = await self.approval_store.get(request_id)
            if request and request.responses:
                return request.responses[-1]
            
            return None
            
        except asyncio.TimeoutError:
            logger.warning(f"Approval wait timed out: {request_id}")
            return None
        finally:
            if request_id in self._approval_events:
                del self._approval_events[request_id]
    
    async def submit_response(
        self,
        request_id: str,
        response: ApprovalResponse
    ) -> bool:
        """
        Submit approval response.
        
        Args:
            request_id: Approval request ID
            response: Approval response
        
        Returns:
            True if response was accepted
        """
        # Get request
        request = await self.approval_store.get(request_id)
        if not request:
            logger.error(f"Approval request not found: {request_id}")
            return False
        
        if request.status != ApprovalStatus.PENDING:
            logger.warning(f"Approval request already processed: {request_id}")
            return False
        
        # Validate approver
        if response.approver_id not in request.approvers:
            logger.error(f"Approver {response.approver_id} not authorized for request {request_id}")
            return False
        
        # Add response
        request.responses.append(response)
        
        # Update status
        if request.is_approved():
            request.status = ApprovalStatus.APPROVED
        elif not request.config.require_all_approvers and not response.approved:
            # Single approver rejection
            request.status = ApprovalStatus.REJECTED
        elif len(request.responses) >= len(request.approvers):
            # All approvers responded
            request.status = ApprovalStatus.REJECTED
        
        # Save updated request
        await self.approval_store.update(request)
        
        # Cancel timeout
        if request_id in self._timeout_tasks:
            self._timeout_tasks[request_id].cancel()
            del self._timeout_tasks[request_id]
        
        # Log audit event
        await self._log_audit_event('approval_responded', request, response)
        
        logger.info(f"Approval response submitted: {request_id} by {response.approver_id}")
        
        return True
    
    async def _send_notifications(self, request: ApprovalRequest) -> None:
        """Send notifications to approvers."""
        message = self._build_notification_message(request)
        
        for approver in request.approvers:
            for channel in request.context.notification_channels:
                service = self.notification_services.get(channel)
                if service:
                    try:
                        await service.send_notification(
                            recipient=approver,
                            message=message,
                            context={
                                'request_id': request.id,
                                'workflow_id': request.workflow_id,
                                'task_id': request.task_id,
                                'approval_url': f"/approval/{request.id}"
                            }
                        )
                    except (OSError, ConnectionError, TimeoutError, ValueError) as e:
                        logger.error(f"Failed to send {channel.value} notification: {e}")
    
    def _build_notification_message(self, request: ApprovalRequest) -> NotificationMessage:
        """Build notification message for approval request."""
        context = request.context
        
        # Build email body
        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background: #f5f5f5; padding: 20px; border-radius: 8px;">
                <h2 style="color: #333; margin-top: 0;">{context.title}</h2>
                <p style="color: #666;">{context.description}</p>
                
                <div style="background: white; padding: 15px; border-radius: 4px; margin: 15px 0;">
                    <h3 style="margin-top: 0;">Details:</h3>
                    <pre style="background: #f9f9f9; padding: 10px; overflow-x: auto;">
{json.dumps(context.details, indent=2)}
                    </pre>
                </div>
                
                <div style="margin: 20px 0;">
                    <p><strong>Priority:</strong> {context.priority.upper()}</p>
                    <p><strong>Timeout:</strong> {request.timeout_at.isoformat() if request.timeout_at else 'None'}</p>
                </div>
                
                <div style="text-align: center; margin: 30px 0;">
                    <a href="/approval/{request.id}?action=approve" 
                       style="background: #4CAF50; color: white; padding: 12px 30px; 
                              text-decoration: none; border-radius: 4px; margin-right: 10px;">
                        Approve
                    </a>
                    <a href="/approval/{request.id}?action=reject" 
                       style="background: #f44336; color: white; padding: 12px 30px; 
                              text-decoration: none; border-radius: 4px;">
                        Reject
                    </a>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Build SMS body
        sms_body = f"""Approval Required: {context.title}
{context.description[:100]}...
Reply APPROVE or REJECT.
ID: {request.id}"""
        
        # Build voice message
        voice_message = f"""Approval request for {context.title}. 
{context.description}. 
To approve, press 1. To reject, press 2."""
        
        return NotificationMessage(
            subject=f"Approval Required: {context.title}",
            body=context.description,
            html_body=html_body,
            sms_body=sms_body,
            voice_message=voice_message,
            actions=[
                {'label': 'Approve', 'url': f'/approval/{request.id}?action=approve', 'primary': True},
                {'label': 'Reject', 'url': f'/approval/{request.id}?action=reject'}
            ]
        )
    
    async def _handle_timeout(self, request_id: str) -> None:
        """Handle approval timeout."""
        try:
            request = await self.approval_store.get(request_id)
            if not request:
                return
            
            # Wait for timeout
            if request.timeout_at:
                wait_seconds = (request.timeout_at - datetime.utcnow()).total_seconds()
                if wait_seconds > 0:
                    await asyncio.sleep(wait_seconds)
            
            # Check if still pending
            request = await self.approval_store.get(request_id)
            if not request or request.status != ApprovalStatus.PENDING:
                return
            
            # Apply escalation policy
            policy = request.config.escalation_policy
            
            if policy:
                if policy.action == EscalationAction.AUTO_APPROVE:
                    await self.submit_response(
                        request_id,
                        ApprovalResponse(
                            approved=True,
                            approver_id='system',
                            comments='Auto-approved due to timeout'
                        )
                    )
                    logger.info(f"Auto-approved due to timeout: {request_id}")
                    
                elif policy.action == EscalationAction.AUTO_REJECT:
                    await self.submit_response(
                        request_id,
                        ApprovalResponse(
                            approved=False,
                            approver_id='system',
                            comments='Auto-rejected due to timeout'
                        )
                    )
                    logger.info(f"Auto-rejected due to timeout: {request_id}")
                    
                elif policy.action == EscalationAction.ESCALATE:
                    # Escalate to next approvers
                    request.escalated_to = policy.next_approvers[0] if policy.next_approvers else None
                    request.status = ApprovalStatus.ESCALATED
                    request.escalation_history.append({
                        'from': request.approvers,
                        'to': policy.next_approvers,
                        'at': datetime.utcnow().isoformat()
                    })
                    request.approvers = policy.next_approvers
                    request.timeout_at = datetime.utcnow() + policy.timeout
                    
                    await self.approval_store.update(request)
                    await self._send_notifications(request)
                    
                    # Set up new timeout
                    timeout_task = asyncio.create_task(
                        self._handle_timeout(request_id)
                    )
                    self._timeout_tasks[request_id] = timeout_task
                    
                    logger.info(f"Approval escalated: {request_id}")
                    
                elif policy.action == EscalationAction.NOTIFY:
                    # Just mark as timed out
                    request.status = ApprovalStatus.TIMED_OUT
                    await self.approval_store.update(request)
                    
                    logger.info(f"Approval timed out: {request_id}")
            else:
                # No policy, just mark as timed out
                request.status = ApprovalStatus.TIMED_OUT
                await self.approval_store.update(request)
                
                logger.info(f"Approval timed out: {request_id}")
            
            # Log audit event
            await self._log_audit_event('approval_timeout', request)
            
        except asyncio.CancelledError:
            logger.debug("Approval timeout handler cancelled")
        except (OSError, ConnectionError, TimeoutError, ValueError) as e:
            logger.error(f"Timeout handler error: {e}")
    
    async def _log_audit_event(
        self,
        event_type: str,
        request: ApprovalRequest,
        response: Optional[ApprovalResponse] = None
    ) -> None:
        """Log audit event."""
        event = {
            'type': event_type,
            'timestamp': datetime.utcnow().isoformat(),
            'request_id': request.id,
            'workflow_id': request.workflow_id,
            'task_id': request.task_id,
            'approvers': request.approvers,
            'status': request.status.value
        }
        
        if response:
            event['response'] = {
                'approved': response.approved,
                'approver_id': response.approver_id,
                'comments': response.comments
            }
        
        for callback in self._audit_callbacks:
            try:
                await callback(event)
            except (OSError, ConnectionError, TimeoutError, ValueError) as e:
                logger.error(f"Audit callback error: {e}")
    
    def register_audit_callback(self, callback: Callable) -> None:
        """Register audit log callback."""
        self._audit_callbacks.append(callback)
    
    async def cancel_approval(self, request_id: str) -> bool:
        """Cancel a pending approval request."""
        request = await self.approval_store.get(request_id)
        if not request or request.status != ApprovalStatus.PENDING:
            return False
        
        # Cancel timeout
        if request_id in self._timeout_tasks:
            self._timeout_tasks[request_id].cancel()
            del self._timeout_tasks[request_id]
        
        # Update status
        request.status = ApprovalStatus.REJECTED
        await self.approval_store.update(request)
        
        logger.info(f"Approval cancelled: {request_id}")
        
        return True
    
    async def get_pending_approvals(
        self,
        approver_id: Optional[str] = None
    ) -> List[ApprovalRequest]:
        """Get pending approval requests."""
        all_approvals = list(self.approval_store._approvals.values())
        
        pending = [a for a in all_approvals if a.status == ApprovalStatus.PENDING]
        
        if approver_id:
            pending = [a for a in pending if approver_id in a.approvers]
        
        return pending


# ============================================================================
# WEB INTERFACE
# ============================================================================

class HITLWebInterface:
    """Web interface for human approvals."""
    
    def __init__(self, hitl_manager: HITLManager):
        self.hitl_manager = hitl_manager
        self.approval_store = hitl_manager.approval_store
    
    def setup_routes(self, app):
        """Setup web routes for approval interface."""
        from fastapi import FastAPI, Request, HTTPException
        from fastapi.responses import HTMLResponse, JSONResponse
        
        @app.get("/approval/{request_id}", response_class=HTMLResponse)
        async def approval_page(request_id: str):
            """Render approval page."""
            approval_request = await self.approval_store.get(request_id)
            
            if not approval_request:
                raise HTTPException(status_code=404, detail="Approval request not found")
            
            if approval_request.status != ApprovalStatus.PENDING:
                return HTMLResponse(f"""
                <html>
                <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
                    <h1>Approval Already Processed</h1>
                    <p>This approval request has already been {approval_request.status.value}.</p>
                </body>
                </html>
                """)
            
            return HTMLResponse(self._render_approval_page(approval_request))
        
        @app.post("/approval/{request_id}/respond")
        async def submit_response(request_id: str, request: Request):
            """Submit approval response."""
            data = await request.json()
            
            response = ApprovalResponse(
                approved=data.get('approved', False),
                approver_id=data.get('approver_id', 'unknown'),
                comments=data.get('comments'),
                additional_data=data.get('additional_data', {})
            )
            
            success = await self.hitl_manager.submit_response(request_id, response)
            
            if not success:
                raise HTTPException(status_code=400, detail="Failed to submit response")
            
            return JSONResponse({
                'success': True,
                'message': 'Response submitted successfully'
            })
        
        @app.get("/api/approvals")
        async def list_approvals(
            status: Optional[str] = None,
            approver: Optional[str] = None
        ):
            """List approval requests."""
            approvals = await self.hitl_manager.get_pending_approvals(approver)
            
            if status:
                approvals = [a for a in approvals if a.status.value == status]
            
            return JSONResponse({
                'approvals': [a.to_dict() for a in approvals],
                'count': len(approvals)
            })
    
    def _render_approval_page(self, request: ApprovalRequest) -> str:
        """Render HTML approval page."""
        context = request.context
        
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Approval Request - {context.title}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }}
        
        .approval-card {{
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            max-width: 600px;
            width: 100%;
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
        }}
        
        .header h1 {{
            font-size: 24px;
            margin-bottom: 8px;
        }}
        
        .header p {{
            opacity: 0.9;
            font-size: 14px;
        }}
        
        .content {{
            padding: 30px;
        }}
        
        .details {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        
        .details h3 {{
            font-size: 14px;
            color: #666;
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .details pre {{
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            padding: 12px;
            font-size: 12px;
            overflow-x: auto;
            max-height: 200px;
            overflow-y: auto;
        }}
        
        .info-row {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            font-size: 14px;
        }}
        
        .info-label {{
            color: #666;
        }}
        
        .info-value {{
            font-weight: 500;
        }}
        
        .priority {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 500;
            text-transform: uppercase;
        }}
        
        .priority.high {{
            background: #fee;
            color: #c33;
        }}
        
        .priority.normal {{
            background: #e8f4fd;
            color: #0066cc;
        }}
        
        .priority.low {{
            background: #e8f5e9;
            color: #2e7d32;
        }}
        
        .priority.critical {{
            background: #ffebee;
            color: #c62828;
            animation: pulse 2s infinite;
        }}
        
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.7; }}
        }}
        
        .timeout {{
            background: #fff3e0;
            border: 1px solid #ff9800;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .timeout-icon {{
            font-size: 20px;
        }}
        
        .timeout-text {{
            font-size: 14px;
            color: #e65100;
        }}
        
        .comments {{
            margin-bottom: 20px;
        }}
        
        .comments label {{
            display: block;
            font-size: 14px;
            color: #666;
            margin-bottom: 8px;
        }}
        
        .comments textarea {{
            width: 100%;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 8px;
            font-size: 14px;
            resize: vertical;
            min-height: 80px;
        }}
        
        .actions {{
            display: flex;
            gap: 12px;
        }}
        
        .btn {{
            flex: 1;
            padding: 14px 24px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
        }}
        
        .btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }}
        
        .btn-approve {{
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
            color: white;
        }}
        
        .btn-reject {{
            background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
            color: white;
        }}
        
        .approver-id {{
            margin-bottom: 20px;
        }}
        
        .approver-id label {{
            display: block;
            font-size: 14px;
            color: #666;
            margin-bottom: 8px;
        }}
        
        .approver-id input {{
            width: 100%;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 8px;
            font-size: 14px;
        }}
        
        .loading {{
            display: none;
            text-align: center;
            padding: 20px;
        }}
        
        .loading.active {{
            display: block;
        }}
        
        .spinner {{
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }}
        
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
    </style>
</head>
<body>
    <div class="approval-card">
        <div class="header">
            <h1>{context.title}</h1>
            <p>{context.description}</p>
        </div>
        
        <div class="content">
            <div class="info-row">
                <span class="info-label">Priority</span>
                <span class="priority {context.priority}">{context.priority}</span>
            </div>
            
            <div class="info-row">
                <span class="info-label">Workflow</span>
                <span class="info-value">{request.workflow_id[:8]}...</span>
            </div>
            
            <div class="info-row">
                <span class="info-label">Task</span>
                <span class="info-value">{request.task_id}</span>
            </div>
            
            <div class="timeout">
                <span class="timeout-icon">⏰</span>
                <span class="timeout-text">
                    Timeout: {request.timeout_at.strftime('%Y-%m-%d %H:%M:%S UTC') if request.timeout_at else 'No timeout'}
                </span>
            </div>
            
            <div class="details">
                <h3>Details</h3>
                <pre>{json.dumps(context.details, indent=2)}</pre>
            </div>
            
            <div class="approver-id">
                <label for="approver-id">Your ID</label>
                <input type="text" id="approver-id" placeholder="Enter your approver ID">
            </div>
            
            <div class="comments">
                <label for="comments">Comments (optional)</label>
                <textarea id="comments" placeholder="Add any comments..."></textarea>
            </div>
            
            <div class="actions">
                <button class="btn btn-approve" onclick="submitResponse(true)">
                    ✅ Approve
                </button>
                <button class="btn btn-reject" onclick="submitResponse(false)">
                    ❌ Reject
                </button>
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Submitting...</p>
            </div>
        </div>
    </div>
    
    <script>
        async function submitResponse(approved) {{
            const approverId = document.getElementById('approver-id').value;
            const comments = document.getElementById('comments').value;
            
            if (!approverId) {{
                alert('Please enter your approver ID');
                return;
            }}
            
            document.getElementById('loading').classList.add('active');
            
            try {{
                const response = await fetch('/approval/{request.id}/respond', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{
                        approved: approved,
                        approver_id: approverId,
                        comments: comments
                    }})
                }});
                
                const result = await response.json();
                
                if (result.success) {{
                    document.querySelector('.content').innerHTML = `
                        <div style="text-align: center; padding: 40px;">
                            <h2 style="color: ${{approved ? '#4CAF50' : '#f44336'}};">
                                ${{approved ? '✅ Approved' : '❌ Rejected'}}
                            </h2>
                            <p>Your response has been recorded.</p>
                        </div>
                    `;
                }} else {{
                    alert('Error: ' + result.message);
                    document.getElementById('loading').classList.remove('active');
                }}
            }} catch (error) {{
                alert('Error submitting response: ' + error.message);
                document.getElementById('loading').classList.remove('active');
            }}
        }}
    </script>
</body>
</html>
"""


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def example_hitl_usage():
    """Example usage of HITL system."""
    
    # Create notification services
    notification_services = {
        NotificationChannel.EMAIL: GmailNotificationService(
            credentials_path="/path/to/credentials.json",
            sender_email="agent@example.com"
        ),
        NotificationChannel.SMS: TwilioSMSService(
            account_sid="your_account_sid",
            auth_token="your_auth_token",
            from_number="+1234567890"
        )
    }
    
    # Create HITL manager
    hitl_manager = HITLManager(
        notification_services=notification_services
    )
    
    # Register audit callback
    async def audit_logger(event):
        print(f"Audit: {event}")
    
    hitl_manager.register_audit_callback(audit_logger)
    
    # Request approval
    approval_request = await hitl_manager.request_approval(
        workflow_id="wf-123",
        task_id="task-456",
        config=ApprovalConfig(
            enabled=True,
            approvers=["admin@example.com", "manager@example.com"],
            timeout=timedelta(hours=2),
            escalation_policy=EscalationPolicy(
                action=EscalationAction.ESCALATE,
                timeout=timedelta(hours=1),
                next_approvers=["director@example.com"]
            )
        ),
        context=ApprovalContext(
            title="Approve Data Export",
            description="A data export has been requested that requires your approval.",
            details={
                "export_type": "customer_data",
                "record_count": 15000,
                "requested_by": "analyst@example.com",
                "reason": "Monthly reporting"
            },
            notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SMS],
            priority="high"
        )
    )
    
    print(f"Approval requested: {approval_request.id}")
    
    # Wait for approval (in real usage, this would be async)
    # response = await hitl_manager.wait_for_approval(approval_request.id, timeout=7200)
    
    # Or submit response programmatically
    await hitl_manager.submit_response(
        approval_request.id,
        ApprovalResponse(
            approved=True,
            approver_id="admin@example.com",
            comments="Approved for monthly reporting"
        )
    )


if __name__ == "__main__":
    asyncio.run(example_hitl_usage())
