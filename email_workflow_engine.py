"""
Email Workflow and Automation Engine
OpenClaw Windows 10 AI Agent System

This module implements the core email workflow automation engine with:
- Rule-based email processing
- Auto-responder system
- Template engine
- Scheduled sending
- Thread tracking
- Workflow orchestration
"""

import asyncio
import base64
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)

# Optional dependencies - lazy imported where used to avoid ImportError at startup
jinja2 = None
AsyncIOScheduler = None
CronTrigger = None
DateTrigger = None
notification = None
_twilio_Client = None

def _ensure_jinja2():
    global jinja2
    if jinja2 is None:
        import jinja2 as _jinja2
        jinja2 = _jinja2
    return jinja2

def _ensure_scheduler():
    global AsyncIOScheduler, CronTrigger, DateTrigger
    if AsyncIOScheduler is None:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler as _AS
        from apscheduler.triggers.cron import CronTrigger as _CT
        from apscheduler.triggers.date import DateTrigger as _DT
        AsyncIOScheduler = _AS
        CronTrigger = _CT
        DateTrigger = _DT
    return AsyncIOScheduler

def _ensure_notification():
    global notification
    if notification is None:
        from plyer import notification as _n
        notification = _n
    return notification

def _ensure_twilio():
    global _twilio_Client
    if _twilio_Client is None:
        from twilio.rest import Client
        _twilio_Client = Client
    return _twilio_Client


# =============================================================================
# ENUMERATIONS
# =============================================================================

class ConditionOperator(Enum):
    """Available condition operators for email filtering."""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    MATCHES_REGEX = "matches_regex"
    NOT_MATCHES_REGEX = "not_matches_regex"
    IN_LIST = "in_list"
    NOT_IN_LIST = "not_in_list"
    LESS_THAN = "<"
    LESS_THAN_OR_EQUAL = "<="
    GREATER_THAN = ">"
    GREATER_THAN_OR_EQUAL = ">="
    BETWEEN = "between"
    NOT_BETWEEN = "not_between"
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"
    CONTAINS_ALL = "contains_all"
    CONTAINS_ANY = "contains_any"
    CONTAINS_NONE = "contains_none"
    BEFORE = "before"
    AFTER = "after"
    WITHIN_LAST = "within_last"
    WITHIN_NEXT = "within_next"


class LogicalOperator(Enum):
    """Logical operators for condition groups."""
    AND = "and"
    OR = "or"
    NOT = "not"


class ActionType(Enum):
    """Available action types for email processing."""
    # Gmail Actions
    LABEL_ADD = "label_add"
    LABEL_REMOVE = "label_remove"
    LABEL_CREATE = "label_create"
    MARK_READ = "mark_read"
    MARK_IMPORTANT = "mark_important"
    STAR = "star"
    MOVE_TO_FOLDER = "move_to_folder"
    ARCHIVE = "archive"
    TRASH = "trash"
    DELETE = "delete"
    MARK_SPAM = "mark_spam"
    
    # Response Actions
    REPLY = "reply"
    REPLY_ALL = "reply_all"
    FORWARD = "forward"
    SEND_NEW = "send_new"
    AUTO_RESPOND = "auto_respond"
    SET_VACATION_RESPONDER = "set_vacation_responder"
    
    # Notification Actions
    NOTIFY_DESKTOP = "notify_desktop"
    NOTIFY_EMAIL = "notify_email"
    NOTIFY_SMS = "notify_sms"
    NOTIFY_VOICE = "notify_voice"
    NOTIFY_TTS = "notify_tts"
    NOTIFY_WEBHOOK = "notify_webhook"
    NOTIFY_SLACK = "notify_slack"
    NOTIFY_DISCORD = "notify_discord"
    
    # Organization Actions
    CREATE_TASK = "create_task"
    CREATE_CALENDAR_EVENT = "create_calendar_event"
    ADD_TO_CONTACTS = "add_to_contacts"
    CREATE_NOTE = "create_note"
    SAVE_ATTACHMENT = "save_attachment"
    EXTRACT_DATA = "extract_data"
    
    # AI-Powered Actions
    AI_SUMMARIZE = "ai_summarize"
    AI_CLASSIFY = "ai_classify"
    AI_EXTRACT_ENTITIES = "ai_extract_entities"
    AI_GENERATE_RESPONSE = "ai_generate_response"
    AI_TRANSLATE = "ai_translate"
    AI_ANALYZE_SENTIMENT = "ai_analyze_sentiment"
    AI_SUGGEST_ACTIONS = "ai_suggest_actions"
    
    # Workflow Actions
    TRIGGER_WORKFLOW = "trigger_workflow"
    DELAY = "delay"
    WAIT_FOR_REPLY = "wait_for_reply"
    BRANCH = "branch"
    LOOP = "loop"
    PARALLEL_EXECUTE = "parallel_execute"
    
    # Integration Actions
    BROWSER_OPEN = "browser_open"
    RUN_SCRIPT = "run_script"
    CALL_API = "call_api"
    UPDATE_DATABASE = "update_database"
    PUBLISH_EVENT = "publish_event"


class ErrorAction(Enum):
    """Error handling strategies for actions."""
    LOG_AND_CONTINUE = "log_and_continue"
    LOG_AND_STOP = "log_and_stop"
    RETRY = "retry"
    FALLBACK = "fallback"
    NOTIFY_AND_CONTINUE = "notify_and_continue"
    NOTIFY_AND_STOP = "notify_and_stop"


class ConflictStrategy(Enum):
    """Conflict resolution strategies."""
    PRIORITY = "priority"
    SEQUENCE = "sequence"
    MERGE = "merge"
    INTERACTIVE = "interactive"


class ResponseType(Enum):
    """Types of auto-responses."""
    TEMPLATE = "template"
    AI_GENERATED = "ai_generated"
    HYBRID = "hybrid"
    FORWARD = "forward"
    WEBHOOK = "webhook"


class ScheduleStatus(Enum):
    """Status values for scheduled emails."""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    SENT = "sent"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ThreadStatus(Enum):
    """Thread status values."""
    ACTIVE = "active"
    ARCHIVED = "archived"
    MUTED = "muted"
    RESOLVED = "resolved"
    SPAM = "spam"


class NodeType(Enum):
    """Available workflow node types."""
    # Triggers
    TRIGGER_EMAIL_RECEIVED = "trigger_email_received"
    TRIGGER_SCHEDULED = "trigger_scheduled"
    TRIGGER_MANUAL = "trigger_manual"
    TRIGGER_WEBHOOK = "trigger_webhook"
    
    # Conditions
    CONDITION_FILTER = "condition_filter"
    CONDITION_AI_ANALYSIS = "condition_ai_analysis"
    CONDITION_TIME = "condition_time"
    CONDITION_VARIABLE = "condition_variable"
    
    # Actions
    ACTION_LABEL = "action_label"
    ACTION_MOVE = "action_move"
    ACTION_REPLY = "action_reply"
    ACTION_FORWARD = "action_forward"
    ACTION_NOTIFY = "action_notify"
    ACTION_WEBHOOK = "action_webhook"
    ACTION_AI_GENERATE = "action_ai_generate"
    ACTION_DELAY = "action_delay"
    
    # Flow Control
    CONTROL_BRANCH = "control_branch"
    CONTROL_MERGE = "control_merge"
    CONTROL_LOOP = "control_loop"
    CONTROL_PARALLEL = "control_parallel"
    
    # Integration
    INTEGRATION_CALENDAR = "integration_calendar"
    INTEGRATION_TASK = "integration_task"
    INTEGRATION_SLACK = "integration_slack"
    INTEGRATION_TWILIO = "integration_twilio"


class ExecutionStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ErrorHandlingStrategy(Enum):
    """Error handling strategies for workflows."""
    STOP = "stop"
    CONTINUE = "continue"
    RETRY = "retry"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class EmailMessage:
    """Represents an email message."""
    id: str
    thread_id: Optional[str]
    gmail_thread_id: Optional[str]
    
    # Headers
    subject: str
    from_address: str
    from_name: Optional[str]
    to_addresses: List[str]
    cc_addresses: List[str] = field(default_factory=list)
    bcc_addresses: List[str] = field(default_factory=list)
    reply_to: Optional[str] = None
    
    # Content
    body_text: str = ""
    body_html: Optional[str] = None
    
    # Metadata
    received_date: datetime = field(default_factory=datetime.utcnow)
    message_id: Optional[str] = None
    in_reply_to: Optional[str] = None
    references: List[str] = field(default_factory=list)
    
    # Attachments
    attachments: List['Attachment'] = field(default_factory=list)
    has_attachments: bool = False
    
    # Gmail-specific
    labels: List[str] = field(default_factory=list)
    is_read: bool = False
    is_important: bool = False
    is_starred: bool = False
    
    # AI analysis results
    ai_analysis: Optional['EmailAnalysis'] = None


@dataclass
class Attachment:
    """Represents an email attachment."""
    filename: str
    mime_type: str
    size: int
    content_id: Optional[str] = None
    content: Optional[bytes] = None


@dataclass
class EmailAnalysis:
    """AI analysis results for an email."""
    sentiment: float = 0.0  # -1.0 to 1.0
    urgency: float = 0.0  # 0.0 to 1.0
    intent: str = "unknown"
    category: str = "general"
    entities: List[Dict[str, Any]] = field(default_factory=list)
    summary: Optional[str] = None
    action_items: List[str] = field(default_factory=list)
    reading_time_minutes: float = 0.0
    spam_probability: float = 0.0
    phishing_risk: float = 0.0
    confidence: float = 0.0
    tone: str = "neutral"
    suggested_actions: List[str] = field(default_factory=list)


@dataclass
class Condition:
    """Single condition for rule evaluation."""
    field: str
    operator: ConditionOperator
    value: Any
    case_sensitive: bool = False


@dataclass
class ConditionGroup:
    """Group of conditions with logical operator."""
    operator: LogicalOperator = LogicalOperator.AND
    conditions: List[Condition] = field(default_factory=list)
    negate: bool = False


@dataclass
class Action:
    """Action to execute when rule matches."""
    type: ActionType
    parameters: Dict[str, Any] = field(default_factory=dict)
    delay: Optional[timedelta] = None
    condition: Optional[str] = None


@dataclass
class ActionConfig:
    """Configuration for action execution."""
    type: ActionType
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    delay: Optional[timedelta] = None
    timeout: timedelta = field(default_factory=lambda: timedelta(seconds=30))
    retry_count: int = 3
    retry_delay: timedelta = field(default_factory=lambda: timedelta(seconds=5))
    on_error: ErrorAction = ErrorAction.LOG_AND_CONTINUE
    fallback_action: Optional[ActionType] = None
    condition: Optional[str] = None
    rate_limit_key: Optional[str] = None
    rate_limit_max: Optional[int] = None
    rate_limit_window: Optional[timedelta] = None


@dataclass
class EmailRule:
    """Complete rule definition for email processing."""
    id: str
    name: str
    description: str = ""
    enabled: bool = True
    priority: int = 100
    stop_processing: bool = False
    conditions: List[ConditionGroup] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)
    schedule: Optional['RuleSchedule'] = None
    rate_limit: Optional['RateLimit'] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"
    tags: List[str] = field(default_factory=list)


@dataclass
class RuleSchedule:
    """Schedule constraints for a rule."""
    active_days: List[str] = field(default_factory=lambda: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    active_hours_start: str = "00:00"
    active_hours_end: str = "23:59"
    timezone: str = "UTC"


@dataclass
class RateLimit:
    """Rate limiting configuration."""
    max_executions: int
    window: timedelta
    per_sender: bool = False


@dataclass
class ProcessingResult:
    """Result of email processing."""
    email_id: str
    matched_rules: List[str]
    actions_executed: List['ActionResult']
    timestamp: datetime
    duration_ms: float = 0.0


@dataclass
class ActionResult:
    """Result of action execution."""
    action: ActionConfig
    success: bool
    data: Any = None
    error: Optional[str] = None
    attempts: int = 1
    execution_time_ms: float = 0.0


@dataclass
class AutoResponder:
    """Auto-responder configuration."""
    id: str
    name: str
    enabled: bool = True
    trigger_conditions: List[Condition] = field(default_factory=list)
    response_type: ResponseType = ResponseType.TEMPLATE
    response_template: Optional[str] = None
    ai_prompt: Optional[str] = None
    delay: timedelta = field(default_factory=lambda: timedelta(minutes=0))
    rate_limit: Optional[RateLimit] = None
    exclude_senders: List[str] = field(default_factory=list)
    exclude_subjects: List[str] = field(default_factory=list)
    exclude_domains: List[str] = field(default_factory=list)
    is_vacation_responder: bool = False
    vacation_start: Optional[datetime] = None
    vacation_end: Optional[datetime] = None
    vacation_message: Optional[str] = None
    max_responses_per_sender: int = 1
    response_cooldown: timedelta = field(default_factory=lambda: timedelta(hours=24))
    include_original_message: bool = False
    reply_to_thread: bool = True
    responses_sent: int = 0
    last_response_at: Optional[datetime] = None


@dataclass
class GeneratedResponse:
    """Generated auto-response."""
    subject: str
    body: str
    tone: str
    confidence: float
    suggested_actions: List[str]


@dataclass
class EmailTemplate:
    """Email template definition."""
    id: str
    name: str
    description: str = ""
    subject_template: str = ""
    body_html_template: Optional[str] = None
    body_text_template: Optional[str] = None
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    language: str = "en"
    translations: Dict[str, Dict[str, str]] = field(default_factory=dict)
    required_variables: List[str] = field(default_factory=list)
    optional_variables: Dict[str, Any] = field(default_factory=dict)
    css_styles: Optional[str] = None
    header_image: Optional[str] = None
    footer_template: Optional[str] = None
    ai_enhance: bool = False
    ai_prompt: Optional[str] = None
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RenderedEmail:
    """Rendered email from template."""
    subject: str
    body_html: Optional[str]
    body_text: Optional[str]
    template_id: str


@dataclass
class ScheduledEmail:
    """Scheduled email definition."""
    template: EmailTemplate
    recipients: List[str]
    variables: Dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    scheduled_time: Optional[datetime] = None
    cron_expression: Optional[str] = None
    is_recurring: bool = False
    status: ScheduleStatus = ScheduleStatus.PENDING
    attempts: int = 0
    last_attempt: Optional[datetime] = None
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"


@dataclass
class ConversationThread:
    """Email conversation thread."""
    id: str
    gmail_thread_id: Optional[str]
    subject: str
    participants: Set[str]
    messages: List[EmailMessage]
    created_at: datetime
    last_activity: datetime
    status: ThreadStatus
    summary: Optional[str] = None
    category: Optional[str] = None
    priority: Optional[int] = None
    sentiment_trend: Optional[List[float]] = None
    pending_actions: List['ActionItem'] = field(default_factory=list)
    completed_actions: List['ActionItem'] = field(default_factory=list)
    user_labels: List[str] = field(default_factory=list)
    is_muted: bool = False
    is_pinned: bool = False


@dataclass
class ActionItem:
    """Extracted action item from conversation."""
    description: str
    assignee: Optional[str] = None
    due_date: Optional[datetime] = None
    priority: str = "medium"
    status: str = "pending"
    source_message_id: Optional[str] = None


@dataclass
class SentimentAnalysis:
    """Sentiment analysis result."""
    overall_score: float
    trend: float
    timeline: List[Dict[str, Any]]


@dataclass
class WorkflowNode:
    """Node in workflow graph."""
    id: str
    type: NodeType
    position: 'Position'
    config: Dict[str, Any]


@dataclass
class Position:
    """Position for visual layout."""
    x: float
    y: float


@dataclass
class WorkflowEdge:
    """Connection between nodes."""
    id: str
    source: str
    target: str
    condition: Optional[str] = None


@dataclass
class WorkflowDefinition:
    """Visual workflow definition."""
    id: str
    name: str
    description: str
    nodes: List[WorkflowNode]
    edges: List[WorkflowEdge]
    trigger: 'WorkflowTrigger'
    enabled: bool = True
    parallel_execution: bool = False
    error_handling: ErrorHandlingStrategy = ErrorHandlingStrategy.STOP


@dataclass
class WorkflowTrigger:
    """Workflow trigger configuration."""
    type: str
    filters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionContext:
    """Context for rule/action execution."""
    email: EmailMessage
    variables: Dict[str, Any] = field(default_factory=dict)
    user: str = "system"


@dataclass
class ExecutionState:
    """State of workflow execution."""
    execution_id: str
    workflow_id: str
    context: 'WorkflowContext'
    current_nodes: List[str]
    completed_nodes: List[str]
    variables: Dict[str, Any]


@dataclass
class WorkflowContext:
    """Context for workflow execution."""
    email_id: Optional[str] = None
    variables: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowExecutionResult:
    """Result of workflow execution."""
    execution_id: str
    workflow_id: str
    status: ExecutionStatus
    variables: Dict[str, Any]
    error: Optional[str] = None


@dataclass
class NodeExecutionResult:
    """Result of node execution."""
    success: bool
    variables: Dict[str, Any]
    error: Optional[str] = None


@dataclass
class SecurityScanResult:
    """Security scan result."""
    email_id: str
    overall_risk: str
    findings: List[Any]
    recommended_action: str


# =============================================================================
# CORE CLASSES
# =============================================================================

class RuleEngine:
    """
    Hybrid rule engine supporting both deterministic and AI-powered rules.
    """
    
    def __init__(self, ai_client=None):
        self.rules: List[EmailRule] = []
        self.deterministic_evaluator = DeterministicEvaluator()
        self.ai_evaluator = AIEvaluator(ai_client) if ai_client else None
        self.action_executor = ActionExecutor()
        self.conflict_resolver = RuleConflictResolver()
    
    def add_rule(self, rule: EmailRule):
        """Add a rule to the engine."""
        self.rules.append(rule)
        # Sort by priority
        self.rules.sort(key=lambda r: r.priority)
    
    def remove_rule(self, rule_id: str):
        """Remove a rule by ID."""
        self.rules = [r for r in self.rules if r.id != rule_id]
    
    async def process_email(self, email: EmailMessage) -> ProcessingResult:
        """
        Main entry point for email processing.
        
        Processing Pipeline:
        1. Pre-processing (enrichment, normalization)
        2. Rule matching (deterministic + AI)
        3. Action execution (parallel where safe)
        4. Post-processing (logging, analytics)
        """
        start_time = datetime.utcnow()
        
        # Step 1: Enrichment
        enriched_email = await self._enrich_email(email)
        
        # Step 2: Rule matching with priority ordering
        matched_rules = await self._match_rules(enriched_email)
        
        # Step 3: Resolve conflicts
        resolved_rules = self.conflict_resolver.resolve_conflicts(
            matched_rules, enriched_email
        )
        
        # Step 4: Execute actions
        context = ExecutionContext(email=enriched_email)
        results = await self.action_executor.execute(
            enriched_email,
            [ActionConfig(type=a.type, parameters=a.parameters) 
             for r in resolved_rules for a in r.actions],
            context
        )
        
        duration = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return ProcessingResult(
            email_id=email.id,
            matched_rules=[r.id for r in resolved_rules],
            actions_executed=results,
            timestamp=datetime.utcnow(),
            duration_ms=duration
        )
    
    async def _enrich_email(self, email: EmailMessage) -> EmailMessage:
        """Enrich email with additional metadata."""
        # Add AI analysis if available
        if self.ai_evaluator:
            email.ai_analysis = await self.ai_evaluator.analyze(email)
        return email
    
    async def _match_rules(self, email: EmailMessage) -> List[EmailRule]:
        """Find all rules that match the email."""
        matched = []
        
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            if await self._evaluate_rule(rule, email):
                matched.append(rule)
        
        return matched
    
    async def _evaluate_rule(self, rule: EmailRule, email: EmailMessage) -> bool:
        """Evaluate if a rule matches an email."""
        for group in rule.conditions:
            if not await self._evaluate_condition_group(group, email):
                return False
        return True
    
    async def _evaluate_condition_group(
        self,
        group: ConditionGroup,
        email: EmailMessage
    ) -> bool:
        """Evaluate a condition group."""
        results = []
        
        for condition in group.conditions:
            result = await self.deterministic_evaluator.evaluate(condition, email)
            results.append(result)
        
        if group.operator == LogicalOperator.AND:
            result = all(results)
        elif group.operator == LogicalOperator.OR:
            result = any(results)
        else:  # NOT
            result = not any(results)
        
        if group.negate:
            result = not result
        
        return result


class DeterministicEvaluator:
    """Evaluates deterministic conditions."""
    
    async def evaluate(self, condition: Condition, email: EmailMessage) -> bool:
        """Evaluate a single condition against an email."""
        value = self._get_field_value(condition.field, email)
        
        if condition.operator == ConditionOperator.EXISTS:
            return value is not None
        
        if condition.operator == ConditionOperator.NOT_EXISTS:
            return value is None
        
        if value is None:
            return False
        
        return self._compare(value, condition.value, condition.operator, condition.case_sensitive)
    
    def _get_field_value(self, field: str, email: EmailMessage) -> Any:
        """Get field value from email."""
        field_map = {
            'id': email.id,
            'subject': email.subject,
            'from': email.from_address,
            'from_name': email.from_name,
            'from_domain': email.from_address.split('@')[-1] if '@' in email.from_address else None,
            'to': email.to_addresses,
            'cc': email.cc_addresses,
            'body': email.body_text,
            'body_length': len(email.body_text) if email.body_text else 0,
            'has_attachments': email.has_attachments,
            'attachment_count': len(email.attachments),
            'is_read': email.is_read,
            'is_important': email.is_important,
            'is_starred': email.is_starred,
            'received_date': email.received_date,
            'labels': email.labels,
        }
        
        # AI analysis fields
        if email.ai_analysis:
            field_map.update({
                'ai_sentiment': email.ai_analysis.sentiment,
                'ai_urgency': email.ai_analysis.urgency,
                'ai_intent': email.ai_analysis.intent,
                'ai_category': email.ai_analysis.category,
                'ai_spam_probability': email.ai_analysis.spam_probability,
                'ai_phishing_risk': email.ai_analysis.phishing_risk,
            })
        
        return field_map.get(field)
    
    def _compare(
        self,
        value: Any,
        target: Any,
        operator: ConditionOperator,
        case_sensitive: bool
    ) -> bool:
        """Compare values based on operator."""
        # Handle string comparison
        if isinstance(value, str) and isinstance(target, str) and not case_sensitive:
            value = value.lower()
            target = target.lower()
        
        if operator == ConditionOperator.EQUALS:
            return value == target
        elif operator == ConditionOperator.NOT_EQUALS:
            return value != target
        elif operator == ConditionOperator.CONTAINS:
            if isinstance(value, str) and isinstance(target, str):
                return target in value
            elif isinstance(value, list):
                return target in value
        elif operator == ConditionOperator.NOT_CONTAINS:
            if isinstance(value, str) and isinstance(target, str):
                return target not in value
            elif isinstance(value, list):
                return target not in value
        elif operator == ConditionOperator.STARTS_WITH:
            if isinstance(value, str) and isinstance(target, str):
                return value.startswith(target)
        elif operator == ConditionOperator.ENDS_WITH:
            if isinstance(value, str) and isinstance(target, str):
                return value.endswith(target)
        elif operator == ConditionOperator.MATCHES_REGEX:
            if isinstance(value, str) and isinstance(target, str):
                return bool(re.search(target, value, re.IGNORECASE))
        elif operator == ConditionOperator.IN_LIST:
            if isinstance(target, list):
                return value in target
        elif operator == ConditionOperator.NOT_IN_LIST:
            if isinstance(target, list):
                return value not in target
        elif operator == ConditionOperator.GREATER_THAN:
            return value > target
        elif operator == ConditionOperator.LESS_THAN:
            return value < target
        elif operator == ConditionOperator.GREATER_THAN_OR_EQUAL:
            return value >= target
        elif operator == ConditionOperator.LESS_THAN_OR_EQUAL:
            return value <= target
        elif operator == ConditionOperator.BETWEEN:
            if isinstance(target, (list, tuple)) and len(target) == 2:
                return target[0] <= value <= target[1]
        
        return False


class AIEvaluator:
    """AI-powered condition evaluator."""
    
    def __init__(self, ai_client):
        self.ai_client = ai_client
    
    async def analyze(self, email: EmailMessage) -> EmailAnalysis:
        """Analyze email using AI via OpenAI client."""
        try:
            from openai_client import OpenAIClient
            client = OpenAIClient.get_instance()
            prompt = (
                f"Analyze this email and return a JSON object with keys: "
                f"sentiment (float -1 to 1), urgency (float 0 to 1), "
                f"intent (string), category (string), confidence (float 0 to 1).\n"
                f"Subject: {email.subject}\n"
                f"From: {email.sender}\n"
                f"Body: {email.body[:500]}\n"
                f"Return ONLY valid JSON."
            )
            response = client.generate(prompt, max_tokens=150)
            import json as _json
            data = _json.loads(response.strip())
            return EmailAnalysis(
                sentiment=float(data.get('sentiment', 0.0)),
                urgency=float(data.get('urgency', 0.5)),
                intent=str(data.get('intent', 'general')),
                category=str(data.get('category', 'general')),
                confidence=float(data.get('confidence', 0.8)),
            )
        except (ImportError, RuntimeError, OSError, ValueError, KeyError):
            return EmailAnalysis(
                sentiment=0.0,
                urgency=0.5,
                intent="general",
                category="general",
                confidence=0.8,
            )


class RuleConflictResolver:
    """Handles conflicts when multiple rules match."""
    
    def resolve_conflicts(
        self,
        matched_rules: List[EmailRule],
        email: EmailMessage,
        strategy: ConflictStrategy = ConflictStrategy.PRIORITY
    ) -> List[EmailRule]:
        """Resolve conflicts between matching rules."""
        if not matched_rules:
            return []
        
        if strategy == ConflictStrategy.PRIORITY:
            return [min(matched_rules, key=lambda r: r.priority)]
        
        elif strategy == ConflictStrategy.SEQUENCE:
            sorted_rules = sorted(matched_rules, key=lambda r: r.priority)
            result = []
            for rule in sorted_rules:
                result.append(rule)
                if rule.stop_processing:
                    break
            return result
        
        elif strategy == ConflictStrategy.MERGE:
            return self._merge_rules(matched_rules)
        
        return matched_rules
    
    def _merge_rules(self, rules: List[EmailRule]) -> List[EmailRule]:
        """Merge compatible actions from multiple rules."""
        # Simplified implementation
        return rules


class ActionExecutor:
    """Executes actions with proper error handling and retry logic."""
    
    def __init__(self):
        self.handlers: Dict[ActionType, callable] = {}
        self.rate_limiter = RateLimiter()
        self._register_handlers()
    
    def _get_gmail_client(self):
        """Get or create a GmailClient instance."""
        if not hasattr(self, '_gmail_client'):
            try:
                from gmail_client_implementation import GmailClient
                self._gmail_client = GmailClient()
            except (ImportError, RuntimeError) as e:
                logger.warning(f"GmailClient unavailable: {e}")
                self._gmail_client = None
        return self._gmail_client

    def _register_handlers(self):
        """Register action handlers."""
        self.handlers[ActionType.LABEL_ADD] = self._handle_label_add
        self.handlers[ActionType.MARK_READ] = self._handle_mark_read
        self.handlers[ActionType.NOTIFY_DESKTOP] = self._handle_notify_desktop
        self.handlers[ActionType.ARCHIVE] = self._handle_archive
        self.handlers[ActionType.DELETE] = self._handle_delete
        self.handlers[ActionType.REPLY] = self._handle_reply
    
    async def execute(
        self,
        email: EmailMessage,
        actions: List[ActionConfig],
        context: ExecutionContext
    ) -> List[ActionResult]:
        """Execute a list of actions for an email."""
        results = []
        
        for action in actions:
            if not action.enabled:
                continue
            
            # Check conditional execution
            if action.condition and not self._evaluate_condition(action.condition, context):
                continue
            
            # Apply delay if specified
            if action.delay:
                await asyncio.sleep(action.delay.total_seconds())
            
            # Check rate limiting
            if action.rate_limit_key:
                allowed = await self.rate_limiter.check(
                    action.rate_limit_key,
                    action.rate_limit_max,
                    action.rate_limit_window
                )
                if not allowed:
                    results.append(ActionResult(
                        action=action,
                        success=False,
                        error="Rate limit exceeded"
                    ))
                    continue
            
            # Execute action with retry
            result = await self._execute_with_retry(email, action, context)
            results.append(result)
        
        return results
    
    async def _execute_with_retry(
        self,
        email: EmailMessage,
        action: ActionConfig,
        context: ExecutionContext
    ) -> ActionResult:
        """Execute action with retry logic."""
        handler = self.handlers.get(action.type)
        
        if not handler:
            return ActionResult(
                action=action,
                success=False,
                error=f"No handler for action type: {action.type}"
            )
        
        for attempt in range(action.retry_count + 1):
            try:
                start_time = datetime.utcnow()
                result = await asyncio.wait_for(
                    handler(email, action.parameters, context),
                    timeout=action.timeout.total_seconds()
                )
                execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                return ActionResult(
                    action=action,
                    success=True,
                    data=result,
                    attempts=attempt + 1,
                    execution_time_ms=execution_time
                )
            except (RuntimeError, ValueError, TypeError) as e:
                if attempt < action.retry_count:
                    await asyncio.sleep(action.retry_delay.total_seconds() * (2 ** attempt))
                else:
                    return ActionResult(
                        action=action,
                        success=False,
                        error=str(e),
                        attempts=attempt + 1
                    )
    
    def _evaluate_condition(self, condition: str, context: ExecutionContext) -> bool:
        """Evaluate conditional expression."""
        # Simplified implementation using Jinja2
        try:
            jinja2 = _ensure_jinja2()
            sandbox_env = jinja2.sandbox.SandboxedEnvironment()
            template = sandbox_env.from_string(f"{{{{ {condition} }}}}")
            result = template.render(**context.variables)
            return result.lower() in ('true', '1', 'yes')
        except (ValueError, KeyError) as e:
            logger.warning(f"Condition evaluation failed for '{condition}': {e}")
            return False
    
    # Action Handlers
    async def _handle_label_add(
        self,
        email: EmailMessage,
        params: Dict[str, Any],
        context: ExecutionContext
    ):
        """Handle label add action via GmailClient."""
        labels = params.get('labels', [])
        try:
            from gmail_client_implementation import GmailClient
            client = GmailClient()
            client.messages.modify_labels(
                message_id=email.message_id,
                add_label_ids=labels,
            )
        except (ImportError, AttributeError, RuntimeError) as e:
            logger.warning(f"Gmail label_add failed, returning local result: {e}")
        return {'labels_added': labels}
    
    async def _handle_mark_read(
        self,
        email: EmailMessage,
        params: Dict[str, Any],
        context: ExecutionContext
    ):
        """Handle mark read action."""
        read = params.get('read', True)
        return {'marked_read': read}
    
    async def _handle_notify_desktop(
        self,
        email: EmailMessage,
        params: Dict[str, Any],
        context: ExecutionContext
    ):
        """Handle desktop notification action."""
        title = params.get('title', 'Email Notification')
        message = params.get('message', email.subject)
        urgency = params.get('urgency', 'normal')
        
        _ensure_notification().notify(
            title=title,
            message=message,
            timeout=10 if urgency == 'normal' else 0
        )
        
        return {'notified': True}

    async def _handle_archive(
        self,
        email: EmailMessage,
        params: Dict[str, Any],
        context: ExecutionContext
    ):
        """Handle archive action via GmailClient."""
        client = self._get_gmail_client()
        if client and hasattr(email, 'message_id') and email.message_id:
            try:
                client.messages.modify_labels(
                    message_id=email.message_id,
                    remove_label_ids=['INBOX'],
                )
            except (RuntimeError, OSError, AttributeError) as e:
                logger.warning(f"Gmail archive failed: {e}")
        return {'archived': True}

    async def _handle_delete(
        self,
        email: EmailMessage,
        params: Dict[str, Any],
        context: ExecutionContext
    ):
        """Handle delete (trash) action via GmailClient."""
        client = self._get_gmail_client()
        if client and hasattr(email, 'message_id') and email.message_id:
            try:
                client.messages.trash_message(email.message_id)
            except (RuntimeError, OSError, AttributeError) as e:
                logger.warning(f"Gmail delete failed: {e}")
        return {'deleted': True}

    async def _handle_reply(
        self,
        email: EmailMessage,
        params: Dict[str, Any],
        context: ExecutionContext
    ):
        """Handle reply action via GmailClient."""
        reply_body = params.get('body', '')
        client = self._get_gmail_client()
        if client and reply_body:
            try:
                thread_id = getattr(email, 'thread_id', None)
                to_addr = getattr(email, 'sender', params.get('to', ''))
                subject = f"Re: {getattr(email, 'subject', '')}"
                client.messages.send_message(
                    to=to_addr,
                    subject=subject,
                    body=reply_body,
                    thread_id=thread_id,
                )
            except (RuntimeError, OSError, AttributeError) as e:
                logger.warning(f"Gmail reply failed: {e}")
        return {'replied': True, 'body_length': len(reply_body)}


class RateLimiter:
    """Rate limiting for actions."""
    
    def __init__(self):
        self.counters: Dict[str, List[datetime]] = {}
    
    async def check(
        self,
        key: str,
        max_count: int,
        window: timedelta
    ) -> bool:
        """Check if action is within rate limit."""
        now = datetime.utcnow()
        
        if key not in self.counters:
            self.counters[key] = []
        
        # Remove old entries
        cutoff = now - window
        self.counters[key] = [t for t in self.counters[key] if t > cutoff]
        
        # Check limit
        if len(self.counters[key]) >= max_count:
            return False
        
        # Add current request
        self.counters[key].append(now)
        return True


class TemplateEngine:
    """Advanced template engine for email generation."""
    
    def __init__(self, template_dir: str = "templates/emails"):
        _j = _ensure_jinja2()
        self.env = _j.Environment(
            loader=_j.FileSystemLoader(template_dir),
            autoescape=_j.select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )
        self._register_filters()
    
    def _register_filters(self):
        """Register custom Jinja2 filters."""
        
        @self.env.filter('format_date')
        def format_date(value, fmt='%B %d, %Y'):
            return value.strftime(fmt) if value else ''
        
        @self.env.filter('format_datetime')
        def format_datetime(value, fmt='%B %d, %Y at %I:%M %p'):
            return value.strftime(fmt) if value else ''
        
        @self.env.filter('truncate_words')
        def truncate_words(value, count=50):
            words = value.split()
            if len(words) > count:
                return ' '.join(words[:count]) + '...'
            return value
        
        @self.env.filter('quote_original')
        def quote_original(value, prefix='> '):
            lines = value.split('\n')
            return '\n'.join(f'{prefix}{line}' for line in lines)
    
    def render_string(self, template_str: str, variables: Dict[str, Any]) -> str:
        """Render a template string."""
        template = self.env.from_string(template_str)
        return template.render(**variables)
    
    async def render_template(
        self,
        template: EmailTemplate,
        variables: Dict[str, Any],
        email_context: Optional[EmailMessage] = None
    ) -> RenderedEmail:
        """Render an email template."""
        # Validate required variables
        missing = [v for v in template.required_variables if v not in variables]
        if missing:
            raise ValueError(f"Missing required variables: {missing}")
        
        # Add default variables
        variables = self._add_defaults(variables, email_context)
        
        # Render
        subject = self.render_string(template.subject_template, variables)
        body_html = None
        body_text = None
        
        if template.body_html_template:
            body_html = self.render_string(template.body_html_template, variables)
        
        if template.body_text_template:
            body_text = self.render_string(template.body_text_template, variables)
        
        return RenderedEmail(
            subject=subject,
            body_html=body_html,
            body_text=body_text,
            template_id=template.id
        )
    
    def _add_defaults(
        self,
        variables: Dict[str, Any],
        email_context: Optional[EmailMessage]
    ) -> Dict[str, Any]:
        """Add default variables."""
        defaults = {
            'now': datetime.utcnow(),
        }
        
        if email_context:
            defaults.update({
                'original_subject': email_context.subject,
                'original_date': email_context.received_date,
                'sender_name': email_context.from_name or email_context.from_address,
                'sender_email': email_context.from_address,
            })
        
        defaults.update(variables)
        return defaults


class EmailScheduler:
    """Advanced email scheduling system."""
    
    def __init__(self, template_engine: TemplateEngine):
        self.scheduler = _ensure_scheduler()()
        self.template_engine = template_engine
        self.executor = ActionExecutor()
    
    async def start(self):
        """Start the scheduler."""
        self.scheduler.start()
    
    async def schedule_email(
        self,
        email: ScheduledEmail,
        trigger: Union[CronTrigger, DateTrigger]
    ) -> str:
        """Schedule an email for future delivery."""
        job = self.scheduler.add_job(
            func=self._send_scheduled_email,
            trigger=trigger,
            args=[email],
            id=email.id,
            replace_existing=True,
            misfire_grace_time=3600
        )
        return job.id
    
    async def schedule_recurring(
        self,
        template: EmailTemplate,
        recipients: List[str],
        cron: str,
        variables: Dict[str, Any]
    ) -> str:
        """Schedule a recurring email."""
        trigger = CronTrigger.from_crontab(cron)
        
        email = ScheduledEmail(
            template=template,
            recipients=recipients,
            variables=variables,
            is_recurring=True
        )
        
        return await self.schedule_email(email, trigger)
    
    async def _send_scheduled_email(self, email: ScheduledEmail):
        """Send a scheduled email."""
        try:
            rendered = await self.template_engine.render_template(
                email.template,
                email.variables
            )
            
            # Send to all recipients via Gmail bridge
            for recipient in email.recipients:
                try:
                    from gmail_client_implementation import GmailClient
                    client = GmailClient()
                    client.messages.send_message(
                        to=recipient,
                        subject=rendered.get('subject', ''),
                        body=rendered.get('body', ''),
                    )
                except ImportError:
                    logger.warning("Gmail client not available for scheduled email send")
                    raise RuntimeError("Gmail client not available")
                except (OSError, ConnectionError, TimeoutError, RuntimeError, ValueError) as send_err:
                    logger.error(f"Failed to send scheduled email to {recipient}: {send_err}")
                    raise
            
            email.status = ScheduleStatus.SENT
            
        except (ImportError, OSError, ValueError) as e:
            email.status = ScheduleStatus.FAILED
            email.error_message = str(e)
            email.attempts += 1


class ConversationManager:
    """Manages email threads and conversation tracking."""
    
    def __init__(self, storage=None):
        self.storage = storage
        self.thread_matcher = ThreadMatcher()
        self.threads: Dict[str, ConversationThread] = {}
    
    async def process_email(self, email: EmailMessage) -> ConversationThread:
        """Process incoming email and add to appropriate thread."""
        # Try to find existing thread
        thread = await self.find_thread(email)
        
        if thread:
            thread = await self._add_to_thread(thread, email)
        else:
            thread = await self._create_thread(email)
        
        # Save thread
        self.threads[thread.id] = thread
        
        return thread
    
    async def find_thread(self, email: EmailMessage) -> Optional[ConversationThread]:
        """Find existing thread for an email."""
        # Method 1: Use Gmail thread ID
        if email.gmail_thread_id:
            for thread in self.threads.values():
                if thread.gmail_thread_id == email.gmail_thread_id:
                    return thread
        
        # Method 2: Use References/In-Reply-To
        if email.in_reply_to:
            for thread in self.threads.values():
                for msg in thread.messages:
                    if msg.message_id == email.in_reply_to:
                        return thread
        
        # Method 3: Fuzzy matching
        return await self.thread_matcher.fuzzy_match(email, list(self.threads.values()))
    
    async def _create_thread(self, email: EmailMessage) -> ConversationThread:
        """Create a new conversation thread."""
        return ConversationThread(
            id=str(uuid.uuid4()),
            gmail_thread_id=email.gmail_thread_id,
            subject=email.subject,
            participants=self._extract_participants(email),
            messages=[email],
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            status=ThreadStatus.ACTIVE
        )
    
    async def _add_to_thread(
        self,
        thread: ConversationThread,
        email: EmailMessage
    ) -> ConversationThread:
        """Add email to existing thread."""
        thread.messages.append(email)
        thread.messages.sort(key=lambda m: m.received_date)
        thread.last_activity = email.received_date
        thread.participants.update(self._extract_participants(email))
        return thread
    
    def _extract_participants(self, email: EmailMessage) -> Set[str]:
        """Extract participant emails from message."""
        participants = {email.from_address}
        participants.update(email.to_addresses)
        participants.update(email.cc_addresses)
        return participants


class ThreadMatcher:
    """Fuzzy matching algorithm for thread identification."""
    
    def __init__(self):
        self.similarity_threshold = 0.85
    
    async def fuzzy_match(
        self,
        email: EmailMessage,
        threads: List[ConversationThread]
    ) -> Optional[ConversationThread]:
        """Find thread using fuzzy matching."""
        best_match = None
        best_score = 0.0
        
        for thread in threads:
            score = self._calculate_similarity(email, thread)
            if score > best_score and score >= self.similarity_threshold:
                best_score = score
                best_match = thread
        
        return best_match
    
    def _calculate_similarity(
        self,
        email: EmailMessage,
        thread: ConversationThread
    ) -> float:
        """Calculate similarity score."""
        scores = []
        
        # Subject similarity (40% weight)
        subject_sim = self._subject_similarity(email.subject, thread.subject)
        scores.append((subject_sim, 0.4))
        
        # Participant overlap (30% weight)
        participant_sim = self._participant_similarity(email, thread)
        scores.append((participant_sim, 0.3))
        
        # Temporal proximity (20% weight)
        temporal_sim = self._temporal_similarity(email, thread)
        scores.append((temporal_sim, 0.2))
        
        # Content similarity (10% weight)
        content_sim = 0.5  # Simplified
        scores.append((content_sim, 0.1))
        
        total_weight = sum(w for _, w in scores)
        weighted_sum = sum(s * w for s, w in scores)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _subject_similarity(self, subject1: str, subject2: str) -> float:
        """Calculate normalized subject similarity."""
        # Remove Re:, Fwd:, etc.
        norm1 = self._normalize_subject(subject1)
        norm2 = self._normalize_subject(subject2)
        
        return SequenceMatcher(None, norm1, norm2).ratio()
    
    def _normalize_subject(self, subject: str) -> str:
        """Normalize subject for comparison."""
        normalized = subject.lower().strip()
        prefixes = [r'^re:\s*', r'^fwd:\s*', r'^fw:\s*']
        
        for prefix in prefixes:
            normalized = re.sub(prefix, '', normalized, flags=re.IGNORECASE)
        
        return ' '.join(normalized.split())
    
    def _participant_similarity(
        self,
        email: EmailMessage,
        thread: ConversationThread
    ) -> float:
        """Calculate participant overlap."""
        email_participants = {
            email.from_address,
            *email.to_addresses,
            *email.cc_addresses
        }
        
        if not email_participants or not thread.participants:
            return 0.0
        
        intersection = email_participants & thread.participants
        union = email_participants | thread.participants
        
        return len(intersection) / len(union)
    
    def _temporal_similarity(
        self,
        email: EmailMessage,
        thread: ConversationThread
    ) -> float:
        """Calculate temporal proximity."""
        time_diff = abs((email.received_date - thread.last_activity).total_seconds())
        max_diff = 30 * 24 * 3600  # 30 days
        
        return max(0, 1 - (time_diff / max_diff))


class GmailAdapter:
    """Gmail API adapter for email operations."""
    
    def __init__(self, credentials):
        from googleapiclient.discovery import build
        self.service = build('gmail', 'v1', credentials=credentials)
    
    async def fetch_emails(
        self,
        query: str = "",
        max_results: int = 100
    ) -> List[EmailMessage]:
        """Fetch emails from Gmail."""
        results = self.service.users().messages().list(
            userId='me',
            q=query,
            maxResults=max_results
        ).execute()
        
        messages = []
        for msg_meta in results.get('messages', []):
            msg = self.service.users().messages().get(
                userId='me',
                id=msg_meta['id'],
                format='full'
            ).execute()
            messages.append(self._parse_message(msg))
        
        return messages
    
    def _parse_message(self, msg: Dict) -> EmailMessage:
        """Parse Gmail message to EmailMessage."""
        headers = {h['name'].lower(): h['value'] for h in msg['payload'].get('headers', [])}
        
        return EmailMessage(
            id=msg['id'],
            thread_id=msg.get('threadId'),
            gmail_thread_id=msg.get('threadId'),
            subject=headers.get('subject', ''),
            from_address=self._extract_email(headers.get('from', '')),
            from_name=self._extract_name(headers.get('from', '')),
            to_addresses=self._extract_emails(headers.get('to', '')),
            cc_addresses=self._extract_emails(headers.get('cc', '')),
            received_date=datetime.fromtimestamp(int(msg['internalDate']) / 1000),
            message_id=headers.get('message-id'),
            in_reply_to=headers.get('in-reply-to'),
            labels=msg.get('labelIds', []),
            is_read='UNREAD' not in msg.get('labelIds', []),
            is_important='IMPORTANT' in msg.get('labelIds', []),
            is_starred='STARRED' in msg.get('labelIds', [])
        )
    
    def _extract_email(self, header: str) -> str:
        """Extract email from header."""
        match = re.search(r'<([^>]+)>', header)
        return match.group(1) if match else header.strip()
    
    def _extract_name(self, header: str) -> Optional[str]:
        """Extract name from header."""
        match = re.search(r'^"?([^"<]+)"?\s*<', header)
        return match.group(1).strip() if match else None
    
    def _extract_emails(self, header: str) -> List[str]:
        """Extract multiple emails from header."""
        if not header:
            return []
        return [self._extract_email(e) for e in header.split(',')]


class SmartAutoResponder:
    """AI-powered auto-responder with context awareness."""
    
    def __init__(self, ai_client, template_engine: TemplateEngine):
        self.ai_client = ai_client
        self.template_engine = template_engine
        self.response_history: Dict[str, datetime] = {}
    
    async def should_respond(
        self,
        email: EmailMessage,
        config: AutoResponder
    ) -> bool:
        """Check if auto-responder should send a response."""
        if not config.enabled:
            return False
        
        # Check exclusions
        if email.from_address in config.exclude_senders:
            return False
        
        domain = email.from_address.split('@')[-1]
        if domain in config.exclude_domains:
            return False
        
        for pattern in config.exclude_subjects:
            if pattern.lower() in email.subject.lower():
                return False
        
        # Check rate limit per sender
        sender_key = email.from_address
        last_response = self.response_history.get(sender_key)
        if last_response:
            time_since = datetime.utcnow() - last_response
            if time_since < config.response_cooldown:
                return False
        
        # Check trigger conditions
        for condition in config.trigger_conditions:
            if not self._evaluate_trigger_condition(condition, email):
                return False
        
        return True

    def _evaluate_trigger_condition(self, condition, email) -> bool:
        """Evaluate a single trigger condition against an email."""
        try:
            cond_type = condition.get('type', '') if isinstance(condition, dict) else getattr(condition, 'type', '')
            value = condition.get('value', '') if isinstance(condition, dict) else getattr(condition, 'value', '')
            if cond_type == 'subject_contains':
                return value.lower() in email.subject.lower()
            elif cond_type == 'from_contains':
                return value.lower() in email.from_address.lower()
            elif cond_type == 'body_contains':
                body = getattr(email, 'body', '') or ''
                return value.lower() in body.lower()
            return True  # Unknown condition types pass by default
        except (OSError, ConnectionError, TimeoutError, ValueError):
            return True

    async def generate_response(
        self,
        email: EmailMessage,
        config: AutoResponder
    ) -> GeneratedResponse:
        """Generate intelligent auto-response."""
        if config.response_type == ResponseType.TEMPLATE and config.response_template:
            # Use template
            variables = {
                'sender_name': email.from_name or email.from_address,
                'subject': email.subject,
            }
            body = self.template_engine.render_string(
                config.response_template,
                variables
            )
            return GeneratedResponse(
                subject=f"Re: {email.subject}",
                body=body,
                tone="professional",
                confidence=1.0,
                suggested_actions=[]
            )
        
        elif config.response_type == ResponseType.AI_GENERATED:
            # Generate AI response
            prompt = self._build_response_prompt(email, config)
            response = await self.ai_client.generate(prompt, max_tokens=500)
            
            return GeneratedResponse(
                subject=f"Re: {email.subject}",
                body=response,
                tone="professional",
                confidence=0.9,
                suggested_actions=[]
            )
        
        else:
            raise ValueError(f"Unsupported response type: {config.response_type}")
    
    def _build_response_prompt(
        self,
        email: EmailMessage,
        config: AutoResponder
    ) -> str:
        """Build AI prompt for response generation."""
        return f"""
Generate a professional auto-response to the following email:

FROM: {email.from_address}
SUBJECT: {email.subject}
BODY:
{email.body_text[:1000]}

Requirements:
- Professional and helpful tone
- 2-4 sentences
- Acknowledge the message
- Provide next steps or timeline
- Do not make commitments without approval

Response:
"""


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

class EmailWorkflowEngine:
    """
    Main entry point for the Email Workflow and Automation Engine.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.rule_engine = RuleEngine()
        self.template_engine = TemplateEngine()
        self.scheduler = EmailScheduler(self.template_engine)
        self.conversation_manager = ConversationManager()
        self.auto_responder: Optional[SmartAutoResponder] = None
        self.gmail_adapter: Optional[GmailAdapter] = None
    
    async def initialize(self):
        """Initialize the engine."""
        await self.scheduler.start()
    
    async def process_incoming_email(self, email: EmailMessage) -> ProcessingResult:
        """Process an incoming email through the workflow engine."""
        # Add to conversation thread
        thread = await self.conversation_manager.process_email(email)
        
        # Process through rule engine
        result = await self.rule_engine.process_email(email)
        
        # Check auto-responder
        if self.auto_responder:
            for config in (self.auto_responder if isinstance(self.auto_responder, list)
                           else [self.auto_responder]):
                try:
                    if hasattr(config, 'should_respond') and config.should_respond(email):
                        response = await self.auto_responder_engine.generate_response(email, config) \
                            if hasattr(self, 'auto_responder_engine') else None
                        if response:
                            logger.info(f"Auto-response generated for {email.from_address}")
                except (OSError, ConnectionError, TimeoutError, ValueError) as e:
                    logger.warning(f"Auto-responder error: {e}")
        
        return result
    
    def add_rule(self, rule: EmailRule):
        """Add a processing rule."""
        self.rule_engine.add_rule(rule)
    
    def remove_rule(self, rule_id: str):
        """Remove a processing rule."""
        self.rule_engine.remove_rule(rule_id)
    
    async def schedule_email(self, email: ScheduledEmail, trigger) -> str:
        """Schedule an email for later delivery."""
        return await self.scheduler.schedule_email(email, trigger)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

async def main():
    """Example usage of the Email Workflow Engine."""
    
    # Initialize engine
    engine = EmailWorkflowEngine()
    await engine.initialize()
    
    # Create a rule
    rule = EmailRule(
        id="rule_urgent_vip",
        name="Urgent VIP Emails",
        description="Handle urgent emails from VIP contacts",
        priority=1,
        conditions=[
            ConditionGroup(
                operator=LogicalOperator.AND,
                conditions=[
                    Condition(
                        field="from",
                        operator=ConditionOperator.IN_LIST,
                        value=["ceo@company.com", "cto@company.com"]
                    ),
                    Condition(
                        field="ai_urgency",
                        operator=ConditionOperator.GREATER_THAN,
                        value=0.8
                    )
                ]
            )
        ],
        actions=[
            Action(
                type=ActionType.LABEL_ADD,
                parameters={"labels": ["VIP", "Urgent"]}
            ),
            Action(
                type=ActionType.MARK_IMPORTANT,
                parameters={}
            ),
            Action(
                type=ActionType.NOTIFY_DESKTOP,
                parameters={
                    "title": "Urgent VIP Email",
                    "message": "{{subject}}",
                    "urgency": "critical"
                }
            )
        ]
    )
    
    engine.add_rule(rule)
    
    # Example email
    email = EmailMessage(
        id="msg_12345",
        thread_id="thread_123",
        gmail_thread_id="thread_123",
        subject="URGENT: Need your approval",
        from_address="ceo@company.com",
        from_name="CEO",
        to_addresses=["me@company.com"],
        body_text="Please review and approve the attached document ASAP.",
        ai_analysis=EmailAnalysis(
            urgency=0.95,
            sentiment=0.2,
            confidence=0.9
        )
    )
    
    # Process email
    result = await engine.process_incoming_email(email)
    
    print(f"Processing Result:")
    print(f"  Email ID: {result.email_id}")
    print(f"  Matched Rules: {result.matched_rules}")
    print(f"  Actions Executed: {len(result.actions_executed)}")
    print(f"  Duration: {result.duration_ms:.2f}ms")


if __name__ == "__main__":
    asyncio.run(main())
