"""
Email Workflow Examples and Use Cases
OpenClaw Windows 10 AI Agent System

This module provides pre-built workflow examples for common email automation scenarios.
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any

from email_workflow_engine import (
    EmailRule, Condition, ConditionGroup, ConditionOperator, LogicalOperator,
    Action, ActionType, EmailMessage, EmailTemplate, AutoResponder, ResponseType,
    ScheduledEmail, CronTrigger, DateTrigger
)


# =============================================================================
# EXAMPLE 1: VIP EMAIL ESCALATION WORKFLOW
# =============================================================================

def create_vip_escalation_rule() -> EmailRule:
    """
    Creates a rule for escalating urgent emails from VIP contacts.
    
    This rule:
    - Matches emails from VIP senders with high urgency
    - Adds VIP and Urgent labels
    - Sends desktop notification
    - Sends SMS notification
    - Marks as important
    """
    return EmailRule(
        id="rule_vip_escalation",
        name="VIP Email Escalation",
        description="Escalate urgent emails from VIP contacts",
        priority=1,  # Highest priority
        enabled=True,
        stop_processing=True,  # Stop after this rule matches
        conditions=[
            ConditionGroup(
                operator=LogicalOperator.AND,
                conditions=[
                    Condition(
                        field="from",
                        operator=ConditionOperator.IN_LIST,
                        value=[
                            "ceo@company.com",
                            "cto@company.com",
                            "cfo@company.com",
                            "president@company.com"
                        ]
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
                    "title": "🚨 Urgent VIP Email",
                    "message": "From: {{sender_name}}\nSubject: {{subject}}",
                    "urgency": "critical",
                    "icon": "icons/vip.png"
                }
            ),
            Action(
                type=ActionType.NOTIFY_SMS,
                parameters={
                    "message": "URGENT: Email from {{sender_name}} - {{subject}}. Check immediately!"
                }
            ),
            Action(
                type=ActionType.NOTIFY_TTS,
                parameters={
                    "message": "Urgent email received from {{sender_name}}"
                }
            )
        ],
        tags=["vip", "urgent", "escalation"]
    )


# =============================================================================
# EXAMPLE 2: NEWSLETTER AUTO-ARCHIVE WORKFLOW
# =============================================================================

def create_newsletter_filter_rule() -> EmailRule:
    """
    Creates a rule for automatically filtering and archiving newsletters.
    
    This rule:
    - Identifies newsletters by various indicators
    - Adds Newsletter label
    - Archives the email (removes from inbox)
    - Optionally marks as read
    """
    return EmailRule(
        id="rule_newsletter_filter",
        name="Newsletter Auto-Archive",
        description="Automatically filter and archive newsletter emails",
        priority=100,  # Lower priority
        enabled=True,
        conditions=[
            ConditionGroup(
                operator=LogicalOperator.OR,
                conditions=[
                    Condition(
                        field="subject",
                        operator=ConditionOperator.CONTAINS,
                        value="unsubscribe",
                        case_sensitive=False
                    ),
                    Condition(
                        field="precedence",
                        operator=ConditionOperator.EQUALS,
                        value="bulk"
                    ),
                    Condition(
                        field="mailing_list",
                        operator=ConditionOperator.EXISTS
                    ),
                    Condition(
                        field="from",
                        operator=ConditionOperator.IN_LIST,
                        value=[
                            "newsletter@",
                            "noreply@",
                            "marketing@",
                            "updates@"
                        ]
                    ),
                    Condition(
                        field="ai_category",
                        operator=ConditionOperator.EQUALS,
                        value="newsletter"
                    )
                ]
            )
        ],
        actions=[
            Action(
                type=ActionType.LABEL_ADD,
                parameters={"labels": ["Newsletters"]}
            ),
            Action(
                type=ActionType.MARK_READ,
                parameters={"read": True}
            ),
            Action(
                type=ActionType.ARCHIVE,
                parameters={}
            ),
            Action(
                type=ActionType.AI_SUMMARIZE,
                parameters={
                    "save_to_notes": True,
                    "extract_links": True
                }
            )
        ],
        tags=["newsletter", "filter", "automation"]
    )


# =============================================================================
# EXAMPLE 3: AFTER-HOURS URGENT NOTIFICATION
# =============================================================================

def create_after_hours_rule() -> EmailRule:
    """
    Creates a rule for notifying about urgent emails received after business hours.
    
    This rule:
    - Matches emails received outside business hours (9 AM - 5 PM)
    - Checks for high urgency using AI analysis
    - Sends SMS notification
    - Optionally makes voice call for critical emails
    """
    return EmailRule(
        id="rule_after_hours",
        name="After-Hours Urgent Notification",
        description="Notify for urgent emails received outside business hours",
        priority=10,
        enabled=True,
        conditions=[
            ConditionGroup(
                operator=LogicalOperator.AND,
                conditions=[
                    Condition(
                        field="received_time",
                        operator=ConditionOperator.NOT_BETWEEN,
                        value=["09:00", "17:00"]
                    ),
                    Condition(
                        field="received_day",
                        operator=ConditionOperator.IN_LIST,
                        value=["Mon", "Tue", "Wed", "Thu", "Fri"]
                    ),
                    Condition(
                        field="ai_urgency",
                        operator=ConditionOperator.GREATER_THAN,
                        value=0.7
                    ),
                    Condition(
                        field="from",
                        operator=ConditionOperator.NOT_IN_LIST,
                        value=[
                            "newsletter@",
                            "noreply@",
                            "marketing@"
                        ]
                    )
                ]
            )
        ],
        actions=[
            Action(
                type=ActionType.LABEL_ADD,
                parameters={"labels": ["After-Hours"]}
            ),
            Action(
                type=ActionType.NOTIFY_SMS,
                parameters={
                    "message": "📧 After-hours urgent email from {{sender_name}}: {{subject}}"
                }
            ),
            Action(
                type=ActionType.NOTIFY_DESKTOP,
                parameters={
                    "title": "After-Hours Email",
                    "message": "Urgent email received after hours",
                    "urgency": "high"
                }
            )
        ],
        tags=["after-hours", "urgent", "notification"]
    )


# =============================================================================
# EXAMPLE 4: MEETING REQUEST AUTO-PROCESSING
# =============================================================================

def create_meeting_request_rule() -> EmailRule:
    """
    Creates a rule for automatically processing meeting requests.
    
    This rule:
    - Identifies meeting request emails
    - Extracts meeting details using AI
    - Creates calendar event
    - Sends acknowledgment response
    - Adds to tasks if action required
    """
    return EmailRule(
        id="rule_meeting_request",
        name="Meeting Request Auto-Processing",
        description="Automatically process meeting request emails",
        priority=20,
        enabled=True,
        conditions=[
            ConditionGroup(
                operator=LogicalOperator.AND,
                conditions=[
                    Condition(
                        field="ai_intent",
                        operator=ConditionOperator.IN_LIST,
                        value=["request_meeting", "schedule_call", "setup_appointment"]
                    ),
                    Condition(
                        field="subject",
                        operator=ConditionOperator.NOT_CONTAINS,
                        value="cancelled",
                        case_sensitive=False
                    ),
                    Condition(
                        field="subject",
                        operator=ConditionOperator.NOT_CONTAINS,
                        value="rescheduled",
                        case_sensitive=False
                    )
                ]
            )
        ],
        actions=[
            Action(
                type=ActionType.LABEL_ADD,
                parameters={"labels": ["Meetings", "Action-Required"]}
            ),
            Action(
                type=ActionType.AI_EXTRACT_ENTITIES,
                parameters={
                    "entity_types": ["date", "time", "location", "attendees"],
                    "save_to_variables": True
                }
            ),
            Action(
                type=ActionType.CREATE_CALENDAR_EVENT,
                parameters={
                    "auto_accept": False,
                    "send_response": True
                }
            ),
            Action(
                type=ActionType.AUTO_RESPOND,
                parameters={
                    "template": "meeting_acknowledgment",
                    "delay": "2m"
                }
            ),
            Action(
                type=ActionType.CREATE_TASK,
                parameters={
                    "title": "Review meeting request from {{sender_name}}",
                    "due": "+1d",
                    "priority": "medium"
                }
            )
        ],
        tags=["meeting", "calendar", "automation"]
    )


# =============================================================================
# EXAMPLE 5: INVOICE PROCESSING WORKFLOW
# =============================================================================

def create_invoice_processing_rule() -> EmailRule:
    """
    Creates a rule for processing invoice emails.
    
    This rule:
    - Identifies invoice emails using AI classification
    - Extracts invoice details
    - Saves attachments to designated folder
    - Creates accounting task
    - Forwards to accounting department
    """
    return EmailRule(
        id="rule_invoice_processing",
        name="Invoice Processing",
        description="Automatically process invoice emails",
        priority=15,
        enabled=True,
        conditions=[
            ConditionGroup(
                operator=LogicalOperator.AND,
                conditions=[
                    Condition(
                        field="ai_category",
                        operator=ConditionOperator.EQUALS,
                        value="invoice"
                    ),
                    Condition(
                        field="has_attachments",
                        operator=ConditionOperator.EQUALS,
                        value=True
                    ),
                    Condition(
                        field="attachment_types",
                        operator=ConditionOperator.CONTAINS,
                        value="pdf"
                    )
                ]
            )
        ],
        actions=[
            Action(
                type=ActionType.LABEL_ADD,
                parameters={"labels": ["Invoices", "Finance"]}
            ),
            Action(
                type=ActionType.SAVE_ATTACHMENT,
                parameters={
                    "path": "C:/Documents/Invoices/{{year}}/{{month}}",
                    "rename_pattern": "{{sender_domain}}_{{date}}_{{filename}}"
                }
            ),
            Action(
                type=ActionType.AI_EXTRACT_ENTITIES,
                parameters={
                    "entity_types": ["amount", "due_date", "invoice_number", "vendor"],
                    "save_to_spreadsheet": True
                }
            ),
            Action(
                type=ActionType.CREATE_TASK,
                parameters={
                    "title": "Process invoice from {{sender_name}}",
                    "assignee": "accounting@company.com",
                    "due": "+3d",
                    "priority": "high"
                }
            ),
            Action(
                type=ActionType.FORWARD,
                parameters={
                    "to": ["accounting@company.com"],
                    "include_attachments": True
                }
            )
        ],
        tags=["invoice", "finance", "accounting"]
    )


# =============================================================================
# EXAMPLE 6: PHISHING DETECTION AND QUARANTINE
# =============================================================================

def create_phishing_protection_rule() -> EmailRule:
    """
    Creates a rule for detecting and handling potential phishing emails.
    
    This rule:
    - Uses AI to detect phishing risk
    - Checks against known phishing patterns
    - Quarantines suspicious emails
    - Notifies security team
    - Logs for analysis
    """
    return EmailRule(
        id="rule_phishing_protection",
        name="Phishing Detection and Quarantine",
        description="Detect and quarantine potential phishing emails",
        priority=1,  # Very high priority
        enabled=True,
        stop_processing=True,
        conditions=[
            ConditionGroup(
                operator=LogicalOperator.OR,
                conditions=[
                    Condition(
                        field="ai_phishing_risk",
                        operator=ConditionOperator.GREATER_THAN,
                        value=0.8
                    ),
                    Condition(
                        field="ai_spam_probability",
                        operator=ConditionOperator.GREATER_THAN,
                        value=0.95
                    ),
                    Condition(
                        field="attachment_types",
                        operator=ConditionOperator.CONTAINS,
                        value="executable"
                    ),
                    Condition(
                        field="link_domains",
                        operator=ConditionOperator.IN_LIST,
                        value=["known-phishing-domain.com", "suspicious-site.net"]
                    )
                ]
            )
        ],
        actions=[
            Action(
                type=ActionType.LABEL_ADD,
                parameters={"labels": ["Quarantine", "Security-Review"]}
            ),
            Action(
                type=ActionType.MARK_SPAM,
                parameters={"spam": True}
            ),
            Action(
                type=ActionType.NOTIFY_EMAIL,
                parameters={
                    "to": "security@company.com",
                    "template": "phishing_alert",
                    "include_headers": True
                }
            ),
            Action(
                type=ActionType.NOTIFY_WEBHOOK,
                parameters={
                    "url": "https://security.company.com/webhook/phishing",
                    "payload": {
                        "alert_type": "phishing_detected",
                        "email_id": "{{email_id}}",
                        "risk_score": "{{ai_phishing_risk}}"
                    }
                }
            ),
            Action(
                type=ActionType.RUN_SCRIPT,
                parameters={
                    "script": "log_security_event.py",
                    "arguments": ["--type=phishing", "--email={{email_id}}"]
                }
            )
        ],
        tags=["security", "phishing", "quarantine"]
    )


# =============================================================================
# EXAMPLE 7: AUTO-RESPONDER CONFIGURATIONS
# =============================================================================

def create_vacation_responder() -> AutoResponder:
    """
    Creates a vacation/out-of-office auto-responder.
    """
    return AutoResponder(
        id="responder_vacation",
        name="Vacation Auto-Responder",
        enabled=False,  # Disabled by default
        is_vacation_responder=True,
        vacation_start=datetime(2025, 2, 1, 9, 0),
        vacation_end=datetime(2025, 2, 15, 17, 0),
        vacation_message="""
        Thank you for your email.
        
        I am currently out of the office until February 15, 2025.
        
        For urgent matters, please contact:
        John Smith (Backup) - john.smith@company.com
        
        I'll respond to your email upon my return.
        
        Best regards,
        [Your Name]
        """,
        exclude_senders=[
            "newsletter@",
            "noreply@",
            "mailer-daemon@"
        ],
        exclude_domains=[
            "mailgun.net",
            "sendgrid.net"
        ],
        max_responses_per_sender=1,
        response_cooldown=timedelta(days=7),
        include_original_message=False,
        reply_to_thread=True
    )


def create_acknowledgment_responder() -> AutoResponder:
    """
    Creates a smart acknowledgment auto-responder.
    """
    return AutoResponder(
        id="responder_acknowledgment",
        name="Smart Acknowledgment",
        enabled=True,
        response_type=ResponseType.AI_GENERATED,
        trigger_conditions=[
            Condition(
                field="from",
                operator=ConditionOperator.NOT_IN_LIST,
                value=["newsletter@", "noreply@", "marketing@"]
            ),
            Condition(
                field="ai_category",
                operator=ConditionOperator.NOT_IN_LIST,
                value=["newsletter", "marketing", "social"]
            ),
            Condition(
                field="is_group_email",
                operator=ConditionOperator.EQUALS,
                value=False
            )
        ],
        ai_prompt="""
        Generate a brief, professional acknowledgment email.
        
        Guidelines:
        - Thank the sender for their email
        - Indicate when they can expect a response
        - Keep it to 2-3 sentences
        - Do not make specific commitments
        - Be warm but professional
        
        Original email context will be provided.
        """,
        delay=timedelta(minutes=5),
        max_responses_per_sender=1,
        response_cooldown=timedelta(hours=24),
        exclude_senders=[
            "newsletter@",
            "noreply@",
            "do-not-reply@",
            "mailer-daemon@"
        ],
        reply_to_thread=True
    )


# =============================================================================
# EXAMPLE 8: SCHEDULED EMAIL WORKFLOWS
# =============================================================================

def create_daily_digest_schedule(template: EmailTemplate) -> ScheduledEmail:
    """
    Creates a scheduled daily email digest.
    """
    return ScheduledEmail(
        template=template,
        recipients=["user@company.com"],
        variables={
            "digest_type": "daily",
            "include_unread": True,
            "include_important": True,
            "include_action_items": True,
            "summary_style": "brief"
        },
        cron_expression="0 8 * * 1-5",  # 8 AM, weekdays
        is_recurring=True
    )


def create_weekly_report_schedule(template: EmailTemplate) -> ScheduledEmail:
    """
    Creates a scheduled weekly activity report.
    """
    return ScheduledEmail(
        template=template,
        recipients=["user@company.com", "manager@company.com"],
        variables={
            "report_type": "weekly",
            "include_metrics": True,
            "include_trends": True,
            "include_charts": True,
            "period": "last_7_days"
        },
        cron_expression="0 9 * * 1",  # Monday 9 AM
        is_recurring=True
    )


def create_follow_up_reminder_schedule(
    template: EmailTemplate,
    recipient: str,
    original_email_id: str
) -> ScheduledEmail:
    """
    Creates a scheduled follow-up reminder.
    """
    return ScheduledEmail(
        template=template,
        recipients=[recipient],
        variables={
            "reminder_type": "follow_up",
            "original_email_id": original_email_id,
            "context": "No response received after 3 days"
        },
        scheduled_time=datetime.utcnow() + timedelta(days=3),
        is_recurring=False
    )


# =============================================================================
# EXAMPLE 9: TEMPLATE DEFINITIONS
# =============================================================================

def create_meeting_confirmation_template() -> EmailTemplate:
    """
    Creates a meeting confirmation email template.
    """
    return EmailTemplate(
        id="template_meeting_confirmation",
        name="Meeting Confirmation",
        description="Template for confirming meeting requests",
        category="meetings",
        subject_template="Re: {{original_subject}} - Confirmed",
        body_text_template="""
Hi {{sender_name}},

I'd be happy to meet with you.

✅ Confirmed Details:
📅 Date: {{meeting_date}}
🕐 Time: {{meeting_time}}
⏱️ Duration: {{meeting_duration}}
{% if meeting_location %}
📍 Location: {{meeting_location}}
{% endif %}
{% if meeting_link %}
🔗 Video Link: {{meeting_link}}
{% endif %}

{% if agenda %}
📝 Agenda:
{{agenda}}
{% endif %}

Looking forward to our discussion!

Best regards,
{{my_name}}
{{my_title}}
{{my_phone}}
        """,
        body_html_template="""
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .header { background: #f4f4f4; padding: 20px; }
        .content { padding: 20px; }
        .detail { margin: 10px 0; }
        .label { font-weight: bold; color: #666; }
    </style>
</head>
<body>
    <div class="header">
        <h2>Meeting Confirmed</h2>
    </div>
    <div class="content">
        <p>Hi {{sender_name}},</p>
        <p>I'd be happy to meet with you.</p>
        
        <h3>Confirmed Details:</h3>
        <div class="detail"><span class="label">📅 Date:</span> {{meeting_date}}</div>
        <div class="detail"><span class="label">🕐 Time:</span> {{meeting_time}}</div>
        <div class="detail"><span class="label">⏱️ Duration:</span> {{meeting_duration}}</div>
        {% if meeting_location %}
        <div class="detail"><span class="label">📍 Location:</span> {{meeting_location}}</div>
        {% endif %}
        {% if meeting_link %}
        <div class="detail"><span class="label">🔗 Video Link:</span> <a href="{{meeting_link}}">Join Meeting</a></div>
        {% endif %}
        
        <p>Looking forward to our discussion!</p>
        
        <p>Best regards,<br>{{my_name}}</p>
    </div>
</body>
</html>
        """,
        required_variables=["sender_name", "meeting_date", "meeting_time", "meeting_duration"],
        optional_variables={
            "meeting_location": None,
            "meeting_link": None,
            "agenda": None,
            "my_name": "Your Name",
            "my_title": "",
            "my_phone": ""
        },
        tags=["meeting", "confirmation"]
    )


def create_follow_up_template() -> EmailTemplate:
    """
    Creates a follow-up reminder email template.
    """
    return EmailTemplate(
        id="template_follow_up",
        name="Follow-up Reminder",
        description="Template for sending follow-up reminders",
        category="follow_up",
        subject_template="Follow-up: {{original_subject}}",
        body_text_template="""
Hi {{sender_name}},

I hope this message finds you well. I wanted to follow up on my previous email:

📧 Subject: {{original_subject}}
📅 Sent: {{original_date | format_date}}

{% if follow_up_context %}
{{follow_up_context}}
{% endif %}

{% if action_needed %}
⚠️ Action Needed: {{action_needed}}
{% endif %}

Please let me know if you need any additional information or have any questions.

Best regards,
{{my_name}}

---
📎 Original Message:
{{original_body | quote_original}}
        """,
        required_variables=["sender_name", "original_subject", "original_date"],
        optional_variables={
            "follow_up_context": "",
            "action_needed": "",
            "original_body": "",
            "my_name": "Your Name"
        },
        tags=["follow_up", "reminder"]
    )


# =============================================================================
# EXAMPLE 10: COMPLEX MULTI-STEP WORKFLOW
# =============================================================================

def create_customer_onboarding_workflow() -> Dict[str, Any]:
    """
    Creates a complex customer onboarding workflow.
    
    This workflow:
    1. Receives new customer signup email
    2. Extracts customer information
    3. Creates CRM entry
    4. Sends welcome email
    5. Creates onboarding tasks
    6. Schedules follow-up emails
    7. Notifies sales team
    """
    return {
        "id": "workflow_customer_onboarding",
        "name": "Customer Onboarding",
        "description": "Automated customer onboarding workflow",
        "trigger": {
            "type": "email_received",
            "filters": {
                "subject_contains": ["new customer", "signup", "registration"],
                "from_domain": "company.com"
            }
        },
        "steps": [
            {
                "id": "extract_info",
                "type": "ai_extract_entities",
                "config": {
                    "entities": ["company_name", "contact_name", "email", "phone", "plan"],
                    "save_to_crm": True
                }
            },
            {
                "id": "create_crm_entry",
                "type": "call_api",
                "config": {
                    "endpoint": "https://crm.company.com/api/customers",
                    "method": "POST",
                    "payload": {
                        "name": "{{extracted.company_name}}",
                        "contact": "{{extracted.contact_name}}",
                        "email": "{{extracted.email}}",
                        "phone": "{{extracted.phone}}",
                        "plan": "{{extracted.plan}}",
                        "source": "email_signup"
                    }
                }
            },
            {
                "id": "send_welcome",
                "type": "send_email",
                "config": {
                    "template": "welcome_new_customer",
                    "to": "{{extracted.email}}",
                    "delay": "5m"
                }
            },
            {
                "id": "create_tasks",
                "type": "parallel_execute",
                "config": {
                    "actions": [
                        {
                            "type": "create_task",
                            "title": "Schedule onboarding call with {{extracted.contact_name}}",
                            "assignee": "sales@company.com",
                            "due": "+2d"
                        },
                        {
                            "type": "create_task",
                            "title": "Send product documentation to {{extracted.contact_name}}",
                            "assignee": "support@company.com",
                            "due": "+1d"
                        },
                        {
                            "type": "create_task",
                            "title": "Set up customer account for {{extracted.company_name}}",
                            "assignee": "tech@company.com",
                            "due": "+1d"
                        }
                    ]
                }
            },
            {
                "id": "schedule_followups",
                "type": "schedule_emails",
                "config": {
                    "emails": [
                        {
                            "template": "onboarding_day3",
                            "delay": "3d"
                        },
                        {
                            "template": "onboarding_week1",
                            "delay": "7d"
                        },
                        {
                            "template": "onboarding_month1",
                            "delay": "30d"
                        }
                    ]
                }
            },
            {
                "id": "notify_sales",
                "type": "notify_slack",
                "config": {
                    "channel": "#sales",
                    "message": "🎉 New customer signup: {{extracted.company_name}} ({{extracted.plan}} plan)"
                }
            }
        ]
    }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_all_example_rules() -> List[EmailRule]:
    """Returns all example rules."""
    return [
        create_vip_escalation_rule(),
        create_newsletter_filter_rule(),
        create_after_hours_rule(),
        create_meeting_request_rule(),
        create_invoice_processing_rule(),
        create_phishing_protection_rule(),
    ]


def get_all_example_responders() -> List[AutoResponder]:
    """Returns all example auto-responders."""
    return [
        create_vacation_responder(),
        create_acknowledgment_responder(),
    ]


def get_all_example_templates() -> List[EmailTemplate]:
    """Returns all example templates."""
    return [
        create_meeting_confirmation_template(),
        create_follow_up_template(),
    ]


# =============================================================================
# MAIN
# =============================================================================

async def main():
    """Demonstrate workflow examples."""
    
    print("=" * 60)
    print("Email Workflow Examples")
    print("=" * 60)
    
    # Get all example rules
    rules = get_all_example_rules()
    print(f"\n📋 Example Rules ({len(rules)}):")
    for rule in rules:
        print(f"  - {rule.name} (Priority: {rule.priority})")
    
    # Get all example responders
    responders = get_all_example_responders()
    print(f"\n📧 Example Auto-Responders ({len(responders)}):")
    for responder in responders:
        print(f"  - {responder.name} (Enabled: {responder.enabled})")
    
    # Get all example templates
    templates = get_all_example_templates()
    print(f"\n📝 Example Templates ({len(templates)}):")
    for template in templates:
        print(f"  - {template.name} (Category: {template.category})")
    
    # Create a sample email
    sample_email = EmailMessage(
        id="msg_sample_001",
        thread_id="thread_001",
        gmail_thread_id="thread_001",
        subject="URGENT: Need approval for Q1 budget",
        from_address="ceo@company.com",
        from_name="CEO",
        to_addresses=["me@company.com"],
        body_text="Please review and approve the attached Q1 budget document ASAP.",
        received_date=datetime.utcnow(),
        ai_analysis=EmailMessage.__dataclass_fields__  # Placeholder
    )
    
    print(f"\n📨 Sample Email:")
    print(f"  From: {sample_email.from_name} <{sample_email.from_address}>")
    print(f"  Subject: {sample_email.subject}")
    print(f"  Body: {sample_email.body_text[:50]}...")
    
    print("\n" + "=" * 60)
    print("Workflow examples loaded successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
