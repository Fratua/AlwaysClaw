"""
Example Workflows for E2E Loop System
OpenClaw-Inspired AI Agent System for Windows 10

This module provides example workflows demonstrating:
- Simple sequential workflows
- Parallel processing workflows
- Human-in-the-loop workflows
- Error handling and compensation
- Long-running workflows
"""

import asyncio
import json
from datetime import datetime
from e2e_loop_core import (
    E2EWorkflowEngine,
    WorkflowDefinition,
    TaskDefinition,
    TaskConfig,
    RetryPolicy,
    create_workflow_definition,
    InMemoryStateBackend
)
from e2e_loop_hitl import (
    HITLManager,
    ApprovalConfig,
    ApprovalContext,
    NotificationChannel,
    EscalationPolicy,
    EscalationAction
)
from e2e_loop_monitoring import (
    WorkflowMetricsCollector,
    AuditLogger,
    AuditStorage,
    AuditEventType
)


# ============================================================================
# EXAMPLE 1: Simple Sequential Workflow
# ============================================================================

def create_simple_workflow() -> WorkflowDefinition:
    """
    Create a simple sequential workflow for data processing.
    
    Flow:
    1. Fetch data from API
    2. Transform data
    3. Save to database
    """
    return create_workflow_definition(
        name="Simple Data Pipeline",
        tasks=[
            {
                'id': 'fetch_data',
                'name': 'Fetch Data',
                'type': 'http',
                'config': {
                    'method': 'GET',
                    'url': 'https://jsonplaceholder.typicode.com/posts',
                    'timeout': 30
                },
                'outputs': [
                    {'name': 'posts', 'value': '${response.data}', 'export': True}
                ]
            },
            {
                'id': 'transform_data',
                'name': 'Transform Data',
                'type': 'python',
                'depends_on': ['fetch_data'],
                'config': {
                    'code': '''
# Transform posts data
posts = context['tasks']['fetch_data']['output']['posts']
transformed = [
    {
        'id': post['id'],
        'title': post['title'].upper(),
        'summary': post['body'][:100] + '...'
    }
    for post in posts[:5]  # Process first 5 posts
]
result = {'transformed_posts': transformed}
'''
                },
                'outputs': [
                    {'name': 'transformed_data', 'value': '${result.transformed_posts}', 'export': True}
                ]
            },
            {
                'id': 'save_data',
                'name': 'Save Data',
                'type': 'python',
                'depends_on': ['transform_data'],
                'config': {
                    'code': '''
# Simulate saving to database
data = context['tasks']['transform_data']['output']['transformed_data']
print(f"Saving {len(data)} records to database")
result = {'saved_count': len(data), 'timestamp': str(datetime.now())}
'''
                },
                'outputs': [
                    {'name': 'save_result', 'value': '${result}', 'export': True}
                ]
            }
        ],
        max_concurrent=1
    )


# ============================================================================
# EXAMPLE 2: Parallel Processing Workflow
# ============================================================================

def create_parallel_workflow() -> WorkflowDefinition:
    """
    Create a workflow with parallel task execution.
    
    Flow:
    1. Fetch source data
    2. Process data in parallel (validation, enrichment, analysis)
    3. Aggregate results
    """
    return create_workflow_definition(
        name="Parallel Data Processing",
        tasks=[
            {
                'id': 'fetch_source',
                'name': 'Fetch Source Data',
                'type': 'python',
                'config': {
                    'code': '''
# Simulate fetching data
source_data = [
    {'id': 1, 'name': 'Item 1', 'value': 100},
    {'id': 2, 'name': 'Item 2', 'value': 200},
    {'id': 3, 'name': 'Item 3', 'value': 300},
]
result = {'data': source_data}
'''
                },
                'outputs': [
                    {'name': 'source_data', 'value': '${result.data}', 'export': True}
                ]
            },
            {
                'id': 'validate_data',
                'name': 'Validate Data',
                'type': 'python',
                'depends_on': ['fetch_source'],
                'config': {
                    'code': '''
# Validate data
data = context['tasks']['fetch_source']['output']['source_data']
valid_items = [item for item in data if item['value'] > 0]
validation_result = {
    'valid': len(valid_items),
    'invalid': len(data) - len(valid_items)
}
result = {'validation': validation_result, 'valid_items': valid_items}
'''
                },
                'outputs': [
                    {'name': 'validation_result', 'value': '${result.validation}', 'export': True}
                ]
            },
            {
                'id': 'enrich_data',
                'name': 'Enrich Data',
                'type': 'python',
                'depends_on': ['fetch_source'],
                'config': {
                    'code': '''
# Enrich data with additional info
data = context['tasks']['fetch_source']['output']['source_data']
enriched = [
    {**item, 'category': 'A' if item['value'] > 150 else 'B'}
    for item in data
]
result = {'enriched_data': enriched}
'''
                },
                'outputs': [
                    {'name': 'enriched_data', 'value': '${result.enriched_data}', 'export': True}
                ]
            },
            {
                'id': 'analyze_data',
                'name': 'Analyze Data',
                'type': 'python',
                'depends_on': ['fetch_source'],
                'config': {
                    'code': '''
# Analyze data
data = context['tasks']['fetch_source']['output']['source_data']
values = [item['value'] for item in data]
analysis = {
    'total': sum(values),
    'average': sum(values) / len(values),
    'max': max(values),
    'min': min(values)
}
result = {'analysis': analysis}
'''
                },
                'outputs': [
                    {'name': 'analysis_result', 'value': '${result.analysis}', 'export': True}
                ]
            },
            {
                'id': 'aggregate_results',
                'name': 'Aggregate Results',
                'type': 'python',
                'depends_on': ['validate_data', 'enrich_data', 'analyze_data'],
                'config': {
                    'code': '''
# Aggregate all parallel results
validation = context['tasks']['validate_data']['output']['validation_result']
enriched = context['tasks']['enrich_data']['output']['enriched_data']
analysis = context['tasks']['analyze_data']['output']['analysis_result']

final_result = {
    'validation': validation,
    'enriched_count': len(enriched),
    'analysis': analysis,
    'processed_at': str(datetime.now())
}
result = final_result
'''
                },
                'outputs': [
                    {'name': 'final_result', 'value': '${result}', 'export': True}
                ]
            }
        ],
        max_concurrent=3  # Allow parallel execution
    )


# ============================================================================
# EXAMPLE 3: Human-in-the-Loop Workflow
# ============================================================================

def create_hitl_workflow() -> WorkflowDefinition:
    """
    Create a workflow with human approval checkpoints.
    
    Flow:
    1. Generate report
    2. Request human approval
    3. If approved, send email
    4. If rejected, log rejection
    """
    return create_workflow_definition(
        name="Report Generation with Approval",
        tasks=[
            {
                'id': 'generate_report',
                'name': 'Generate Report',
                'type': 'python',
                'config': {
                    'code': '''
# Generate a sample report
report = {
    'title': 'Monthly Sales Report',
    'period': '2025-01',
    'total_sales': 150000,
    'transactions': 1250,
    'top_products': [
        {'name': 'Product A', 'sales': 45000},
        {'name': 'Product B', 'sales': 38000},
        {'name': 'Product C', 'sales': 32000}
    ]
}
result = {'report': report}
'''
                },
                'outputs': [
                    {'name': 'report', 'value': '${result.report}', 'export': True}
                ]
            },
            {
                'id': 'human_approval',
                'name': 'Manager Approval',
                'type': 'python',
                'depends_on': ['generate_report'],
                'requires_approval': True,
                'approvers': ['manager@example.com'],
                'approval_timeout': 3600,  # 1 hour
                'config': {
                    'code': '''
# This task requires human approval
report = context['tasks']['generate_report']['output']['report']
print(f"Report generated: {report['title']}")
print("Waiting for manager approval...")
result = {'approved': True, 'approved_by': 'manager@example.com'}
'''
                },
                'outputs': [
                    {'name': 'approval_result', 'value': '${result}', 'export': True}
                ]
            },
            {
                'id': 'send_report',
                'name': 'Send Report Email',
                'type': 'python',
                'depends_on': ['human_approval'],
                'config': {
                    'code': '''
# Send approved report
report = context['tasks']['generate_report']['output']['report']
approval = context['tasks']['human_approval']['output']['approval_result']

if approval.get('approved'):
    print(f"Sending report '{report['title']}' to stakeholders")
    result = {'sent': True, 'recipients': ['team@example.com']}
else:
    print("Report was not approved, skipping send")
    result = {'sent': False, 'reason': 'not_approved'}
'''
                },
                'outputs': [
                    {'name': 'send_result', 'value': '${result}', 'export': True}
                ]
            }
        ],
        max_concurrent=1
    )


# ============================================================================
# EXAMPLE 4: Error Handling and Compensation Workflow
# ============================================================================

def create_compensation_workflow() -> WorkflowDefinition:
    """
    Create a workflow demonstrating error handling and compensation.
    
    Flow:
    1. Create order
    2. Reserve inventory
    3. Process payment (may fail)
    4. If payment fails, compensate (release inventory, cancel order)
    """
    return create_workflow_definition(
        name="Order Processing with Compensation",
        tasks=[
            {
                'id': 'create_order',
                'name': 'Create Order',
                'type': 'python',
                'config': {
                    'code': '''
# Create order
order = {
    'id': 'ORD-' + str(datetime.now().timestamp()),
    'items': [
        {'sku': 'ITEM-001', 'qty': 2, 'price': 50},
        {'sku': 'ITEM-002', 'qty': 1, 'price': 100}
    ],
    'total': 200
}
print(f"Order created: {order['id']}")
result = {'order': order}
'''
                },
                'outputs': [
                    {'name': 'order', 'value': '${result.order}', 'export': True}
                ]
            },
            {
                'id': 'reserve_inventory',
                'name': 'Reserve Inventory',
                'type': 'python',
                'depends_on': ['create_order'],
                'config': {
                    'code': '''
# Reserve inventory
order = context['tasks']['create_order']['output']['order']
for item in order['items']:
    print(f"Reserving {item['qty']} units of {item['sku']}")
result = {'reserved': True}
'''
                },
                'outputs': [
                    {'name': 'reservation', 'value': '${result}', 'export': True}
                ]
            },
            {
                'id': 'process_payment',
                'name': 'Process Payment',
                'type': 'python',
                'depends_on': ['reserve_inventory'],
                'config': {
                    'code': '''
# Simulate payment processing (may fail)
import random
order = context['tasks']['create_order']['output']['order']

# Simulate 30% failure rate for demo
if random.random() < 0.3:
    raise Exception("Payment declined: Insufficient funds")

print(f"Payment processed for order {order['id']}: ${order['total']}")
result = {'payment_id': 'PAY-' + str(datetime.now().timestamp()), 'amount': order['total']}
'''
                },
                'retries': 2,
                'retry_strategy': 'fixed',
                'outputs': [
                    {'name': 'payment', 'value': '${result}', 'export': True}
                ]
            },
            {
                'id': 'send_confirmation',
                'name': 'Send Confirmation',
                'type': 'python',
                'depends_on': ['process_payment'],
                'config': {
                    'code': '''
# Send order confirmation
order = context['tasks']['create_order']['output']['order']
payment = context['tasks']['process_payment']['output']['payment']
print(f"Sending confirmation for order {order['id']}")
result = {'sent': True}
'''
                },
                'outputs': [
                    {'name': 'confirmation', 'value': '${result}', 'export': True}
                ]
            },
            # Compensation tasks
            {
                'id': 'release_inventory',
                'name': 'Release Inventory',
                'type': 'python',
                'config': {
                    'code': '''
# Release reserved inventory
order = context['tasks']['create_order']['output']['order']
for item in order['items']:
    print(f"Releasing {item['qty']} units of {item['sku']}")
result = {'released': True}
'''
                },
                'outputs': []
            },
            {
                'id': 'cancel_order',
                'name': 'Cancel Order',
                'type': 'python',
                'config': {
                    'code': '''
# Cancel order
order = context['tasks']['create_order']['output']['order']
print(f"Cancelling order {order['id']}")
result = {'cancelled': True}
'''
                },
                'outputs': []
            }
        ],
        max_concurrent=1
    )


# ============================================================================
# EXAMPLE 5: Conditional Workflow
# ============================================================================

def create_conditional_workflow() -> WorkflowDefinition:
    """
    Create a workflow with conditional execution paths.
    
    Flow:
    1. Check data quality
    2. If quality is good -> process normally
    3. If quality is poor -> clean data first
    4. Generate report
    """
    return create_workflow_definition(
        name="Conditional Data Processing",
        tasks=[
            {
                'id': 'check_quality',
                'name': 'Check Data Quality',
                'type': 'python',
                'config': {
                    'code': '''
# Check data quality
import random
data_quality = random.choice(['good', 'poor', 'excellent'])
print(f"Data quality: {data_quality}")
result = {'quality': data_quality, 'needs_cleaning': data_quality == 'poor'}
'''
                },
                'outputs': [
                    {'name': 'quality_check', 'value': '${result}', 'export': True}
                ]
            },
            {
                'id': 'clean_data',
                'name': 'Clean Data',
                'type': 'python',
                'depends_on': ['check_quality'],
                'when': '${tasks.check_quality.output.quality_check.needs_cleaning}',
                'config': {
                    'code': '''
# Clean data
print("Cleaning data...")
result = {'cleaned': True, 'records_processed': 1000}
'''
                },
                'outputs': [
                    {'name': 'cleaning_result', 'value': '${result}', 'export': True}
                ]
            },
            {
                'id': 'process_data',
                'name': 'Process Data',
                'type': 'python',
                'depends_on': ['check_quality', 'clean_data'],
                'config': {
                    'code': '''
# Process data
quality = context['tasks']['check_quality']['output']['quality_check']
cleaned = context['tasks'].get('clean_data', {}).get('output', {})

if cleaned:
    print(f"Processing cleaned data: {cleaned}")
else:
    print(f"Processing data with quality: {quality['quality']}")

result = {'processed': True, 'output_records': 950}
'''
                },
                'outputs': [
                    {'name': 'processing_result', 'value': '${result}', 'export': True}
                ]
            },
            {
                'id': 'generate_report',
                'name': 'Generate Report',
                'type': 'python',
                'depends_on': ['process_data'],
                'config': {
                    'code': '''
# Generate report
processing = context['tasks']['process_data']['output']['processing_result']
report = {
    'records_processed': processing['output_records'],
    'success_rate': 0.95,
    'generated_at': str(datetime.now())
}
result = {'report': report}
'''
                },
                'outputs': [
                    {'name': 'report', 'value': '${result.report}', 'export': True}
                ]
            }
        ],
        max_concurrent=2
    )


# ============================================================================
# RUN EXAMPLES
# ============================================================================

async def run_example_workflow(example_name: str):
    """Run an example workflow."""
    
    print(f"\n{'='*60}")
    print(f"Running Example: {example_name}")
    print('='*60)
    
    # Create workflow engine
    engine = E2EWorkflowEngine()
    
    # Create metrics collector
    metrics = WorkflowMetricsCollector()
    
    # Create audit logger
    audit_storage = AuditStorage()
    audit_logger = AuditLogger(audit_storage)
    
    # Get workflow definition
    workflows = {
        'simple': create_simple_workflow,
        'parallel': create_parallel_workflow,
        'hitl': create_hitl_workflow,
        'compensation': create_compensation_workflow,
        'conditional': create_conditional_workflow
    }
    
    workflow_def = workflows[example_name]()
    
    # Log workflow start
    await audit_logger.log(
        AuditEventType.WORKFLOW_STARTED,
        workflow_id=workflow_def.id,
        details={'workflow_name': workflow_def.name}
    )
    
    # Submit workflow
    workflow_id = await engine.submit_workflow(
        definition=workflow_def,
        inputs={'example': example_name, 'timestamp': str(datetime.now())}
    )
    
    print(f"Workflow submitted: {workflow_id}")
    
    # Wait for completion
    from e2e_loop_core import WorkflowStatus
    
    max_wait = 60  # seconds
    waited = 0
    
    while waited < max_wait:
        status = await engine.get_workflow_status(workflow_id)
        
        if status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]:
            break
        
        await asyncio.sleep(0.5)
        waited += 0.5
    
    # Get final state
    state = await engine.state_backend.load_workflow_state(workflow_id)
    
    print(f"\nFinal Status: {state.status.value}")
    print(f"Execution Time: {(state.completed_at - state.created_at).total_seconds():.2f}s")
    
    if state.error_message:
        print(f"Error: {state.error_message}")
    
    print("\nTask Results:")
    for task_id, task_state in state.task_states.items():
        status_icon = "✅" if task_state.status.value == 'completed' else "❌"
        print(f"  {status_icon} {task_id}: {task_state.status.value}")
        if task_state.outputs:
            print(f"      Output: {json.dumps(task_state.outputs, indent=6)[:200]}...")
    
    print(f"\nWorkflow Outputs:")
    print(json.dumps(state.outputs, indent=2))
    
    # Log workflow completion
    await audit_logger.log(
        AuditEventType.WORKFLOW_COMPLETED if state.status == WorkflowStatus.COMPLETED else AuditEventType.WORKFLOW_FAILED,
        workflow_id=workflow_id,
        details={
            'status': state.status.value,
            'duration': (state.completed_at - state.created_at).total_seconds() if state.completed_at else None
        }
    )
    
    return state


async def run_all_examples():
    """Run all example workflows."""
    
    examples = ['simple', 'parallel', 'conditional', 'compensation']
    
    for example in examples:
        try:
            await run_example_workflow(example)
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"\n❌ Example '{example}' failed: {e}")
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)


if __name__ == "__main__":
    # Run all examples
    asyncio.run(run_all_examples())
