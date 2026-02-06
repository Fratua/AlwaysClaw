# AlwaysClaw Loop Contract Schema

## Overview

`loop.contract.schema.json` defines the canonical JSON Schema (draft 2020-12) for all hardcoded loop contracts in the AlwaysClaw loop kernel.

## Structure

Every loop contract is a JSON object with these field groups:

- **Identity**: `loopId`, `version`, `displayName`, `description`, `ownerAgent`
- **Classification**: `category` (14 values), `tier` (A/B), `dependencyLayer` (11 layers), `priority`
- **Triggers**: `triggerTypes` (6 types), `triggerClass` (4 classes), `cadence`
- **Entry/Exit**: `entryCriteria`, `hardStops`, `requiredContext`
- **Execution**: `thinkingPolicy`, `allowedTools`, `forbiddenTools`, `toolTierBudget`, `maxSteps`, `maxDurationSec`
- **Governance**: `approvalPoints`, `approvalRequired`, `riskTier`, `terminationPolicy`
- **Outcomes**: `successCriteria`, `rollbackPlan`, `fallbackLoopChain`
- **Writebacks**: `writebacks` (memory, logs, actionArtifacts, auditTrail)
- **Output**: `outputContract` (summary, confidence, evidence, actions, next)
- **Lifecycle**: `loopLifecycleStates` (10 states from queued to awaiting_approval)
- **Safety**: `guardrails` (confidenceThreshold, toolFailureBudget, sideEffectApprovalRequired)
- **Composition**: `composition` (canBeCalledBy, canCall, isSubLoopOf)

## Validation

Install ajv and ajv-formats, then validate any contract:

```js
import Ajv from 'ajv';
import addFormats from 'ajv-formats';
import schema from './loop.contract.schema.json' assert { type: 'json' };

const ajv = new Ajv({ allErrors: true });
addFormats(ajv);
const validate = ajv.compile(schema);
const valid = validate(contract);
if (!valid) console.error(validate.errors);
```

## Enum Reference

- **category**: strategic, knowledge, delivery, autonomy, runtime, context-memory, integration-gmail, integration-twilio, integration-browser, integration-voice, security, reliability, quality, governance
- **tier**: A (mandatory v1), B (advanced phase-in)
- **riskTier**: safe, moderate, high
- **triggerClass**: always-on, event-triggered, scheduled, manual-scheduled-hybrid
- **terminationPolicy**: safe_stop_and_report, hard_stop_and_alert, retry_then_escalate, park_and_await_approval
- **toolTierBudget**: tier-0 (read-only), tier-1 (reversible writes), tier-2 (irreversible/high-risk)
