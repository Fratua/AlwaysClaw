#!/usr/bin/env node

/**
 * generate_artifacts.mjs
 *
 * Generates the complete AlwaysClaw loop artifact pack:
 * - loops/registry.json (105 loop manifest)
 * - loops/contracts/*.json (105 individual contract files)
 *
 * All loops are derived from OPENCLAW_WIN10_GPT52_MASTERPLAN.md sections 19, 26, 36, 37, 49, 52-58.
 */

import { writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..');

// ─────────────────────────────────────────────────────────────────────────────
// Tier A loop numbers (from master plan section 53)
// ─────────────────────────────────────────────────────────────────────────────

const TIER_A_NUMBERS = new Set([
  19,20,21,22,23,24,28,30,31,32,33,35,38,39,40,
  47,48,49,50,52,53,54,55,57,58,59,60,61,62,
  66,67,68,69,70,71,73,76,77,81,83,91,97,99
]);

// ─────────────────────────────────────────────────────────────────────────────
// Full lifecycle (used by core loops)
// ─────────────────────────────────────────────────────────────────────────────

const FULL_LIFECYCLE = [
  'queued','preflight','context-build','reason','act','verify','commit','reflect','archive'
];

const STANDARD_LIFECYCLE = [
  'queued','preflight','context-build','reason','act','verify','commit','archive'
];

const APPROVAL_LIFECYCLE = [
  'queued','preflight','context-build','reason','act','awaiting_approval','verify','commit','reflect','archive'
];

// ─────────────────────────────────────────────────────────────────────────────
// 15 Core loop definitions (deep spec from master plan section 19, 26, 49)
// ─────────────────────────────────────────────────────────────────────────────

const CORE_LOOPS = [
  {
    num: 1, id: 'ralph-loop-v1', displayName: 'Ralph Loop',
    description: 'Macro-loop coordinator: rapid assess, learn, plan, act, reflect, harden. Used as the strategic orchestration layer for multi-loop planning and system-wide decision-making.',
    ownerAgent: 'loop-kernel-agent', category: 'strategic', dependencyLayer: 'strategic', priority: 1,
    triggerTypes: ['manual', 'cron', 'event'], triggerClass: 'manual-scheduled-hybrid',
    cadence: { default: 'weekly', quietHours: 'suppress', override: 'on-incident' },
    entryCriteria: ['strategic_review_scheduled_or_incident_detected', 'system_health_baseline_available'],
    hardStops: ['operator_abort', 'budget_exhausted', 'critical_incident_override'],
    requiredContext: ['AGENTS.md', 'SOUL.md', 'system_telemetry', 'loop_scorecards', 'backlog_state'],
    thinkingPolicy: { default: 'xhigh', escalateTo: 'xhigh', escalateOn: ['complex_multi_domain_decision'], deescalateOn: ['budget_pressure'] },
    allowedTools: ['memory.search', 'memory.write', 'scheduler.list', 'tasks.read', 'tasks.write', 'metrics.query', 'loop.invoke'],
    forbiddenTools: ['host.exec', 'browser.navigate', 'email.send'],
    toolTierBudget: 'tier-1', approvalPoints: ['policy_changes', 'tool_scope_modifications', 'loop_priority_reorder'],
    approvalRequired: true, maxSteps: 30, maxDurationSec: 1200, riskTier: 'moderate',
    successCriteria: ['prioritized_strategy_graph_emitted', 'seven_day_action_plan_generated', 'memory_writeback_complete'],
    rollbackPlan: 'Revert strategy graph to previous version and notify operator of failed planning cycle.',
    fallbackLoopChain: ['research-loop-v1', 'planning-loop-v1'],
    terminationPolicy: 'safe_stop_and_report',
    writebacks: { memory: true, logs: true, actionArtifacts: true, auditTrail: true },
    outputContract: { summaryRequired: true, confidenceRequired: true, evidenceRequired: true, actionsRequired: true, nextRequired: true },
    loopLifecycleStates: FULL_LIFECYCLE,
    guardrails: { confidenceThreshold: 0.7, toolFailureBudget: 3, sideEffectApprovalRequired: true },
    composition: { canBeCalledBy: [], canCall: ['research-loop-v1','discovery-loop-v1','planning-loop-v1','meta-cognition-loop-v1'] }
  },
  {
    num: 2, id: 'research-loop-v1', displayName: 'Research Loop',
    description: 'Collects, grades, and normalizes evidence from multiple sources. Requires source quality scoring and produces structured evidence packs for downstream decision loops.',
    ownerAgent: 'research-agent', category: 'knowledge', dependencyLayer: 'knowledge', priority: 5,
    triggerTypes: ['manual', 'event'], triggerClass: 'event-triggered',
    cadence: { default: 'on-demand', quietHours: 'defer', override: 'immediate-on-uncertainty' },
    entryCriteria: ['fact_gap_identified_or_uncertainty_threshold_exceeded', 'research_budget_available'],
    hardStops: ['source_access_failure_cascade', 'budget_limit_reached', 'operator_abort'],
    requiredContext: ['AGENTS.md', 'research_brief', 'prior_evidence_packs', 'domain_context'],
    thinkingPolicy: { default: 'high', escalateTo: 'xhigh', escalateOn: ['extreme_ambiguity', 'conflicting_sources'], deescalateOn: ['budget_pressure'] },
    allowedTools: ['memory.search', 'memory.write', 'web.search', 'web.fetch', 'file.read'],
    forbiddenTools: ['host.exec', 'browser.formSubmit', 'email.send'],
    toolTierBudget: 'tier-0', approvalPoints: [],
    approvalRequired: false, maxSteps: 24, maxDurationSec: 720, riskTier: 'safe',
    successCriteria: ['evidence_pack_emitted_with_quality_scores', 'minimum_three_sources_consulted', 'confidence_above_threshold'],
    rollbackPlan: 'Discard partial evidence pack and return uncertainty signal to caller.',
    fallbackLoopChain: ['discovery-loop-v1'],
    terminationPolicy: 'safe_stop_and_report',
    writebacks: { memory: true, logs: true, actionArtifacts: true, auditTrail: false },
    outputContract: { summaryRequired: true, confidenceRequired: true, evidenceRequired: true, actionsRequired: false, nextRequired: true },
    loopLifecycleStates: FULL_LIFECYCLE,
    guardrails: { confidenceThreshold: 0.6, toolFailureBudget: 5, sideEffectApprovalRequired: false },
    composition: { canBeCalledBy: ['ralph-loop-v1','planning-loop-v1','bug-finder-loop-v1'], canCall: ['discovery-loop-v1'] }
  },
  {
    num: 3, id: 'discovery-loop-v1', displayName: 'Discovery Loop',
    description: 'Scans unknown option-space and emits ranked hypotheses with risk/cost matrices. Explores solution spaces when the problem domain is poorly understood.',
    ownerAgent: 'discovery-agent', category: 'knowledge', dependencyLayer: 'knowledge', priority: 6,
    triggerTypes: ['manual', 'event'], triggerClass: 'event-triggered',
    cadence: { default: 'on-demand', quietHours: 'defer', override: 'immediate-on-ambiguity' },
    entryCriteria: ['solution_space_uncertainty_detected', 'discovery_budget_available'],
    hardStops: ['hypothesis_count_exceeds_limit', 'budget_exhausted', 'operator_abort'],
    requiredContext: ['AGENTS.md', 'problem_statement', 'existing_research_packs', 'constraint_set'],
    thinkingPolicy: { default: 'high', escalateTo: 'xhigh', escalateOn: ['high_ambiguity_persistent'], deescalateOn: ['budget_pressure'] },
    allowedTools: ['memory.search', 'memory.write', 'web.search', 'web.fetch', 'file.read'],
    forbiddenTools: ['host.exec', 'browser.formSubmit', 'email.send'],
    toolTierBudget: 'tier-0', approvalPoints: [],
    approvalRequired: false, maxSteps: 24, maxDurationSec: 720, riskTier: 'safe',
    successCriteria: ['ranked_hypothesis_set_emitted', 'risk_cost_matrix_generated', 'at_least_two_viable_options_identified'],
    rollbackPlan: 'Discard hypothesis set and signal discovery failure to caller.',
    fallbackLoopChain: [],
    terminationPolicy: 'safe_stop_and_report',
    writebacks: { memory: true, logs: true, actionArtifacts: true, auditTrail: false },
    outputContract: { summaryRequired: true, confidenceRequired: true, evidenceRequired: true, actionsRequired: false, nextRequired: true },
    loopLifecycleStates: FULL_LIFECYCLE,
    guardrails: { confidenceThreshold: 0.5, toolFailureBudget: 5, sideEffectApprovalRequired: false },
    composition: { canBeCalledBy: ['ralph-loop-v1','research-loop-v1','planning-loop-v1'], canCall: [] }
  },
  {
    num: 4, id: 'bug-finder-loop-v1', displayName: 'Bug Finder Loop',
    description: 'Reproducibility-first defect hunter that creates minimal failing test cases from SLO violations, user reports, or automated detection signals.',
    ownerAgent: 'bug-finder-agent', category: 'delivery', dependencyLayer: 'delivery', priority: 8,
    triggerTypes: ['event', 'manual'], triggerClass: 'event-triggered',
    cadence: { default: 'event-driven', quietHours: 'allow', override: 'immediate-on-slo-breach' },
    entryCriteria: ['failed_slo_or_test_or_user_defect_report_exists', 'debugging_budget_available'],
    hardStops: ['reproduction_impossible_after_max_attempts', 'budget_exhausted', 'operator_abort'],
    requiredContext: ['AGENTS.md', 'defect_signal', 'relevant_source_code', 'test_suite_state', 'recent_change_log'],
    thinkingPolicy: { default: 'high', escalateTo: 'xhigh', escalateOn: ['complex_reproduction_path'], deescalateOn: ['budget_pressure'] },
    allowedTools: ['memory.search', 'memory.write', 'file.read', 'file.write', 'test.run', 'code.search'],
    forbiddenTools: ['host.exec', 'browser.navigate', 'email.send'],
    toolTierBudget: 'tier-1', approvalPoints: [],
    approvalRequired: false, maxSteps: 24, maxDurationSec: 900, riskTier: 'safe',
    successCriteria: ['reproducible_defect_card_created', 'minimal_failing_test_written', 'root_cause_hypothesis_documented'],
    rollbackPlan: 'Discard incomplete defect card and escalate to human triage.',
    fallbackLoopChain: ['research-loop-v1'],
    terminationPolicy: 'safe_stop_and_report',
    writebacks: { memory: true, logs: true, actionArtifacts: true, auditTrail: true },
    outputContract: { summaryRequired: true, confidenceRequired: true, evidenceRequired: true, actionsRequired: true, nextRequired: true },
    loopLifecycleStates: FULL_LIFECYCLE,
    guardrails: { confidenceThreshold: 0.6, toolFailureBudget: 4, sideEffectApprovalRequired: false },
    composition: { canBeCalledBy: ['ralph-loop-v1','end-to-end-loop-v1'], canCall: ['research-loop-v1','debugging-loop-v1'] }
  },
  {
    num: 5, id: 'debugging-loop-v1', displayName: 'Debugging Loop',
    description: 'Isolates root cause from defect card, implements targeted fix, and verifies no regression is introduced through the reproduce-isolate-fix-verify cycle.',
    ownerAgent: 'debugging-agent', category: 'delivery', dependencyLayer: 'delivery', priority: 9,
    triggerTypes: ['event', 'manual'], triggerClass: 'event-triggered',
    cadence: { default: 'event-driven', quietHours: 'allow', override: 'immediate-on-critical-bug' },
    entryCriteria: ['defect_card_exists_with_reproduction_steps', 'fix_budget_available'],
    hardStops: ['fix_introduces_new_failures', 'budget_exhausted', 'operator_abort'],
    requiredContext: ['AGENTS.md', 'defect_card', 'source_code_under_test', 'test_suite', 'recent_changes'],
    thinkingPolicy: { default: 'xhigh', escalateTo: 'xhigh', escalateOn: ['multi_root_cause_interaction'], deescalateOn: ['budget_pressure'] },
    allowedTools: ['memory.search', 'memory.write', 'file.read', 'file.write', 'test.run', 'code.search', 'code.edit'],
    forbiddenTools: ['host.exec', 'browser.navigate', 'email.send'],
    toolTierBudget: 'tier-1', approvalPoints: ['external_side_effects', 'writes_outside_sandbox'],
    approvalRequired: true, maxSteps: 30, maxDurationSec: 1200, riskTier: 'moderate',
    successCriteria: ['candidate_fix_implemented', 'verification_report_shows_no_regression', 'defect_card_marked_resolved'],
    rollbackPlan: 'Revert fix patch and restore codebase to pre-fix state.',
    fallbackLoopChain: ['research-loop-v1', 'bug-finder-loop-v1'],
    terminationPolicy: 'safe_stop_and_report',
    writebacks: { memory: true, logs: true, actionArtifacts: true, auditTrail: true },
    outputContract: { summaryRequired: true, confidenceRequired: true, evidenceRequired: true, actionsRequired: true, nextRequired: true },
    loopLifecycleStates: APPROVAL_LIFECYCLE,
    guardrails: { confidenceThreshold: 0.75, toolFailureBudget: 3, sideEffectApprovalRequired: true },
    composition: { canBeCalledBy: ['ralph-loop-v1','bug-finder-loop-v1'], canCall: ['research-loop-v1','end-to-end-loop-v1'] }
  },
  {
    num: 6, id: 'end-to-end-loop-v1', displayName: 'End-to-End Loop',
    description: 'Executes full scenario from trigger to user-visible completion. Validates feature correctness across the entire stack with structured execution traces.',
    ownerAgent: 'e2e-agent', category: 'delivery', dependencyLayer: 'delivery', priority: 10,
    triggerTypes: ['event', 'cron', 'manual'], triggerClass: 'manual-scheduled-hybrid',
    cadence: { default: 'per-release', quietHours: 'suppress', override: 'nightly' },
    entryCriteria: ['feature_marked_implementation_complete', 'test_environment_available'],
    hardStops: ['environment_unavailable', 'critical_infrastructure_failure', 'operator_abort'],
    requiredContext: ['AGENTS.md', 'feature_spec', 'test_scenarios', 'deployment_state', 'integration_endpoints'],
    thinkingPolicy: { default: 'xhigh', escalateTo: 'xhigh', escalateOn: ['complex_multi_service_failure'], deescalateOn: ['budget_pressure'] },
    allowedTools: ['memory.search', 'memory.write', 'test.run', 'file.read', 'browser.navigate', 'browser.screenshot', 'api.call'],
    forbiddenTools: ['host.exec', 'email.send', 'browser.formSubmit'],
    toolTierBudget: 'tier-1', approvalPoints: ['external_side_effects'],
    approvalRequired: true, maxSteps: 30, maxDurationSec: 1200, riskTier: 'moderate',
    successCriteria: ['end_to_end_execution_trace_recorded', 'pass_fail_verdict_emitted', 'all_assertions_passed'],
    rollbackPlan: 'Flag feature as validation-failed and notify release gate.',
    fallbackLoopChain: ['bug-finder-loop-v1', 'debugging-loop-v1'],
    terminationPolicy: 'safe_stop_and_report',
    writebacks: { memory: true, logs: true, actionArtifacts: true, auditTrail: true },
    outputContract: { summaryRequired: true, confidenceRequired: true, evidenceRequired: true, actionsRequired: true, nextRequired: true },
    loopLifecycleStates: APPROVAL_LIFECYCLE,
    guardrails: { confidenceThreshold: 0.8, toolFailureBudget: 3, sideEffectApprovalRequired: true },
    composition: { canBeCalledBy: ['ralph-loop-v1','debugging-loop-v1'], canCall: ['bug-finder-loop-v1'] }
  },
  {
    num: 7, id: 'meta-cognition-loop-v1', displayName: 'Meta-Cognition Loop',
    description: 'Critiques loop execution strategy and rewrites future run strategy rules. Analyzes repeated failures to derive systemic improvements across the loop kernel.',
    ownerAgent: 'meta-cognition-agent', category: 'strategic', dependencyLayer: 'strategic', priority: 3,
    triggerTypes: ['cron', 'event'], triggerClass: 'scheduled',
    cadence: { default: 'daily', quietHours: 'suppress', override: 'on-repeat-failure-threshold' },
    entryCriteria: ['repeated_loop_failure_detected_or_periodic_review_due', 'loop_telemetry_available'],
    hardStops: ['strategy_rewrite_causes_validation_failure', 'budget_exhausted', 'operator_abort'],
    requiredContext: ['AGENTS.md', 'SOUL.md', 'loop_scorecards', 'failure_telemetry', 'current_strategy_rules'],
    thinkingPolicy: { default: 'xhigh', escalateTo: 'xhigh', escalateOn: ['systemic_pattern_detected'], deescalateOn: ['budget_pressure'] },
    allowedTools: ['memory.search', 'memory.write', 'metrics.query', 'config.read', 'config.propose'],
    forbiddenTools: ['host.exec', 'browser.navigate', 'email.send', 'config.write'],
    toolTierBudget: 'tier-1', approvalPoints: ['strategy_deltas', 'policy_amendments'],
    approvalRequired: false, maxSteps: 22, maxDurationSec: 900, riskTier: 'safe',
    successCriteria: ['strategy_deltas_emitted', 'policy_amendments_proposed', 'improvement_hypothesis_documented'],
    rollbackPlan: 'Discard proposed strategy changes and retain current policy.',
    fallbackLoopChain: ['research-loop-v1'],
    terminationPolicy: 'safe_stop_and_report',
    writebacks: { memory: true, logs: true, actionArtifacts: true, auditTrail: true },
    outputContract: { summaryRequired: true, confidenceRequired: true, evidenceRequired: true, actionsRequired: true, nextRequired: true },
    loopLifecycleStates: FULL_LIFECYCLE,
    guardrails: { confidenceThreshold: 0.65, toolFailureBudget: 3, sideEffectApprovalRequired: false },
    composition: { canBeCalledBy: ['ralph-loop-v1'], canCall: ['research-loop-v1','self-learning-loop-v1'] }
  },
  {
    num: 8, id: 'exploration-loop-v1', displayName: 'Exploration Loop',
    description: 'Controlled capability probing under strict non-destructive policy. Tests tool and system boundaries during innovation windows or when capability gaps are detected.',
    ownerAgent: 'exploration-agent', category: 'knowledge', dependencyLayer: 'knowledge', priority: 12,
    triggerTypes: ['manual', 'cron'], triggerClass: 'scheduled',
    cadence: { default: 'controlled-windows', quietHours: 'suppress', override: 'on-capability-gap' },
    entryCriteria: ['innovation_window_open_or_capability_gap_hypothesis', 'exploration_budget_available'],
    hardStops: ['destructive_action_attempted', 'budget_exhausted', 'operator_abort'],
    requiredContext: ['AGENTS.md', 'capability_map', 'tool_registry', 'safety_constraints'],
    thinkingPolicy: { default: 'high', escalateTo: 'xhigh', escalateOn: ['unexpected_capability_discovered'], deescalateOn: ['budget_pressure'] },
    allowedTools: ['memory.search', 'memory.write', 'file.read', 'web.search', 'tool.probe'],
    forbiddenTools: ['host.exec', 'file.write', 'email.send', 'browser.formSubmit'],
    toolTierBudget: 'tier-0', approvalPoints: [],
    approvalRequired: false, maxSteps: 20, maxDurationSec: 600, riskTier: 'safe',
    successCriteria: ['safe_experiment_results_documented', 'feasibility_notes_generated', 'no_side_effects_produced'],
    rollbackPlan: 'Discard exploration artifacts and log exploration boundary violations.',
    fallbackLoopChain: ['research-loop-v1'],
    terminationPolicy: 'safe_stop_and_report',
    writebacks: { memory: true, logs: true, actionArtifacts: true, auditTrail: false },
    outputContract: { summaryRequired: true, confidenceRequired: true, evidenceRequired: true, actionsRequired: false, nextRequired: true },
    loopLifecycleStates: FULL_LIFECYCLE,
    guardrails: { confidenceThreshold: 0.5, toolFailureBudget: 5, sideEffectApprovalRequired: false },
    composition: { canBeCalledBy: ['ralph-loop-v1','planning-loop-v1'], canCall: ['research-loop-v1'] }
  },
  {
    num: 9, id: 'self-driven-loop-v1', displayName: 'Self-Driven Loop',
    description: 'Proposes autonomous tasks from backlog analysis and telemetry signals during idle windows. Cannot execute tier-2 side effects without approval.',
    ownerAgent: 'self-driven-agent', category: 'autonomy', dependencyLayer: 'autonomy', priority: 15,
    triggerTypes: ['heartbeat', 'self-driven'], triggerClass: 'scheduled',
    cadence: { default: '30m', quietHours: 'suppress', override: 'on-idle-window' },
    entryCriteria: ['idle_window_detected_and_backlog_opportunity_exists', 'autonomy_policy_permits'],
    hardStops: ['tier2_tool_requested_without_approval', 'budget_exhausted', 'operator_abort'],
    requiredContext: ['AGENTS.md', 'backlog_state', 'telemetry_summary', 'autonomy_policy'],
    thinkingPolicy: { default: 'high', escalateTo: 'xhigh', escalateOn: ['complex_task_proposal'], deescalateOn: ['budget_pressure'] },
    allowedTools: ['memory.search', 'memory.write', 'tasks.read', 'tasks.propose', 'scheduler.list'],
    forbiddenTools: ['host.exec', 'browser.navigate', 'email.send', 'config.write'],
    toolTierBudget: 'tier-1', approvalPoints: ['task_execution_escalation'],
    approvalRequired: true, maxSteps: 16, maxDurationSec: 480, riskTier: 'moderate',
    successCriteria: ['proposed_autonomous_tasks_with_justification_emitted', 'no_unapproved_side_effects'],
    rollbackPlan: 'Discard task proposals and return to idle state.',
    fallbackLoopChain: ['planning-loop-v1'],
    terminationPolicy: 'safe_stop_and_report',
    writebacks: { memory: true, logs: true, actionArtifacts: true, auditTrail: true },
    outputContract: { summaryRequired: true, confidenceRequired: true, evidenceRequired: false, actionsRequired: true, nextRequired: true },
    loopLifecycleStates: APPROVAL_LIFECYCLE,
    guardrails: { confidenceThreshold: 0.6, toolFailureBudget: 2, sideEffectApprovalRequired: true },
    composition: { canBeCalledBy: [], canCall: ['planning-loop-v1','research-loop-v1'] }
  },
  {
    num: 10, id: 'self-learning-loop-v1', displayName: 'Self-Learning Loop',
    description: 'Converts loop outcomes into reusable memory cards and playbooks. Runs continuously after any loop completion to capture lessons learned.',
    ownerAgent: 'self-learning-agent', category: 'autonomy', dependencyLayer: 'autonomy', priority: 16,
    triggerTypes: ['event'], triggerClass: 'event-triggered',
    cadence: { default: 'continuous', quietHours: 'allow', override: 'on-loop-completion' },
    entryCriteria: ['loop_completion_event_received', 'learning_pipeline_available'],
    hardStops: ['memory_write_failure', 'budget_exhausted', 'operator_abort'],
    requiredContext: ['AGENTS.md', 'completed_loop_output', 'existing_playbooks', 'memory_index'],
    thinkingPolicy: { default: 'medium', escalateTo: 'high', escalateOn: ['novel_pattern_detected'], deescalateOn: ['routine_outcome'] },
    allowedTools: ['memory.search', 'memory.write', 'file.read'],
    forbiddenTools: ['host.exec', 'browser.navigate', 'email.send', 'config.write', 'file.write'],
    toolTierBudget: 'tier-0', approvalPoints: [],
    approvalRequired: false, maxSteps: 12, maxDurationSec: 360, riskTier: 'safe',
    successCriteria: ['memory_cards_written', 'playbook_entries_updated_or_created'],
    rollbackPlan: 'Discard new memory cards and retain existing playbooks unchanged.',
    fallbackLoopChain: [],
    terminationPolicy: 'safe_stop_and_report',
    writebacks: { memory: true, logs: true, actionArtifacts: false, auditTrail: false },
    outputContract: { summaryRequired: true, confidenceRequired: false, evidenceRequired: false, actionsRequired: false, nextRequired: true },
    loopLifecycleStates: STANDARD_LIFECYCLE,
    guardrails: { confidenceThreshold: 0.4, toolFailureBudget: 3, sideEffectApprovalRequired: false },
    composition: { canBeCalledBy: ['meta-cognition-loop-v1'], canCall: [] }
  },
  {
    num: 11, id: 'self-updating-loop-v1', displayName: 'Self-Updating Loop',
    description: 'Drafts prompt/config changes and opens approval requests. Changes require explicit approval gate plus regression suite validation before application.',
    ownerAgent: 'self-updating-agent', category: 'autonomy', dependencyLayer: 'autonomy', priority: 18,
    triggerTypes: ['cron', 'event'], triggerClass: 'scheduled',
    cadence: { default: 'daily', quietHours: 'suppress', override: 'on-drift-signal' },
    entryCriteria: ['config_or_prompt_drift_signal_detected', 'update_proposal_budget_available'],
    hardStops: ['regression_suite_failure', 'approval_denied', 'operator_abort'],
    requiredContext: ['AGENTS.md', 'current_config', 'current_prompts', 'drift_metrics', 'regression_suite'],
    thinkingPolicy: { default: 'high', escalateTo: 'xhigh', escalateOn: ['complex_prompt_revision'], deescalateOn: ['budget_pressure'] },
    allowedTools: ['memory.search', 'memory.write', 'config.read', 'config.propose', 'test.run', 'file.read'],
    forbiddenTools: ['host.exec', 'config.write', 'file.write', 'email.send'],
    toolTierBudget: 'tier-1', approvalPoints: ['config_change_proposal', 'prompt_change_proposal'],
    approvalRequired: true, maxSteps: 16, maxDurationSec: 600, riskTier: 'high',
    successCriteria: ['change_proposal_patch_generated', 'rollback_notes_included', 'regression_suite_passed'],
    rollbackPlan: 'Discard proposed patch and retain current config/prompt versions.',
    fallbackLoopChain: ['research-loop-v1'],
    terminationPolicy: 'park_and_await_approval',
    writebacks: { memory: true, logs: true, actionArtifacts: true, auditTrail: true },
    outputContract: { summaryRequired: true, confidenceRequired: true, evidenceRequired: true, actionsRequired: true, nextRequired: true },
    loopLifecycleStates: APPROVAL_LIFECYCLE,
    guardrails: { confidenceThreshold: 0.8, toolFailureBudget: 2, sideEffectApprovalRequired: true },
    composition: { canBeCalledBy: [], canCall: ['research-loop-v1'] }
  },
  {
    num: 12, id: 'self-upgrading-loop-v1', displayName: 'Self-Upgrading Loop',
    description: 'Proposes binary/dependency/plugin upgrades with rollback plan. Must include smoke test and e2e verification plan before approval request.',
    ownerAgent: 'self-upgrading-agent', category: 'autonomy', dependencyLayer: 'autonomy', priority: 19,
    triggerTypes: ['cron', 'event'], triggerClass: 'scheduled',
    cadence: { default: 'weekly', quietHours: 'suppress', override: 'on-security-patch' },
    entryCriteria: ['security_patch_or_performance_bottleneck_detected', 'upgrade_proposal_budget_available'],
    hardStops: ['upgrade_test_failure', 'approval_denied', 'operator_abort'],
    requiredContext: ['AGENTS.md', 'dependency_manifest', 'vulnerability_reports', 'performance_baselines'],
    thinkingPolicy: { default: 'high', escalateTo: 'xhigh', escalateOn: ['major_version_upgrade_required'], deescalateOn: ['budget_pressure'] },
    allowedTools: ['memory.search', 'memory.write', 'file.read', 'dependency.scan', 'test.run'],
    forbiddenTools: ['host.exec', 'dependency.install', 'file.write', 'email.send'],
    toolTierBudget: 'tier-1', approvalPoints: ['upgrade_proposal', 'rollback_plan_review'],
    approvalRequired: true, maxSteps: 18, maxDurationSec: 720, riskTier: 'high',
    successCriteria: ['upgrade_proposal_with_rollback_plan_emitted', 'smoke_test_plan_included', 'e2e_verification_plan_included'],
    rollbackPlan: 'Discard upgrade proposal and retain current dependency versions.',
    fallbackLoopChain: ['research-loop-v1'],
    terminationPolicy: 'park_and_await_approval',
    writebacks: { memory: true, logs: true, actionArtifacts: true, auditTrail: true },
    outputContract: { summaryRequired: true, confidenceRequired: true, evidenceRequired: true, actionsRequired: true, nextRequired: true },
    loopLifecycleStates: APPROVAL_LIFECYCLE,
    guardrails: { confidenceThreshold: 0.85, toolFailureBudget: 2, sideEffectApprovalRequired: true },
    composition: { canBeCalledBy: [], canCall: ['research-loop-v1'] }
  },
  {
    num: 13, id: 'planning-loop-v1', displayName: 'Planning Loop',
    description: 'Reprioritizes execution backlog with cost/risk/confidence scoring. Transforms research bundles into ordered executable plans.',
    ownerAgent: 'planning-agent', category: 'strategic', dependencyLayer: 'strategic', priority: 2,
    triggerTypes: ['cron', 'manual', 'self-driven', 'heartbeat'], triggerClass: 'manual-scheduled-hybrid',
    cadence: { default: 'daily', quietHours: 'suppress', override: 'on-major-context-change' },
    entryCriteria: ['backlog_exists_and_policy_allows_autonomy', 'planning_budget_available'],
    hardStops: ['plan_conflict_with_active_loops', 'budget_exhausted', 'operator_abort'],
    requiredContext: ['AGENTS.md', 'backlog_state', 'resource_budget', 'active_loop_states', 'priority_policy'],
    thinkingPolicy: { default: 'xhigh', escalateTo: 'xhigh', escalateOn: ['repeat_failure', 'high_ambiguity'], deescalateOn: ['budget_pressure'] },
    allowedTools: ['memory.search', 'memory.write', 'scheduler.list', 'tasks.write', 'tasks.read'],
    forbiddenTools: ['host.exec', 'browser.navigate', 'email.send'],
    toolTierBudget: 'tier-1', approvalPoints: ['external_side_effects', 'tier2_tools'],
    approvalRequired: false, maxSteps: 20, maxDurationSec: 720, riskTier: 'safe',
    successCriteria: ['prioritized_plan_emitted', 'memory_writeback_complete', 'execution_backlog_ordered_with_budgets'],
    rollbackPlan: 'Revert to previous backlog ordering and notify operator.',
    fallbackLoopChain: ['research-loop-v1', 'discovery-loop-v1'],
    terminationPolicy: 'safe_stop_and_report',
    writebacks: { memory: true, logs: true, actionArtifacts: true, auditTrail: true },
    outputContract: { summaryRequired: true, confidenceRequired: true, evidenceRequired: false, actionsRequired: true, nextRequired: true },
    loopLifecycleStates: FULL_LIFECYCLE,
    guardrails: { confidenceThreshold: 0.65, toolFailureBudget: 3, sideEffectApprovalRequired: false },
    composition: { canBeCalledBy: ['ralph-loop-v1','self-driven-loop-v1'], canCall: ['research-loop-v1','discovery-loop-v1'] }
  },
  {
    num: 14, id: 'context-engineering-loop-v1', displayName: 'Context Engineering Loop',
    description: 'Optimizes context packing, retrieval windows, and compaction strategy. Responds to token pressure or context retrieval misses with policy revisions.',
    ownerAgent: 'context-engineering-agent', category: 'knowledge', dependencyLayer: 'knowledge', priority: 7,
    triggerTypes: ['cron', 'event'], triggerClass: 'scheduled',
    cadence: { default: 'daily', quietHours: 'suppress', override: 'on-token-pressure-threshold' },
    entryCriteria: ['token_pressure_or_context_retrieval_misses_detected', 'context_optimization_budget_available'],
    hardStops: ['context_corruption_detected', 'budget_exhausted', 'operator_abort'],
    requiredContext: ['AGENTS.md', 'context_metrics', 'retrieval_miss_log', 'compaction_policy'],
    thinkingPolicy: { default: 'high', escalateTo: 'xhigh', escalateOn: ['severe_context_degradation'], deescalateOn: ['budget_pressure'] },
    allowedTools: ['memory.search', 'memory.write', 'config.read', 'config.propose', 'metrics.query'],
    forbiddenTools: ['host.exec', 'browser.navigate', 'email.send'],
    toolTierBudget: 'tier-1', approvalPoints: ['policy_changes'],
    approvalRequired: true, maxSteps: 18, maxDurationSec: 600, riskTier: 'moderate',
    successCriteria: ['retrieval_packing_policy_revision_emitted', 'context_miss_rate_reduction_projected'],
    rollbackPlan: 'Revert to previous context packing policy.',
    fallbackLoopChain: ['research-loop-v1'],
    terminationPolicy: 'safe_stop_and_report',
    writebacks: { memory: true, logs: true, actionArtifacts: true, auditTrail: true },
    outputContract: { summaryRequired: true, confidenceRequired: true, evidenceRequired: true, actionsRequired: true, nextRequired: true },
    loopLifecycleStates: APPROVAL_LIFECYCLE,
    guardrails: { confidenceThreshold: 0.7, toolFailureBudget: 3, sideEffectApprovalRequired: true },
    composition: { canBeCalledBy: ['ralph-loop-v1','planning-loop-v1'], canCall: ['research-loop-v1'] }
  },
  {
    num: 15, id: 'context-prompt-engineering-loop-v1', displayName: 'Context Prompt Engineering Loop',
    description: 'Iterates prompt templates using regression suite gates. Detects quality regressions in responses and proposes template revisions with test evidence.',
    ownerAgent: 'prompt-engineering-agent', category: 'knowledge', dependencyLayer: 'knowledge', priority: 11,
    triggerTypes: ['cron', 'event'], triggerClass: 'scheduled',
    cadence: { default: 'nightly', quietHours: 'allow', override: 'on-quality-regression' },
    entryCriteria: ['response_quality_regression_detected_or_nightly_review_due', 'prompt_revision_budget_available'],
    hardStops: ['regression_suite_degradation', 'budget_exhausted', 'operator_abort'],
    requiredContext: ['AGENTS.md', 'prompt_templates', 'regression_suite', 'quality_metrics', 'response_samples'],
    thinkingPolicy: { default: 'high', escalateTo: 'xhigh', escalateOn: ['widespread_quality_degradation'], deescalateOn: ['budget_pressure'] },
    allowedTools: ['memory.search', 'memory.write', 'file.read', 'test.run', 'config.propose'],
    forbiddenTools: ['host.exec', 'browser.navigate', 'email.send', 'config.write'],
    toolTierBudget: 'tier-1', approvalPoints: ['prompt_baseline_publish'],
    approvalRequired: true, maxSteps: 22, maxDurationSec: 840, riskTier: 'moderate',
    successCriteria: ['prompt_template_revisions_proposed', 'regression_test_results_included', 'quality_improvement_projected'],
    rollbackPlan: 'Retain current prompt templates and log revision attempt.',
    fallbackLoopChain: ['research-loop-v1'],
    terminationPolicy: 'safe_stop_and_report',
    writebacks: { memory: true, logs: true, actionArtifacts: true, auditTrail: true },
    outputContract: { summaryRequired: true, confidenceRequired: true, evidenceRequired: true, actionsRequired: true, nextRequired: true },
    loopLifecycleStates: APPROVAL_LIFECYCLE,
    guardrails: { confidenceThreshold: 0.7, toolFailureBudget: 3, sideEffectApprovalRequired: true },
    composition: { canBeCalledBy: ['ralph-loop-v1','meta-cognition-loop-v1'], canCall: ['research-loop-v1'] }
  },
];

// ─────────────────────────────────────────────────────────────────────────────
// 90 Derived loop definitions (from master plan section 52)
// ─────────────────────────────────────────────────────────────────────────────

const DERIVED_LOOPS = [
  // --- Runtime and orchestration loops (16-35) ---
  { num: 16, id: 'intake-normalization-loop-v1', displayName: 'Intake Normalization Loop', category: 'runtime', dependencyLayer: 'runtime-intake', ownerAgent: 'gateway-core-agent', description: 'Normalizes incoming commands and messages into canonical internal format before routing through the intake pipeline.' },
  { num: 17, id: 'intent-classification-loop-v1', displayName: 'Intent Classification Loop', category: 'runtime', dependencyLayer: 'runtime-intake', ownerAgent: 'gateway-core-agent', description: 'Classifies normalized intake messages by intent type to determine appropriate routing and processing pipeline.' },
  { num: 18, id: 'route-resolution-loop-v1', displayName: 'Route Resolution Loop', category: 'runtime', dependencyLayer: 'runtime-intake', ownerAgent: 'gateway-core-agent', description: 'Resolves classified intents to specific agent routes and session targets using routing policy rules.' },
  { num: 19, id: 'session-key-resolution-loop-v1', displayName: 'Session Key Resolution Loop', category: 'runtime', dependencyLayer: 'runtime-orchestration', ownerAgent: 'session-engine-agent', description: 'Resolves session keys using main, per-peer, or per-channel-peer strategies based on agent session policy.' },
  { num: 20, id: 'session-isolation-guard-loop-v1', displayName: 'Session Isolation Guard Loop', category: 'runtime', dependencyLayer: 'runtime-orchestration', ownerAgent: 'session-engine-agent', description: 'Enforces session isolation boundaries to prevent cross-session data leakage and unauthorized access between sessions.' },
  { num: 21, id: 'queue-admission-control-loop-v1', displayName: 'Queue Admission Control Loop', category: 'runtime', dependencyLayer: 'runtime-orchestration', ownerAgent: 'queue-control-agent', description: 'Controls admission of tasks into queue lanes enforcing concurrency limits: main=4, subagent=8, unconfigured=1.' },
  { num: 22, id: 'queue-backpressure-relief-loop-v1', displayName: 'Queue Backpressure Relief Loop', category: 'runtime', dependencyLayer: 'runtime-orchestration', ownerAgent: 'queue-control-agent', description: 'Detects queue depth thresholds and applies backpressure relief by throttling intake and shedding low-priority work.' },
  { num: 23, id: 'retry-orchestration-loop-v1', displayName: 'Retry Orchestration Loop', category: 'runtime', dependencyLayer: 'runtime-orchestration', ownerAgent: 'recovery-agent', description: 'Orchestrates retry logic with exponential backoff and jitter for failed command executions and tool invocations.' },
  { num: 24, id: 'dead-letter-recovery-loop-v1', displayName: 'Dead-Letter Recovery Loop', category: 'runtime', dependencyLayer: 'runtime-orchestration', ownerAgent: 'recovery-agent', description: 'Processes messages that have exhausted retry budgets from the dead-letter queue for manual review or automated recovery.' },
  { num: 25, id: 'presence-refresh-loop-v1', displayName: 'Presence Refresh Loop', category: 'runtime', dependencyLayer: 'runtime-orchestration', ownerAgent: 'presence-agent', description: 'Refreshes ephemeral presence entries before TTL expiration (5 min) and prunes stale entries beyond 200 cap.' },
  { num: 26, id: 'hook-dispatch-loop-v1', displayName: 'Hook Dispatch Loop', category: 'runtime', dependencyLayer: 'runtime-orchestration', ownerAgent: 'gateway-core-agent', description: 'Dispatches registered hook callbacks when matching events are emitted by the runtime event bus.' },
  { num: 27, id: 'hook-retry-replay-loop-v1', displayName: 'Hook Retry and Replay Loop', category: 'runtime', dependencyLayer: 'runtime-orchestration', ownerAgent: 'gateway-core-agent', description: 'Retries failed hook dispatches and supports replay of hook events from the event log for recovery.' },
  { num: 28, id: 'heartbeat-synthesis-loop-v1', displayName: 'Heartbeat Synthesis Loop', category: 'runtime', dependencyLayer: 'runtime-orchestration', ownerAgent: 'gateway-core-agent', description: 'Generates heartbeat events at 30-minute default cadence with HEARTBEAT_OK suppression for quiet check-ins.' },
  { num: 29, id: 'heartbeat-escalation-loop-v1', displayName: 'Heartbeat Escalation Loop', category: 'runtime', dependencyLayer: 'runtime-orchestration', ownerAgent: 'gateway-core-agent', description: 'Escalates missed heartbeat signals into incident events after configurable grace period threshold.' },
  { num: 30, id: 'cron-dispatch-loop-v1', displayName: 'Cron Dispatch Loop', category: 'runtime', dependencyLayer: 'runtime-orchestration', ownerAgent: 'gateway-core-agent', description: 'Dispatches scheduled cron jobs at their configured times with support for main and isolated session execution.' },
  { num: 31, id: 'cron-catch-up-loop-v1', displayName: 'Cron Catch-Up Loop', category: 'runtime', dependencyLayer: 'runtime-orchestration', ownerAgent: 'gateway-core-agent', description: 'Identifies and executes missed cron jobs after system downtime or restart to maintain scheduling reliability.' },
  { num: 32, id: 'cron-isolated-session-bootstrap-loop-v1', displayName: 'Cron Isolated Session Bootstrap Loop', category: 'runtime', dependencyLayer: 'runtime-orchestration', ownerAgent: 'gateway-core-agent', description: 'Bootstraps isolated execution sessions (cron:<jobId>) for cron jobs requiring autonomous background work.' },
  { num: 33, id: 'model-fallback-loop-v1', displayName: 'Model Fallback Loop', category: 'runtime', dependencyLayer: 'runtime-orchestration', ownerAgent: 'gpt52-router-agent', description: 'Routes model requests through primary/fallback/codex chain: gpt-5.2 primary, gpt-5.2-pro fallback, gpt-5.2-codex for coding.' },
  { num: 34, id: 'thinking-level-controller-loop-v1', displayName: 'Thinking-Level Controller Loop', category: 'runtime', dependencyLayer: 'runtime-orchestration', ownerAgent: 'thinking-control-agent', description: 'Selects thinking effort level (medium/high/xhigh) per request based on task complexity and budget constraints.' },
  { num: 35, id: 'tool-eligibility-resolver-loop-v1', displayName: 'Tool Eligibility Resolver Loop', category: 'runtime', dependencyLayer: 'runtime-orchestration', ownerAgent: 'tool-selection-agent', description: 'Resolves tool eligibility per execution context using allow/deny policies and tier-based authorization rules.' },
  // --- Context, memory, and persona loops (36-46) ---
  { num: 36, id: 'context-budget-manager-loop-v1', displayName: 'Context Budget Manager Loop', category: 'context-memory', dependencyLayer: 'context-memory', ownerAgent: 'context-compiler-agent', description: 'Manages token budget allocation across context slices ensuring prompt assembly stays within model limits.' },
  { num: 37, id: 'context-snapshot-loop-v1', displayName: 'Context Snapshot Loop', category: 'context-memory', dependencyLayer: 'context-memory', ownerAgent: 'context-compiler-agent', description: 'Captures point-in-time snapshots of assembled context for replay, debugging, and forensic analysis.' },
  { num: 38, id: 'memory-writeback-loop-v1', displayName: 'Memory Writeback Loop', category: 'context-memory', dependencyLayer: 'context-memory', ownerAgent: 'memory-runtime-agent', description: 'Persists loop outputs and conversation insights to disk memory and vector store with structured indexing.' },
  { num: 39, id: 'memory-retrieval-validation-loop-v1', displayName: 'Memory Retrieval Validation Loop', category: 'context-memory', dependencyLayer: 'context-memory', ownerAgent: 'memory-runtime-agent', description: 'Validates retrieved memory entries for relevance, recency, and accuracy before injection into active context.' },
  { num: 40, id: 'memory-compaction-trigger-loop-v1', displayName: 'Memory Compaction Trigger Loop', category: 'context-memory', dependencyLayer: 'context-memory', ownerAgent: 'compaction-agent', description: 'Monitors context pressure thresholds and triggers auto-compaction with summary/archival/usage tracking behavior.' },
  { num: 41, id: 'memory-consolidation-loop-v1', displayName: 'Memory Consolidation Loop', category: 'context-memory', dependencyLayer: 'context-memory', ownerAgent: 'compaction-agent', description: 'Consolidates fragmented memory entries into coherent knowledge structures during maintenance windows.' },
  { num: 42, id: 'identity-drift-detection-loop-v1', displayName: 'Identity Drift Detection Loop', category: 'context-memory', dependencyLayer: 'context-memory', ownerAgent: 'identity-runtime-agent', description: 'Detects drift between active agent behavior and IDENTITY.md/SOUL.md baseline specifications over time.' },
  { num: 43, id: 'soul-consistency-enforcement-loop-v1', displayName: 'Soul Consistency Enforcement Loop', category: 'context-memory', dependencyLayer: 'context-memory', ownerAgent: 'identity-runtime-agent', description: 'Enforces SOUL.md temperament and boundary constraints during agent execution preventing personality divergence.' },
  { num: 44, id: 'user-profile-refresh-loop-v1', displayName: 'User Profile Refresh Loop', category: 'context-memory', dependencyLayer: 'context-memory', ownerAgent: 'identity-runtime-agent', description: 'Periodically refreshes USER.md profile data from observed interactions and explicit user preference updates.' },
  { num: 45, id: 'preference-learning-loop-v1', displayName: 'Preference Learning Loop', category: 'context-memory', dependencyLayer: 'context-memory', ownerAgent: 'identity-runtime-agent', description: 'Learns user preferences from interaction patterns and updates preference models in the user profile store.' },
  { num: 46, id: 'relationship-continuity-loop-v1', displayName: 'Relationship Continuity Loop', category: 'context-memory', dependencyLayer: 'context-memory', ownerAgent: 'identity-runtime-agent', description: 'Maintains conversation relationship context across sessions ensuring natural interaction continuity.' },
  // --- Gmail loops (47-51) ---
  { num: 47, id: 'gmail-watch-renewal-loop-v1', displayName: 'Gmail Watch Renewal Loop', category: 'integration-gmail', dependencyLayer: 'integration', ownerAgent: 'gmail-integration-agent', description: 'Renews Gmail Pub/Sub watch subscriptions before 7-day expiration with jittered daily renewal cadence.' },
  { num: 48, id: 'gmail-notification-intake-loop-v1', displayName: 'Gmail Notification Intake Loop', category: 'integration-gmail', dependencyLayer: 'integration', ownerAgent: 'gmail-integration-agent', description: 'Receives and ACKs Gmail Pub/Sub push notifications immediately then enqueues mailbox sync jobs.' },
  { num: 49, id: 'gmail-history-delta-sync-loop-v1', displayName: 'Gmail History Delta Sync Loop', category: 'integration-gmail', dependencyLayer: 'integration', ownerAgent: 'gmail-integration-agent', description: 'Performs incremental mailbox sync using history.list from stored lastHistoryId to process new messages.' },
  { num: 50, id: 'gmail-full-resync-loop-v1', displayName: 'Gmail Full Resync Loop', category: 'integration-gmail', dependencyLayer: 'integration', ownerAgent: 'gmail-integration-agent', description: 'Executes full mailbox resync when history API returns 404 for stale/invalid startHistoryId pointers.' },
  { num: 51, id: 'gmail-action-execution-loop-v1', displayName: 'Gmail Action Execution Loop', category: 'integration-gmail', dependencyLayer: 'integration', ownerAgent: 'gmail-integration-agent', description: 'Executes approved Gmail actions (reply, forward, label, archive) with delivery confirmation tracking.' },
  // --- Twilio loops (52-56) ---
  { num: 52, id: 'twilio-webhook-verification-loop-v1', displayName: 'Twilio Webhook Verification Loop', category: 'integration-twilio', dependencyLayer: 'integration', ownerAgent: 'twilio-agent', description: 'Validates X-Twilio-Signature on all incoming webhooks and rejects unsigned requests as security events.' },
  { num: 53, id: 'sms-inbound-processing-loop-v1', displayName: 'SMS Inbound Processing Loop', category: 'integration-twilio', dependencyLayer: 'integration', ownerAgent: 'twilio-sms-agent', description: 'Processes verified inbound SMS messages through normalization, session routing, and response generation pipeline.' },
  { num: 54, id: 'sms-delivery-reconciliation-loop-v1', displayName: 'SMS Delivery Reconciliation Loop', category: 'integration-twilio', dependencyLayer: 'integration', ownerAgent: 'twilio-sms-agent', description: 'Reconciles SMS delivery status callbacks with sent messages using message SID and status timestamp dedup.' },
  { num: 55, id: 'voice-call-orchestration-loop-v1', displayName: 'Voice Call Orchestration Loop', category: 'integration-voice', dependencyLayer: 'integration', ownerAgent: 'twilio-voice-agent', description: 'Orchestrates inbound and outbound voice calls through Twilio Calls API with TwiML flow or stream bridge.' },
  { num: 56, id: 'voice-callback-reconciliation-loop-v1', displayName: 'Voice Callback Reconciliation Loop', category: 'integration-voice', dependencyLayer: 'integration', ownerAgent: 'twilio-voice-agent', description: 'Reconciles voice call status callbacks as idempotent state transitions in the call lifecycle state machine.' },
  // --- Browser loops (57-60) ---
  { num: 57, id: 'browser-session-lifecycle-loop-v1', displayName: 'Browser Session Lifecycle Loop', category: 'integration-browser', dependencyLayer: 'integration', ownerAgent: 'browser-control-agent', description: 'Manages isolated browser profile lifecycle including creation, suspension, resumption, and teardown.' },
  { num: 58, id: 'browser-workflow-executor-loop-v1', displayName: 'Browser Workflow Executor Loop', category: 'integration-browser', dependencyLayer: 'integration', ownerAgent: 'browser-control-agent', description: 'Executes deterministic browser workflow scripts with navigation, form interaction, and data extraction steps.' },
  { num: 59, id: 'browser-artifact-verification-loop-v1', displayName: 'Browser Artifact Verification Loop', category: 'integration-browser', dependencyLayer: 'integration', ownerAgent: 'browser-control-agent', description: 'Verifies browser action outcomes by comparing screenshots, DOM digests, and action transcripts against expectations.' },
  { num: 60, id: 'browser-prompt-injection-scanner-loop-v1', displayName: 'Browser Prompt-Injection Scanner Loop', category: 'integration-browser', dependencyLayer: 'integration', ownerAgent: 'browser-risk-agent', description: 'Scans browser page content for prompt injection attempts before allowing agent interaction with page elements.' },
  // --- Voice/STT/TTS loops (61-65) ---
  { num: 61, id: 'stt-stream-processing-loop-v1', displayName: 'STT Stream Processing Loop', category: 'integration-voice', dependencyLayer: 'integration', ownerAgent: 'stt-agent', description: 'Processes real-time speech-to-text audio streams with chunked transcription and confidence scoring.' },
  { num: 62, id: 'tts-synthesis-loop-v1', displayName: 'TTS Synthesis Loop', category: 'integration-voice', dependencyLayer: 'integration', ownerAgent: 'tts-agent', description: 'Synthesizes text responses into audio output using configured voice persona with latency optimization.' },
  { num: 63, id: 'wakeword-arbitration-loop-v1', displayName: 'Wakeword Arbitration Loop', category: 'integration-voice', dependencyLayer: 'integration', ownerAgent: 'wake-word-agent', description: 'Arbitrates wake word detection events filtering false positives and routing confirmed activations to talk loop.' },
  { num: 64, id: 'voice-barge-in-coordination-loop-v1', displayName: 'Voice Barge-In Coordination Loop', category: 'integration-voice', dependencyLayer: 'integration', ownerAgent: 'interruption-agent', description: 'Coordinates barge-in interruptions during voice interactions ensuring safe conversation state handoff.' },
  { num: 65, id: 'voice-transcript-redaction-loop-v1', displayName: 'Voice Transcript Redaction Loop', category: 'integration-voice', dependencyLayer: 'integration', ownerAgent: 'voice-safety-agent', description: 'Applies PII and sensitive content redaction to voice transcripts before memory writeback and storage.' },
  // --- Security and trust loops (66-75) ---
  { num: 66, id: 'pairing-challenge-trust-loop-v1', displayName: 'Pairing Challenge and Trust Loop', category: 'security', dependencyLayer: 'security', ownerAgent: 'auth-boundary-agent', description: 'Executes device/sender pairing challenge protocol establishing trust relationship for unknown senders.' },
  { num: 67, id: 'mention-gate-enforcement-loop-v1', displayName: 'Mention-Gate Enforcement Loop', category: 'security', dependencyLayer: 'security', ownerAgent: 'access-policy-agent', description: 'Enforces mention-gating policy in group contexts requiring explicit @mention for non-allowlisted senders.' },
  { num: 68, id: 'privilege-escalation-approval-loop-v1', displayName: 'Privilege Escalation Approval Loop', category: 'security', dependencyLayer: 'security', ownerAgent: 'elevated-exec-guard-agent', description: 'Gates privilege escalation requests through explicit human approval workflow with command audit trail.' },
  { num: 69, id: 'command-risk-scoring-loop-v1', displayName: 'Command Risk Scoring Loop', category: 'security', dependencyLayer: 'security', ownerAgent: 'elevated-exec-guard-agent', description: 'Scores incoming commands for risk level based on tool tier, target scope, and historical pattern analysis.' },
  { num: 70, id: 'secret-rotation-loop-v1', displayName: 'Secret Rotation Loop', category: 'security', dependencyLayer: 'security', ownerAgent: 'secret-hygiene-agent', description: 'Rotates secrets and credentials on schedule or on-demand with target completion under 15 minutes.' },
  { num: 71, id: 'exfiltration-guard-loop-v1', displayName: 'Exfiltration Guard Loop', category: 'security', dependencyLayer: 'security', ownerAgent: 'secret-hygiene-agent', description: 'Monitors data flow patterns for potential exfiltration attempts and blocks unauthorized data transfers.' },
  { num: 72, id: 'access-anomaly-detection-loop-v1', displayName: 'Access Anomaly Detection Loop', category: 'security', dependencyLayer: 'security', ownerAgent: 'threat-intel-agent', description: 'Detects anomalous access patterns including unusual tool spikes and off-hours privileged operations.' },
  { num: 73, id: 'incident-containment-loop-v1', displayName: 'Incident Containment Loop', category: 'security', dependencyLayer: 'security', ownerAgent: 'incident-response-agent', description: 'Executes contain, rotate, audit incident response procedures with one-command connector kill switch.' },
  { num: 74, id: 'forensics-package-builder-loop-v1', displayName: 'Forensics Package Builder Loop', category: 'security', dependencyLayer: 'security', ownerAgent: 'incident-response-agent', description: 'Assembles forensics package with command history, loop state snapshots, and channel event traces for investigation.' },
  { num: 75, id: 'policy-drift-audit-loop-v1', displayName: 'Policy Drift Audit Loop', category: 'security', dependencyLayer: 'security', ownerAgent: 'security-audit-agent', description: 'Audits active security policies against baseline detecting unauthorized drift in tool tiers and access rules.' },
  // --- Reliability and operations loops (76-90) ---
  { num: 76, id: 'health-probing-loop-v1', displayName: 'Health Probing Loop', category: 'reliability', dependencyLayer: 'reliability', ownerAgent: 'observability-agent', description: 'Probes all service components for liveness and readiness with structured health check reports every 5 minutes.' },
  { num: 77, id: 'watchdog-recovery-loop-v1', displayName: 'Watchdog Recovery Loop', category: 'reliability', dependencyLayer: 'reliability', ownerAgent: 'observability-agent', description: 'Executes watchdog recovery with max 5 restart attempts in 15 minutes before entering degraded mode.' },
  { num: 78, id: 'restart-governor-loop-v1', displayName: 'Restart Governor Loop', category: 'reliability', dependencyLayer: 'reliability', ownerAgent: 'observability-agent', description: 'Governs restart decisions with exponential backoff and circuit breaker preventing restart storm cascades.' },
  { num: 79, id: 'backup-snapshot-loop-v1', displayName: 'Backup Snapshot Loop', category: 'reliability', dependencyLayer: 'reliability', ownerAgent: 'observability-agent', description: 'Creates periodic state backup snapshots of config, memory, and session data for disaster recovery.' },
  { num: 80, id: 'restore-validation-loop-v1', displayName: 'Restore Validation Loop', category: 'reliability', dependencyLayer: 'reliability', ownerAgent: 'observability-agent', description: 'Validates backup integrity and restore procedures during monthly disaster recovery simulation drills.' },
  { num: 81, id: 'log-compaction-loop-v1', displayName: 'Log Compaction Loop', category: 'reliability', dependencyLayer: 'reliability', ownerAgent: 'observability-agent', description: 'Compacts and archives old log entries during daily maintenance windows to manage storage growth.' },
  { num: 82, id: 'metrics-export-loop-v1', displayName: 'Metrics Export Loop', category: 'reliability', dependencyLayer: 'reliability', ownerAgent: 'observability-agent', description: 'Exports runtime metrics (loop runs, queue depth, tool failures, approval delays) to observability pipeline.' },
  { num: 83, id: 'slo-burn-rate-alert-loop-v1', displayName: 'SLO Burn-Rate Alert Loop', category: 'reliability', dependencyLayer: 'reliability', ownerAgent: 'slo-agent', description: 'Monitors SLO burn rate against error budgets: command success >= 99%, loop latency <= 30s, cron reliability >= 99.5%.' },
  { num: 84, id: 'cost-budget-control-loop-v1', displayName: 'Cost Budget Control Loop', category: 'reliability', dependencyLayer: 'reliability', ownerAgent: 'cost-controller-agent', description: 'Tracks token and infrastructure spend against configured budgets triggering alerts and auto-throttling.' },
  { num: 85, id: 'latency-regression-detection-loop-v1', displayName: 'Latency Regression Detection Loop', category: 'reliability', dependencyLayer: 'reliability', ownerAgent: 'observability-agent', description: 'Detects latency regression trends across loop execution and model inference response times.' },
  { num: 86, id: 'capacity-forecast-loop-v1', displayName: 'Capacity Forecast Loop', category: 'reliability', dependencyLayer: 'reliability', ownerAgent: 'observability-agent', description: 'Projects resource utilization trends and forecasts capacity needs for proactive scaling decisions.' },
  { num: 87, id: 'canary-release-validation-loop-v1', displayName: 'Canary Release Validation Loop', category: 'reliability', dependencyLayer: 'reliability', ownerAgent: 'release-agent', description: 'Validates canary deployments by comparing canary metrics against baseline with automated rollback trigger.' },
  { num: 88, id: 'rollback-verification-loop-v1', displayName: 'Rollback Verification Loop', category: 'reliability', dependencyLayer: 'reliability', ownerAgent: 'rollback-agent', description: 'Verifies rollback completeness after deployment reversal ensuring system state matches pre-deployment baseline.' },
  { num: 89, id: 'chaos-drill-loop-v1', displayName: 'Chaos Drill Loop', category: 'reliability', dependencyLayer: 'reliability', ownerAgent: 'chaos-agent', description: 'Executes controlled failure injection drills validating system resilience and recovery procedures.' },
  { num: 90, id: 'postmortem-synthesis-loop-v1', displayName: 'Postmortem Synthesis Loop', category: 'reliability', dependencyLayer: 'reliability', ownerAgent: 'postmortem-agent', description: 'Synthesizes incident postmortem reports with root cause analysis and corrective action tracking.' },
  // --- Delivery and quality engineering loops (91-95) ---
  { num: 91, id: 'regression-suite-orchestration-loop-v1', displayName: 'Regression Suite Orchestration Loop', category: 'quality', dependencyLayer: 'governance', ownerAgent: 'unit-test-agent', description: 'Orchestrates full regression test suite execution after code changes or deployment events.' },
  { num: 92, id: 'test-gap-discovery-loop-v1', displayName: 'Test Gap Discovery Loop', category: 'quality', dependencyLayer: 'governance', ownerAgent: 'unit-test-agent', description: 'Identifies gaps in test coverage by analyzing code paths lacking assertion coverage.' },
  { num: 93, id: 'config-drift-reconciliation-loop-v1', displayName: 'Config Drift Reconciliation Loop', category: 'quality', dependencyLayer: 'governance', ownerAgent: 'observability-agent', description: 'Detects and reconciles configuration drift between running state and declared config baseline.' },
  { num: 94, id: 'dependency-vulnerability-tracking-loop-v1', displayName: 'Dependency Vulnerability Tracking Loop', category: 'quality', dependencyLayer: 'governance', ownerAgent: 'security-audit-agent', description: 'Tracks known vulnerabilities in project dependencies and generates upgrade recommendations.' },
  { num: 95, id: 'schema-migration-safety-loop-v1', displayName: 'Schema Migration Safety Loop', category: 'quality', dependencyLayer: 'governance', ownerAgent: 'unit-test-agent', description: 'Validates schema migrations for backward compatibility and data integrity before application.' },
  // --- Autonomy governance loops (96-105) ---
  { num: 96, id: 'proposal-aggregation-loop-v1', displayName: 'Proposal Aggregation Loop', category: 'governance', dependencyLayer: 'governance', ownerAgent: 'portfolio-orchestrator-agent', description: 'Aggregates autonomous task proposals from self-driven loops into unified review queue for prioritization.' },
  { num: 97, id: 'approval-queue-manager-loop-v1', displayName: 'Approval Queue Manager Loop', category: 'governance', dependencyLayer: 'governance', ownerAgent: 'change-board-agent', description: 'Manages the approval queue processing pending approvals with timeout escalation and priority ordering.' },
  { num: 98, id: 'human-in-the-loop-timeout-escalation-loop-v1', displayName: 'Human-in-the-Loop Timeout Escalation Loop', category: 'governance', dependencyLayer: 'governance', ownerAgent: 'change-board-agent', description: 'Escalates stale human approval requests after configurable timeout with increasing urgency notifications.' },
  { num: 99, id: 'safe-mode-guardian-loop-v1', displayName: 'Safe-Mode Guardian Loop', category: 'governance', dependencyLayer: 'governance', ownerAgent: 'evolution-governor-agent', description: 'Monitors system health and autonomy boundaries enforcing safe-mode restrictions when violations detected.' },
  { num: 100, id: 'goal-priority-rebalancer-loop-v1', displayName: 'Goal Priority Rebalancer Loop', category: 'governance', dependencyLayer: 'governance', ownerAgent: 'portfolio-orchestrator-agent', description: 'Rebalances goal priorities across active loops based on changing context and resource availability.' },
  { num: 101, id: 'objective-conflict-resolver-loop-v1', displayName: 'Objective Conflict Resolver Loop', category: 'governance', dependencyLayer: 'governance', ownerAgent: 'portfolio-orchestrator-agent', description: 'Detects and resolves conflicting objectives between concurrent autonomous loops using priority arbitration.' },
  { num: 102, id: 'learning-quality-auditor-loop-v1', displayName: 'Learning Quality Auditor Loop', category: 'governance', dependencyLayer: 'governance', ownerAgent: 'evolution-governor-agent', description: 'Audits quality of self-learning outputs ensuring memory cards and playbooks meet accuracy standards.' },
  { num: 103, id: 'self-change-rollback-loop-v1', displayName: 'Self-Change Rollback Loop', category: 'governance', dependencyLayer: 'governance', ownerAgent: 'evolution-governor-agent', description: 'Rolls back self-applied changes when regression detected after self-update or self-upgrade execution.' },
  { num: 104, id: 'knowledge-decay-cleanup-loop-v1', displayName: 'Knowledge Decay Cleanup Loop', category: 'governance', dependencyLayer: 'governance', ownerAgent: 'knowledge-curator-agent', description: 'Identifies and archives decayed knowledge entries that are outdated or contradicted by newer evidence.' },
  { num: 105, id: 'weekly-strategy-renewal-loop-v1', displayName: 'Weekly Strategy Renewal Loop', category: 'governance', dependencyLayer: 'governance', ownerAgent: 'portfolio-orchestrator-agent', description: 'Conducts weekly strategic review and renewal cycle updating long-term objectives and resource allocation.' },
];

// ─────────────────────────────────────────────────────────────────────────────
// Derived loop defaults per category
// ─────────────────────────────────────────────────────────────────────────────

const CATEGORY_DEFAULTS = {
  'runtime': {
    triggerTypes: ['event'], triggerClass: 'always-on',
    cadence: { default: 'continuous' },
    thinkingPolicy: { default: 'medium', escalateTo: 'high', escalateOn: ['repeated_failure'], deescalateOn: ['budget_pressure'] },
    allowedTools: ['memory.search', 'metrics.query', 'config.read'],
    forbiddenTools: ['host.exec', 'browser.navigate', 'email.send'],
    toolTierBudget: 'tier-0',
    maxSteps: 10, maxDurationSec: 120,
    riskTier: 'safe', approvalRequired: false, approvalPoints: [],
    rollbackPlan: 'Revert to previous state and emit error event.',
    terminationPolicy: 'retry_then_escalate',
    writebacks: { memory: false, logs: true, actionArtifacts: false, auditTrail: true },
    outputContract: { summaryRequired: true, confidenceRequired: false, evidenceRequired: false, actionsRequired: false, nextRequired: false },
    loopLifecycleStates: STANDARD_LIFECYCLE,
    guardrails: { confidenceThreshold: 0.5, toolFailureBudget: 3, sideEffectApprovalRequired: false },
  },
  'context-memory': {
    triggerTypes: ['event', 'cron'], triggerClass: 'scheduled',
    cadence: { default: 'daily' },
    thinkingPolicy: { default: 'medium', escalateTo: 'high', escalateOn: ['data_inconsistency'], deescalateOn: ['budget_pressure'] },
    allowedTools: ['memory.search', 'memory.write', 'file.read', 'config.read'],
    forbiddenTools: ['host.exec', 'browser.navigate', 'email.send'],
    toolTierBudget: 'tier-0',
    maxSteps: 14, maxDurationSec: 300,
    riskTier: 'safe', approvalRequired: false, approvalPoints: [],
    rollbackPlan: 'Discard pending writes and retain current state.',
    terminationPolicy: 'safe_stop_and_report',
    writebacks: { memory: true, logs: true, actionArtifacts: false, auditTrail: false },
    outputContract: { summaryRequired: true, confidenceRequired: false, evidenceRequired: false, actionsRequired: false, nextRequired: true },
    loopLifecycleStates: STANDARD_LIFECYCLE,
    guardrails: { confidenceThreshold: 0.5, toolFailureBudget: 3, sideEffectApprovalRequired: false },
  },
  'integration-gmail': {
    triggerTypes: ['event'], triggerClass: 'event-triggered',
    cadence: { default: 'event-driven' },
    thinkingPolicy: { default: 'medium', escalateTo: 'high', escalateOn: ['auth_failure'], deescalateOn: ['budget_pressure'] },
    allowedTools: ['gmail.api', 'memory.write', 'config.read'],
    forbiddenTools: ['host.exec', 'browser.navigate'],
    toolTierBudget: 'tier-1',
    maxSteps: 12, maxDurationSec: 180,
    riskTier: 'safe', approvalRequired: false, approvalPoints: [],
    rollbackPlan: 'Abort Gmail operation and log failure event.',
    terminationPolicy: 'retry_then_escalate',
    writebacks: { memory: true, logs: true, actionArtifacts: true, auditTrail: true },
    outputContract: { summaryRequired: true, confidenceRequired: false, evidenceRequired: false, actionsRequired: true, nextRequired: false },
    loopLifecycleStates: STANDARD_LIFECYCLE,
    guardrails: { confidenceThreshold: 0.5, toolFailureBudget: 3, sideEffectApprovalRequired: false },
  },
  'integration-twilio': {
    triggerTypes: ['event'], triggerClass: 'event-triggered',
    cadence: { default: 'event-driven' },
    thinkingPolicy: { default: 'medium', escalateTo: 'high', escalateOn: ['signature_mismatch'], deescalateOn: ['budget_pressure'] },
    allowedTools: ['twilio.api', 'memory.write', 'config.read'],
    forbiddenTools: ['host.exec', 'browser.navigate'],
    toolTierBudget: 'tier-1',
    maxSteps: 12, maxDurationSec: 180,
    riskTier: 'safe', approvalRequired: false, approvalPoints: [],
    rollbackPlan: 'Abort Twilio operation and log security event if signature invalid.',
    terminationPolicy: 'retry_then_escalate',
    writebacks: { memory: true, logs: true, actionArtifacts: true, auditTrail: true },
    outputContract: { summaryRequired: true, confidenceRequired: false, evidenceRequired: false, actionsRequired: true, nextRequired: false },
    loopLifecycleStates: STANDARD_LIFECYCLE,
    guardrails: { confidenceThreshold: 0.5, toolFailureBudget: 3, sideEffectApprovalRequired: false },
  },
  'integration-browser': {
    triggerTypes: ['event', 'manual'], triggerClass: 'event-triggered',
    cadence: { default: 'on-demand' },
    thinkingPolicy: { default: 'medium', escalateTo: 'high', escalateOn: ['page_injection_detected'], deescalateOn: ['budget_pressure'] },
    allowedTools: ['browser.navigate', 'browser.screenshot', 'browser.readDom', 'memory.write'],
    forbiddenTools: ['host.exec', 'email.send', 'browser.formSubmit'],
    toolTierBudget: 'tier-1',
    maxSteps: 15, maxDurationSec: 300,
    riskTier: 'moderate', approvalRequired: false, approvalPoints: ['form_submission', 'purchase_action'],
    rollbackPlan: 'Close browser session and discard pending actions.',
    terminationPolicy: 'safe_stop_and_report',
    writebacks: { memory: true, logs: true, actionArtifacts: true, auditTrail: true },
    outputContract: { summaryRequired: true, confidenceRequired: true, evidenceRequired: true, actionsRequired: true, nextRequired: false },
    loopLifecycleStates: STANDARD_LIFECYCLE,
    guardrails: { confidenceThreshold: 0.6, toolFailureBudget: 3, sideEffectApprovalRequired: true },
  },
  'integration-voice': {
    triggerTypes: ['event'], triggerClass: 'event-triggered',
    cadence: { default: 'event-driven' },
    thinkingPolicy: { default: 'medium', escalateTo: 'high', escalateOn: ['transcription_confidence_low'], deescalateOn: ['budget_pressure'] },
    allowedTools: ['voice.stream', 'stt.transcribe', 'tts.synthesize', 'memory.write'],
    forbiddenTools: ['host.exec', 'browser.navigate', 'email.send'],
    toolTierBudget: 'tier-1',
    maxSteps: 12, maxDurationSec: 300,
    riskTier: 'safe', approvalRequired: false, approvalPoints: [],
    rollbackPlan: 'End voice stream gracefully and log incomplete interaction.',
    terminationPolicy: 'safe_stop_and_report',
    writebacks: { memory: true, logs: true, actionArtifacts: true, auditTrail: false },
    outputContract: { summaryRequired: true, confidenceRequired: true, evidenceRequired: false, actionsRequired: false, nextRequired: false },
    loopLifecycleStates: STANDARD_LIFECYCLE,
    guardrails: { confidenceThreshold: 0.5, toolFailureBudget: 3, sideEffectApprovalRequired: false },
  },
  'security': {
    triggerTypes: ['event'], triggerClass: 'always-on',
    cadence: { default: 'continuous' },
    thinkingPolicy: { default: 'high', escalateTo: 'xhigh', escalateOn: ['active_threat_detected'], deescalateOn: ['false_positive_confirmed'] },
    allowedTools: ['memory.search', 'memory.write', 'config.read', 'metrics.query', 'auth.verify'],
    forbiddenTools: ['host.exec', 'browser.navigate'],
    toolTierBudget: 'tier-1',
    maxSteps: 15, maxDurationSec: 300,
    riskTier: 'moderate', approvalRequired: false, approvalPoints: ['privilege_escalation'],
    rollbackPlan: 'Revert to last known secure state and alert operator.',
    terminationPolicy: 'hard_stop_and_alert',
    writebacks: { memory: true, logs: true, actionArtifacts: true, auditTrail: true },
    outputContract: { summaryRequired: true, confidenceRequired: true, evidenceRequired: true, actionsRequired: true, nextRequired: true },
    loopLifecycleStates: STANDARD_LIFECYCLE,
    guardrails: { confidenceThreshold: 0.7, toolFailureBudget: 2, sideEffectApprovalRequired: true },
  },
  'reliability': {
    triggerTypes: ['cron', 'event'], triggerClass: 'scheduled',
    cadence: { default: 'daily' },
    thinkingPolicy: { default: 'medium', escalateTo: 'high', escalateOn: ['slo_breach_imminent'], deescalateOn: ['budget_pressure'] },
    allowedTools: ['metrics.query', 'memory.write', 'config.read', 'health.check'],
    forbiddenTools: ['host.exec', 'browser.navigate', 'email.send'],
    toolTierBudget: 'tier-0',
    maxSteps: 14, maxDurationSec: 300,
    riskTier: 'safe', approvalRequired: false, approvalPoints: [],
    rollbackPlan: 'Log failure and defer to next scheduled run.',
    terminationPolicy: 'safe_stop_and_report',
    writebacks: { memory: false, logs: true, actionArtifacts: true, auditTrail: true },
    outputContract: { summaryRequired: true, confidenceRequired: false, evidenceRequired: true, actionsRequired: false, nextRequired: true },
    loopLifecycleStates: STANDARD_LIFECYCLE,
    guardrails: { confidenceThreshold: 0.5, toolFailureBudget: 3, sideEffectApprovalRequired: false },
  },
  'quality': {
    triggerTypes: ['event', 'cron'], triggerClass: 'event-triggered',
    cadence: { default: 'on-change' },
    thinkingPolicy: { default: 'medium', escalateTo: 'high', escalateOn: ['test_failure_spike'], deescalateOn: ['budget_pressure'] },
    allowedTools: ['test.run', 'file.read', 'memory.write', 'config.read'],
    forbiddenTools: ['host.exec', 'browser.navigate', 'email.send'],
    toolTierBudget: 'tier-0',
    maxSteps: 16, maxDurationSec: 360,
    riskTier: 'safe', approvalRequired: false, approvalPoints: [],
    rollbackPlan: 'Log test results and defer to next quality gate.',
    terminationPolicy: 'safe_stop_and_report',
    writebacks: { memory: true, logs: true, actionArtifacts: true, auditTrail: false },
    outputContract: { summaryRequired: true, confidenceRequired: false, evidenceRequired: true, actionsRequired: true, nextRequired: true },
    loopLifecycleStates: STANDARD_LIFECYCLE,
    guardrails: { confidenceThreshold: 0.5, toolFailureBudget: 3, sideEffectApprovalRequired: false },
  },
  'governance': {
    triggerTypes: ['event', 'cron'], triggerClass: 'scheduled',
    cadence: { default: 'daily' },
    thinkingPolicy: { default: 'high', escalateTo: 'xhigh', escalateOn: ['policy_violation_detected'], deescalateOn: ['budget_pressure'] },
    allowedTools: ['memory.search', 'memory.write', 'config.read', 'tasks.read'],
    forbiddenTools: ['host.exec', 'browser.navigate', 'email.send', 'config.write'],
    toolTierBudget: 'tier-1',
    maxSteps: 14, maxDurationSec: 300,
    riskTier: 'moderate', approvalRequired: false, approvalPoints: ['policy_change'],
    rollbackPlan: 'Retain current governance policy and log audit failure.',
    terminationPolicy: 'safe_stop_and_report',
    writebacks: { memory: true, logs: true, actionArtifacts: true, auditTrail: true },
    outputContract: { summaryRequired: true, confidenceRequired: true, evidenceRequired: true, actionsRequired: true, nextRequired: true },
    loopLifecycleStates: STANDARD_LIFECYCLE,
    guardrails: { confidenceThreshold: 0.6, toolFailureBudget: 2, sideEffectApprovalRequired: true },
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// Per-loop overrides for specific derived loops
// ─────────────────────────────────────────────────────────────────────────────

const DERIVED_OVERRIDES = {
  19: { triggerClass: 'always-on', riskTier: 'safe' },
  20: { triggerClass: 'always-on', riskTier: 'safe' },
  21: { triggerClass: 'always-on', riskTier: 'safe' },
  22: { triggerClass: 'always-on', riskTier: 'safe' },
  23: { triggerClass: 'always-on', riskTier: 'safe' },
  24: { triggerClass: 'always-on', riskTier: 'safe' },
  28: { triggerClass: 'always-on', cadence: { default: '30m' } },
  30: { triggerClass: 'always-on' },
  31: { triggerClass: 'always-on' },
  32: { triggerClass: 'always-on' },
  33: { triggerClass: 'always-on', riskTier: 'safe' },
  35: { triggerClass: 'always-on', riskTier: 'safe' },
  38: { triggerClass: 'always-on' },
  39: { triggerClass: 'always-on' },
  40: { triggerClass: 'always-on' },
  47: { cadence: { default: 'daily' }, triggerTypes: ['cron'], triggerClass: 'scheduled' },
  48: { triggerClass: 'always-on' },
  49: { triggerClass: 'event-triggered' },
  50: { triggerTypes: ['event'], triggerClass: 'event-triggered' },
  51: { triggerTypes: ['event', 'manual'], riskTier: 'moderate', approvalRequired: true, approvalPoints: ['gmail_action_execution'] },
  52: { triggerClass: 'always-on' },
  53: { triggerClass: 'always-on' },
  54: { triggerClass: 'event-triggered' },
  55: { triggerClass: 'always-on' },
  57: { triggerClass: 'always-on' },
  58: { triggerClass: 'event-triggered' },
  59: { triggerClass: 'event-triggered' },
  60: { triggerClass: 'always-on', riskTier: 'safe' },
  61: { triggerClass: 'always-on' },
  62: { triggerClass: 'always-on' },
  66: { triggerClass: 'always-on' },
  67: { triggerClass: 'always-on' },
  68: { triggerClass: 'always-on', riskTier: 'high', approvalRequired: true },
  69: { triggerClass: 'always-on' },
  70: { triggerClass: 'always-on', riskTier: 'high' },
  71: { triggerClass: 'always-on' },
  73: { triggerClass: 'always-on', riskTier: 'high' },
  76: { triggerClass: 'always-on', cadence: { default: '5m' } },
  77: { triggerClass: 'always-on' },
  81: { cadence: { default: 'daily' } },
  83: { triggerClass: 'always-on' },
  91: { triggerClass: 'event-triggered' },
  97: { triggerClass: 'always-on' },
  99: { triggerClass: 'always-on' },
};

// ─────────────────────────────────────────────────────────────────────────────
// Per-loop entry/success criteria (unique per loop)
// ─────────────────────────────────────────────────────────────────────────────

function getEntryCriteria(num, def) {
  const map = {
    16: ['raw_command_or_message_received_at_gateway'],
    17: ['normalized_message_available_for_classification'],
    18: ['intent_classified_and_routing_policy_loaded'],
    19: ['routed_message_requires_session_key_resolution'],
    20: ['session_key_resolved_and_isolation_check_required'],
    21: ['task_submitted_to_queue_for_admission_check'],
    22: ['queue_depth_exceeds_backpressure_threshold'],
    23: ['command_execution_failed_with_retryable_error'],
    24: ['message_exhausted_retry_budget_in_dead_letter_queue'],
    25: ['presence_entry_approaching_ttl_expiration'],
    26: ['runtime_event_emitted_matching_registered_hook'],
    27: ['hook_dispatch_failed_with_retryable_error'],
    28: ['heartbeat_interval_elapsed_since_last_synthesis'],
    29: ['heartbeat_missed_beyond_grace_period_threshold'],
    30: ['cron_job_scheduled_time_reached'],
    31: ['system_restart_detected_with_missed_cron_windows'],
    32: ['cron_job_requires_isolated_session_execution'],
    33: ['model_request_failed_on_primary_provider'],
    34: ['new_request_requires_thinking_level_selection'],
    35: ['execution_context_requires_tool_eligibility_check'],
    36: ['prompt_assembly_approaching_token_budget_limit'],
    37: ['context_state_checkpoint_interval_reached'],
    38: ['loop_output_or_insight_ready_for_persistence'],
    39: ['memory_entries_retrieved_pending_validation'],
    40: ['context_pressure_exceeds_compaction_threshold'],
    41: ['fragmented_memory_entries_exceed_consolidation_threshold'],
    42: ['identity_behavior_samples_collected_for_drift_check'],
    43: ['agent_execution_requires_soul_boundary_verification'],
    44: ['user_interaction_data_available_for_profile_refresh'],
    45: ['interaction_pattern_data_accumulated_for_learning'],
    46: ['cross_session_context_requires_continuity_check'],
    47: ['gmail_watch_expiration_approaching_renewal_window'],
    48: ['gmail_pubsub_push_notification_received'],
    49: ['gmail_sync_job_enqueued_with_valid_history_id'],
    50: ['gmail_history_api_returned_404_stale_history'],
    51: ['approved_gmail_action_queued_for_execution'],
    52: ['twilio_webhook_request_received_pending_verification'],
    53: ['verified_inbound_sms_message_ready_for_processing'],
    54: ['sms_delivery_status_callback_received'],
    55: ['voice_call_event_received_for_orchestration'],
    56: ['voice_call_status_callback_received_for_reconciliation'],
    57: ['browser_session_action_requested_create_or_teardown'],
    58: ['browser_workflow_script_queued_for_execution'],
    59: ['browser_action_completed_pending_artifact_verification'],
    60: ['browser_page_loaded_pending_injection_scan'],
    61: ['audio_stream_data_available_for_transcription'],
    62: ['text_response_ready_for_voice_synthesis'],
    63: ['wake_word_detection_event_received_for_arbitration'],
    64: ['barge_in_interrupt_signal_detected_during_speech'],
    65: ['voice_transcript_ready_for_redaction_processing'],
    66: ['unknown_sender_contact_attempt_detected'],
    67: ['group_message_received_without_explicit_mention'],
    68: ['privilege_escalation_request_submitted'],
    69: ['incoming_command_requires_risk_assessment'],
    70: ['secret_rotation_schedule_reached_or_breach_detected'],
    71: ['data_transfer_pattern_matches_exfiltration_heuristic'],
    72: ['access_pattern_deviates_from_baseline_profile'],
    73: ['security_incident_detected_requiring_containment'],
    74: ['incident_containment_complete_forensics_required'],
    75: ['policy_audit_schedule_reached_or_drift_signal'],
    76: ['health_check_interval_elapsed_for_component_probing'],
    77: ['component_health_check_failed_recovery_needed'],
    78: ['restart_requested_for_failed_component'],
    79: ['backup_schedule_reached_for_state_snapshot'],
    80: ['disaster_recovery_drill_scheduled_for_restore_test'],
    81: ['log_maintenance_window_reached_for_compaction'],
    82: ['metrics_export_interval_elapsed'],
    83: ['slo_burn_rate_check_interval_elapsed'],
    84: ['cost_tracking_interval_elapsed_for_budget_check'],
    85: ['latency_measurement_window_completed_for_analysis'],
    86: ['capacity_forecast_interval_reached'],
    87: ['canary_deployment_initiated_pending_validation'],
    88: ['rollback_executed_pending_state_verification'],
    89: ['chaos_drill_window_open_for_failure_injection'],
    90: ['incident_resolved_postmortem_synthesis_required'],
    91: ['code_change_or_deployment_event_triggers_regression'],
    92: ['coverage_report_indicates_test_gaps'],
    93: ['config_state_check_interval_elapsed'],
    94: ['dependency_scan_interval_reached_or_advisory_received'],
    95: ['schema_change_proposed_pending_safety_validation'],
    96: ['autonomous_task_proposals_accumulated_for_review'],
    97: ['approval_requests_pending_in_queue'],
    98: ['human_approval_request_exceeded_timeout_threshold'],
    99: ['system_health_or_autonomy_boundary_check_required'],
    100: ['goal_context_change_detected_requiring_rebalance'],
    101: ['concurrent_loop_objectives_conflict_detected'],
    102: ['self_learning_outputs_accumulated_for_audit'],
    103: ['regression_detected_after_self_applied_change'],
    104: ['knowledge_entry_age_exceeds_decay_threshold'],
    105: ['weekly_strategy_review_interval_reached'],
  };
  return map[num] || [`loop_${num}_entry_condition_met`];
}

function getSuccessCriteria(num, def) {
  const map = {
    16: ['message_normalized_to_canonical_format'],
    17: ['intent_classification_completed_with_confidence'],
    18: ['route_resolved_to_target_agent_and_session'],
    19: ['session_key_resolved_using_configured_strategy'],
    20: ['session_isolation_verified_no_leakage_detected'],
    21: ['task_admitted_to_queue_within_concurrency_limits'],
    22: ['queue_depth_reduced_below_safe_threshold'],
    23: ['retry_completed_successfully_or_budget_exhausted'],
    24: ['dead_letter_message_processed_or_archived'],
    25: ['presence_entries_refreshed_stale_pruned'],
    26: ['hook_callbacks_dispatched_for_matching_events'],
    27: ['failed_hooks_retried_or_escalated'],
    28: ['heartbeat_event_synthesized_and_dispatched'],
    29: ['missed_heartbeat_escalated_to_incident_event'],
    30: ['cron_job_dispatched_to_target_session'],
    31: ['missed_cron_jobs_caught_up_and_executed'],
    32: ['isolated_session_bootstrapped_for_cron_job'],
    33: ['model_request_routed_to_fallback_provider'],
    34: ['thinking_level_selected_for_request_context'],
    35: ['tool_eligibility_resolved_for_execution_scope'],
    36: ['token_budget_allocated_within_model_limits'],
    37: ['context_snapshot_captured_and_indexed'],
    38: ['loop_output_persisted_to_memory_and_vector_store'],
    39: ['retrieved_memory_validated_for_relevance_and_accuracy'],
    40: ['compaction_triggered_context_pressure_reduced'],
    41: ['fragmented_memories_consolidated_into_structures'],
    42: ['identity_drift_assessment_completed_with_score'],
    43: ['soul_boundary_constraints_verified_no_violations'],
    44: ['user_profile_refreshed_from_latest_interactions'],
    45: ['preference_model_updated_from_patterns'],
    46: ['cross_session_continuity_context_verified'],
    47: ['gmail_watch_renewed_before_expiration'],
    48: ['gmail_notification_acked_and_sync_job_enqueued'],
    49: ['mailbox_delta_synced_via_history_api'],
    50: ['full_mailbox_resync_completed_baseline_reset'],
    51: ['gmail_action_executed_with_delivery_confirmation'],
    52: ['twilio_webhook_signature_validated'],
    53: ['inbound_sms_processed_through_session_pipeline'],
    54: ['sms_delivery_status_reconciled_with_sent_messages'],
    55: ['voice_call_orchestrated_through_twiml_flow'],
    56: ['voice_callback_reconciled_in_call_state_machine'],
    57: ['browser_session_lifecycle_action_completed'],
    58: ['browser_workflow_executed_successfully'],
    59: ['browser_artifacts_verified_against_expectations'],
    60: ['page_content_scanned_no_injection_threats_found'],
    61: ['audio_stream_transcribed_with_confidence_scores'],
    62: ['text_synthesized_to_audio_with_target_latency'],
    63: ['wake_word_arbitrated_activation_routed'],
    64: ['barge_in_handled_conversation_state_preserved'],
    65: ['transcript_redacted_pii_removed_before_storage'],
    66: ['pairing_challenge_completed_trust_established'],
    67: ['mention_gate_enforced_unauthorized_messages_blocked'],
    68: ['privilege_escalation_approved_or_denied_with_audit'],
    69: ['command_risk_score_computed_and_routing_decided'],
    70: ['secrets_rotated_within_target_time_window'],
    71: ['exfiltration_attempt_blocked_and_logged'],
    72: ['access_anomaly_assessment_completed'],
    73: ['incident_contained_kill_switch_engaged_if_needed'],
    74: ['forensics_package_assembled_for_investigation'],
    75: ['policy_audit_completed_drift_report_generated'],
    76: ['all_components_probed_health_report_generated'],
    77: ['failed_component_recovered_or_degraded_mode_entered'],
    78: ['restart_decision_made_with_backoff_circuit_check'],
    79: ['state_backup_snapshot_created_and_verified'],
    80: ['restore_procedure_validated_during_drill'],
    81: ['old_log_entries_compacted_and_archived'],
    82: ['runtime_metrics_exported_to_observability_pipeline'],
    83: ['slo_burn_rate_assessed_alerts_fired_if_needed'],
    84: ['cost_budget_assessed_throttling_applied_if_needed'],
    85: ['latency_regression_assessment_completed'],
    86: ['capacity_forecast_generated_with_projections'],
    87: ['canary_metrics_compared_rollback_decided_if_needed'],
    88: ['rollback_state_verified_matches_pre_deployment'],
    89: ['chaos_drill_completed_resilience_report_generated'],
    90: ['postmortem_report_synthesized_with_action_items'],
    91: ['regression_suite_executed_results_reported'],
    92: ['test_gaps_identified_and_prioritized'],
    93: ['config_drift_detected_and_reconciliation_proposed'],
    94: ['vulnerability_report_generated_with_recommendations'],
    95: ['schema_migration_validated_for_compatibility'],
    96: ['proposals_aggregated_into_unified_review_queue'],
    97: ['approval_queue_processed_decisions_dispatched'],
    98: ['stale_approvals_escalated_with_notifications'],
    99: ['safe_mode_assessment_completed_restrictions_applied'],
    100: ['goal_priorities_rebalanced_across_active_loops'],
    101: ['objective_conflicts_resolved_via_arbitration'],
    102: ['learning_outputs_audited_quality_verified'],
    103: ['self_change_rolled_back_regression_resolved'],
    104: ['decayed_knowledge_entries_archived_or_removed'],
    105: ['weekly_strategy_renewed_objectives_updated'],
  };
  return map[num] || [`loop_${num}_completed_successfully`];
}

// ─────────────────────────────────────────────────────────────────────────────
// Fallback chain mapping (derived loops reference core loops)
// ─────────────────────────────────────────────────────────────────────────────

function getFallbackChain(num, category) {
  const chainMap = {
    'runtime': ['retry-orchestration-loop-v1'],
    'context-memory': ['research-loop-v1'],
    'integration-gmail': ['research-loop-v1'],
    'integration-twilio': ['research-loop-v1'],
    'integration-browser': ['research-loop-v1'],
    'integration-voice': [],
    'security': ['incident-containment-loop-v1'],
    'reliability': ['watchdog-recovery-loop-v1'],
    'quality': ['research-loop-v1'],
    'governance': ['planning-loop-v1'],
  };
  // Some specific overrides
  const specific = {
    22: ['dead-letter-recovery-loop-v1'],
    23: ['dead-letter-recovery-loop-v1'],
    24: [],
    29: ['incident-containment-loop-v1'],
    33: [],
    50: ['gmail-history-delta-sync-loop-v1'],
    73: [],
    77: ['restart-governor-loop-v1'],
    78: ['incident-containment-loop-v1'],
    88: ['chaos-drill-loop-v1'],
    103: [],
  };
  if (specific[num] !== undefined) return specific[num];
  return chainMap[category] || [];
}

// ─────────────────────────────────────────────────────────────────────────────
// Composition helpers (who can call whom)
// ─────────────────────────────────────────────────────────────────────────────

function getComposition(num, id) {
  // Core loops have rich composition in their definitions.
  // Derived loops get minimal composition.
  return {
    canBeCalledBy: ['ralph-loop-v1'],
    canCall: [],
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Build a full contract from a derived definition
// ─────────────────────────────────────────────────────────────────────────────

function buildDerivedContract(def) {
  const defaults = CATEGORY_DEFAULTS[def.category];
  if (!defaults) throw new Error(`No category defaults for: ${def.category}`);

  const overrides = DERIVED_OVERRIDES[def.num] || {};
  const tier = TIER_A_NUMBERS.has(def.num) ? 'A' : 'B';

  const contract = {
    loopId: def.id,
    version: '1.0.0',
    displayName: def.displayName,
    description: def.description,
    ownerAgent: def.ownerAgent,
    category: def.category,
    tier,
    dependencyLayer: def.dependencyLayer,
    priority: Math.min(20 + Math.floor(def.num * 0.7), 100),
    triggerTypes: overrides.triggerTypes || defaults.triggerTypes,
    triggerClass: overrides.triggerClass || defaults.triggerClass,
    cadence: overrides.cadence || defaults.cadence,
    entryCriteria: getEntryCriteria(def.num),
    hardStops: [`budget_exhausted`, `operator_abort`, `${def.id.replace(/-v\d+$/, '')}_critical_failure`],
    requiredContext: ['AGENTS.md', `${def.category}_context`],
    thinkingPolicy: defaults.thinkingPolicy,
    allowedTools: defaults.allowedTools,
    forbiddenTools: defaults.forbiddenTools,
    toolTierBudget: defaults.toolTierBudget,
    approvalPoints: overrides.approvalPoints || defaults.approvalPoints,
    approvalRequired: overrides.approvalRequired ?? defaults.approvalRequired,
    maxSteps: defaults.maxSteps,
    maxDurationSec: defaults.maxDurationSec,
    riskTier: overrides.riskTier || defaults.riskTier,
    successCriteria: getSuccessCriteria(def.num),
    rollbackPlan: defaults.rollbackPlan,
    fallbackLoopChain: getFallbackChain(def.num, def.category),
    terminationPolicy: defaults.terminationPolicy,
    writebacks: defaults.writebacks,
    outputContract: defaults.outputContract,
    loopLifecycleStates: defaults.loopLifecycleStates,
    guardrails: defaults.guardrails,
    composition: getComposition(def.num, def.id),
  };

  return contract;
}

// ─────────────────────────────────────────────────────────────────────────────
// Build a full contract from a core loop definition
// ─────────────────────────────────────────────────────────────────────────────

function buildCoreContract(def) {
  const { num, id, ...rest } = def;
  return {
    loopId: id,
    version: '1.0.0',
    tier: 'core',
    ...rest,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Build registry entry
// ─────────────────────────────────────────────────────────────────────────────

function buildRegistryEntry(contract) {
  return {
    loopId: contract.loopId,
    displayName: contract.displayName,
    tier: contract.tier,
    category: contract.category,
    loopClass: contract.tier === 'core' ? 'core' : 'derived',
    riskTier: contract.riskTier,
    triggerClass: contract.triggerClass,
    dependencyEdges: contract.fallbackLoopChain || [],
    contractFile: `loops/contracts/${contract.loopId}.json`,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Main generation
// ─────────────────────────────────────────────────────────────────────────────

function main() {
  console.log('Generating AlwaysClaw loop artifacts...');

  // Ensure directories exist
  const contractsDir = join(ROOT, 'loops', 'contracts');
  if (!existsSync(contractsDir)) {
    mkdirSync(contractsDir, { recursive: true });
  }

  // Build all contracts
  const allContracts = [];

  // Core loops (1-15)
  for (const def of CORE_LOOPS) {
    allContracts.push(buildCoreContract(def));
  }

  // Derived loops (16-105)
  for (const def of DERIVED_LOOPS) {
    allContracts.push(buildDerivedContract(def));
  }

  console.log(`Built ${allContracts.length} contracts`);

  // Write individual contract files
  for (const contract of allContracts) {
    const filePath = join(contractsDir, `${contract.loopId}.json`);
    writeFileSync(filePath, JSON.stringify(contract, null, 2) + '\n', 'utf-8');
  }
  console.log(`Wrote ${allContracts.length} contract files to loops/contracts/`);

  // Build registry
  const registryEntries = allContracts.map(buildRegistryEntry);

  const registry = {
    $schema: '../schema/loop.contract.schema.json',
    version: '1.0.0',
    generatedAt: new Date().toISOString(),
    counts: {
      total: registryEntries.length,
      core: registryEntries.filter(e => e.tier === 'core').length,
      derived: registryEntries.filter(e => e.tier !== 'core').length,
      tierA: registryEntries.filter(e => e.tier === 'A').length,
      tierB: registryEntries.filter(e => e.tier === 'B').length,
    },
    dependencyLayers: [
      'strategic', 'knowledge', 'delivery', 'autonomy',
      'runtime-intake', 'runtime-orchestration', 'context-memory',
      'integration', 'security', 'reliability', 'governance'
    ],
    loops: registryEntries,
  };

  const registryPath = join(ROOT, 'loops', 'registry.json');
  writeFileSync(registryPath, JSON.stringify(registry, null, 2) + '\n', 'utf-8');
  console.log(`Wrote registry with ${registryEntries.length} entries to loops/registry.json`);

  // Summary
  console.log('\n--- Generation Summary ---');
  console.log(`Total contracts: ${allContracts.length}`);
  console.log(`Core: ${registry.counts.core}`);
  console.log(`Tier A: ${registry.counts.tierA}`);
  console.log(`Tier B: ${registry.counts.tierB}`);
  console.log(`Total: ${registry.counts.core + registry.counts.tierA + registry.counts.tierB}`);
  console.log('Done.');
}

main();
