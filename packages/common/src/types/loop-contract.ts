/**
 * @module loop-contract
 *
 * TypeScript interfaces mirroring the canonical loop contract JSON schema
 * for the AlwaysClaw hardcoded loop kernel.
 *
 * Every loop is a deterministic state machine with a fixed contract
 * defining its identity, triggers, guardrails, tool budgets, approval
 * requirements, and composition rules.
 *
 * Design reference: Master Plan Sections 6, 18, 26, 36, 37, 49 -
 * loop framework, lifecycle, contracts, composition, and execution matrix.
 */

import { ToolTier } from './tool-tiers.js';

// ---------------------------------------------------------------------------
// Enumerations and Literal Types
// ---------------------------------------------------------------------------

/**
 * All 14 loop categories derived from the 15-loop catalog and dependency
 * graph layers (strategic, knowledge, delivery, autonomy/evolution).
 */
export type LoopCategory =
  | 'strategic'
  | 'knowledge'
  | 'delivery'
  | 'autonomy'
  | 'evolution'
  | 'runtime'
  | 'security'
  | 'operations'
  | 'integration'
  | 'memory'
  | 'context'
  | 'governance'
  | 'quality'
  | 'observation';

/**
 * Loop implementation tier. Tier A loops are mandatory for OpenClaw parity
 * and safety (v1); Tier B loops are advanced autonomy/optimization (post-v1).
 */
export type LoopTier = 'A' | 'B';

/**
 * Dependency layer indicating where this loop sits in the execution topology.
 * Derived from Master Plan Section 37 dependency graph.
 */
export type DependencyLayer =
  | 'intake'
  | 'routing'
  | 'session'
  | 'queue'
  | 'model'
  | 'domain'
  | 'memory'
  | 'verification'
  | 'approval'
  | 'output'
  | 'meta';

/**
 * How a loop can be triggered.
 */
export type TriggerType =
  | 'manual'
  | 'event'
  | 'cron'
  | 'heartbeat'
  | 'self-driven'
  | 'threshold';

/**
 * Trigger class grouping from Master Plan Section 57.
 */
export type TriggerClass =
  | 'always-on'
  | 'event-triggered'
  | 'scheduled'
  | 'manual-scheduled-hybrid';

/**
 * Thinking / reasoning effort level for the model during loop execution.
 */
export type ThinkingLevel = 'medium' | 'high' | 'xhigh';

/**
 * Canonical loop lifecycle states. Every loop transitions through a
 * subset of these states during execution.
 *
 * Design reference: Master Plan Section 18 - canonical loop lifecycle.
 */
export enum LoopLifecycleState {
  Queued = 'queued',
  Preflight = 'preflight',
  ContextBuild = 'context-build',
  Reason = 'reason',
  Act = 'act',
  Verify = 'verify',
  Commit = 'commit',
  Reflect = 'reflect',
  Archive = 'archive',
  AwaitingApproval = 'awaiting-approval',
}

/**
 * Policy for what happens when a loop completes or fails.
 */
export enum TerminationPolicy {
  /** Stop gracefully and emit a structured report. */
  SafeStopAndReport = 'safe_stop_and_report',

  /** Retry from the last checkpoint. */
  RetryFromCheckpoint = 'retry_from_checkpoint',

  /** Chain into a fallback loop. */
  ChainFallback = 'chain_fallback',

  /** Raise an incident and halt. */
  RaiseIncident = 'raise_incident',
}

/**
 * Risk classification for a loop based on its potential side effects.
 */
export enum RiskTier {
  /** Read-only or reversible, no external side effects. */
  Safe = 'safe',

  /** Reversible writes, constrained side effects. */
  Moderate = 'moderate',

  /** Irreversible or privilege-elevating effects. */
  High = 'high',
}

/**
 * Maximum tool tier budget allowed during a loop run.
 * Maps to the three {@link ToolTier} values.
 */
export type ToolTierBudget = 'tier0-readonly' | 'tier1-reversible' | 'tier2-irreversible';

// ---------------------------------------------------------------------------
// Supporting Interfaces
// ---------------------------------------------------------------------------

/**
 * Thinking / reasoning effort policy for a loop, including dynamic
 * escalation rules.
 */
export interface ThinkingPolicy {
  /** Default thinking level for the loop. */
  default: ThinkingLevel;

  /** Level to escalate to when escalation conditions are met. */
  escalateTo?: ThinkingLevel;

  /** Conditions that trigger thinking level escalation. */
  escalateOn?: string[];
}

/**
 * Cadence definition for scheduled loop execution.
 */
export interface LoopCadence {
  /** Cron expression or human-readable interval (e.g. `"30m"`, `"daily"`). */
  schedule: string;

  /** Timezone for schedule interpretation. */
  timezone?: string;

  /** Whether the loop can be suppressed during quiet hours. */
  quietHoursSuppressible?: boolean;
}

/**
 * Defines what a loop writes back to durable storage after execution.
 */
export interface LoopWritebacks {
  /** Whether the loop writes to the memory subsystem. */
  memory: boolean;

  /** Whether the loop writes structured log artifacts. */
  logs: boolean;

  /** Whether the loop produces action artifacts (screenshots, diffs, etc.). */
  actionArtifacts: boolean;

  /** Additional writeback targets. */
  additional?: string[];
}

/**
 * Contract for the structured output a loop must produce.
 */
export interface LoopOutputContract {
  /** Summary of what the loop accomplished. */
  summary: boolean;

  /** Confidence score (0-1) for the loop outcome. */
  confidence: boolean;

  /** Evidence collected during execution. */
  evidence: boolean;

  /** Actions taken or proposed. */
  actions: boolean;

  /** Recommended next steps or follow-up loops. */
  next: boolean;
}

/**
 * Guardrails that constrain loop behavior at runtime.
 */
export interface LoopGuardrails {
  /** Confidence threshold below which the loop auto-falls back. */
  minConfidenceThreshold?: number;

  /** Maximum number of tool failures before the loop raises an incident. */
  maxToolFailures?: number;

  /** Whether unapproved side effects automatically park the loop. */
  parkOnUnapprovedSideEffects: boolean;

  /** Additional custom guardrail descriptions. */
  custom?: string[];
}

/**
 * Composition rules defining how this loop relates to other loops
 * in the execution topology.
 */
export interface LoopComposition {
  /** Loops that may be invoked as sub-loops during execution. */
  canInvoke?: string[];

  /** Loops that must complete before this loop can start. */
  dependsOn?: string[];

  /** Loops that this loop can chain into upon completion. */
  chainsTo?: string[];

  /** Whether this loop acts as a macro-orchestrator. */
  isMacroOrchestrator?: boolean;
}

// ---------------------------------------------------------------------------
// Loop Contract (Main Interface)
// ---------------------------------------------------------------------------

/**
 * The canonical loop contract that fully defines a hardcoded loop in the
 * AlwaysClaw loop kernel. Every loop in the system (core and derived)
 * must conform to this interface.
 *
 * Design reference: Master Plan Sections 6, 18, 26, 36 - loop contracts.
 */
export interface LoopContract {
  /** Unique loop identifier (e.g. `"planning-loop-v1"`). */
  loopId: string;

  /** Semantic version of the loop contract. */
  version: string;

  /** Human-readable display name for the loop. */
  displayName: string;

  /** Description of the loop's purpose and behavior. */
  description: string;

  /** Agent that owns and executes this loop. */
  ownerAgent: string;

  /** Category classification for the loop. */
  category: LoopCategory;

  /** Implementation tier (A = mandatory v1, B = advanced post-v1). */
  tier: LoopTier;

  /** Position in the execution topology dependency graph. */
  dependencyLayer: DependencyLayer;

  /** Execution priority relative to other loops. */
  priority: number;

  /** How this loop can be triggered. */
  triggerTypes: TriggerType[];

  /** Trigger class grouping for scheduling policy. */
  triggerClass: TriggerClass;

  /** Optional cadence for scheduled execution. */
  cadence?: LoopCadence;

  /** Conditions that must be true for the loop to start. */
  entryCriteria: string[];

  /** Conditions that force immediate loop termination. */
  hardStops: string[];

  /** Context files, events, or memory items required before execution. */
  requiredContext: string[];

  /** Thinking / reasoning effort policy. */
  thinkingPolicy: ThinkingPolicy;

  /** Strict allowlist of tools the loop may invoke. */
  allowedTools: string[];

  /** Hard deny list of tools the loop must never invoke. */
  forbiddenTools: string[];

  /** Maximum tool tier permitted during execution. */
  toolTierBudget: ToolTierBudget;

  /** Named checkpoints where side effects require approval. */
  approvalPoints: string[];

  /** Whether this loop requires human approval before commit. */
  approvalRequired: boolean;

  /** Maximum number of discrete steps the loop may execute. */
  maxSteps: number;

  /** Maximum wall-clock duration in seconds for the loop run. */
  maxDurationSec: number;

  /** Risk classification for the loop. */
  riskTier: RiskTier;

  /** Deterministic pass conditions for loop success. */
  successCriteria: string[];

  /** Description of how to undo the loop's effects if needed. */
  rollbackPlan: string;

  /** Ordered list of fallback loop IDs to chain into on failure. */
  fallbackLoopChain: string[];

  /** Policy for what happens when the loop terminates. */
  terminationPolicy: TerminationPolicy;

  /** What the loop writes back to durable storage. */
  writebacks: LoopWritebacks;

  /** Structured output contract the loop must produce. */
  outputContract: LoopOutputContract;

  /** Valid lifecycle states this loop may transition through. */
  loopLifecycleStates: LoopLifecycleState[];

  /** Runtime guardrails constraining loop behavior. */
  guardrails: LoopGuardrails;

  /** Composition rules linking this loop to the broader topology. */
  composition: LoopComposition;
}

// ---------------------------------------------------------------------------
// Type Guard
// ---------------------------------------------------------------------------

/**
 * Runtime type guard that validates whether an unknown value conforms
 * to the {@link LoopContract} shape. Checks all required top-level
 * fields for presence and basic type correctness.
 *
 * @param obj - The value to validate.
 * @returns `true` if `obj` satisfies the LoopContract interface.
 */
export function isValidLoopContract(obj: unknown): obj is LoopContract {
  if (typeof obj !== 'object' || obj === null) return false;

  const o = obj as Record<string, unknown>;

  return (
    typeof o.loopId === 'string' &&
    typeof o.version === 'string' &&
    typeof o.displayName === 'string' &&
    typeof o.description === 'string' &&
    typeof o.ownerAgent === 'string' &&
    typeof o.category === 'string' &&
    typeof o.tier === 'string' &&
    typeof o.dependencyLayer === 'string' &&
    typeof o.priority === 'number' &&
    Array.isArray(o.triggerTypes) &&
    typeof o.triggerClass === 'string' &&
    Array.isArray(o.entryCriteria) &&
    Array.isArray(o.hardStops) &&
    Array.isArray(o.requiredContext) &&
    typeof o.thinkingPolicy === 'object' && o.thinkingPolicy !== null &&
    Array.isArray(o.allowedTools) &&
    Array.isArray(o.forbiddenTools) &&
    typeof o.toolTierBudget === 'string' &&
    Array.isArray(o.approvalPoints) &&
    typeof o.approvalRequired === 'boolean' &&
    typeof o.maxSteps === 'number' &&
    typeof o.maxDurationSec === 'number' &&
    typeof o.riskTier === 'string' &&
    Array.isArray(o.successCriteria) &&
    typeof o.rollbackPlan === 'string' &&
    Array.isArray(o.fallbackLoopChain) &&
    typeof o.terminationPolicy === 'string' &&
    typeof o.writebacks === 'object' && o.writebacks !== null &&
    typeof o.outputContract === 'object' && o.outputContract !== null &&
    Array.isArray(o.loopLifecycleStates) &&
    typeof o.guardrails === 'object' && o.guardrails !== null &&
    typeof o.composition === 'object' && o.composition !== null
  );
}
