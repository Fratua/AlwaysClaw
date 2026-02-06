/**
 * @module approval-matrix
 *
 * Approval workflow types for the AlwaysClaw governance layer.
 *
 * Every loop that performs side effects or uses elevated tools must pass
 * through the approval matrix. The matrix maps risk tiers to approval
 * requirements, escalation paths, and dual-checkpoint controls.
 *
 * Design reference: Master Plan Section 58 - approval and risk matrix,
 * Section 45 - self-* loop guardrails.
 */

import { RiskTier } from './loop-contract.js';

// ---------------------------------------------------------------------------
// Approval Decision
// ---------------------------------------------------------------------------

/**
 * Possible outcomes of an approval evaluation.
 */
export enum ApprovalDecision {
  /** The request has been approved for execution. */
  Approved = 'approved',

  /** The request has been denied. */
  Denied = 'denied',

  /** The request has been escalated to a higher authority. */
  Escalated = 'escalated',

  /** The request is awaiting a decision. */
  Pending = 'pending',
}

// ---------------------------------------------------------------------------
// Approval Request
// ---------------------------------------------------------------------------

/**
 * A formal request for approval, generated when a loop reaches an
 * approval checkpoint or attempts to use a restricted tool.
 */
export interface ApprovalRequest {
  /** Unique identifier for this approval request. */
  requestId: string;

  /** The loop that generated this approval request. */
  loopId: string;

  /** The specific loop run instance. */
  loopRunId: string;

  /** Risk tier of the action requiring approval. */
  riskTier: RiskTier;

  /** Agent or service that created the request. */
  requestedBy: string;

  /** ISO 8601 timestamp of when the request was created. */
  requestedAt: string;

  /** List of tool names the loop is requesting to use. */
  toolsRequested: string[];

  /** Description of anticipated side effects. */
  sideEffects: string[];

  /** Current decision status. */
  decision?: ApprovalDecision;

  /** Identity of the approver (human or policy engine). */
  decidedBy?: string;

  /** ISO 8601 timestamp of when the decision was made. */
  decidedAt?: string;
}

// ---------------------------------------------------------------------------
// Escalation Path
// ---------------------------------------------------------------------------

/**
 * Defines the escalation behavior when an approval cannot be resolved
 * at the current level or when failures occur.
 */
export interface EscalationPath {
  /** Action to take on approval failure or timeout. */
  onFailure: 'retry' | 'escalate' | 'abort';

  /** Whether to escalate to a human operator. */
  humanEscalation: boolean;

  /** Maximum retry attempts before escalating or aborting. */
  maxRetries: number;
}

// ---------------------------------------------------------------------------
// Approval Policy
// ---------------------------------------------------------------------------

/**
 * Policy entry mapping a risk tier to its approval requirements,
 * conditions, checkpoint controls, and escalation path.
 */
export interface ApprovalPolicy {
  /** The risk tier this policy applies to. */
  riskTier: RiskTier;

  /**
   * Whether manual approval is required for this risk tier.
   *
   * - `safe`     = no manual approval required.
   * - `moderate` = pre-approved policy route or manual approval.
   * - `high`     = explicit manual approval always required.
   */
  approvalRequired: boolean;

  /** Additional conditions that must be met for automatic approval. */
  conditions: string[];

  /**
   * Whether dual checkpoint verification is required.
   * When true, approval is checked both before execution and after
   * verification but before final commit.
   */
  dualCheckpoint?: boolean;

  /** Escalation behavior when approval is not granted in time. */
  escalationPath: EscalationPath;
}
