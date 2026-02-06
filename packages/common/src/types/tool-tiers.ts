/**
 * @module tool-tiers
 *
 * Tool tier classification, registry entry types, and policy interfaces
 * for the AlwaysClaw tool execution service.
 *
 * Tools are classified into three tiers based on risk and reversibility.
 * Per-agent-role overrides allow fine-grained control over which tools
 * are available in different trust contexts.
 *
 * Design reference: Master Plan Section 17 - tool tiers, Section 27 -
 * tool policy and agent roles.
 */

// ---------------------------------------------------------------------------
// Tool Tier
// ---------------------------------------------------------------------------

/**
 * Classification tier for a tool based on its risk and reversibility.
 *
 * - `Tier0ReadOnly`     - Read-only tools with no side effects.
 * - `Tier1Reversible`   - Write tools whose effects can be undone.
 * - `Tier2Irreversible` - High-risk tools with irreversible side effects.
 */
export enum ToolTier {
  /** Read-only tools: normal automation, no write side effects. */
  Tier0ReadOnly = 'tier0-readonly',

  /** Reversible write tools: allowed in approved loops. */
  Tier1Reversible = 'tier1-reversible',

  /** Irreversible or high-risk tools: explicit human approval always required. */
  Tier2Irreversible = 'tier2-irreversible',
}

// ---------------------------------------------------------------------------
// Agent Role
// ---------------------------------------------------------------------------

/**
 * Trust level assigned to an agent, controlling its baseline tool access.
 *
 * - `ownerTrustedAgent`   - Full host permissions with approval gates for destructive actions.
 * - `specialistAgent`     - Constrained permissions by tool tier and path allowlist.
 * - `publicFacingAgent`   - No direct host exec; browser read-only unless elevated.
 */
export type AgentRole =
  | 'ownerTrustedAgent'
  | 'specialistAgent'
  | 'publicFacingAgent';

// ---------------------------------------------------------------------------
// Tool Registry Entry
// ---------------------------------------------------------------------------

/**
 * A single entry in the tool registry describing a tool's identity,
 * tier classification, and approval requirements.
 */
export interface ToolRegistryEntry {
  /** Unique tool name (e.g. `"memory.search"`, `"browser.navigate"`). */
  name: string;

  /** Risk tier classification for this tool. */
  tier: ToolTier;

  /** Human-readable description of what the tool does. */
  description: string;

  /**
   * Whether explicit human approval is required before each invocation.
   * Always `true` for {@link ToolTier.Tier2Irreversible} tools.
   */
  requiresApproval: boolean;
}

// ---------------------------------------------------------------------------
// Tool Policy
// ---------------------------------------------------------------------------

/**
 * Per-agent-role tool policy that overrides default tier-based access.
 * Each role maps to a set of explicitly allowed and denied tool names.
 */
export interface ToolPolicy {
  /** The agent role this policy applies to. */
  role: AgentRole;

  /**
   * Tool names explicitly allowed for this role, even if tier would
   * normally restrict access.
   */
  allowedTools: string[];

  /**
   * Tool names explicitly denied for this role, even if tier would
   * normally permit access.
   */
  deniedTools: string[];

  /**
   * Maximum tool tier this role can use without additional approval.
   * Tools above this tier require explicit approval regardless of allowlist.
   */
  maxAutoApprovedTier: ToolTier;
}

// ---------------------------------------------------------------------------
// Tool Eligibility Result
// ---------------------------------------------------------------------------

/**
 * Result of evaluating whether an agent is eligible to invoke a specific tool.
 */
export interface ToolEligibilityResult {
  /** Whether the agent is eligible to use the tool. */
  eligible: boolean;

  /** Human-readable reason when the tool is not eligible. */
  reason?: string;

  /**
   * When `eligible` is `true` but the tool requires approval, this field
   * describes the type of approval needed (e.g. `"human"`, `"policy"`).
   */
  requiredApproval?: string;
}
