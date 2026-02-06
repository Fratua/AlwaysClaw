/**
 * @module session-keys
 *
 * Session strategy types and key derivation for the AlwaysClaw runtime.
 *
 * Sessions scope conversation context, memory, and state isolation.
 * The runtime supports four strategies that determine how session keys
 * are derived from incoming message metadata.
 *
 * Design reference: Master Plan Sections 3, 32, 40 - session strategies,
 * DM scopes, group behavior, and mention gating.
 */

// ---------------------------------------------------------------------------
// Session Strategy
// ---------------------------------------------------------------------------

/**
 * Session isolation strategy that determines how session keys are derived.
 *
 * - `main`                     - single shared session for all DMs.
 * - `per-peer`                 - one session per unique peer identity.
 * - `per-channel-peer`         - one session per (channel, peer) pair.
 * - `per-account-channel-peer` - one session per (account, channel, peer) triple.
 */
export type SessionStrategy =
  | 'main'
  | 'per-peer'
  | 'per-channel-peer'
  | 'per-account-channel-peer';

// ---------------------------------------------------------------------------
// Session Key Parameters
// ---------------------------------------------------------------------------

/**
 * Parameters used to derive a session key from incoming message context.
 * Which fields are used depends on the active {@link SessionStrategy}.
 */
export interface SessionKeyParams {
  /** Unique identifier of the remote peer (sender). */
  peerId?: string;

  /** Channel or group identifier for group-context sessions. */
  channelId?: string;

  /** Account identifier for multi-account routing scenarios. */
  accountId?: string;
}

// ---------------------------------------------------------------------------
// Mention Gating Configuration
// ---------------------------------------------------------------------------

/**
 * Configuration for mention-based gating in group/channel contexts.
 * When enabled, the agent only responds to messages that explicitly
 * mention it, unless the sender is exempted.
 */
export interface MentionGatingConfig {
  /** Whether mention gating is active. */
  enabled: boolean;

  /** Sender identifiers that bypass the mention requirement. */
  exemptSenders: string[];

  /** When true, only respond if the agent is explicitly @-mentioned. */
  requireExplicitMention: boolean;
}

// ---------------------------------------------------------------------------
// Session Configuration
// ---------------------------------------------------------------------------

/**
 * Full session configuration for an agent, controlling DM scope,
 * group scope, and mention gating behavior.
 */
export interface SessionConfig {
  /** The session strategy that governs key derivation. */
  strategy: SessionStrategy;

  /**
   * Session scope applied to direct messages.
   * Source-locked default: `'main'`.
   */
  dmScope: SessionStrategy;

  /**
   * Session scope applied to group/channel messages.
   * Default: `'per-channel-peer'` for isolation per group context.
   */
  groupScope: SessionStrategy;

  /** Mention gating configuration for group contexts. */
  mentionGating: MentionGatingConfig;
}

// ---------------------------------------------------------------------------
// Default Configuration
// ---------------------------------------------------------------------------

/**
 * Source-locked default session configuration derived from verified
 * OpenClaw session behavior (Master Plan Sections 32, 40).
 *
 * - DM scope defaults to `main` (shared primary DM session).
 * - Group scope defaults to `per-channel-peer`.
 * - Mention gating is enabled by default for group contexts.
 */
export const DEFAULT_SESSION_CONFIG: Readonly<SessionConfig> = {
  strategy: 'main',
  dmScope: 'main',
  groupScope: 'per-channel-peer',
  mentionGating: {
    enabled: true,
    exemptSenders: [],
    requireExplicitMention: true,
  },
} as const;

// ---------------------------------------------------------------------------
// Session Key Deriver
// ---------------------------------------------------------------------------

/**
 * Function type that derives a session key string from the active strategy
 * and incoming message parameters.
 *
 * @param strategy - The session strategy to apply.
 * @param params   - Context parameters extracted from the incoming message.
 * @returns The derived session key string.
 */
export type SessionKeyDeriver = (
  strategy: SessionStrategy,
  params: SessionKeyParams,
) => string;
