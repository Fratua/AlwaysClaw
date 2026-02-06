/**
 * @module queue-lanes
 *
 * Queue lane types and configuration for the AlwaysClaw command queue.
 *
 * The runtime separates traffic into named lanes to prevent starvation
 * and enforce per-lane concurrency limits. The three built-in lanes are
 * `main`, `subagent`, and dynamic `session:<key>` lanes.
 *
 * Design reference: Master Plan Sections 5, 32, 40 - queue lanes,
 * concurrency defaults, and backpressure strategy.
 */

// ---------------------------------------------------------------------------
// Lane Names
// ---------------------------------------------------------------------------

/**
 * Named queue lane. `main` and `subagent` are well-known fixed lanes;
 * any other lane is a per-session dynamic lane using the template literal
 * pattern `session:${string}`.
 */
export type QueueLaneName = 'main' | 'subagent' | `session:${string}`;

// ---------------------------------------------------------------------------
// Lane Configuration
// ---------------------------------------------------------------------------

/**
 * Configuration for a single queue lane.
 */
export interface QueueLaneConfig {
  /** Lane identifier. */
  name: QueueLaneName;

  /**
   * Maximum number of concurrent commands that may execute in this lane.
   * Source-locked defaults: main=4, subagent=8, unconfigured session lanes=1.
   */
  concurrency: number;

  /** Priority class for this lane when competing for shared resources. */
  priority: 'critical' | 'high' | 'medium' | 'low';

  /** Maximum number of queued (waiting) commands before admission is refused. */
  maxQueueDepth: number;

  /**
   * Queue depth at which backpressure signaling begins. Must be less than
   * or equal to {@link maxQueueDepth}.
   */
  backpressureThreshold: number;
}

// ---------------------------------------------------------------------------
// Default Configurations
// ---------------------------------------------------------------------------

/**
 * Source-locked default lane configurations derived from verified OpenClaw
 * queue behavior (Master Plan Section 32/40/47).
 *
 * - `main` concurrency: 4
 * - `subagent` concurrency: 8
 * - default session lane concurrency: 1
 */
export const DEFAULT_LANE_CONFIGS: ReadonlyArray<QueueLaneConfig> = [
  {
    name: 'main',
    concurrency: 4,
    priority: 'high',
    maxQueueDepth: 64,
    backpressureThreshold: 48,
  },
  {
    name: 'subagent',
    concurrency: 8,
    priority: 'medium',
    maxQueueDepth: 128,
    backpressureThreshold: 96,
  },
  {
    name: 'session:default',
    concurrency: 1,
    priority: 'medium',
    maxQueueDepth: 16,
    backpressureThreshold: 12,
  },
] as const;

// ---------------------------------------------------------------------------
// Admission Decision
// ---------------------------------------------------------------------------

/**
 * Result of the queue admission control evaluation.
 *
 * - `admit`  - command is immediately dispatched for execution.
 * - `queue`  - command is accepted into the waiting queue.
 * - `reject` - command is rejected due to backpressure or policy.
 */
export type QueueAdmissionDecision = 'admit' | 'queue' | 'reject';

// ---------------------------------------------------------------------------
// Backpressure Policy
// ---------------------------------------------------------------------------

/**
 * Defines behavior when a queue lane exceeds its backpressure threshold.
 */
export interface QueueBackpressurePolicy {
  /**
   * Strategy to apply when queue depth exceeds the threshold.
   *
   * - `drop-oldest`      - evict the oldest queued command.
   * - `reject-new`       - refuse new commands until depth decreases.
   * - `signal-producer`  - notify the producing service to slow down.
   */
  strategy: 'drop-oldest' | 'reject-new' | 'signal-producer';

  /** Maximum time in milliseconds a command may wait in the queue. */
  maxWaitMs: number;

  /**
   * Callback invoked when the backpressure threshold is breached.
   *
   * @param laneName  - The lane that breached its threshold.
   * @param depth     - Current queue depth at the time of breach.
   */
  onThresholdBreached: (laneName: QueueLaneName, depth: number) => void;
}

// ---------------------------------------------------------------------------
// Lane Name Resolver
// ---------------------------------------------------------------------------

/**
 * Resolves an arbitrary string into a well-known {@link QueueLaneName}.
 *
 * - `"main"` and `"subagent"` pass through unchanged.
 * - Any other string is wrapped as `session:<name>`.
 *
 * @param name - Raw lane name string from the caller.
 * @returns A typed {@link QueueLaneName}.
 */
export function resolveLaneName(name: string): QueueLaneName {
  if (name === 'main' || name === 'subagent') {
    return name;
  }
  return `session:${name}`;
}
