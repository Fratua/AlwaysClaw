/**
 * @module event-envelope
 *
 * Typed event envelope system for AlwaysClaw internal service communication.
 * All inter-service events are wrapped in an {@link EventEnvelope} to support
 * replay, forensics, tracing, and deterministic ordering.
 *
 * Design reference: Master Plan Section 24 - typed event envelopes across
 * internal services.
 */

// ---------------------------------------------------------------------------
// Event Priority
// ---------------------------------------------------------------------------

/**
 * Priority level for event processing. Higher priorities are dequeued first
 * when backpressure or lane contention occurs.
 */
export type EventPriority = 'critical' | 'high' | 'medium' | 'low';

// ---------------------------------------------------------------------------
// Event Type
// ---------------------------------------------------------------------------

/**
 * Exhaustive union of all event types emitted within the AlwaysClaw runtime.
 *
 * Naming convention: `<domain>.<verb>` where domain maps to a runtime
 * subsystem and verb describes the lifecycle transition.
 */
export type EventType =
  // Command lifecycle
  | 'command.received'
  | 'command.queued'
  | 'command.started'
  | 'command.completed'
  | 'command.failed'
  // Loop lifecycle
  | 'loop.queued'
  | 'loop.preflight'
  | 'loop.context-build'
  | 'loop.reason'
  | 'loop.act'
  | 'loop.verify'
  | 'loop.commit'
  | 'loop.reflect'
  | 'loop.archive'
  | 'loop.awaiting-approval'
  | 'loop.failed'
  // Session lifecycle
  | 'session.created'
  | 'session.resumed'
  | 'session.compacted'
  // Memory lifecycle
  | 'memory.written'
  | 'memory.retrieved'
  | 'memory.compacted'
  // Heartbeat
  | 'heartbeat.tick'
  | 'heartbeat.ok'
  // Cron
  | 'cron.dispatched'
  | 'cron.completed'
  // Incident management
  | 'incident.created'
  | 'incident.contained'
  | 'incident.resolved'
  // Approval workflow
  | 'approval.requested'
  | 'approval.granted'
  | 'approval.denied'
  // Health monitoring
  | 'health.check'
  | 'health.degraded'
  | 'health.recovered';

// ---------------------------------------------------------------------------
// Event Metadata
// ---------------------------------------------------------------------------

/**
 * Optional tracing and priority metadata attached to an event envelope.
 * Supports distributed tracing via OpenTelemetry-compatible span identifiers.
 */
export interface EventMetadata {
  /** Distributed trace identifier (W3C Trace Context compatible). */
  traceId: string;

  /** Span identifier for this specific event within the trace. */
  spanId: string;

  /** Parent span identifier for causal chaining. */
  parentSpanId?: string;

  /** Processing priority for this event. */
  priority: EventPriority;

  /** Time-to-live in milliseconds. Events older than TTL may be discarded. */
  ttlMs?: number;
}

// ---------------------------------------------------------------------------
// Event Envelope
// ---------------------------------------------------------------------------

/**
 * The canonical event envelope that wraps every inter-service message in
 * the AlwaysClaw runtime.
 *
 * @typeParam T - The shape of the domain-specific payload.
 *
 * @example
 * ```ts
 * const envelope: EventEnvelope<{ commandText: string }> = {
 *   eventId: '550e8400-e29b-41d4-a716-446655440000',
 *   timestamp: '2026-02-06T12:00:00.000Z',
 *   source: 'alwaysclaw-gateway',
 *   type: 'command.received',
 *   sessionKey: 'main',
 *   payload: { commandText: 'check email' },
 * };
 * ```
 */
export interface EventEnvelope<T> {
  /** Globally unique event identifier (UUID v4). */
  eventId: string;

  /** ISO 8601 timestamp of event creation. */
  timestamp: string;

  /** Service or component that produced the event. */
  source: string;

  /** Discriminated event type from the {@link EventType} union. */
  type: EventType;

  /** Agent that originated or is the target of the event. */
  agentId?: string;

  /** Session key that scopes this event's processing context. */
  sessionKey: string;

  /** Correlation identifier for grouping related events across services. */
  correlationId?: string;

  /** Loop run identifier when the event is part of a loop execution. */
  loopRunId?: string;

  /** Domain-specific payload. */
  payload: T;

  /** Optional tracing and priority metadata. */
  metadata?: EventMetadata;
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/**
 * Create a new {@link EventEnvelope} with an auto-generated UUID v4 eventId
 * and an ISO 8601 timestamp set to the current time.
 *
 * @typeParam T - The shape of the domain-specific payload.
 * @param fields - All envelope fields except `eventId` and `timestamp`.
 * @returns A fully-formed event envelope ready for dispatch.
 */
export function createEventEnvelope<T>(
  fields: Omit<EventEnvelope<T>, 'eventId' | 'timestamp'>,
): EventEnvelope<T> {
  return {
    eventId: crypto.randomUUID(),
    timestamp: new Date().toISOString(),
    ...fields,
  };
}

// ---------------------------------------------------------------------------
// Type Guard
// ---------------------------------------------------------------------------

/**
 * Runtime type guard that validates whether an unknown value conforms to the
 * {@link EventEnvelope} shape. Checks all required top-level fields for
 * presence and correct primitive types.
 *
 * @param value - The value to validate.
 * @returns `true` if `value` satisfies the EventEnvelope contract.
 */
export function isValidEventEnvelope(value: unknown): value is EventEnvelope<unknown> {
  if (typeof value !== 'object' || value === null) return false;

  const obj = value as Record<string, unknown>;

  return (
    typeof obj.eventId === 'string' &&
    typeof obj.timestamp === 'string' &&
    typeof obj.source === 'string' &&
    typeof obj.type === 'string' &&
    typeof obj.sessionKey === 'string' &&
    'payload' in obj
  );
}
