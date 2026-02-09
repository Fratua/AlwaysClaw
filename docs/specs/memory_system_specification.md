# MEMORY SYSTEM ARCHITECTURE SPECIFICATION
## Windows 10 OpenClaw-Inspired AI Agent System
### Version 1.0 - Technical Design Document

---

## TABLE OF CONTENTS

1. [Executive Summary](#1-executive-summary)
2. [Memory Architecture Overview](#2-memory-architecture-overview)
3. [Memory File Structures](#3-memory-file-structures)
4. [Episodic Memory System](#4-episodic-memory-system)
5. [Semantic Memory System](#5-semantic-memory-system)
6. [Procedural Memory System](#6-procedural-memory-system)
7. [Memory Consolidation](#7-memory-consolidation)
8. [Memory Retrieval and Search](#8-memory-retrieval-and-search)
9. [Memory Update Mechanisms](#9-memory-update-mechanisms)
10. [Forgetting and Archiving](#10-forgetting-and-archiving)
11. [Context Management](#11-context-management)
12. [Implementation Details](#12-implementation-details)
13. [Windows 10 Specific Considerations](#13-windows-10-specific-considerations)

---

## 1. EXECUTIVE SUMMARY

This specification defines a comprehensive memory system architecture for a Windows 10-based OpenClaw-inspired AI agent. The system implements a **file-first, multi-tier memory architecture** that mirrors human cognitive memory types while optimized for 24/7 autonomous operation with GPT-5.2.

### Key Design Principles

| Principle | Description |
|-----------|-------------|
| **File-First** | Markdown files are the source of truth; databases are indexing layers |
| **Human-Readable** | All memory is inspectable and editable by humans |
| **Version-Controllable** | Git-compatible for tracking memory evolution |
| **Hybrid Search** | Combines BM25 lexical + vector semantic search |
| **Automatic Consolidation** | Pre-compaction memory flush prevents context loss |
| **Multi-Agent Ready** | Per-agent memory isolation with shared context capability |

---

## 2. MEMORY ARCHITECTURE OVERVIEW

### 2.1 Memory Hierarchy

```
+-------------------------------------------------------------+
|                    MEMORY HIERARCHY                          |
+-------------------------------------------------------------+
|                                                               |
|  +-------------------------------------------------------+   |
|  |  TIER 1: WORKING MEMORY (Context Window)               |   |
|  |  - Active conversation context                         |   |
|  |  - Current task state                                  |   |
|  |  - Temporary computation                               |   |
|  |  Size: ~200K tokens (GPT-5.2)                          |   |
|  +-------------------------------------------------------+   |
|                              |                                 |
|  +-------------------------------------------------------+   |
|  |  TIER 2: SHORT-TERM MEMORY (Daily Context)             |   |
|  |  - Today + Yesterday logs (memory/YYYY-MM-DD.md)       |   |
|  |  - Session transcripts                                 |   |
|  |  - Recent interactions                                 |   |
|  |  Retention: 2 days active, 30 days indexed             |   |
|  +-------------------------------------------------------+   |
|                              |                                 |
|  +-------------------------------------------------------+   |
|  |  TIER 3: LONG-TERM MEMORY (Curated Knowledge)          |   |
|  |  - MEMORY.md (decisions, preferences, durable facts)   |   |
|  |  - SEMANTIC.md (facts, concepts, relationships)        |   |
|  |  - PROCEDURAL.md (skills, workflows, procedures)       |   |
|  |  - USER.md (user profile and preferences)              |   |
|  |  Retention: Permanent (with archiving)                 |   |
|  +-------------------------------------------------------+   |
|                              |                                 |
|  +-------------------------------------------------------+   |
|  |  TIER 4: ARCHIVAL MEMORY (Historical Storage)          |   |
|  |  - Compressed daily logs (>30 days)                    |   |
|  |  - Session archives                                    |   |
|  |  - Episodic summaries                                  |   |
|  |  Retention: Indefinite (compressed)                    |   |
|  +-------------------------------------------------------+   |
|                                                               |
+-------------------------------------------------------------+
```

### 2.2 Memory Types Matrix

| Memory Type | Storage Location | Format | Retention | Update Frequency |
|-------------|------------------|--------|-----------|------------------|
| **Working** | RAM/Context | Tokens | Session | Real-time |
| **Episodic** | memory/*.md + SQLite | Markdown + Vectors | 30 days active | Event-driven |
| **Semantic** | SEMANTIC.md + graph | Markdown + Graph | Permanent | Daily consolidation |
| **Procedural** | PROCEDURAL.md | Markdown | Permanent | Weekly refinement |
| **User Profile** | USER.md | Markdown | Permanent | Per-interaction |
| **Identity** | IDENTITY.md + SOUL.md | Markdown | Permanent | Manual |

---

## 3. MEMORY FILE STRUCTURES

### 3.1 Directory Structure

```
C:/OpenClaw/
+-- memory/
|   +-- MEMORY.md              # Curated long-term memory
|   +-- SEMANTIC.md            # Facts, concepts, knowledge graph
|   +-- PROCEDURAL.md          # Skills, workflows, procedures
|   +-- EPISODIC.md            # Significant events and experiences
|   +-- USER.md                # User profile and preferences
|   +-- IDENTITY.md            # Agent identity definition
|   +-- SOUL.md                # Agent persona and behavior
|   +-- AGENTS.md              # Multi-agent coordination
|   +-- 2025-01-15.md          # Daily log (YYYY-MM-DD.md)
|   +-- 2025-01-16.md          # Daily log
|   +-- archive/
|       +-- 2024-12/           # Monthly archives
|       +-- compressed/        # Compressed old memories
+-- sessions/
|   +-- 2025-01-16-morning-check.md
|   +-- 2025-01-16-email-processing.md
|   +-- index.json             # Session index
+-- index/
|   +-- memory.sqlite          # SQLite vector database
|   +-- cache.json             # Embedding cache
|   +-- embeddings/            # Embedding storage
+-- config/
    +-- memory.yaml            # Memory configuration
    +-- retention.yaml         # Retention policies
```

### 3.2 MEMORY.md - Curated Long-Term Memory

```markdown
# Memory

> Curated long-term memory for decisions, preferences, and durable facts.
> Last updated: 2025-01-16 14:32:00 UTC

## Decisions

### Architecture Decisions

- **[2025-01-10]** Use SQLite with sqlite-vec for vector storage (not external DB)
  - Rationale: File-first approach, no external dependencies
  - Impact: Simpler backup, human-readable source of truth
  - Status: Active

- **[2025-01-12]** Implement hybrid search (BM25 + Vector)
  - Rationale: Better precision/recall than either alone
  - Config: BM25 weight 0.3, Vector weight 0.7
  - Status: Active

### User Preferences

- **[2025-01-08]** User prefers email summaries at 9 AM daily
  - Source: User explicitly requested
  - Priority: High
  - Status: Active

- **[2025-01-14]** User dislikes verbose technical explanations
  - Source: Feedback on 2025-01-13 interaction
  - Priority: Medium
  - Status: Active

## Facts

### System Facts

- Windows 10 Pro 22H2 (Build 19045.3803)
- Python 3.11.7
- Node.js 20.10.0
- Primary browser: Chrome 120.0

### User Facts

- Name: [REDACTED]
- Timezone: America/New_York (EST/EDT)
- Preferred communication: Email for async, Voice for urgent
- Work hours: 9 AM - 6 PM EST

## Goals

### Active Goals

- [ ] Implement full Gmail automation by 2025-01-20
- [ ] Set up Twilio voice integration by 2025-01-18
- [ ] Create 15 agentic loops by 2025-01-25

### Completed Goals

- [x] Basic memory system architecture (2025-01-15)
- [x] SQLite vector database setup (2025-01-14)

## Patterns

### Communication Patterns

- User typically checks emails between 8-9 AM
- User prefers actionable summaries over detailed reports
- User responds faster to voice messages than text

### Technical Patterns

- API rate limits: 100 req/min for Gmail, 50 req/min for Twilio
- Retry pattern: 3 attempts with exponential backoff
- Error handling: Log -> Notify -> Fallback

## Relationships

- Gmail API <-> Email processing loop
- Twilio API <-> Voice/SMS loops
- Browser control <-> Web automation loops
```

### 3.3 Daily Log Structure (memory/YYYY-MM-DD.md)

```markdown
# Daily Log: 2025-01-16

> Auto-generated daily activity log
> Agent: OpenClaw-Win10-v1.0

## Summary

- **Active Hours**: 06:00 - 23:45 EST
- **Interactions**: 47
- **Tasks Completed**: 12
- **Errors**: 2 (both recovered)
- **Heartbeat Status**: Healthy

## Timeline

### 06:00 - Morning Startup

- [06:00:15] Heartbeat initiated
- [06:00:30] Loaded MEMORY.md, USER.md, SOUL.md
- [06:00:45] Loaded daily logs: 2025-01-16.md, 2025-01-15.md
- [06:01:00] Gmail connection established (47 unread)
- [06:01:15] Twilio client initialized
- [06:01:30] Browser controller ready

### 06:05 - Morning Email Processing

- [06:05:00] Started email_loop_01 (morning_summary)
- [06:05:15] Filtered 47 emails -> 12 important
- [06:05:30] Generated summary for user
- [06:05:45] Sent summary via preferred channel (email)

### 09:00 - User Interaction

- [09:15:23] User query: "Check my calendar for today"
- [09:15:25] Retrieved calendar data
- [09:15:30] Response: "You have 3 meetings today..."
- [09:15:35] User feedback: Positive

### 12:30 - Afternoon Tasks

- [12:30:00] Cron job: hourly_system_check
- [12:30:05] System status: All green
- [12:30:10] Disk usage: 67% (healthy)
- [12:30:15] Memory usage: 4.2GB / 16GB

### 18:00 - Evening Processing

- [18:00:00] Started evening_summary loop
- [18:00:30] Processed 23 new emails
- [18:01:00] Generated daily report
- [18:01:15] Archived processed emails

## Events

### Significant Events

- **[09:15]** User asked about calendar - first calendar query this week
- **[14:30]** Gmail API rate limit hit - implemented backoff
- **[19:45]** User requested voice call via Twilio

### Errors & Recoveries

- **[14:30:15]** Error: Gmail API rate limit (429)
  - Recovery: Exponential backoff, retry at 14:32
  - Status: Recovered
  
- **[21:15:22]** Error: Browser timeout on page load
  - Recovery: Refreshed connection, retry successful
  - Status: Recovered

## Decisions Made Today

1. **[12:45]** Increased email batch size from 10 to 20
   - Rationale: User has high email volume
   - Impact: Faster processing

2. **[15:30]** Added "urgent" keyword filter
   - Rationale: User missed important email yesterday
   - Impact: Better priority detection

## Context for Tomorrow

- User has important meeting at 10 AM (from calendar)
- 3 emails flagged for follow-up
- System update scheduled for 03:00
- User mentioned interest in Slack integration
```

### 3.4 SEMANTIC.md - Facts and Knowledge

```markdown
# Semantic Memory

> Facts, concepts, and structured knowledge
> Format: Markdown with embedded knowledge graph references

## Entities

### People

#### User
- **Type**: Person
- **Properties**:
  - name: [REDACTED]
  - timezone: America/New_York
  - work_hours: 09:00-18:00
  - preferred_communication: email_async, voice_urgent
- **Relationships**:
  - uses: Gmail, Calendar, Slack
  - manages: Multiple projects

### Systems

#### Gmail API
- **Type**: External Service
- **Properties**:
  - rate_limit: 100 req/min
  - auth_method: OAuth2
  - scopes: readonly, send, modify
- **Relationships**:
  - used_by: email_processing_loop
  - depends_on: Google Cloud Console

#### Twilio
- **Type**: External Service
- **Properties**:
  - rate_limit: 50 req/min
  - capabilities: voice, sms, whatsapp
  - pricing: per-minute voice, per-segment SMS
- **Relationships**:
  - used_by: voice_loop, sms_loop

## Concepts

### Agentic Loops

#### Definition
An agentic loop is a self-contained, repeatable workflow that the agent can execute autonomously.

#### Types
1. **Monitoring Loops**: Continuous observation (email, system health)
2. **Processing Loops**: Handle incoming data (email processing, SMS handling)
3. **Action Loops**: Execute tasks (send messages, make calls)
4. **Maintenance Loops**: System upkeep (cleanup, backups)

#### Current Loops
- email_loop_01 (morning_summary)
- email_loop_02 (continuous_monitor)
- voice_loop_01 (incoming_call_handler)
- system_loop_01 (heartbeat)
- system_loop_02 (cleanup)

### Memory Types

#### Episodic Memory
- Event-based, temporal
- Stores specific experiences
- Decays over time
- Example: "User asked about calendar on 2025-01-16"

#### Semantic Memory
- Fact-based, atemporal
- Stores general knowledge
- Persistent
- Example: "User prefers email summaries at 9 AM"

#### Procedural Memory
- Skill-based
- Stores how-to knowledge
- Improves with practice
- Example: "How to process Gmail inbox efficiently"

## Facts

### System Facts

| Fact | Value | Source | Confidence |
|------|-------|--------|------------|
| OS Version | Windows 10 Pro 22H2 | System query | 100% |
| Python Version | 3.11.7 | sys.version | 100% |
| Gmail Rate Limit | 100 req/min | API docs | 100% |
| User Wake Time | ~06:30 EST | Pattern analysis | 85% |
| User Sleep Time | ~23:00 EST | Pattern analysis | 80% |

### Domain Facts

| Domain | Fact | Evidence |
|--------|------|----------|
| Email | User receives 50-100 emails/day | 30-day average |
| Communication | User responds faster to voice | Response time analysis |
| Schedule | Most meetings on Tue/Thu | Calendar analysis |

## Relationships

```
[User] --uses--> [Gmail]
[User] --prefers--> [Email Summaries @ 9 AM]
[Email Loop] --queries--> [Gmail API]
[Email Loop] --writes--> [Daily Log]
[Daily Log] --feeds--> [Memory Consolidation]
[Memory Consolidation] --updates--> [MEMORY.md]
```

## Ontology

```yaml
entities:
  Person:
    properties: [name, timezone, preferences]
    relationships: [uses, manages, prefers]
  
  Service:
    properties: [rate_limit, auth_method, capabilities]
    relationships: [used_by, depends_on]
  
  Loop:
    properties: [type, frequency, status]
    relationships: [queries, writes, triggers]

relationships:
  uses:
    domain: [Person]
    range: [Service, Tool]
  
  triggers:
    domain: [Event]
    range: [Loop, Action]
```
```

### 3.5 PROCEDURAL.md - Skills and Workflows

```markdown
# Procedural Memory

> Skills, workflows, procedures, and learned patterns
> Version: 1.0 | Last refined: 2025-01-16

## Skills

### Email Processing

#### Skill: Process Gmail Inbox

**Version**: 1.2
**Success Rate**: 94% (last 100 executions)
**Average Time**: 45 seconds

**Procedure**:

1. **Authenticate**
   ```python
   # Check OAuth token validity
   if token_expired():
       refresh_token()
   ```

2. **Fetch Emails**
   ```python
   # Get unread emails from last 24h
   emails = gmail.query(
       query="is:unread newer_than:1d",
       max_results=50
   )
   ```

3. **Classify**
   - Important: From known contacts + keywords
   - Promotional: Unsubscribe links present
   - Spam: ML classifier confidence > 0.9
   - Newsletter: Regular sender pattern

4. **Generate Summary**
   - Group by sender domain
   - Extract action items
   - Flag urgent items

5. **Deliver**
   - Format as markdown
   - Send via preferred channel
   - Archive processed emails

**Error Handling**:
- Rate limit: Exponential backoff (1s, 2s, 4s)
- Auth failure: Trigger re-auth flow
- Timeout: Retry once, then queue

**Optimization Notes**:
- Batch size increased from 10 to 20 (2025-01-16)
- Added "urgent" keyword filter (2025-01-16)
- Cache sender classification for 1 hour

---

### Voice Handling

#### Skill: Handle Incoming Twilio Call

**Version**: 1.0
**Success Rate**: 98%
**Average Time**: 3 minutes

**Procedure**:

1. **Answer Call**
   - Play greeting (TTS)
   - Identify caller from caller ID
   - Load user context

2. **Intent Recognition**
   - STT transcription
   - Intent classification
   - Confidence threshold: 0.7

3. **Execute**
   - Query relevant systems
   - Formulate response
   - TTS response

4. **Follow-up**
   - Log call details
   - Send summary to user
   - Schedule callbacks if needed

---

### System Maintenance

#### Skill: Daily Memory Consolidation

**Version**: 1.0
**Trigger**: Daily at 03:00

**Procedure**:

1. **Load Recent Logs**
   - Today + yesterday's daily logs
   - Last 7 days of session transcripts

2. **Extract Patterns**
   - User behavior patterns
   - System performance trends
   - Error frequency analysis

3. **Update Semantic Memory**
   - New facts -> SEMANTIC.md
   - Updated relationships
   - Confidence adjustments

4. **Update Procedural Memory**
   - Successful procedures -> refine
   - Failed procedures -> flag for review
   - New patterns -> document

5. **Archive Old Data**
   - Compress logs > 30 days
   - Move to archive directory
   - Update indexes

---

## Workflows

### Multi-Step Workflows

#### Workflow: Morning Routine

**Trigger**: 06:00 daily
**Estimated Duration**: 5 minutes

```
[06:00:00] START
    |
[06:00:15] System Check
    |
[06:01:00] Email Summary Generation
    |
[06:02:00] Calendar Check
    |
[06:02:30] Weather & News (if configured)
    |
[06:03:00] Deliver Summary to User
    |
[06:05:00] END
```

#### Workflow: Emergency Response

**Trigger**: "urgent" keyword or user command
**Priority**: Highest
**Estimated Duration**: Variable

```
[DETECT] Urgent signal detected
    |
[INTERRUPT] Pause non-critical loops
    |
[NOTIFY] Alert user via all channels
    |
[GATHER] Collect relevant context
    |
[EXECUTE] Perform requested action
    |
[REPORT] Send completion status
    |
[RESUME] Resume paused loops
```

---

## Decision Trees

### Decision: Which Communication Channel?

```
User Message Received
        |
Is it marked urgent?
    +-- YES -> Use voice call
    | NO
Is user currently active?
    +-- YES -> Use same channel
    | NO
What is user's preference for this time?
    +-- Morning -> Email summary
    +-- Work hours -> Slack/Teams
    +-- Evening -> Deferred to morning
```

### Decision: Which Loop to Execute?

```
Incoming Event
        |
Is it email-related?
    +-- YES -> Email Loop
    | NO
Is it voice/SMS?
    +-- YES -> Twilio Loop
    | NO
Is it system-related?
    +-- YES -> System Loop
    | NO
Is it scheduled?
    +-- YES -> Cron Loop
    | NO
-> Create ad-hoc task
```

---

## Learned Patterns

### Pattern: User Email Checking Behavior

**Observation**: User checks emails in batches
- Morning: 8-9 AM (comprehensive review)
- Lunch: 12-1 PM (quick scan)
- Evening: 6-7 PM (final check)

**Adaptation**: 
- Generate comprehensive summary at 8:30 AM
- Quick digest at 12:30 PM
- Final summary at 6:30 PM

**Confidence**: 87% (based on 30 days of data)

### Pattern: Error Recovery Success

**Observation**: 94% of Gmail rate limit errors recover on first retry

**Adaptation**:
- Default retry: 1 attempt
- Escalate to exponential backoff only on second failure

**Confidence**: 94% (last 50 rate limit events)
```

### 3.6 EPISODIC.md - Events and Experiences

```markdown
# Episodic Memory

> Significant events and experiences with full context
> Format: Structured event log with embeddings

## Recent Episodes (Last 7 Days)

### Episode: 2025-01-16-001

**Timestamp**: 2025-01-16 09:15:23 EST
**Type**: User Interaction
**Importance**: High
**Embedding**: [vector:768]

**Context**:
- User was at desk (mouse activity detected)
- Previous interaction: 2 hours ago
- System status: All green

**Event**:
User asked: "Check my calendar for today"

**Agent Response**:
Retrieved calendar and responded: "You have 3 meetings today:
1. 10:00 AM - Team standup
2. 2:00 PM - Client call
3. 4:30 PM - Project review"

**Outcome**:
- User feedback: Positive ("Thanks!")
- No follow-up needed

**Related Memories**:
- USER.preferences.calendar_access: frequent
- PATTERN.morning_calendar_queries: increasing

---

### Episode: 2025-01-16-002

**Timestamp**: 2025-01-16 14:30:15 EST
**Type**: System Event
**Importance**: Medium
**Embedding**: [vector:768]

**Context**:
- Processing email batch #3 of the day
- 23 emails in current batch
- System load: Normal

**Event**:
Gmail API returned rate limit error (429)

**Agent Response**:
1. Logged error
2. Implemented exponential backoff
3. Retried at 14:32:00
4. Success on retry

**Outcome**:
- Recovery: Successful
- User impact: None (background processing)
- Action taken: Increased backoff delay

**Lessons Learned**:
- Current batch size may be too aggressive
- Consider implementing request queuing

---

## Episode Index

| Date | ID | Type | Importance | Status |
|------|-----|------|------------|--------|
| 2025-01-16 | 001 | User Interaction | High | Active |
| 2025-01-16 | 002 | System Event | Medium | Active |
| 2025-01-15 | 001 | User Request | High | Active |
| 2025-01-15 | 002 | Error Recovery | Low | Archived |
| 2025-01-14 | 001 | Decision | High | Consolidated |

## Episode Retrieval

Episodes can be retrieved by:
1. **Semantic similarity**: Vector search on episode embeddings
2. **Temporal proximity**: Events near a specific date/time
3. **Type filter**: User interaction, system event, error, etc.
4. **Importance threshold**: High/Medium/Low
```

---

## 4. EPISODIC MEMORY SYSTEM

### 4.1 Architecture

```
+-------------------------------------------------------------+
|                    EPISODIC MEMORY SYSTEM                    |
+-------------------------------------------------------------+
|                                                               |
|  +-------------+    +-------------+    +-------------+      |
|  |   Event     |--->|   Episode   |--->|  Embedding  |      |
|  |  Detection  |    |   Creation  |    |  Generation |      |
|  +-------------+    +-------------+    +-------------+      |
|         |                                              |      |
|         |                                       +---------+   |
|         |                                       | SQLite  |   |
|         |                                       |  Store  |   |
|         |                                       +----+----+   |
|         |                                            |        |
|  +-------------+    +-------------+    +-------------+      |
|  |   Action    |<---|   Similar   |<---|   Vector    |      |
|  |   Taken     |    |   Episodes  |    |   Search    |      |
|  +-------------+    +-------------+    +-------------+      |
|                                                               |
+-------------------------------------------------------------+
```

### 4.2 Episode Structure

```python
@dataclass
class Episode:
    """A single episodic memory entry"""
    
    # Identification
    id: str                    # Format: YYYY-MM-DD-NNN
    timestamp: datetime
    
    # Content
    type: EpisodeType          # USER_INTERACTION, SYSTEM_EVENT, ERROR, DECISION
    description: str           # Human-readable summary
    context: Dict[str, Any]    # Full context at time of event
    
    # Metadata
    importance: Importance     # HIGH, MEDIUM, LOW
    embedding: List[float]     # Vector representation (768-dim)
    
    # Relationships
    related_episodes: List[str]  # IDs of related episodes
    consolidated_into: Optional[str]  # Semantic memory entry if consolidated
    
    # Lifecycle
    created_at: datetime
    last_accessed: datetime
    access_count: int

class EpisodeType(Enum):
    USER_INTERACTION = "user_interaction"
    SYSTEM_EVENT = "system_event"
    ERROR = "error"
    DECISION = "decision"
    LEARNING = "learning"

class Importance(Enum):
    HIGH = 3
    MEDIUM = 2
    LOW = 1
```

### 4.3 Episode Creation Pipeline

```python
class EpisodeCreator:
    """Creates and stores new episodic memories"""
    
    async def create_episode(
        self,
        event: Event,
        context: Context
    ) -> Episode:
        """Create a new episode from an event"""
        
        # 1. Determine importance
        importance = self._calculate_importance(event)
        
        # 2. Generate description
        description = await self._generate_description(event, context)
        
        # 3. Create embedding
        embedding = await self._create_embedding(description)
        
        # 4. Find related episodes
        related = await self._find_related(embedding)
        
        # 5. Create episode
        episode = Episode(
            id=self._generate_id(),
            timestamp=datetime.now(),
            type=event.type,
            description=description,
            context=context.to_dict(),
            importance=importance,
            embedding=embedding,
            related_episodes=related,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=0
        )
        
        # 6. Store
        await self._store(episode)
        
        return episode
    
    def _calculate_importance(self, event: Event) -> Importance:
        """Calculate importance score for an event"""
        score = 0
        
        # User interaction importance
        if event.type == EpisodeType.USER_INTERACTION:
            score += 2
            if event.user_feedback == "positive":
                score += 1
        
        # Error importance
        if event.type == EpisodeType.ERROR:
            score += 2
            if event.recovered:
                score -= 1
        
        # Decision importance
        if event.type == EpisodeType.DECISION:
            score += 2
        
        # Time-based decay consideration
        if event.is_recurring:
            score -= 1
        
        return Importance(min(score, 3))
```

### 4.4 Episodic Retrieval

```python
class EpisodicRetrieval:
    """Retrieve relevant episodic memories"""
    
    async def retrieve(
        self,
        query: str,
        context: Optional[Context] = None,
        max_results: int = 5,
        time_window: Optional[timedelta] = None,
        min_importance: Importance = Importance.LOW
    ) -> List[Episode]:
        """Retrieve relevant episodes"""
        
        # 1. Create query embedding
        query_embedding = await self.embedder.embed(query)
        
        # 2. Build SQL query
        sql = """
        SELECT e.*, 
               vec_distance_cosine(e.embedding, ?) as similarity
        FROM episodes e
        WHERE e.importance >= ?
        """
        params = [query_embedding, min_importance.value]
        
        # 3. Add time filter if specified
        if time_window:
            cutoff = datetime.now() - time_window
            sql += " AND e.timestamp >= ?"
            params.append(cutoff)
        
        # 4. Add context filters
        if context:
            if context.user_id:
                sql += " AND json_extract(e.context, '$.user_id') = ?"
                params.append(context.user_id)
        
        # 5. Order and limit
        sql += """
        ORDER BY 
            (similarity * 0.7 + 
             (e.importance / 3.0) * 0.2 + 
             (1.0 / (1 + (julianday('now') - julianday(e.timestamp)))) * 0.1) DESC
        LIMIT ?
        """
        params.append(max_results)
        
        # 6. Execute
        episodes = await self.db.fetchall(sql, params)
        
        # 7. Update access stats
        for episode in episodes:
            await self._update_access_stats(episode.id)
        
        return episodes
```

---

## 5. SEMANTIC MEMORY SYSTEM

### 5.1 Architecture

```
+-------------------------------------------------------------+
|                   SEMANTIC MEMORY SYSTEM                     |
+-------------------------------------------------------------+
|                                                               |
|  +-----------------------------------------------------+    |
|  |              KNOWLEDGE GRAPH LAYER                   |    |
|  |                                                      |    |
|  |   [Entity: User] ----uses----> [Entity: Gmail]      |    |
|  |        |                      |                      |    |
|  |   prefers              rate_limit: 100/min           |    |
|  |        |                      |                      |    |
|  |        v                      v                      |    |
|  |   [Fact: Summary @ 9 AM]  [Fact: OAuth2 Auth]       |    |
|  |                                                      |    |
|  +-----------------------------------------------------+    |
|                          |                                    |
|  +-----------------------------------------------------+    |
|  |              EMBEDDING LAYER                         |    |
|  |                                                      |    |
|  |   Entity embeddings -> Concept embeddings           |    |
|  |   Relationship embeddings -> Fact embeddings        |    |
|  |                                                      |    |
|  +-----------------------------------------------------+    |
|                          |                                    |
|  +-----------------------------------------------------+    |
|  |              STORAGE LAYER                           |    |
|  |                                                      |    |
|  |   SEMANTIC.md (human-readable source)               |    |
|  |   SQLite (vector + graph storage)                   |    |
|  |   Cache (frequent access)                           |    |
|  |                                                      |    |
|  +-----------------------------------------------------+    |
|                                                               |
+-------------------------------------------------------------+
```

### 5.2 Entity-Relationship Model

```python
@dataclass
class Entity:
    """A semantic entity (person, place, thing, concept)"""
    
    id: str
    type: str                    # Person, Service, Concept, etc.
    name: str
    properties: Dict[str, Any]
    embedding: List[float]
    created_at: datetime
    updated_at: datetime
    confidence: float            # 0.0 - 1.0

@dataclass
class Relationship:
    """A relationship between entities"""
    
    id: str
    source_id: str
    target_id: str
    type: str                    # uses, prefers, triggers, etc.
    properties: Dict[str, Any]
    embedding: List[float]
    created_at: datetime
    confidence: float

@dataclass
class Fact:
    """A factual statement"""
    
    id: str
    statement: str
    category: str                # System, User, Domain
    source: str                  # Where this fact came from
    confidence: float
    embedding: List[float]
    created_at: datetime
    last_verified: datetime
    verification_count: int
```

### 5.3 Knowledge Graph Operations

```python
class KnowledgeGraph:
    """Semantic knowledge graph operations"""
    
    async def add_entity(self, entity: Entity) -> None:
        """Add a new entity to the graph"""
        
        # 1. Generate embedding
        entity.embedding = await self.embedder.embed(
            f"{entity.type}: {entity.name}. Properties: {entity.properties}"
        )
        
        # 2. Store in SQLite
        await self.db.execute("""
            INSERT INTO entities (id, type, name, properties, embedding, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [
            entity.id, entity.type, entity.name,
            json.dumps(entity.properties),
            entity.embedding, entity.confidence
        ])
        
        # 3. Update SEMANTIC.md
        await self._update_markdown()
    
    async def query(
        self,
        query_type: QueryType,
        **params
    ) -> List[Union[Entity, Relationship, Fact]]:
        """Query the knowledge graph"""
        
        if query_type == QueryType.ENTITY_BY_NAME:
            return await self._query_entity_by_name(params["name"])
        
        elif query_type == QueryType.RELATED_ENTITIES:
            return await self._query_related_entities(
                params["entity_id"],
                params.get("relationship_type")
            )
        
        elif query_type == QueryType.SEMANTIC_SEARCH:
            return await self._semantic_search(params["query"])
        
        elif query_type == QueryType.PATH_FINDING:
            return await self._find_path(
                params["source_id"],
                params["target_id"]
            )
    
    async def infer(self, premise: str) -> List[Fact]:
        """Infer new facts from existing knowledge"""
        
        # 1. Find relevant entities and relationships
        relevant = await self._semantic_search(premise)
        
        # 2. Build context for inference
        context = self._build_inference_context(relevant)
        
        # 3. Use LLM to generate inferences
        inferences = await self.llm.generate_inference(premise, context)
        
        # 4. Validate and store new facts
        new_facts = []
        for inference in inferences:
            fact = await self._validate_and_create_fact(inference)
            if fact:
                new_facts.append(fact)
        
        return new_facts
```

### 5.4 Fact Verification

```python
class FactVerifier:
    """Verify and validate facts"""
    
    async def verify(
        self,
        fact: Fact,
        verification_method: VerificationMethod
    ) -> VerificationResult:
        """Verify a fact's accuracy"""
        
        if verification_method == VerificationMethod.USER_CONFIRMATION:
            return await self._verify_with_user(fact)
        
        elif verification_method == VerificationMethod.CROSS_REFERENCE:
            return await self._cross_reference(fact)
        
        elif verification_method == VerificationMethod.TEMPORAL_CHECK:
            return await self._temporal_check(fact)
        
        elif verification_method == VerificationMethod.CONSISTENCY_CHECK:
            return await self._consistency_check(fact)
    
    async def _consistency_check(self, fact: Fact) -> VerificationResult:
        """Check if fact is consistent with existing knowledge"""
        
        # 1. Find related facts
        related = await self.kg.query(
            QueryType.SEMANTIC_SEARCH,
            query=fact.statement
        )
        
        # 2. Check for contradictions
        contradictions = []
        for related_fact in related:
            if self._are_contradictory(fact, related_fact):
                contradictions.append(related_fact)
        
        # 3. Calculate consistency score
        if contradictions:
            confidence = 1.0 / (1 + len(contradictions))
            return VerificationResult(
                verified=False,
                confidence=confidence,
                contradictions=contradictions
            )
        
        return VerificationResult(verified=True, confidence=0.9)
```

---

## 6. PROCEDURAL MEMORY SYSTEM

### 6.1 Architecture

```
+-------------------------------------------------------------+
|                  PROCEDURAL MEMORY SYSTEM                    |
+-------------------------------------------------------------+
|                                                               |
|  +-------------+    +-------------+    +-------------+      |
|  |   Skill     |    |  Workflow   |    |  Decision   |      |
|  |   Storage   |    |   Engine    |    |    Tree     |      |
|  +------+------+    +------+------+    +------+------+      |
|         |                  |                  |              |
|         +------------------+------------------+              |
|                            |                                 |
|                   +-----------------+                        |
|                   |  Execution      |                        |
|                   |  Monitor        |                        |
|                   +--------+--------+                        |
|                            |                                 |
|         +------------------+------------------+              |
|         |                  |                  |              |
|  +-------------+    +-------------+    +-------------+      |
|  |  Success    |    |   Failure   |    |  Refinement |      |
|  |  Tracking   |    |   Analysis  |    |   Engine    |      |
|  +-------------+    +-------------+    +-------------+      |
|                                                               |
+-------------------------------------------------------------+
```

### 6.2 Skill Representation

```python
@dataclass
class Skill:
    """A learned skill or procedure"""
    
    id: str
    name: str
    version: str
    description: str
    
    # Procedure
    steps: List[Step]
    preconditions: List[Condition]
    postconditions: List[Condition]
    
    # Performance metrics
    success_rate: float          # 0.0 - 1.0
    execution_count: int
    average_execution_time: timedelta
    last_executed: datetime
    
    # Learning
    created_at: datetime
    refined_at: datetime
    refinement_history: List[Refinement]
    
    # Error handling
    error_patterns: List[ErrorPattern]
    recovery_procedures: Dict[str, RecoveryProcedure]

@dataclass
class Step:
    """A single step in a procedure"""
    
    id: str
    order: int
    description: str
    action: str                    # Code or natural language
    expected_outcome: str
    timeout: timedelta
    
    # Conditional execution
    condition: Optional[str]       # Execute if condition met
    
    # Error handling for this step
    on_error: str                  # retry, skip, abort, fallback

@dataclass
class Refinement:
    """A refinement made to a skill"""
    
    timestamp: datetime
    type: RefinementType           # OPTIMIZATION, BUGFIX, ADAPTATION
    description: str
    before: Optional[str]
    after: str
    trigger: str                   # What caused this refinement
    success_rate_change: float
```

### 6.3 Skill Learning

```python
class SkillLearner:
    """Learn and refine skills from experience"""
    
    async def learn_from_episode(
        self,
        episode: Episode
    ) -> Optional[Skill]:
        """Extract a skill from a successful episode"""
        
        # 1. Check if episode represents a procedure
        if not self._is_procedure_episode(episode):
            return None
        
        # 2. Extract steps
        steps = self._extract_steps(episode)
        
        # 3. Check if similar skill exists
        existing = await self._find_similar_skill(steps)
        
        if existing:
            # 4a. Refine existing skill
            return await self._refine_skill(existing, episode)
        else:
            # 4b. Create new skill
            return await self._create_skill(episode, steps)
    
    async def refine_skill(
        self,
        skill: Skill,
        new_episodes: List[Episode]
    ) -> Skill:
        """Refine a skill based on new experiences"""
        
        # 1. Analyze success/failure patterns
        success_patterns = self._analyze_successes(new_episodes)
        failure_patterns = self._analyze_failures(new_episodes)
        
        # 2. Identify optimization opportunities
        optimizations = self._identify_optimizations(skill, success_patterns)
        
        # 3. Identify bug fixes
        bugfixes = self._identify_bugfixes(skill, failure_patterns)
        
        # 4. Apply refinements
        for optimization in optimizations:
            skill = await self._apply_optimization(skill, optimization)
        
        for bugfix in bugfixes:
            skill = await self._apply_bugfix(skill, bugfix)
        
        # 5. Update metrics
        skill.success_rate = self._calculate_success_rate(new_episodes)
        skill.refined_at = datetime.now()
        
        return skill
    
    def _identify_optimizations(
        self,
        skill: Skill,
        success_patterns: List[Pattern]
    ) -> List[Optimization]:
        """Identify potential optimizations"""
        
        optimizations = []
        
        # Check for redundant steps
        redundant = self._find_redundant_steps(skill, success_patterns)
        if redundant:
            optimizations.append(Optimization(
                type="remove_redundancy",
                steps=redundant,
                expected_improvement="10-20% faster"
            ))
        
        # Check for batching opportunities
        batchable = self._find_batchable_steps(skill, success_patterns)
        if batchable:
            optimizations.append(Optimization(
                type="batch_operations",
                steps=batchable,
                expected_improvement="30-50% fewer API calls"
            ))
        
        # Check for caching opportunities
        cacheable = self._find_cacheable_steps(skill, success_patterns)
        if cacheable:
            optimizations.append(Optimization(
                type="add_caching",
                steps=cacheable,
                expected_improvement="40-60% faster on repeat"
            ))
        
        return optimizations
```

### 6.4 Workflow Execution

```python
class WorkflowEngine:
    """Execute workflows and track performance"""
    
    async def execute(
        self,
        workflow: Workflow,
        context: ExecutionContext
    ) -> ExecutionResult:
        """Execute a workflow"""
        
        execution_id = self._generate_execution_id()
        start_time = datetime.now()
        
        try:
            # 1. Validate preconditions
            if not await self._check_preconditions(workflow, context):
                return ExecutionResult(
                    success=False,
                    error="Preconditions not met"
                )
            
            # 2. Execute steps
            results = []
            for step in workflow.steps:
                result = await self._execute_step(step, context)
                results.append(result)
                
                if not result.success:
                    # Handle step failure
                    recovery = await self._attempt_recovery(step, result)
                    if not recovery.success:
                        return ExecutionResult(
                            success=False,
                            error=f"Step {step.id} failed: {result.error}",
                            partial_results=results
                        )
            
            # 3. Validate postconditions
            if not await self._check_postconditions(workflow, results):
                return ExecutionResult(
                    success=False,
                    error="Postconditions not met",
                    partial_results=results
                )
            
            # 4. Record success
            await self._record_execution(
                workflow, execution_id, start_time, results, success=True
            )
            
            return ExecutionResult(success=True, results=results)
            
        except Exception as e:
            # Record failure
            await self._record_execution(
                workflow, execution_id, start_time, [], success=False, error=str(e)
            )
            return ExecutionResult(success=False, error=str(e))
```

---

## 7. MEMORY CONSOLIDATION

### 7.1 Consolidation Pipeline

```
+-------------------------------------------------------------+
|                  MEMORY CONSOLIDATION                        |
+-------------------------------------------------------------+
|                                                               |
|  +-------------+    +-------------+    +-------------+      |
|  |   Daily     |--->|   Pattern   |--->|   Extract   |      |
|  |    Logs     |    |  Detection  |    |   Facts     |      |
|  +-------------+    +-------------+    +-------------+      |
|         |                                     |               |
|         |                              +-------------+       |
|         |                              |   Update    |       |
|         |                              |   Semantic  |       |
|         |                              |   Memory    |       |
|         |                              +------+------+       |
|         |                                     |               |
|  +-------------+    +-------------+    +-------------+      |
|  |   Archive   |<---|   Compress  |<---|   Mark for  |      |
|  |   Storage   |    |   Old Data  |    |   Archive   |      |
|  +-------------+    +-------------+    +-------------+      |
|                                                               |
+-------------------------------------------------------------+
```

### 7.2 Consolidation Scheduler

```python
class ConsolidationScheduler:
    """Schedule and manage memory consolidation"""
    
    CONSOLIDATION_SCHEDULE = {
        "daily": {
            "time": "03:00",
            "tasks": [
                "process_daily_logs",
                "extract_patterns",
                "update_semantic",
                "archive_old_episodes"
            ]
        },
        "weekly": {
            "day": "sunday",
            "time": "02:00",
            "tasks": [
                "consolidate_semantic",
                "refine_procedures",
                "compress_archives",
                "generate_reports"
            ]
        },
        "monthly": {
            "day": 1,
            "time": "01:00",
            "tasks": [
                "deep_consolidation",
                "knowledge_graph_optimization",
                "full_backup",
                "retention_enforcement"
            ]
        }
    }
    
    async def run_consolidation(
        self,
        consolidation_type: str
    ) -> ConsolidationResult:
        """Run consolidation tasks"""
        
        schedule = self.CONSOLIDATION_SCHEDULE[consolidation_type]
        results = []
        
        for task_name in schedule["tasks"]:
            task = self._get_task(task_name)
            result = await task.execute()
            results.append(result)
        
        return ConsolidationResult(
            type=consolidation_type,
            tasks_completed=len(results),
            details=results,
            timestamp=datetime.now()
        )
```

### 7.3 Pattern Extraction

```python
class PatternExtractor:
    """Extract patterns from episodic memories"""
    
    async def extract_patterns(
        self,
        episodes: List[Episode],
        min_confidence: float = 0.7
    ) -> List[Pattern]:
        """Extract patterns from episodes"""
        
        patterns = []
        
        # 1. Temporal patterns
        temporal = await self._extract_temporal_patterns(episodes)
        patterns.extend(temporal)
        
        # 2. Behavioral patterns
        behavioral = await self._extract_behavioral_patterns(episodes)
        patterns.extend(behavioral)
        
        # 3. Causal patterns
        causal = await self._extract_causal_patterns(episodes)
        patterns.extend(causal)
        
        # 4. Filter by confidence
        filtered = [p for p in patterns if p.confidence >= min_confidence]
        
        # 5. Merge similar patterns
        merged = self._merge_similar_patterns(filtered)
        
        return merged
    
    async def _extract_temporal_patterns(
        self,
        episodes: List[Episode]
    ) -> List[Pattern]:
        """Extract time-based patterns"""
        
        patterns = []
        
        # Group episodes by hour of day
        hourly = self._group_by_hour(episodes)
        for hour, eps in hourly.items():
            if len(eps) >= 5:  # Minimum threshold
                pattern = TemporalPattern(
                    type="hourly_activity",
                    hour=hour,
                    frequency=len(eps),
                    confidence=len(eps) / len(episodes),
                    description=f"Activity peak at {hour}:00"
                )
                patterns.append(pattern)
        
        # Group by day of week
        daily = self._group_by_day_of_week(episodes)
        for day, eps in daily.items():
            if len(eps) >= 3:
                pattern = TemporalPattern(
                    type="daily_activity",
                    day=day,
                    frequency=len(eps),
                    confidence=len(eps) / len(episodes),
                    description=f"High activity on {day}"
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _extract_behavioral_patterns(
        self,
        episodes: List[Episode]
    ) -> List[Pattern]:
        """Extract user behavior patterns"""
        
        patterns = []
        
        # Sequence patterns
        sequences = self._find_sequences(episodes)
        for seq in sequences:
            pattern = BehavioralPattern(
                type="action_sequence",
                sequence=seq.events,
                frequency=seq.frequency,
                confidence=seq.confidence,
                description=f"User often does: {' -> '.join(seq.events)}"
            )
            patterns.append(pattern)
        
        # Preference patterns
        preferences = self._extract_preferences(episodes)
        for pref in preferences:
            pattern = BehavioralPattern(
                type="preference",
                subject=pref.subject,
                value=pref.value,
                confidence=pref.confidence,
                description=f"User prefers {pref.subject}: {pref.value}"
            )
            patterns.append(pattern)
        
        return patterns
```

### 7.4 Fact Extraction

```python
class FactExtractor:
    """Extract facts from episodes and patterns"""
    
    async def extract_facts(
        self,
        episodes: List[Episode],
        patterns: List[Pattern]
    ) -> List[Fact]:
        """Extract facts from processed data"""
        
        facts = []
        
        # 1. Extract from high-confidence patterns
        for pattern in patterns:
            if pattern.confidence >= 0.8:
                fact = await self._pattern_to_fact(pattern)
                if fact:
                    facts.append(fact)
        
        # 2. Extract from significant episodes
        for episode in episodes:
            if episode.importance == Importance.HIGH:
                episode_facts = await self._episode_to_facts(episode)
                facts.extend(episode_facts)
        
        # 3. Deduplicate
        unique_facts = self._deduplicate_facts(facts)
        
        # 4. Verify
        verified = []
        for fact in unique_facts:
            result = await self.verifier.verify(fact, VerificationMethod.CONSISTENCY_CHECK)
            if result.verified:
                verified.append(fact)
        
        return verified
```

---

## 8. MEMORY RETRIEVAL AND SEARCH

### 8.1 Hybrid Search Architecture

```
+-------------------------------------------------------------+
|                  HYBRID SEARCH SYSTEM                        |
+-------------------------------------------------------------+
|                                                               |
|   Query: "What did the user decide about email summaries?"   |
|                          |                                    |
|              +---------------------+                         |
|              |   Query Analyzer    |                         |
|              +----------+----------+                         |
|                         |                                    |
|         +---------------+---------------+                    |
|         |               |               |                    |
|  +-------------+ +-------------+ +-------------+            |
|  |   Lexical   | |   Semantic  | |   Temporal  |            |
|  |   (BM25)    | |  (Vector)   | |   Search    |            |
|  +------+------+ +------+------+ +------+------+            |
|         |               |               |                    |
|         +---------------+---------------+                    |
|                         |                                    |
|              +---------------------+                         |
|              |   Result Fusion     |                         |
|              |   (RRF Algorithm)   |                         |
|              +----------+----------+                         |
|                         |                                    |
|              +---------------------+                         |
|              |   Ranked Results    |                         |
|              +---------------------+                         |
|                                                               |
+-------------------------------------------------------------+
```

### 8.2 Search Implementation

```python
class HybridSearch:
    """Hybrid BM25 + Vector search"""
    
    # Weights for result fusion
    BM25_WEIGHT = 0.3
    VECTOR_WEIGHT = 0.7
    
    async def search(
        self,
        query: str,
        memory_types: List[MemoryType] = None,
        max_results: int = 10,
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> List[SearchResult]:
        """Perform hybrid search across memory types"""
        
        # 1. BM25 lexical search
        bm25_results = await self._bm25_search(query, memory_types, max_results * 2)
        
        # 2. Vector semantic search
        vector_results = await self._vector_search(query, memory_types, max_results * 2)
        
        # 3. Temporal search (if time range specified)
        temporal_results = []
        if time_range:
            temporal_results = await self._temporal_search(time_range, memory_types)
        
        # 4. Fuse results using Reciprocal Rank Fusion
        fused = self._reciprocal_rank_fusion(
            bm25_results, vector_results, temporal_results
        )
        
        # 5. Return top results
        return fused[:max_results]
    
    async def _bm25_search(
        self,
        query: str,
        memory_types: List[MemoryType],
        max_results: int
    ) -> List[SearchResult]:
        """BM25 lexical search using SQLite FTS5"""
        
        # Tokenize query
        tokens = self._tokenize(query)
        
        # Build FTS5 query
        fts_query = " OR ".join(tokens)
        
        # Search
        results = await self.db.fetchall("""
            SELECT 
                m.id, m.type, m.content, m.source_file,
                rank as bm25_score
            FROM memory_fts m
            WHERE m.content MATCH ?
            AND m.type IN (?)
            ORDER BY rank
            LIMIT ?
        """, [fts_query, ",".join(memory_types), max_results])
        
        return [self._to_search_result(r) for r in results]
    
    async def _vector_search(
        self,
        query: str,
        memory_types: List[MemoryType],
        max_results: int
    ) -> List[SearchResult]:
        """Vector similarity search using sqlite-vec"""
        
        # Create query embedding
        query_embedding = await self.embedder.embed(query)
        
        # Search
        results = await self.db.fetchall("""
            SELECT 
                m.id, m.type, m.content, m.source_file,
                vec_distance_cosine(m.embedding, ?) as similarity
            FROM memory_vectors m
            WHERE m.type IN (?)
            ORDER BY similarity DESC
            LIMIT ?
        """, [query_embedding, ",".join(memory_types), max_results])
        
        return [self._to_search_result(r) for r in results]
    
    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[SearchResult],
        vector_results: List[SearchResult],
        temporal_results: List[SearchResult] = None,
        k: int = 60
    ) -> List[SearchResult]:
        """Fuse results using Reciprocal Rank Fusion"""
        
        scores = defaultdict(float)
        
        # Add BM25 scores
        for rank, result in enumerate(bm25_results):
            scores[result.id] += self.BM25_WEIGHT * (1 / (k + rank + 1))
        
        # Add vector scores
        for rank, result in enumerate(vector_results):
            scores[result.id] += self.VECTOR_WEIGHT * (1 / (k + rank + 1))
        
        # Add temporal boost
        if temporal_results:
            for result in temporal_results:
                scores[result.id] *= 1.1  # 10% boost for temporal relevance
        
        # Sort by fused score
        sorted_results = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Fetch full results
        return [self._fetch_result(id) for id, _ in sorted_results]
```

### 8.3 Contextual Retrieval

```python
class ContextualRetrieval:
    """Retrieve memories with context awareness"""
    
    async def retrieve_for_context(
        self,
        current_context: Context,
        query: Optional[str] = None,
        max_results: int = 5
    ) -> List[RetrievedMemory]:
        """Retrieve memories relevant to current context"""
        
        # 1. Build context query
        context_query = self._build_context_query(current_context)
        
        # 2. Combine with explicit query if provided
        if query:
            full_query = f"{context_query} {query}"
        else:
            full_query = context_query
        
        # 3. Search with context boost
        results = await self.hybrid_search.search(
            query=full_query,
            max_results=max_results * 2
        )
        
        # 4. Apply context-specific ranking
        ranked = self._rank_by_context(results, current_context)
        
        # 5. Add contextual snippets
        contextualized = await self._add_context_snippets(ranked)
        
        return contextualized[:max_results]
    
    def _build_context_query(self, context: Context) -> str:
        """Build a query from current context"""
        
        parts = []
        
        # Add user context
        if context.user_id:
            parts.append(f"user:{context.user_id}")
        
        # Add temporal context
        if context.time_of_day:
            parts.append(f"time:{context.time_of_day}")
        
        # Add task context
        if context.current_task:
            parts.append(f"task:{context.current_task}")
        
        # Add recent topics
        if context.recent_topics:
            parts.extend(context.recent_topics)
        
        return " ".join(parts)
    
    def _rank_by_context(
        self,
        results: List[SearchResult],
        context: Context
    ) -> List[SearchResult]:
        """Re-rank results based on context relevance"""
        
        scored = []
        for result in results:
            score = result.score
            
            # Boost for user match
            if context.user_id and result.user_id == context.user_id:
                score *= 1.2
            
            # Boost for temporal proximity
            if context.current_time and result.timestamp:
                time_diff = abs((context.current_time - result.timestamp).days)
                time_boost = 1.0 / (1 + time_diff / 7)  # Decay over weeks
                score *= (1 + time_boost)
            
            # Boost for topic overlap
            if context.recent_topics:
                topic_overlap = len(set(result.topics) & set(context.recent_topics))
                score *= (1 + topic_overlap * 0.1)
            
            scored.append((result, score))
        
        # Sort by adjusted score
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [r for r, _ in scored]
```

---

## 9. MEMORY UPDATE MECHANISMS

### 9.1 Update Pipeline

```
+-------------------------------------------------------------+
|                  MEMORY UPDATE PIPELINE                      |
+-------------------------------------------------------------+
|                                                               |
|  +-------------+    +-------------+    +-------------+      |
|  |   Event     |--->|   Validate  |--->|   Create    |      |
|  |   Source    |    |   Update    |    |   Patch     |      |
|  +-------------+    +-------------+    +-------------+      |
|                                               |               |
|  +-------------+    +-------------+    +-------------+      |
|  |   Notify    |<---|   Commit    |<---|   Apply     |      |
|  |   Listeners |    |   Changes   |    |   Patch     |      |
|  +-------------+    +-------------+    +-------------+      |
|                                                               |
+-------------------------------------------------------------+
```

### 9.2 Update Manager

```python
class MemoryUpdateManager:
    """Manage memory updates with validation and conflict resolution"""
    
    async def update_memory(
        self,
        update: MemoryUpdate,
        source: UpdateSource
    ) -> UpdateResult:
        """Apply a memory update"""
        
        # 1. Validate update
        validation = await self._validate_update(update)
        if not validation.valid:
            return UpdateResult(
                success=False,
                error=validation.error
            )
        
        # 2. Check for conflicts
        conflicts = await self._detect_conflicts(update)
        if conflicts:
            resolution = await self._resolve_conflicts(update, conflicts)
            if not resolution.resolved:
                return UpdateResult(
                    success=False,
                    error="Conflicts could not be resolved",
                    conflicts=conflicts
                )
            update = resolution.merged_update
        
        # 3. Create patch
        patch = await self._create_patch(update)
        
        # 4. Apply patch
        try:
            await self._apply_patch(patch)
        except Exception as e:
            await self._rollback(patch)
            return UpdateResult(success=False, error=str(e))
        
        # 5. Update indexes
        await self._update_indexes(update)
        
        # 6. Notify listeners
        await self._notify_listeners(update)
        
        # 7. Record update
        await self._record_update(update, source)
        
        return UpdateResult(success=True, patch=patch)
    
    async def _validate_update(self, update: MemoryUpdate) -> ValidationResult:
        """Validate a memory update"""
        
        errors = []
        
        # Check required fields
        if not update.memory_type:
            errors.append("Memory type is required")
        
        if not update.content:
            errors.append("Content is required")
        
        # Validate content format
        if update.memory_type == MemoryType.SEMANTIC:
            if not self._is_valid_semantic_content(update.content):
                errors.append("Invalid semantic memory format")
        
        # Check size limits
        content_size = len(update.content)
        if content_size > self.MAX_CONTENT_SIZE:
            errors.append(f"Content exceeds maximum size: {content_size}")
        
        # Validate against schema
        if update.schema:
            schema_errors = self._validate_schema(update.content, update.schema)
            errors.extend(schema_errors)
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors
        )
    
    async def _detect_conflicts(
        self,
        update: MemoryUpdate
    ) -> List[Conflict]:
        """Detect conflicts with existing memory"""
        
        conflicts = []
        
        # Find existing memory at same location
        existing = await self._get_existing(update.target_id)
        
        if existing:
            # Check for concurrent modifications
            if existing.version != update.base_version:
                conflicts.append(Conflict(
                    type="concurrent_modification",
                    existing=existing,
                    incoming=update
                ))
            
            # Check for contradictory content
            if self._are_contradictory(existing.content, update.content):
                conflicts.append(Conflict(
                    type="contradiction",
                    existing=existing,
                    incoming=update
                ))
        
        return conflicts
    
    async def _resolve_conflicts(
        self,
        update: MemoryUpdate,
        conflicts: List[Conflict]
    ) -> ConflictResolution:
        """Resolve detected conflicts"""
        
        for conflict in conflicts:
            if conflict.type == "concurrent_modification":
                # Use three-way merge
                merged = self._three_way_merge(
                    base=conflict.existing.base_version,
                    ours=conflict.existing.content,
                    theirs=update.content
                )
                update.content = merged
            
            elif conflict.type == "contradiction":
                # Use LLM to resolve
                resolution = await self.llm.resolve_contradiction(
                    existing=conflict.existing.content,
                    incoming=update.content,
                    context=update.context
                )
                
                if resolution.resolved:
                    update.content = resolution.content
                else:
                    return ConflictResolution(resolved=False)
        
        return ConflictResolution(resolved=True, merged_update=update)
```

### 9.3 Automatic Memory Flush

```python
class MemoryFlushManager:
    """Handle automatic memory flush before context compaction"""
    
    # Configuration
    SOFT_THRESHOLD_TOKENS = 4000
    RESERVE_TOKENS_FLOOR = 20000
    
    def __init__(self, context_window: int = 200000):
        self.context_window = context_window
        self.flush_threshold = (
            context_window - 
            self.RESERVE_TOKENS_FLOOR - 
            self.SOFT_THRESHOLD_TOKENS
        )
    
    async def check_and_flush(
        self,
        current_tokens: int,
        context: Context
    ) -> Optional[FlushResult]:
        """Check if flush is needed and execute"""
        
        if current_tokens < self.flush_threshold:
            return None  # No flush needed
        
        # Execute memory flush
        return await self._execute_flush(context)
    
    async def _execute_flush(self, context: Context) -> FlushResult:
        """Execute memory flush"""
        
        # 1. Generate flush prompt
        flush_prompt = self._generate_flush_prompt(context)
        
        # 2. Query LLM for memories to preserve
        memories_to_preserve = await self.llm.generate(
            prompt=flush_prompt,
            system="Extract important memories that should be preserved"
        )
        
        # 3. Parse and validate memories
        parsed_memories = self._parse_memories(memories_to_preserve)
        
        # 4. Store memories
        stored = []
        for memory in parsed_memories:
            result = await self.update_manager.update_memory(
                update=MemoryUpdate(
                    memory_type=memory.type,
                    content=memory.content,
                    target_file=memory.target_file
                ),
                source=UpdateSource.AUTO_FLUSH
            )
            if result.success:
                stored.append(memory)
        
        return FlushResult(
            triggered=True,
            memories_preserved=len(stored),
            memories=stored,
            timestamp=datetime.now()
        )
    
    def _generate_flush_prompt(self, context: Context) -> str:
        """Generate prompt for memory flush"""
        
        return f"""
Session nearing compaction. Current context will be truncated.

Review the conversation and extract important information that should be preserved:

1. Decisions made
2. Preferences expressed
3. Facts learned
4. Tasks created or completed
5. Errors encountered and how they were resolved

For each item, specify:
- Type: decision | preference | fact | task | error
- Content: The information to preserve
- Target: Which memory file to store it in

Current context summary:
{context.summary}

Respond in this format:
```
TYPE: decision
CONTENT: [decision details]
TARGET: MEMORY.md

TYPE: preference
CONTENT: [preference details]
TARGET: USER.md
```

If nothing important to preserve, respond with: NO_REPLY
"""
```

---

## 10. FORGETTING AND ARCHIVING

### 10.1 Retention Policies

```yaml
# retention.yaml - Memory Retention Policies

retention_policies:
  # Episodic memory retention
  episodic:
    active_period: 30 days
    archive_after: 30 days
    compress_after: 90 days
    delete_after: 365 days
    
    importance_override:
      HIGH: never_delete
      MEDIUM: archive_after_60_days
      LOW: archive_after_7_days
  
  # Daily logs retention
  daily_logs:
    keep_active: 2 days
    archive_after: 7 days
    compress_after: 30 days
    delete_after: 180 days
  
  # Session transcripts
  sessions:
    keep_active: 7 days
    archive_after: 30 days
    compress_after: 90 days
    delete_after: 365 days
  
  # Semantic memory
  semantic:
    retention: permanent
    archive: false
    
    confidence_pruning:
      enabled: true
      threshold: 0.3
      check_interval: 30 days
  
  # Procedural memory
  procedural:
    retention: permanent
    archive: false
    
    success_rate_pruning:
      enabled: true
      threshold: 0.1
      min_executions: 10
      check_interval: 30 days

compression:
  algorithm: gzip
  level: 9
  
  # What to compress
  compress:
    - old_daily_logs
    - archived_episodes
    - old_sessions
  
  # What to exclude from compression
  exclude:
    - MEMORY.md
    - SEMANTIC.md
    - PROCEDURAL.md
    - USER.md
    - current_daily_logs

archival:
  location: memory/archive/
  
  # Archive structure
  structure:
    by_month: "%Y-%m"
    by_year: "%Y"
  
  # Indexing
  maintain_index: true
  index_location: archive/index.json
```

### 10.2 Archival System

```python
class ArchivalSystem:
    """Manage memory archival and compression"""
    
    async def archive_old_memories(self) -> ArchivalResult:
        """Archive memories past retention period"""
        
        results = []
        
        # 1. Archive episodic memories
        episodic_result = await self._archive_episodic()
        results.append(episodic_result)
        
        # 2. Archive daily logs
        logs_result = await self._archive_daily_logs()
        results.append(logs_result)
        
        # 3. Archive sessions
        sessions_result = await self._archive_sessions()
        results.append(sessions_result)
        
        # 4. Update archive index
        await self._update_archive_index()
        
        return ArchivalResult(
            timestamp=datetime.now(),
            details=results,
            total_archived=sum(r.count for r in results)
        )
    
    async def _archive_episodic(self) -> ArchiveOperationResult:
        """Archive old episodic memories"""
        
        # Find episodes to archive
        cutoff = datetime.now() - timedelta(days=30)
        
        episodes = await self.db.fetchall("""
            SELECT * FROM episodes
            WHERE timestamp < ?
            AND importance < ?
            AND archived = FALSE
        """, [cutoff, Importance.HIGH.value])
        
        archived = 0
        for episode in episodes:
            # Archive to file
            archive_path = self._get_archive_path(episode.timestamp)
            await self._append_to_archive(archive_path, episode)
            
            # Mark as archived
            await self.db.execute("""
                UPDATE episodes
                SET archived = TRUE, archived_at = ?
                WHERE id = ?
            """, [datetime.now(), episode.id])
            
            archived += 1
        
        return ArchiveOperationResult(
            type="episodic",
            count=archived
        )
    
    async def compress_archives(self) -> CompressionResult:
        """Compress old archives"""
        
        # Find archives to compress
        archives = await self._find_compressible_archives()
        
        compressed = 0
        space_saved = 0
        
        for archive in archives:
            original_size = archive.size
            
            # Compress
            compressed_path = await self._compress_file(
                archive.path,
                algorithm="gzip",
                level=9
            )
            
            compressed_size = os.path.getsize(compressed_path)
            space_saved += original_size - compressed_size
            
            # Update index
            await self._update_compression_index(archive, compressed_path)
            
            compressed += 1
        
        return CompressionResult(
            files_compressed=compressed,
            space_saved_bytes=space_saved,
            timestamp=datetime.now()
        )
```

### 10.3 Selective Forgetting

```python
class SelectiveForgetting:
    """Implement selective forgetting based on relevance"""
    
    async def apply_forgetting(
        self,
        memory_type: MemoryType,
        strategy: ForgettingStrategy
    ) -> ForgettingResult:
        """Apply forgetting to memories"""
        
        if strategy == ForgettingStrategy.AGE_BASED:
            return await self._age_based_forgetting(memory_type)
        
        elif strategy == ForgettingStrategy.RELEVANCE_BASED:
            return await self._relevance_based_forgetting(memory_type)
        
        elif strategy == ForgettingStrategy.IMPORTANCE_BASED:
            return await self._importance_based_forgetting(memory_type)
        
        elif strategy == ForgettingStrategy.ACCESS_BASED:
            return await self._access_based_forgetting(memory_type)
    
    async def _relevance_based_forgetting(
        self,
        memory_type: MemoryType
    ) -> ForgettingResult:
        """Forget memories with low relevance to current context"""
        
        # Get current context embedding
        context_embedding = await self._get_current_context_embedding()
        
        # Find low-relevance memories
        low_relevance = await self.db.fetchall("""
            SELECT m.*, 
                   vec_distance_cosine(m.embedding, ?) as relevance
            FROM memories m
            WHERE m.type = ?
            AND m.last_accessed < ?
            ORDER BY relevance ASC
            LIMIT 1000
        """, [
            context_embedding,
            memory_type.value,
            datetime.now() - timedelta(days=30)
        ])
        
        forgotten = []
        for memory in low_relevance:
            if memory.relevance > 0.8:  # Still somewhat relevant
                continue
            
            # Archive before forgetting
            await self.archival.archive_memory(memory)
            
            # Remove from active memory
            await self.db.execute("""
                DELETE FROM memories WHERE id = ?
            """, [memory.id])
            
            forgotten.append(memory.id)
        
        return ForgettingResult(
            strategy=ForgettingStrategy.RELEVANCE_BASED,
            forgotten_count=len(forgotten),
            forgotten_ids=forgotten
        )
    
    async def _access_based_forgetting(
        self,
        memory_type: MemoryType
    ) -> ForgettingResult:
        """Forget memories that haven't been accessed recently"""
        
        # Calculate access-based relevance score
        memories = await self.db.fetchall("""
            SELECT * FROM memories
            WHERE type = ?
        """, [memory_type.value])
        
        forgotten = []
        for memory in memories:
            # Calculate forgetting score
            days_since_access = (datetime.now() - memory.last_accessed).days
            access_frequency = memory.access_count / max(days_since_access, 1)
            
            # Forgetting curve: higher score = more likely to forget
            forgetting_score = (
                days_since_access * 0.5 +
                (1 / (access_frequency + 1)) * 0.5
            )
            
            if forgetting_score > 30:  # Threshold
                await self.archival.archive_memory(memory)
                await self.db.execute("""
                    DELETE FROM memories WHERE id = ?
                """, [memory.id])
                forgotten.append(memory.id)
        
        return ForgettingResult(
            strategy=ForgettingStrategy.ACCESS_BASED,
            forgotten_count=len(forgotten),
            forgotten_ids=forgotten
        )
```

---

## 11. CONTEXT MANAGEMENT

### 11.1 Context Structure

```python
@dataclass
class AgentContext:
    """Complete context for agent operation"""
    
    # Identity
    agent_id: str
    agent_name: str
    session_id: str
    
    # Temporal
    current_time: datetime
    timezone: str
    
    # User
    user_id: str
    user_profile: UserProfile
    
    # System
    system_status: SystemStatus
    active_loops: List[AgenticLoop]
    
    # Memory
    working_memory: WorkingMemory
    recent_memories: List[Memory]
    
    # Task
    current_task: Optional[Task]
    task_history: List[Task]
    
    # Environment
    environment_state: EnvironmentState
    
    @property
    def summary(self) -> str:
        """Generate context summary for prompts"""
        return f"""
Agent: {self.agent_name} ({self.agent_id})
Session: {self.session_id}
Time: {self.current_time} ({self.timezone})
User: {self.user_profile.display_name}
Active Loops: {len(self.active_loops)}
Current Task: {self.current_task.name if self.current_task else "None"}
System Status: {self.system_status.overall}
"""

@dataclass
class WorkingMemory:
    """Short-term working memory"""
    
    # Recent conversation
    conversation_history: List[Message]
    
    # Active facts
    active_facts: List[Fact]
    
    # Current focus
    current_topic: Optional[str]
    related_topics: List[str]
    
    # Temporary state
    scratchpad: Dict[str, Any]
    
    def to_prompt_context(self, max_tokens: int = 4000) -> str:
        """Convert to prompt-friendly context"""
        
        parts = []
        
        # Add recent conversation
        conversation_text = self._format_conversation(
            self.conversation_history,
            max_tokens=max_tokens * 0.5
        )
        parts.append(f"Recent Conversation:\n{conversation_text}")
        
        # Add active facts
        facts_text = "\n".join([
            f"- {fact.statement}" 
            for fact in self.active_facts[:10]
        ])
        parts.append(f"Active Facts:\n{facts_text}")
        
        # Add current focus
        if self.current_topic:
            parts.append(f"Current Topic: {self.current_topic}")
            parts.append(f"Related: {', '.join(self.related_topics[:5])}")
        
        return "\n\n".join(parts)
```

### 11.2 Context Loading

```python
class ContextLoader:
    """Load context at session start"""
    
    async def load_session_context(
        self,
        agent_id: str,
        user_id: str
    ) -> AgentContext:
        """Load complete context for a new session"""
        
        # 1. Load identity
        identity = await self._load_identity(agent_id)
        
        # 2. Load user profile
        user_profile = await self._load_user_profile(user_id)
        
        # 3. Load recent memories
        recent_memories = await self._load_recent_memories(
            days=2,
            max_results=50
        )
        
        # 4. Load daily logs (today + yesterday)
        daily_logs = await self._load_daily_logs(days=2)
        
        # 5. Load active loops
        active_loops = await self._load_active_loops(agent_id)
        
        # 6. Load system status
        system_status = await self._check_system_status()
        
        # 7. Build working memory
        working_memory = WorkingMemory(
            conversation_history=[],
            active_facts=self._extract_active_facts(recent_memories),
            current_topic=None,
            related_topics=[],
            scratchpad={}
        )
        
        return AgentContext(
            agent_id=agent_id,
            agent_name=identity.name,
            session_id=self._generate_session_id(),
            current_time=datetime.now(),
            timezone=user_profile.timezone,
            user_id=user_id,
            user_profile=user_profile,
            system_status=system_status,
            active_loops=active_loops,
            working_memory=working_memory,
            recent_memories=recent_memories,
            current_task=None,
            task_history=[],
            environment_state=await self._get_environment_state()
        )
    
    async def _load_recent_memories(
        self,
        days: int,
        max_results: int
    ) -> List[Memory]:
        """Load recent memories from all types"""
        
        cutoff = datetime.now() - timedelta(days=days)
        
        # Load from episodic
        episodic = await self.db.fetchall("""
            SELECT * FROM episodes
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, [cutoff, max_results // 2])
        
        # Load from daily logs
        logs = await self._load_daily_logs(days=days)
        
        # Load from semantic (relevant facts)
        semantic = await self.semantic_memory.get_relevant_facts(
            context=self._get_current_context(),
            max_results=max_results // 4
        )
        
        # Combine and sort
        all_memories = [
            Memory.from_episode(e) for e in episodic
        ] + [
            Memory.from_log(l) for l in logs
        ] + [
            Memory.from_fact(f) for f in semantic
        ]
        
        all_memories.sort(key=lambda m: m.relevance_score, reverse=True)
        
        return all_memories[:max_results]
```

### 11.3 Context Persistence

```python
class ContextPersistence:
    """Persist context across sessions"""
    
    async def save_session_state(
        self,
        context: AgentContext
    ) -> None:
        """Save current session state"""
        
        # 1. Save session transcript
        await self._save_session_transcript(context)
        
        # 2. Update daily log
        await self._update_daily_log(context)
        
        # 3. Save working memory
        await self._save_working_memory(context.working_memory)
        
        # 4. Save task state
        if context.current_task:
            await self._save_task_state(context.current_task)
    
    async def _save_session_transcript(
        self,
        context: AgentContext
    ) -> None:
        """Save session transcript to file"""
        
        transcript = SessionTranscript(
            session_id=context.session_id,
            agent_id=context.agent_id,
            user_id=context.user_id,
            start_time=context.session_start,
            end_time=datetime.now(),
            messages=context.working_memory.conversation_history,
            metadata={
                "tasks_completed": len(context.task_history),
                "loops_executed": len(context.active_loops)
            }
        )
        
        # Generate filename
        slug = await self._generate_slug(transcript)
        filename = f"sessions/{transcript.start_time.strftime('%Y-%m-%d')}-{slug}.md"
        
        # Save
        await self._write_transcript_file(filename, transcript)
        
        # Index for search
        await self.memory_index.index_session(filename)
    
    async def _update_daily_log(self, context: AgentContext) -> None:
        """Append to daily log"""
        
        today = datetime.now().strftime('%Y-%m-%d')
        log_file = f"memory/{today}.md"
        
        # Generate log entry
        entry = self._generate_log_entry(context)
        
        # Append to file
        await self._append_to_file(log_file, entry)
```

---

## 12. IMPLEMENTATION DETAILS

### 12.1 SQLite Schema

```sql
-- Core memory tables

-- Episodic memories
CREATE TABLE episodes (
    id TEXT PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    type TEXT NOT NULL,
    description TEXT NOT NULL,
    context TEXT,  -- JSON
    importance INTEGER NOT NULL,
    embedding FLOAT[],  -- sqlite-vec
    related_episodes TEXT,  -- JSON array
    consolidated_into TEXT,
    created_at DATETIME NOT NULL,
    last_accessed DATETIME NOT NULL,
    access_count INTEGER DEFAULT 0,
    archived BOOLEAN DEFAULT FALSE,
    archived_at DATETIME
);

-- Semantic entities
CREATE TABLE entities (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    name TEXT NOT NULL,
    properties TEXT,  -- JSON
    embedding FLOAT[],
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    confidence REAL NOT NULL
);

-- Relationships
CREATE TABLE relationships (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    type TEXT NOT NULL,
    properties TEXT,  -- JSON
    embedding FLOAT[],
    created_at DATETIME NOT NULL,
    confidence REAL NOT NULL,
    FOREIGN KEY (source_id) REFERENCES entities(id),
    FOREIGN KEY (target_id) REFERENCES entities(id)
);

-- Facts
CREATE TABLE facts (
    id TEXT PRIMARY KEY,
    statement TEXT NOT NULL,
    category TEXT NOT NULL,
    source TEXT NOT NULL,
    confidence REAL NOT NULL,
    embedding FLOAT[],
    created_at DATETIME NOT NULL,
    last_verified DATETIME,
    verification_count INTEGER DEFAULT 0
);

-- Skills/Procedures
CREATE TABLE skills (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    description TEXT NOT NULL,
    steps TEXT,  -- JSON
    success_rate REAL NOT NULL,
    execution_count INTEGER DEFAULT 0,
    average_execution_time_ms INTEGER,
    last_executed DATETIME,
    created_at DATETIME NOT NULL,
    refined_at DATETIME,
    refinement_history TEXT  -- JSON
);

-- FTS5 for BM25 search
CREATE VIRTUAL TABLE memory_fts USING fts5(
    content,
    type,
    source_file,
    content_rowid=rowid
);

-- Vector search using sqlite-vec
CREATE VIRTUAL TABLE memory_vectors USING vec0(
    embedding FLOAT[768]
);

-- Indexes
CREATE INDEX idx_episodes_timestamp ON episodes(timestamp);
CREATE INDEX idx_episodes_type ON episodes(type);
CREATE INDEX idx_episodes_importance ON episodes(importance);
CREATE INDEX idx_entities_type ON entities(type);
CREATE INDEX idx_facts_category ON facts(category);
CREATE INDEX idx_skills_name ON skills(name);
```

### 12.2 Configuration

```yaml
# memory.yaml - Memory System Configuration

memory:
  # Storage
  storage:
    type: sqlite
    path: index/memory.sqlite
    backup:
      enabled: true
      interval: daily
      keep_count: 7
  
  # Embeddings
  embeddings:
    provider: auto  # auto, local, openai, gemini
    local:
      model: hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf
      download_if_missing: true
    openai:
      model: text-embedding-3-small
      batch_size: 100
    gemini:
      model: gemini-embedding-001
    dimensions: 768
  
  # Search
  search:
    hybrid:
      enabled: true
      bm25_weight: 0.3
      vector_weight: 0.7
    max_results: 10
    min_score: 0.5
    snippet_max_chars: 700
  
  # Chunking
  chunking:
    tokens: 400
    overlap: 80
    preserve_headers: true
  
  # Context
  context:
    working_memory_tokens: 4000
    recent_memories: 50
    daily_logs_to_load: 2
  
  # Consolidation
  consolidation:
    daily:
      time: "03:00"
      enabled: true
    weekly:
      day: sunday
      time: "02:00"
      enabled: true
  
  # Flush
  flush:
    enabled: true
    soft_threshold_tokens: 4000
    reserve_tokens_floor: 20000
  
  # Retention
  retention:
    config_file: config/retention.yaml
```

### 12.3 Python Implementation Structure

```
openclaw/
+-- memory/
|   +-- __init__.py
|   +-- core/
|   |   +-- __init__.py
|   |   +-- types.py           # Data types and enums
|   |   +-- context.py         # Context management
|   |   +-- exceptions.py      # Custom exceptions
|   +-- storage/
|   |   +-- __init__.py
|   |   +-- sqlite_store.py    # SQLite storage
|   |   +-- file_store.py      # Markdown file storage
|   |   +-- index_manager.py   # Index management
|   +-- embeddings/
|   |   +-- __init__.py
|   |   +-- provider.py        # Embedding provider interface
|   |   +-- local.py           # Local embeddings
|   |   +-- openai.py          # OpenAI embeddings
|   |   +-- gemini.py          # Gemini embeddings
|   +-- search/
|   |   +-- __init__.py
|   |   +-- hybrid.py          # Hybrid search
|   |   +-- bm25.py            # BM25 implementation
|   |   +-- vector.py          # Vector search
|   +-- episodic/
|   |   +-- __init__.py
|   |   +-- episode.py         # Episode management
|   |   +-- creator.py         # Episode creation
|   |   +-- retrieval.py       # Episode retrieval
|   +-- semantic/
|   |   +-- __init__.py
|   |   +-- knowledge_graph.py # Knowledge graph
|   |   +-- entity.py          # Entity management
|   |   +-- fact.py            # Fact management
|   |   +-- inference.py       # Inference engine
|   +-- procedural/
|   |   +-- __init__.py
|   |   +-- skill.py           # Skill management
|   |   +-- workflow.py        # Workflow engine
|   |   +-- learner.py         # Skill learning
|   +-- consolidation/
|   |   +-- __init__.py
|   |   +-- scheduler.py       # Consolidation scheduler
|   |   +-- pattern.py         # Pattern extraction
|   |   +-- fact_extractor.py  # Fact extraction
|   +-- archival/
|   |   +-- __init__.py
|   |   +-- archiver.py        # Archival system
|   |   +-- compression.py     # Compression
|   |   +-- forgetting.py      # Selective forgetting
|   +-- update/
|       +-- __init__.py
|       +-- manager.py         # Update manager
|       +-- validator.py       # Update validation
|       +-- flush.py           # Memory flush
+-- config/
    +-- memory.yaml
    +-- retention.yaml
```

---

## 13. WINDOWS 10 SPECIFIC CONSIDERATIONS

### 13.1 File System Integration

```python
class WindowsFileSystem:
    """Windows 10 specific file system operations"""
    
    BASE_PATH = Path("C:/OpenClaw")
    
    def __init__(self):
        self.ensure_directories()
    
    def ensure_directories(self):
        """Create required directory structure"""
        directories = [
            self.BASE_PATH / "memory",
            self.BASE_PATH / "memory" / "archive",
            self.BASE_PATH / "sessions",
            self.BASE_PATH / "index",
            self.BASE_PATH / "config",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    async def watch_files(self):
        """Watch memory files for changes"""
        # Use Windows file system notifications
        # or polling for compatibility
        pass
    
    async def atomic_write(self, path: Path, content: str):
        """Atomic file write for Windows"""
        temp_path = path.with_suffix('.tmp')
        
        # Write to temp file
        async with aiofiles.open(temp_path, 'w', encoding='utf-8') as f:
            await f.write(content)
        
        # Atomic rename
        temp_path.replace(path)
```

### 13.2 Windows Service Integration

```python
class WindowsService:
    """Run as Windows service for 24/7 operation"""
    
    def __init__(self):
        self.memory_system = None
        self.heartbeat = None
    
    async def start(self):
        """Start the memory system"""
        
        # Initialize memory system
        self.memory_system = await MemorySystem.initialize()
        
        # Start heartbeat
        self.heartbeat = Heartbeat(self.memory_system)
        await self.heartbeat.start()
        
        # Load agentic loops
        await self._load_loops()
    
    async def stop(self):
        """Graceful shutdown"""
        
        # Save context
        await self.memory_system.save_session_state()
        
        # Stop heartbeat
        await self.heartbeat.stop()
        
        # Close connections
        await self.memory_system.close()
```

### 13.3 Windows Task Scheduler Integration

```python
class WindowsTaskScheduler:
    """Integrate with Windows Task Scheduler for cron-like behavior"""
    
    async def schedule_consolidation(self):
        """Schedule daily consolidation"""
        
        # Use schtasks command or taskscheduler library
        task = {
            "task_name": "OpenClaw_MemoryConsolidation",
            "schedule": "DAILY",
            "start_time": "03:00",
            "action": "python -m openclaw.memory.consolidate"
        }
        
        await self._create_task(task)
```

---

## APPENDIX A: MEMORY FILE TEMPLATES

### A.1 MEMORY.md Template

```markdown
# Memory

> Curated long-term memory for decisions, preferences, and durable facts.
> Last updated: {timestamp}

## Decisions

<!-- Record important decisions here -->

## User Preferences

<!-- Record user preferences here -->

## Facts

### System Facts

<!-- System configuration facts -->

### User Facts

<!-- Facts about the user -->

## Goals

### Active Goals

<!-- Current goals -->

### Completed Goals

<!-- Completed goals -->

## Patterns

<!-- Recognized patterns -->

## Relationships

<!-- Important relationships -->
```

### A.2 Daily Log Template

```markdown
# Daily Log: {date}

> Auto-generated daily activity log
> Agent: {agent_name}

## Summary

- **Active Hours**: {start} - {end}
- **Interactions**: {count}
- **Tasks Completed**: {count}
- **Errors**: {count}
- **Heartbeat Status**: {status}

## Timeline

<!-- Chronological events -->

## Events

### Significant Events

<!-- Important events -->

### Errors & Recoveries

<!-- Errors and how they were resolved -->

## Decisions Made Today

<!-- Decisions from today -->

## Context for Tomorrow

<!-- Information for next session -->
```

---

## APPENDIX B: API REFERENCE

### B.1 Memory System API

```python
class MemorySystem:
    """Main memory system interface"""
    
    # Initialization
    @classmethod
    async def initialize(cls, config: Config) -> "MemorySystem": ...
    
    # Episodic memory
    async def create_episode(self, event: Event) -> Episode: ...
    async def get_episodes(
        self, 
        query: str, 
        max_results: int = 10
    ) -> List[Episode]: ...
    
    # Semantic memory
    async def add_entity(self, entity: Entity) -> None: ...
    async def query_knowledge(
        self, 
        query: str
    ) -> List[Union[Entity, Fact]]: ...
    
    # Procedural memory
    async def learn_skill(self, episode: Episode) -> Skill: ...
    async def execute_workflow(
        self, 
        workflow: Workflow
    ) -> ExecutionResult: ...
    
    # Search
    async def search(
        self, 
        query: str, 
        max_results: int = 10
    ) -> List[SearchResult]: ...
    
    # Context
    async def load_context(
        self, 
        agent_id: str, 
        user_id: str
    ) -> AgentContext: ...
    async def save_context(self, context: AgentContext) -> None: ...
    
    # Maintenance
    async def consolidate(self) -> ConsolidationResult: ...
    async def archive_old_memories(self) -> ArchivalResult: ...
```

---

## SUMMARY

This specification defines a comprehensive memory system architecture for a Windows 10 OpenClaw-inspired AI agent. Key features include:

1. **File-First Design**: Markdown files as source of truth with SQLite for indexing
2. **Multi-Tier Memory**: Working, short-term, long-term, and archival tiers
3. **Human Memory Types**: Episodic, semantic, and procedural memory
4. **Hybrid Search**: BM25 + vector search for optimal retrieval
5. **Automatic Consolidation**: Pre-compaction memory flush
6. **Selective Forgetting**: Relevance-based memory management
7. **Windows Integration**: Service-based 24/7 operation

The system is designed for GPT-5.2 with 200K context window, supporting 15 agentic loops and full Windows 10 system access.
