# Email Search and Indexing System Technical Specification
## OpenClaw Windows 10 AI Agent Framework

**Version:** 1.0  
**Date:** 2024  
**Status:** Technical Specification

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture Overview](#system-architecture-overview)
3. [Full-Text Search Implementation](#full-text-search-implementation)
4. [Email Indexing Strategies](#email-indexing-strategies)
5. [Search Query Processing](#search-query-processing)
6. [Faceted Search Design](#faceted-search-design)
7. [Ranking and Relevance](#ranking-and-relevance)
8. [Incremental Indexing](#incremental-indexing)
9. [Search Caching](#search-caching)
10. [Gmail Search Query DSL](#gmail-search-query-dsl)
11. [Implementation Details](#implementation-details)
12. [Performance Optimization](#performance-optimization)
13. [Security Considerations](#security-considerations)

---

## Executive Summary

This specification defines a high-performance email search and indexing system for the OpenClaw Windows 10 AI agent framework. The system provides:

- **Sub-100ms search latency** for indexed emails
- **Full-text search** across email bodies, subjects, and metadata
- **Advanced query capabilities** including boolean logic, wildcards, and fuzzy matching
- **Faceted search** by date, sender, labels, attachments, and custom fields
- **Gmail search DSL compatibility** for seamless integration
- **Real-time incremental indexing** for new emails
- **Intelligent caching** for frequently accessed searches

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EMAIL SEARCH SYSTEM ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   Gmail API  │    │  IMAP/POP3   │    │  Local PST   │                   │
│  │   Connector  │    │  Connector   │    │   Parser     │                   │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                   │
│         │                   │                    │                           │
│         └───────────────────┼────────────────────┘                           │
│                             ▼                                                │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │                    EMAIL INGESTION PIPELINE                   │           │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │           │
│  │  │ Fetcher  │→│ Parser   │→│ Enricher │→│ Normalizer│      │           │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘      │           │
│  └───────────────────────────┬──────────────────────────────────┘           │
│                              ▼                                               │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │                    INDEXING ENGINE                            │           │
│  │  ┌──────────────────────────────────────────────────────┐    │           │
│  │  │              Elasticsearch Cluster                      │    │           │
│  │  │  ┌────────────┐  ┌────────────┐  ┌────────────┐       │    │           │
│  │  │  │  Primary   │  │  Replica 1 │  │  Replica 2 │       │    │           │
│  │  │  │   Node     │  │    Node    │  │    Node    │       │    │           │
│  │  │  └────────────┘  └────────────┘  └────────────┘       │    │           │
│  │  └──────────────────────────────────────────────────────┘    │           │
│  │                                                              │           │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │           │
│  │  │   Inverted   │  │   Document   │  │   Facet      │       │           │
│  │  │    Index     │  │    Store     │  │   Indices    │       │           │
│  │  └──────────────┘  └──────────────┘  └──────────────┘       │           │
│  └───────────────────────────┬──────────────────────────────────┘           │
│                              ▼                                               │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │                    SEARCH LAYER                               │           │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │           │
│  │  │  Query   │→│  Parser  │→│ Optimizer│→│ Executor │      │           │
│  │  │  Router  │  │          │  │          │  │          │      │           │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘      │           │
│  │                                                              │           │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                    │           │
│  │  │  Cache   │  │  Ranker  │  │  Merger  │                    │           │
│  │  │  Layer   │  │          │  │          │                    │           │
│  │  └──────────┘  └──────────┘  └──────────┘                    │           │
│  └───────────────────────────┬──────────────────────────────────┘           │
│                              ▼                                               │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │                    API LAYER                                  │           │
│  │  REST API │ GraphQL │ WebSocket │ gRPC                       │           │
│  └──────────────────────────────────────────────────────────────┘           │
│                              ▼                                               │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │                    AI AGENT INTEGRATION                       │           │
│  │  GPT-5.2 │ Agent Loops │ Memory │ Context Manager            │           │
│  └──────────────────────────────────────────────────────────────┘           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Full-Text Search Implementation

### 3.1 Technology Selection: Elasticsearch

**Primary Choice:** Elasticsearch 8.x

**Rationale:**
- Native full-text search with Lucene engine
- Distributed architecture for scalability
- Real-time indexing capabilities
- Advanced query DSL
- Built-in faceting and aggregation
- Windows-compatible with WSL2/Docker

**Alternative for Lightweight Deployments:** Lunr.js (client-side)

### 3.2 Elasticsearch Configuration

```yaml
# elasticsearch.yml
cluster.name: openclaw-email-search
node.name: ${HOSTNAME}
path.data: /var/lib/elasticsearch
path.logs: /var/log/elasticsearch

# Memory settings (adjust based on available RAM)
bootstrap.memory_lock: true
indices.memory.index_buffer_size: 30%

# Search performance
indices.query.bool.max_clause_count: 4096
search.default_search_timeout: 30s

# Indexing performance
index.refresh_interval: 5s
index.translog.durability: async
index.number_of_shards: 3
index.number_of_replicas: 1
```

### 3.3 Search Index Mapping

```json
{
  "mappings": {
    "dynamic": "strict",
    "properties": {
      "email_id": { "type": "keyword", "index": true, "store": true },
      "thread_id": { "type": "keyword", "index": true },
      "message_id": { "type": "keyword", "index": true },
      "subject": {
        "type": "text",
        "analyzer": "email_analyzer",
        "fields": {
          "keyword": { "type": "keyword", "ignore_above": 512 },
          "suggest": { "type": "completion" }
        }
      },
      "body": {
        "type": "text",
        "analyzer": "email_analyzer",
        "store": false,
        "term_vector": "with_positions_offsets"
      },
      "body_plain": { "type": "text", "analyzer": "standard" },
      "from": {
        "type": "nested",
        "properties": {
          "email": { "type": "keyword" },
          "name": { "type": "text" },
          "domain": { "type": "keyword" }
        }
      },
      "to": {
        "type": "nested",
        "properties": {
          "email": { "type": "keyword" },
          "name": { "type": "text" },
          "domain": { "type": "keyword" }
        }
      },
      "cc": {
        "type": "nested",
        "properties": {
          "email": { "type": "keyword" },
          "name": { "type": "text" }
        }
      },
      "bcc": {
        "type": "nested",
        "properties": {
          "email": { "type": "keyword" },
          "name": { "type": "text" }
        }
      },
      "date_sent": { "type": "date", "format": "strict_date_optional_time||epoch_millis" },
      "date_received": { "type": "date", "format": "strict_date_optional_time||epoch_millis" },
      "labels": { "type": "keyword", "index": true },
      "categories": { "type": "keyword" },
      "importance": { "type": "keyword" },
      "priority": { "type": "integer" },
      "is_read": { "type": "boolean" },
      "is_starred": { "type": "boolean" },
      "has_attachments": { "type": "boolean" },
      "attachments": {
        "type": "nested",
        "properties": {
          "filename": { "type": "keyword" },
          "content_type": { "type": "keyword" },
          "size": { "type": "long" },
          "content_extracted": { "type": "text" }
        }
      },
      "in_reply_to": { "type": "keyword" },
      "references": { "type": "keyword" },
      "conversation_index": { "type": "integer" },
      "folder": { "type": "keyword" },
      "size_bytes": { "type": "long" },
      "language": { "type": "keyword" },
      "sentiment_score": { "type": "float" },
      "entities": {
        "type": "nested",
        "properties": {
          "type": { "type": "keyword" },
          "text": { "type": "keyword" },
          "start": { "type": "integer" },
          "end": { "type": "integer" }
        }
      },
      "extracted_dates": { "type": "date" },
      "extracted_urls": { "type": "keyword" },
      "extracted_emails": { "type": "keyword" },
      "indexed_at": { "type": "date" },
      "source": { "type": "keyword" },
      "user_id": { "type": "keyword" }
    }
  },
  "settings": {
    "number_of_shards": 3,
    "number_of_replicas": 1,
    "analysis": {
      "analyzer": {
        "email_analyzer": {
          "type": "custom",
          "tokenizer": "uax_url_email",
          "filter": ["lowercase", "stop", "word_delimiter_graph", "porter_stem"]
        }
      }
    }
  }
}
```

---

## Email Indexing Strategies

### 4.1 Multi-Tier Indexing Architecture

| Tier | Age Range | Storage | Refresh | Shards |
|------|-----------|---------|---------|--------|
| HOT | 0-30 days | SSD | 1s | 3 |
| WARM | 31-90 days | Standard | 30s | 2 |
| COLD | 91+ days | Compressed | Manual | 1 |

### 4.2 Indexing Pipeline Stages

1. **Parse**: Extract structured data from raw email
2. **Enrich**: Add metadata (URLs, emails, dates, language)
3. **Analyze**: NLP processing (sentiment, entities, topics)
4. **Index**: Store in appropriate tier

### 4.3 Priority Levels

- **REALTIME**: Immediate indexing (< 1s)
- **HIGH**: Within 5 seconds
- **NORMAL**: Within 30 seconds
- **BATCH**: Next batch cycle

---

## Search Query Processing

### 5.1 Query Types Supported

| Type | Example | Description |
|------|---------|-------------|
| Term | `meeting` | Simple keyword search |
| Phrase | `"quarterly report"` | Exact phrase match |
| Field | `from:john@example.com` | Field-specific search |
| Boolean | `meeting AND urgent` | Logical operators |
| Wildcard | `proj*` | Prefix/suffix matching |
| Fuzzy | `meeting~` | Approximate matching |
| Range | `after:2024-01-01` | Date/size ranges |

### 5.2 Query Parser Features

- Natural language query parsing
- Field prefix recognition (from:, to:, subject:, etc.)
- Boolean operator support (AND, OR, NOT)
- Quoted phrase handling
- Date range parsing

---

## Faceted Search Design

### 6.1 Available Facets

| Facet | Field | Type | Multi-Select |
|-------|-------|------|--------------|
| Sender | from.email | terms | Yes |
| Domain | from.domain | terms | Yes |
| Labels | labels | terms | Yes |
| Date | date_received | date_histogram | No |
| Attachments | has_attachments | terms | No |
| Importance | importance | terms | Yes |
| Folder | folder | terms | Yes |

---

## Ranking and Relevance

### 7.1 Relevance Factors

| Factor | Weight | Description |
|--------|--------|-------------|
| Subject match | 3.0x | Higher weight for subject matches |
| Recency | 2.0x | Exponential decay with age |
| Starred | 1.5x | Boost starred emails |
| Unread | 1.1x | Slight boost for unread |
| Priority | Variable | Based on calculated priority score |
| Sentiment | Variable | Positive sentiment boost |

### 7.2 Personalization Signals

- Frequent sender boost (1.2x)
- Important domain boost (1.3x)
- Click history boost (1.2x)

---

## Incremental Indexing

### 8.1 Gmail API Sync Strategy

1. **Initial Sync**: Full sync of last 30 days
2. **Continuous Sync**: Poll Gmail History API every 30s
3. **Change Types Handled**:
   - messageAdded
   - messageDeleted
   - labelsAdded
   - labelsRemoved

### 8.2 Sync State Management

```python
SyncState = {
    "last_sync": datetime,
    "last_history_id": str,
    "total_indexed": int,
    "pending_changes": List[Dict]
}
```

---

## Search Caching

### 9.1 Multi-Layer Cache Architecture

| Layer | Technology | Size | TTL |
|-------|------------|------|-----|
| L1 | In-memory (LRU) | 100 entries | Variable |
| L2 | Redis | Configurable | 5-60 min |

### 9.2 Cache Key Generation

- SHA256 hash of normalized query
- Includes query, sort, filters, facets

### 9.3 TTL Strategy

| Query Type | TTL |
|------------|-----|
| Real-time (date sort) | 60s |
| Historical search | 1 hour |
| Default | 5 minutes |

---

## Gmail Search Query DSL

### 10.1 Supported Operators

| Operator | Example | Description |
|----------|---------|-------------|
| from: | `from:john@example.com` | Sender email |
| to: | `to:jane@example.com` | Recipient email |
| subject: | `subject:meeting` | Subject contains |
| label: | `label:important` | Has label |
| has:attachment | `has:attachment` | Has attachments |
| filename: | `filename:pdf` | Attachment filename |
| after: | `after:2024/01/01` | After date |
| before: | `before:2024/02/01` | Before date |
| newer_than: | `newer_than:7d` | Within last N days |
| older_than: | `older_than:1y` | Older than N time |
| is:starred | `is:starred` | Starred emails |
| is:unread | `is:unread` | Unread emails |
| is:important | `is:important` | Important emails |
| larger: | `larger:5M` | Larger than size |
| smaller: | `smaller:100K` | Smaller than size |
| in: | `in:inbox` | In folder |

### 10.2 Complex Query Examples

```
# Recent emails from boss with attachments
from:boss@company.com has:attachment newer_than:7d

# Unread urgent emails
is:unread subject:urgent

# Large emails from last month
larger:5M newer_than:30d

# Emails about project excluding spam
project -label:spam
```

---

## Implementation Details

### 11.1 Core Service API

```python
class EmailSearchService:
    async def search(query: str, options: Dict) -> Dict
    async def search_stream(query: str, options: Dict) -> AsyncGenerator
    async def index_email(email: EmailDocument, priority: str) -> bool
    async def index_batch(emails: List[EmailDocument]) -> Dict
    async def delete_email(email_id: str) -> bool
    async def get_suggestions(prefix: str, field: str, limit: int) -> List[str]
    async def get_stats() -> Dict
```

### 11.2 Configuration Options

```python
config = {
    "elasticsearch": {
        "hosts": ["localhost:9200"],
        "username": None,
        "password": None
    },
    "cache": {
        "l1_max_size": 100,
        "l2_enabled": True,
        "redis_host": "localhost",
        "redis_port": 6379
    },
    "indexing": {
        "batch_size": 100,
        "poll_interval_seconds": 30
    },
    "relevance": {
        "recency_boost_enabled": True,
        "recency_half_life_days": 7,
        "starred_boost": 1.5
    }
}
```

---

## Performance Optimization

### 12.1 Performance Targets

| Metric | Target |
|--------|--------|
| Search P50 latency | < 50ms |
| Search P95 latency | < 200ms |
| Search P99 latency | < 500ms |
| Indexing latency | < 100ms |
| Cache hit rate | > 80% |

### 12.2 Optimization Strategies

1. **Index Tiering**: Route queries to appropriate indices based on date
2. **Bulk Indexing**: Batch document indexing for efficiency
3. **Query Caching**: Cache frequent queries with smart TTL
4. **Connection Pooling**: Reuse Elasticsearch connections
5. **Async Processing**: Non-blocking I/O for all operations

---

## Security Considerations

### 13.1 Input Sanitization

- Maximum query length: 1000 characters
- Forbidden patterns: script tags, javascript:, etc.
- HTML entity encoding for output

### 13.2 Rate Limiting

| Action | Limit |
|--------|-------|
| Search per minute | 60 |
| Search per hour | 1000 |
| Export per day | 10 |

### 13.3 Access Control

- User ID filtering on all queries
- ACL-based result filtering
- Admin permission overrides

---

## Summary

This technical specification provides a comprehensive email search and indexing system for the OpenClaw Windows 10 AI agent framework.

### Key Features

1. **Elasticsearch-based Full-Text Search** with custom analyzers and multi-tier index architecture
2. **Advanced Indexing Pipeline** with NLP enrichment (sentiment, entities, topics)
3. **Flexible Query Processing** supporting natural language and structured queries
4. **Faceted Search** with dynamic facets and multi-select support
5. **Relevance Ranking** with recency, engagement, and personalization signals
6. **Incremental Indexing** via Gmail API History for real-time sync
7. **Multi-Layer Caching** with L1 in-memory and L2 Redis
8. **Gmail DSL Support** with full compatibility for Gmail search operators

### Performance Targets

- Search latency: P50 < 50ms, P95 < 200ms
- Indexing latency: < 100ms per email
- Cache hit rate: > 80%
- Support for millions of emails

### Integration Points

- Gmail API connector
- AI agent context integration
- REST/GraphQL/WebSocket APIs
- Windows 10 native integration

---

**Document Version:** 1.0  
**Last Updated:** 2024  
**Author:** Email Systems Expert Agent
