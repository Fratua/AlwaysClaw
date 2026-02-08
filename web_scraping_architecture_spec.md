# Web Scraping & Data Extraction System Architecture
## OpenClaw Windows 10 AI Agent Framework

**Version:** 1.0.0  
**Date:** 2025-01-20  
**Classification:** Technical Specification

---

## Executive Summary

This document outlines the complete web scraping and data extraction architecture for the Windows 10 OpenClaw-inspired AI agent system. The architecture supports 24/7 autonomous operation with 15 hardcoded agentic loops, handling JavaScript-rendered content, anti-bot evasion, and structured data extraction.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [HTML Parsing Layer](#2-html-parsing-layer)
3. [JavaScript Rendering Engine](#3-javascript-rendering-engine)
4. [Data Extraction Patterns](#4-data-extraction-patterns)
5. [Structured Data Extraction](#5-structured-data-extraction)
6. [Table Extraction & Normalization](#6-table-extraction--normalization)
7. [Anti-Bot Evasion System](#7-anti-bot-evasion-system)
8. [Rate Limiting & Politeness](#8-rate-limiting--politeness)
9. [Content Deduplication](#9-content-deduplication)
10. [Scraping Pipelines](#10-scraping-pipelines)
11. [Implementation Reference](#11-implementation-reference)

---

## 1. System Overview

### 1.1 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         WEB SCRAPING ORCHESTRATOR                           │
│                    (Part of OpenClaw Agent System)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Agent 1    │  │   Agent 2    │  │   Agent 3    │  │   Agent N    │    │
│  │ (News Loop)  │  │(Product Loop)│  │ (Social Loop)│  │ (Custom)     │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         │                 │                 │                 │            │
│         └─────────────────┴─────────────────┴─────────────────┘            │
│                                   │                                         │
│                    ┌──────────────▼──────────────┐                         │
│                    │    Request Queue Manager    │                         │
│                    │    (Priority + Round-Robin) │                         │
│                    └──────────────┬──────────────┘                         │
│                                   │                                         │
│         ┌─────────────────────────┼─────────────────────────┐              │
│         │                         │                         │              │
│  ┌──────▼──────┐         ┌────────▼────────┐      ┌────────▼───────┐      │
│  │   Static    │         │   Dynamic       │      │   API-Based    │      │
│  │   Scraper   │         │   Renderer      │      │   Extractor    │      │
│  │  (Cheerio)  │         │  (Playwright)   │      │  (REST/GraphQL)│      │
│  └──────┬──────┘         └────────┬────────┘      └────────┬───────┘      │
│         │                         │                         │              │
│         └─────────────────────────┼─────────────────────────┘              │
│                                   │                                         │
│                    ┌──────────────▼──────────────┐                         │
│                    │    Content Processor        │                         │
│                    │  (Parse → Extract → Store)  │                         │
│                    └──────────────┬──────────────┘                         │
│                                   │                                         │
│         ┌─────────────────────────┼─────────────────────────┐              │
│         │                         │                         │              │
│  ┌──────▼──────┐         ┌────────▼────────┐      ┌────────▼───────┐      │
│  │  Structured │         │   Unstructured  │      │   Metadata     │      │
│  │   Storage   │         │    Storage      │      │    Cache       │      │
│  │  (SQLite)   │         │   (File System) │      │   (Redis)      │      │
│  └─────────────┘         └─────────────────┘      └────────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Core Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| Static Parser | Cheerio + JSDOM | Fast HTML parsing for static content |
| Dynamic Renderer | Playwright | JavaScript execution, SPA handling |
| API Extractor | Axios + Custom | Direct API endpoint consumption |
| Queue Manager | Bull/BullMQ | Request scheduling and prioritization |
| Data Store | SQLite + Files | Structured and unstructured storage |
| Cache Layer | Redis | Session management, deduplication |

---

## 2. HTML Parsing Layer

### 2.1 Parser Selection Matrix

```javascript
// Parser Selection Strategy
const PARSER_STRATEGY = {
  CHEERIO: {
    useWhen: [
      'Static HTML content',
      'Server-side rendered pages',
      'High throughput required',
      'No JavaScript dependencies'
    ],
    performance: '10,000+ pages/minute',
    memory: 'Low (~50MB per 1000 pages)',
    jsSupport: false
  },
  
  JSDOM: {
    useWhen: [
      'DOM manipulation needed',
      'jQuery-style selectors',
      'Moderate JavaScript execution',
      'Form interaction simulation'
    ],
    performance: '500-1000 pages/minute',
    memory: 'Medium (~200MB per 1000 pages)',
    jsSupport: 'Limited'
  },
  
  PLAYWRIGHT: {
    useWhen: [
      'Single Page Applications (SPA)',
      'Heavy JavaScript frameworks',
      'User interaction required',
      'Anti-bot evasion needed'
    ],
    performance: '50-200 pages/minute',
    memory: 'High (~1GB per 10 browsers)',
    jsSupport: 'Full'
  }
};
```

### 2.2 Parser Implementations

See source files:
- `src/parsers/CheerioParser.ts` - Fast static HTML parsing
- `src/parsers/JSDOMParser.ts` - DOM manipulation with limited JS

---

## 3. JavaScript Rendering Engine

### 3.1 Playwright Browser Pool

Key features:
- **Pool Management**: Configurable max browsers (default: 5) and pages per browser (default: 10)
- **Stealth Mode**: Automatic fingerprint randomization and anti-detection
- **Health Monitoring**: Automatic cleanup of stale browser instances
- **Resource Blocking**: Block images, fonts, and other non-essential resources

### 3.2 Dynamic Content Renderer

Capabilities:
- JavaScript execution and SPA handling
- Wait conditions (selector, function, network idle)
- Automatic scrolling to bottom
- Screenshot capture
- Custom script execution

---

## 4. Data Extraction Patterns

### 4.1 CSS Selector Engine

Features:
- CSS3 selector support
- Attribute extraction
- Built-in transformers (text, html, number, url, date)
- Custom transformation functions
- Fallback values

### 4.2 XPath Engine

Features:
- Full XPath 1.0 support
- Namespace handling
- Multiple extraction modes (text, attribute, count, exists)
- Fallback to CSS for simple expressions

### 4.3 Regex Pattern Engine

Common patterns included:
- Email addresses
- Phone numbers
- URLs
- Prices
- ISO dates
- IPv4 addresses

---

## 5. Structured Data Extraction

### 5.1 JSON-LD Extractor

Supports Schema.org types:
- Article, NewsArticle, BlogPosting
- Product, Offer, Review
- Organization, Person, LocalBusiness
- Event, JobPosting, Recipe
- VideoObject, ImageObject, AudioObject
- WebPage, WebSite, BreadcrumbList
- FAQPage, HowTo, Course

### 5.2 Microdata Extractor

- itemscope/itemprop parsing
- Nested item handling
- Value extraction by element type

### 5.3 RDFa Extractor

- typeof/about/property parsing
- CURIE expansion
- Vocabulary support (schema.org, foaf, dc, og)

---

## 6. Table Extraction & Normalization

### 6.1 Table Detection and Extraction

Features:
- Automatic header detection (th tags, first row heuristics)
- Whitespace cleaning
- Number normalization
- Link extraction
- CSV and JSON export

### 6.2 Table Normalization

Transformations:
- Fill missing cells
- Remove empty rows/columns
- Deduplicate headers
- Standardize types (date, number, boolean)
- Merge compatible tables

---

## 7. Anti-Bot Evasion System

### 7.1 Fingerprint Randomization

Randomized properties:
- User-Agent (Chrome, Firefox, Safari)
- Viewport dimensions
- Hardware concurrency (4-16 cores)
- Device memory (4-16 GB)
- Platform (Windows, Mac, Linux)
- Locale and timezone
- Color depth
- Plugins
- Canvas fingerprint noise

### 7.2 Request Behavior Mimicry

Human-like behaviors:
- Random delays (500ms - 3s)
- Bezier curve mouse movements
- Variable click positions
- Typing with occasional typos
- Reading pauses
- Natural scrolling patterns

### 7.3 CAPTCHA Detection and Handling

Supported CAPTCHA types:
- reCAPTCHA v2/v3
- hCaptcha
- Image CAPTCHAs
- Text CAPTCHAs

Integration points for:
- 2Captcha
- Anti-Captcha
- Manual solving

---

## 8. Rate Limiting & Politeness

### 8.1 Rate Limiter

Configurable limits:
- Requests per second (default: 1)
- Requests per minute (default: 20)
- Requests per hour (default: 200)
- Requests per day (default: 2000)
- Concurrent requests (default: 3)
- Domain-specific overrides

### 8.2 Robots.txt Parser

Features:
- User-agent matching
- Allow/Disallow pattern matching
- Crawl-delay extraction
- Sitemap discovery
- Host directive support

---

## 9. Content Deduplication

### 9.1 Deduplication Engine

Methods:
- **Hash**: SHA-256 exact match
- **SimHash**: Near-duplicate detection (64-bit)
- **MinHash**: Jaccard similarity estimation
- **Hybrid**: Combined approach (recommended)

Configuration:
- Similarity threshold (default: 0.85)
- TTL for fingerprints (default: 24 hours)
- Storage backend (memory/redis/sqlite)

---

## 10. Scraping Pipelines

### 10.1 Pipeline Orchestrator

Pipeline stages:
1. URL pattern matching
2. Rate limiting
3. Deduplication check
4. Content fetching (static/dynamic)
5. Data extraction
6. Post-processing (clean, validate, enrich)
7. Output formatting (JSON/CSV/XML)

### 10.2 Event System

Events emitted:
- `pipelineRegistered`
- `pipelineStarted`
- `pipelineCompleted`
- `pipelineFailed`
- `duplicateDetected`
- `rateLimited`

---

## 11. Implementation Reference

### 11.1 Directory Structure

```
src/
├── parsers/
│   ├── CheerioParser.ts
│   ├── JSDOMParser.ts
│   └── types.ts
├── renderers/
│   ├── BrowserPool.ts
│   ├── DynamicRenderer.ts
│   └── types.ts
├── extractors/
│   ├── CSSSelectorEngine.ts
│   ├── XPathEngine.ts
│   ├── RegexEngine.ts
│   ├── JSONLDExtractor.ts
│   ├── MicrodataExtractor.ts
│   ├── RDFaExtractor.ts
│   ├── TableExtractor.ts
│   ├── TableNormalizer.ts
│   └── types.ts
├── evasion/
│   ├── FingerprintManager.ts
│   ├── BehaviorMimicry.ts
│   └── CaptchaHandler.ts
├── politeness/
│   ├── RateLimiter.ts
│   └── RobotsParser.ts
├── dedup/
│   └── DeduplicationEngine.ts
├── pipelines/
│   ├── PipelineOrchestrator.ts
│   └── configs/
├── types/
│   └── index.ts
└── index.ts
```

### 11.2 Performance Benchmarks

| Component | Throughput | Memory Usage | Notes |
|-----------|------------|--------------|-------|
| Cheerio Parser | 10,000+ pages/min | ~50MB/1000 pages | Static HTML only |
| JSDOM Parser | 500-1000 pages/min | ~200MB/1000 pages | Limited JS support |
| Playwright | 50-200 pages/min | ~1GB/10 browsers | Full JS support |
| Rate Limiter | 10,000+ requests/sec | ~10MB | In-memory tracking |
| Deduplication | 50,000+ checks/sec | ~100MB/1M fingerprints | Hybrid method |

### 11.3 Security Considerations

1. **Data Privacy**: Ensure compliance with GDPR, CCPA when scraping personal data
2. **Terms of Service**: Respect website ToS and robots.txt directives
3. **Rate Limiting**: Implement conservative limits to avoid being blocked
4. **Data Storage**: Encrypt sensitive data at rest and in transit
5. **Access Control**: Implement authentication for scraping APIs

---

## Source Code Files

Full TypeScript implementations are provided in separate files:

1. `src/parsers/CheerioParser.ts` - Cheerio-based HTML parser
2. `src/parsers/JSDOMParser.ts` - JSDOM-based parser
3. `src/renderers/BrowserPool.ts` - Playwright browser pool
4. `src/renderers/DynamicRenderer.ts` - Dynamic content renderer
5. `src/extractors/CSSSelectorEngine.ts` - CSS selector extraction
6. `src/extractors/XPathEngine.ts` - XPath extraction
7. `src/extractors/RegexEngine.ts` - Regex pattern extraction
8. `src/extractors/JSONLDExtractor.ts` - JSON-LD structured data
9. `src/extractors/MicrodataExtractor.ts` - Microdata extraction
10. `src/extractors/RDFaExtractor.ts` - RDFa extraction
11. `src/extractors/TableExtractor.ts` - Table extraction
12. `src/extractors/TableNormalizer.ts` - Table normalization
13. `src/evasion/FingerprintManager.ts` - Fingerprint randomization
14. `src/evasion/BehaviorMimicry.ts` - Human behavior mimicry
15. `src/evasion/CaptchaHandler.ts` - CAPTCHA detection
16. `src/politeness/RateLimiter.ts` - Rate limiting
17. `src/politeness/RobotsParser.ts` - robots.txt parsing
18. `src/dedup/DeduplicationEngine.ts` - Content deduplication
19. `src/pipelines/PipelineOrchestrator.ts` - Pipeline orchestration
20. `src/types/index.ts` - Type definitions

---

*Document Version: 1.0.0*  
*Last Updated: 2025-01-20*  
*Author: OpenClay AI Agent System*
