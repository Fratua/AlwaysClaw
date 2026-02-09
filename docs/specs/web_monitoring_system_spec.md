# Web Monitoring and Change Detection System
## Technical Specification for Windows 10 OpenClaw AI Agent

**Version:** 1.0  
**Date:** 2025  
**System:** OpenClaw Windows 10 AI Agent Framework  
**AI Core:** GPT-5.2 with Extended Thinking Capability

---

## Table of Contents
1. [System Overview](#1-system-overview)
2. [Architecture Design](#2-architecture-design)
3. [DOM Monitoring & Change Detection](#3-dom-monitoring--change-detection)
4. [Visual Regression Testing](#4-visual-regression-testing)
5. [Content Change Detection Algorithms](#5-content-change-detection-algorithms)
6. [Monitoring Schedule & Frequency](#6-monitoring-schedule--frequency)
7. [Alert Generation & Notification](#7-alert-generation--notification)
8. [Change Categorization System](#8-change-categorization-system)
9. [Monitoring Dashboard](#9-monitoring-dashboard)
10. [Historical Change Tracking](#10-historical-change-tracking)
11. [Implementation Code](#11-implementation-code)

---

## 1. System Overview

### 1.1 Purpose
The Web Monitoring and Change Detection System provides continuous surveillance of target websites, detecting and categorizing changes across DOM structure, visual appearance, and content. Integrated with the OpenClaw AI agent framework, it enables autonomous web intelligence gathering with intelligent alerting.

### 1.2 Key Capabilities
- **Real-time DOM monitoring** with configurable selectors
- **Visual regression testing** using image comparison algorithms
- **Content fingerprinting** for text-based change detection
- **Multi-channel alerting** (Gmail, Twilio SMS/Voice, TTS)
- **Historical tracking** with diff visualization
- **AI-powered change analysis** using GPT-5.2

### 1.3 System Components
```
┌─────────────────────────────────────────────────────────────────┐
│                    WEB MONITORING SYSTEM                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Monitor    │  │   Change     │  │    Alert     │          │
│  │   Engine     │──│   Detector   │──│   Manager    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         │                 │                 │                  │
│  ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐          │
│  │   Browser   │   │   Visual    │   │ Notification│          │
│  │   Control   │   │  Comparator │   │   Router    │          │
│  └─────────────┘   └─────────────┘   └─────────────┘          │
│         │                 │                 │                  │
│  ┌──────▼─────────────────▼─────────────────▼──────┐          │
│  │              Data Storage Layer                 │          │
│  │  (Snapshots │ Diffs │ History │ Configurations) │          │
│  └─────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Architecture Design

### 2.1 Core Modules

#### 2.1.1 MonitorEngine
Central orchestrator managing all monitoring tasks, scheduling, and coordination between components.

#### 2.1.2 DOMWatcher
Specialized module for DOM structure monitoring using CSS selectors and XPath queries.

#### 2.1.3 VisualComparator
Image processing module for screenshot comparison and visual change detection.

#### 2.1.4 ContentAnalyzer
Text analysis module for content fingerprinting and semantic change detection.

#### 2.1.5 AlertManager
Notification routing system with multi-channel support and escalation policies.

#### 2.1.6 HistoryTracker
Data persistence module for change history, snapshots, and trend analysis.

### 2.2 Data Flow
```
Target URL → Browser Capture → Parallel Analysis → Change Detection → Alert Decision → Notification
                │                    │                    │                │
                ├─ DOM Snapshot      ├─ DOM Diff          ├─ Categorize    ├─ Priority
                ├─ Screenshot        ├─ Image Diff        ├─ Severity      ├─ Channel Select
                └─ Content Hash      └─ Text Diff         └─ Store         └─ Send
```

---

## 3. DOM Monitoring & Change Detection

### 3.1 DOM Watcher Architecture

```python
class DOMWatcher:
    """
    Monitors DOM changes using multiple detection strategies:
    - Full DOM hash comparison
    - Selective element monitoring
    - Attribute change tracking
    - Text content monitoring
    """
```

### 3.2 Detection Strategies

#### 3.2.1 Full DOM Monitoring
- Captures complete HTML document
- Generates cryptographic hash (SHA-256)
- Compares against baseline hash
- Triggers on any structural change

#### 3.2.2 Selective Element Monitoring
- CSS selector-based targeting
- XPath query support
- Multi-element batch monitoring
- Configurable depth and scope

#### 3.2.3 Attribute Tracking
- Monitors specific HTML attributes
- Class name changes
- Style property modifications
- Data attribute updates

### 3.3 DOM Change Types

| Change Type | Description | Detection Method |
|-------------|-------------|------------------|
| `ELEMENT_ADDED` | New DOM elements appeared | Node count + selector match |
| `ELEMENT_REMOVED` | Existing elements disappeared | Selector not found |
| `TEXT_CHANGED` | Text content modified | InnerText comparison |
| `ATTRIBUTE_CHANGED` | HTML attributes modified | Attribute hash comparison |
| `STRUCTURE_CHANGED` | Parent-child relationships altered | DOM tree hash |
| `CLASS_CHANGED` | CSS classes modified | ClassList comparison |

### 3.4 DOM Monitoring Configuration

```yaml
dom_monitoring:
  strategies:
    full_dom:
      enabled: true
      hash_algorithm: "sha256"
      ignore_whitespace: true
      
    selective_elements:
      enabled: true
      selectors:
        - "#main-content"
        - ".article-body"
        - "[data-track='price']"
        - "//div[@class='status']"
      
    attribute_tracking:
      enabled: true
      attributes:
        - "class"
        - "style"
        - "data-status"
        - "aria-label"
      
  filters:
    ignore_elements:
      - "script"
      - "style"
      - "noscript"
      - ".advertisement"
    ignore_attributes:
      - "data-timestamp"
      - "data-session"
```

---

## 4. Visual Regression Testing

### 4.1 Screenshot Capture System

#### 4.1.1 Capture Modes
- **Full Page**: Complete scrollable content
- **Viewport**: Visible area only
- **Element**: Specific DOM element
- **Region**: Custom bounding box

#### 4.1.2 Capture Settings
```yaml
screenshot_config:
  full_page:
    enabled: true
    format: "png"
    quality: 90
    scale: 1.0
    
  viewport:
    enabled: true
    width: 1920
    height: 1080
    device_scale_factor: 1
    
  element:
    enabled: true
    selectors:
      - "#hero-section"
      - ".pricing-table"
      
  wait_conditions:
    network_idle: true
    dom_stable: true
    custom_selectors_visible: []
```

### 4.2 Image Comparison Algorithms

#### 4.2.1 Pixel-by-Pixel Comparison
```
Algorithm: PixelDiff
- Compares each pixel RGB value
- Configurable threshold (0-255)
- Fast but sensitive to noise
- Use case: Precise visual matching
```

#### 4.2.2 Perceptual Hash (pHash)
```
Algorithm: PerceptualHash
- Generates 64-bit image fingerprint
- Resistant to minor variations
- Hamming distance comparison
- Use case: Similar image detection
```

#### 4.2.3 Structural Similarity (SSIM)
```
Algorithm: SSIM
- Measures structural similarity
- Range: 0.0 to 1.0
- Accounts for luminance, contrast, structure
- Use case: Human-perceived similarity
```

#### 4.2.4 Histogram Comparison
```
Algorithm: HistogramDiff
- Compares color distribution
- Chi-square or correlation methods
- Ignores spatial information
- Use case: Color scheme changes
```

### 4.3 Visual Change Detection Matrix

| Algorithm | Speed | Accuracy | Noise Resistant | Best For |
|-----------|-------|----------|-----------------|----------|
| PixelDiff | Fast | High | No | Exact matching |
| pHash | Medium | Medium | Yes | Similar images |
| SSIM | Slow | High | Yes | Visual quality |
| Histogram | Fast | Low | Yes | Color changes |

### 4.4 Visual Diff Output

```python
class VisualDiffResult:
    """
    Visual comparison result structure
    """
    similarity_score: float      # 0.0 - 1.0
    diff_percentage: float       # 0.0 - 100.0
    changed_regions: List[Region]  # Bounding boxes of changes
    diff_image_path: str         # Path to diff visualization
    baseline_image_path: str     # Path to baseline screenshot
    current_image_path: str      # Path to current screenshot
    algorithm_used: str          # Comparison algorithm name
    threshold_exceeded: bool     # Whether change exceeds threshold
```

---

## 5. Content Change Detection Algorithms

### 5.1 Text Content Analysis

#### 5.1.1 Content Fingerprinting
```python
class ContentFingerprint:
    """
    Creates multiple fingerprints for robust change detection
    """
    methods = {
        'simhash': SimHash(),           # Near-duplicate detection
        'minhash': MinHash(),           # Similarity estimation
        'shingling': Shingling(),       # N-gram based
        'tfidf': TFIDFHash(),           # Semantic fingerprint
        'semantic': SemanticEmbedding()  # AI-powered similarity
    }
```

#### 5.1.2 Fingerprint Comparison Methods

| Method | Description | Collision Rate | Use Case |
|--------|-------------|----------------|----------|
| SimHash | Locality-sensitive hashing | Low | Near-duplicate detection |
| MinHash | Jaccard similarity estimation | Medium | Set similarity |
| Shingling | N-gram fingerprinting | Low | Text comparison |
| TF-IDF | Term frequency weighting | Very Low | Content relevance |
| Semantic | AI embeddings | Very Low | Meaning-based detection |

### 5.2 Text Diff Algorithms

#### 5.2.1 Unified Diff
- Line-by-line comparison
- Context lines configuration
- Standard patch format

#### 5.2.2 Word-Level Diff
- Token-based comparison
- Inline change highlighting
- Better for prose content

#### 5.2.3 Character-Level Diff
- Finest granularity
- Maximum precision
- Higher computational cost

### 5.3 Content Extraction

```yaml
content_extraction:
  enabled: true
  
  text_content:
    enabled: true
    selectors:
      - "article"
      - "main"
      - "[role='main']"
    exclude:
      - "nav"
      - "footer"
      - ".sidebar"
      
  metadata:
    enabled: true
    extract:
      - title
      - description
      - keywords
      - author
      - publish_date
      - last_modified
      
  structured_data:
    enabled: true
    formats:
      - json_ld
      - microdata
      - opengraph
      - twitter_cards
```

### 5.4 Change Significance Scoring

```python
def calculate_significance_score(change) -> float:
    """
    Calculate significance score (0.0 - 1.0) based on:
    - Content type (title, body, metadata)
    - Change magnitude (additions, deletions)
    - Semantic importance
    - Historical patterns
    """
    scores = {
        'title_change': 0.9,
        'price_change': 0.85,
        'availability_change': 0.8,
        'major_content_update': 0.7,
        'minor_text_edit': 0.3,
        'formatting_change': 0.2,
        'timestamp_update': 0.1
    }
    return weighted_score(scores, change)
```

---

## 6. Monitoring Schedule & Frequency

### 6.1 Scheduling Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SCHEDULING ENGINE                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Cron      │  │  Adaptive   │  │   Event     │         │
│  │  Scheduler  │  │  Frequency  │  │  Triggered  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Cron-Based Scheduling

```yaml
scheduling:
  cron_jobs:
    # High-frequency monitoring (critical sites)
    critical:
      pattern: "*/5 * * * *"  # Every 5 minutes
      sites:
        - "https://status.example.com"
        - "https://api.example.com/health"
      
    # Standard monitoring
    standard:
      pattern: "0 * * * *"  # Every hour
      sites:
        - "https://news.example.com"
        - "https://blog.example.com"
      
    # Daily monitoring
    daily:
      pattern: "0 9 * * *"  # 9 AM daily
      sites:
        - "https://docs.example.com"
        - "https://changelog.example.com"
      
    # Weekly monitoring
    weekly:
      pattern: "0 9 * * 1"  # Monday 9 AM
      sites:
        - "https://careers.example.com"
```

### 6.3 Adaptive Frequency Algorithm

```python
class AdaptiveFrequency:
    """
    Dynamically adjusts monitoring frequency based on:
    - Historical change patterns
    - Time-of-day patterns
    - Recent activity levels
    - User-defined importance
    """
    
    def calculate_next_check(self, monitor):
        base_interval = monitor.base_interval
        
        # Adjust based on change frequency
        if monitor.change_frequency > 0.5:  # Frequent changes
            return base_interval * 0.5  # Check more often
        elif monitor.change_frequency < 0.1:  # Rare changes
            return base_interval * 2.0  # Check less often
            
        # Adjust based on time patterns
        if self.is_business_hours():
            return base_interval * 0.8
            
        return base_interval
```

### 6.4 Event-Triggered Monitoring

```yaml
event_triggers:
  webhook_received:
    enabled: true
    endpoint: "/webhooks/monitor-trigger"
    
  file_system_events:
    enabled: true
    watch_paths:
      - "./config/sites.json"
      
  system_events:
    enabled: true
    triggers:
      - name: "on_system_startup"
        action: "run_all_monitors"
      - name: "on_network_restore"
        action: "run_critical_monitors"
```

### 6.5 Frequency Presets

| Preset | Interval | Use Case |
|--------|----------|----------|
| `realtime` | 30 seconds | Critical systems |
| `frequent` | 5 minutes | High-priority sites |
| `regular` | 1 hour | Standard monitoring |
| `daily` | 24 hours | Low-priority content |
| `weekly` | 7 days | Archive monitoring |
| `adaptive` | Dynamic | AI-optimized |

---

## 7. Alert Generation & Notification

### 7.1 Alert Pipeline

```
Change Detected → Severity Assessment → Alert Generation → Channel Selection → Notification Sent
                       │                       │                  │
                       ├─ Critical             ├─ Immediate       ├─ SMS
                       ├─ High                 ├─ High Priority   ├─ Voice
                       ├─ Medium               ├─ Standard        ├─ Email
                       └─ Low                  └─ Digest          └─ Dashboard
```

### 7.2 Severity Levels

```python
class SeverityLevel(Enum):
    CRITICAL = 5   # Immediate notification, all channels
    HIGH = 4       # Priority notification, SMS + Email
    MEDIUM = 3     # Standard notification, Email
    LOW = 2        # Digest notification, Dashboard
    INFO = 1       # Log only, no notification
```

### 7.3 Severity Determination

```python
def determine_severity(change) -> SeverityLevel:
    """
    Determine alert severity based on multiple factors
    """
    score = 0
    
    # Content type weight
    if change.type == 'PRICE_CHANGE':
        score += 40
    elif change.type == 'AVAILABILITY_CHANGE':
        score += 35
    elif change.type == 'SECURITY_CHANGE':
        score += 50
    elif change.type == 'CONTENT_UPDATE':
        score += 20
        
    # Change magnitude
    score += min(change.percentage_changed * 0.5, 30)
    
    # Site importance
    score += change.site.importance_score * 10
    
    # Historical significance
    if change.is_unusual_pattern:
        score += 15
        
    # Map score to severity
    if score >= 80:
        return SeverityLevel.CRITICAL
    elif score >= 60:
        return SeverityLevel.HIGH
    elif score >= 40:
        return SeverityLevel.MEDIUM
    elif score >= 20:
        return SeverityLevel.LOW
    else:
        return SeverityLevel.INFO
```

### 7.4 Notification Channels

#### 7.4.1 Gmail Integration
```python
class GmailNotifier:
    """
    Gmail notification handler with rich HTML formatting
    """
    config = {
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'use_tls': True,
        'templates': {
            'critical': 'alert_critical.html',
            'standard': 'alert_standard.html',
            'digest': 'alert_digest.html'
        }
    }
```

#### 7.4.2 Twilio SMS
```python
class TwilioSMSNotifier:
    """
    SMS notifications for critical alerts
    """
    config = {
        'max_length': 1600,
        'include_link': True,
        'shorten_urls': True,
        'template': "ALERT: {severity} change detected on {site}. {summary}. View: {link}"
    }
```

#### 7.4.3 Twilio Voice
```python
class TwilioVoiceNotifier:
    """
    Voice call notifications for critical alerts
    """
    config = {
        'enabled_for_severity': ['CRITICAL'],
        'tts_voice': 'Polly.Joanna',
        'max_duration': 60,
        'retry_count': 3,
        'template': "This is an alert from OpenClaw monitoring. A critical change has been detected on {site_name}. Please check your dashboard immediately."
    }
```

#### 7.4.4 Text-to-Speech (TTS)
```python
class TTSNotifier:
    """
    Local TTS notifications for the AI agent
    """
    config = {
        'enabled': True,
        'voice_id': 'default',
        'speak_on_severity': ['CRITICAL', 'HIGH'],
        'template': "Attention: {severity} level change detected on {site}"
    }
```

### 7.5 Alert Templates

```html
<!-- Critical Alert Email Template -->
<!DOCTYPE html>
<html>
<head>
    <style>
        .alert-critical { background: #dc3545; color: white; padding: 20px; }
        .alert-high { background: #fd7e14; color: white; padding: 15px; }
        .diff-visual { border: 1px solid #ddd; margin: 10px 0; }
        .change-stats { background: #f8f9fa; padding: 15px; }
    </style>
</head>
<body>
    <div class="alert-{{severity}}">
        <h1>{{severity}} Alert: Change Detected</h1>
        <p>Site: {{site_name}}</p>
        <p>Detected at: {{timestamp}}</p>
    </div>
    
    <div class="change-stats">
        <h3>Change Summary</h3>
        <ul>
            <li>Type: {{change_type}}</li>
            <li>Elements affected: {{affected_count}}</li>
            <li>Similarity: {{similarity_score}}%</li>
        </ul>
    </div>
    
    <div class="diff-visual">
        <img src="cid:diff-image" alt="Visual Diff" />
    </div>
    
    <a href="{{dashboard_link}}" class="btn">View in Dashboard</a>
</body>
</html>
```

### 7.6 Alert Throttling & Deduplication

```python
class AlertThrottler:
    """
    Prevents alert fatigue through intelligent throttling
    """
    
    def should_send_alert(self, alert) -> bool:
        # Check for recent similar alerts
        recent_alerts = self.get_recent_alerts(
            site=alert.site,
            window_minutes=30
        )
        
        # Deduplicate similar changes
        for recent in recent_alerts:
            if self.is_similar_change(alert, recent):
                return False
                
        # Rate limiting per site
        if self.rate_limit_exceeded(alert.site):
            return False
            
        # Cooldown period
        if self.in_cooldown(alert.site, alert.type):
            return False
            
        return True
```

---

## 8. Change Categorization System

### 8.1 Change Type Taxonomy

```python
class ChangeType(Enum):
    # Content Changes
    TEXT_ADDED = "text_added"
    TEXT_REMOVED = "text_removed"
    TEXT_MODIFIED = "text_modified"
    
    # Structural Changes
    ELEMENT_ADDED = "element_added"
    ELEMENT_REMOVED = "element_removed"
    ELEMENT_REORDERED = "element_reordered"
    
    # Visual Changes
    STYLE_CHANGED = "style_changed"
    LAYOUT_CHANGED = "layout_changed"
    IMAGE_CHANGED = "image_changed"
    
    # Functional Changes
    LINK_CHANGED = "link_changed"
    FORM_CHANGED = "form_changed"
    SCRIPT_CHANGED = "script_changed"
    
    # Business Logic Changes
    PRICE_CHANGED = "price_changed"
    AVAILABILITY_CHANGED = "availability_changed"
    STATUS_CHANGED = "status_changed"
    
    # Metadata Changes
    META_CHANGED = "meta_changed"
    TITLE_CHANGED = "title_changed"
    DESCRIPTION_CHANGED = "description_changed"
```

### 8.2 Auto-Categorization Rules

```yaml
change_categories:
  price_change:
    patterns:
      - selector: "[data-track='price']"
      - selector: ".price"
      - selector: "meta[property='product:price']"
    severity: HIGH
    
  availability_change:
    patterns:
      - selector: ".in-stock"
      - selector: ".out-of-stock"
      - selector: "[data-availability]"
    severity: HIGH
    
  content_update:
    patterns:
      - selector: "article"
      - selector: ".blog-post"
      - selector: ".news-item"
    severity: MEDIUM
    
  security_related:
    patterns:
      - selector: ".security-notice"
      - selector: ".privacy-policy"
      - selector: ".terms-of-service"
    severity: CRITICAL
```

### 8.3 Change Classification AI

```python
class ChangeClassifier:
    """
    GPT-5.2 powered change classification
    """
    
    async def classify_change(self, change) -> ClassificationResult:
        prompt = f"""
        Analyze this website change and classify it:
        
        Site: {change.site_url}
        Change Type: {change.detection_method}
        Diff Summary: {change.summary}
        
        Classify into categories:
        1. Primary category (business/technical/visual/content)
        2. Urgency level (critical/high/medium/low)
        3. User impact (none/low/medium/high)
        4. Recommended action
        
        Return JSON format.
        """
        
        response = await self.gpt52.generate(
            prompt=prompt,
            temperature=0.3,
            response_format="json"
        )
        
        return ClassificationResult.parse(response)
```

### 8.4 Change Impact Matrix

| Change Category | User Impact | Business Impact | Notification Priority |
|----------------|-------------|-----------------|----------------------|
| Price Change | High | Critical | Immediate |
| Availability | High | High | Immediate |
| Security | High | Critical | Immediate |
| Content Update | Medium | Low | Standard |
| Visual Refresh | Low | Low | Digest |
| Bug Fix | Medium | Medium | Standard |
| Performance | Low | Medium | Low |

---

## 9. Monitoring Dashboard

### 9.1 Dashboard Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      MONITORING DASHBOARD                          │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │   Overview   │  │  Site List   │  │  Alert Feed  │              │
│  │   Panel      │  │   Panel      │  │   Panel      │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
├─────────────────────────────────────────────────────────────────────┤
│  ┌────────────────────────┐  ┌────────────────────────┐            │
│  │     Diff Viewer        │  │    History Timeline    │            │
│  │    (Side-by-side)      │  │    (Change History)    │            │
│  └────────────────────────┘  └────────────────────────┘            │
├─────────────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────────────┐    │
│  │                   Configuration Panel                       │    │
│  └────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### 9.2 Dashboard Components

#### 9.2.1 Overview Panel
- Total monitors active
- Changes detected (24h)
- Alert count by severity
- System health status
- Average response time

#### 9.2.2 Site List Panel
- Monitored sites table
- Last check timestamp
- Status indicator
- Change frequency graph
- Quick actions

#### 9.2.3 Alert Feed Panel
- Real-time alert stream
- Severity color coding
- One-click acknowledgment
- Alert filtering
- Bulk actions

#### 9.2.4 Diff Viewer
- Side-by-side comparison
- Screenshot overlay
- DOM tree diff
- Text diff highlighting
- Change navigation

#### 9.2.5 History Timeline
- Chronological change log
- Filter by site/type/date
- Trend visualization
- Export functionality

### 9.3 Dashboard API Endpoints

```yaml
api_endpoints:
  # Overview
  GET /api/dashboard/overview:
    returns: site_count, alert_count, health_status
    
  # Sites
  GET /api/sites:
    returns: list of monitored sites
    
  POST /api/sites:
    body: {url, config}
    creates: new monitor
    
  GET /api/sites/{id}/status:
    returns: current status, last_check, last_change
    
  # Changes
  GET /api/changes:
    query: site_id, type, severity, date_range
    returns: paginated change list
    
  GET /api/changes/{id}/diff:
    returns: diff details, images, comparison
    
  # Alerts
  GET /api/alerts:
    query: status, severity, site
    returns: alert list
    
  POST /api/alerts/{id}/acknowledge:
    acknowledges: alert
    
  # History
  GET /api/history:
    query: site_id, period
    returns: historical data
```

### 9.4 Dashboard UI Components

```html
<!-- Main Dashboard Layout -->
<div class="dashboard">
    <header class="dashboard-header">
        <h1>OpenClaw Web Monitor</h1>
        <div class="status-indicators">
            <span class="status-dot active"></span>
            <span>System Active</span>
        </div>
    </header>
    
    <div class="dashboard-grid">
        <!-- Stats Cards -->
        <div class="stat-card">
            <h3>Active Monitors</h3>
            <div class="stat-value" id="monitor-count">0</div>
        </div>
        
        <div class="stat-card alert-critical">
            <h3>Critical Alerts</h3>
            <div class="stat-value" id="critical-count">0</div>
        </div>
        
        <div class="stat-card">
            <h3>Changes Today</h3>
            <div class="stat-value" id="changes-count">0</div>
        </div>
        
        <!-- Sites Table -->
        <div class="panel sites-panel">
            <h2>Monitored Sites</h2>
            <table id="sites-table">
                <thead>
                    <tr>
                        <th>Site</th>
                        <th>Status</th>
                        <th>Last Check</th>
                        <th>Changes</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody></tbody>
            </table>
        </div>
        
        <!-- Alert Feed -->
        <div class="panel alerts-panel">
            <h2>Recent Alerts</h2>
            <div id="alert-feed"></div>
        </div>
    </div>
</div>
```

---

## 10. Historical Change Tracking

### 10.1 Data Storage Schema

```sql
-- Monitored Sites Table
CREATE TABLE monitored_sites (
    id UUID PRIMARY KEY,
    url VARCHAR(2048) NOT NULL,
    name VARCHAR(255),
    config JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE
);

-- Snapshots Table
CREATE TABLE snapshots (
    id UUID PRIMARY KEY,
    site_id UUID REFERENCES monitored_sites(id),
    captured_at TIMESTAMP DEFAULT NOW(),
    dom_hash VARCHAR(64),
    content_hash VARCHAR(64),
    visual_hash VARCHAR(64),
    screenshot_path VARCHAR(512),
    dom_path VARCHAR(512),
    content_path VARCHAR(512),
    metadata JSONB
);

-- Changes Table
CREATE TABLE changes (
    id UUID PRIMARY KEY,
    site_id UUID REFERENCES monitored_sites(id),
    snapshot_before_id UUID REFERENCES snapshots(id),
    snapshot_after_id UUID REFERENCES snapshots(id),
    detected_at TIMESTAMP DEFAULT NOW(),
    change_type VARCHAR(50),
    severity VARCHAR(20),
    category VARCHAR(50),
    description TEXT,
    diff_data JSONB,
    significance_score FLOAT,
    is_acknowledged BOOLEAN DEFAULT FALSE
);

-- Alerts Table
CREATE TABLE alerts (
    id UUID PRIMARY KEY,
    change_id UUID REFERENCES changes(id),
    sent_at TIMESTAMP DEFAULT NOW(),
    channel VARCHAR(50),
    recipient VARCHAR(255),
    status VARCHAR(20),
    content TEXT
);

-- Indexes for performance
CREATE INDEX idx_snapshots_site_time ON snapshots(site_id, captured_at);
CREATE INDEX idx_changes_site_time ON changes(site_id, detected_at);
CREATE INDEX idx_changes_severity ON changes(severity);
CREATE INDEX idx_alerts_change ON alerts(change_id);
```

### 10.2 Storage Management

```yaml
storage_config:
  retention_policy:
    snapshots:
      keep_last: 100
      keep_days: 90
      archive_after_days: 30
      
    changes:
      keep_days: 365
      archive_after_days: 180
      
    alerts:
      keep_days: 90
      
  compression:
    enabled: true
    algorithm: "gzip"
    level: 6
    
  cleanup_schedule:
    enabled: true
    cron: "0 2 * * *"  # Daily at 2 AM
```

### 10.3 Historical Analysis

```python
class HistoricalAnalyzer:
    """
    Analyze historical change patterns
    """
    
    def generate_trend_report(self, site_id, days=30):
        """Generate change trend analysis"""
        changes = self.get_changes(site_id, days)
        
        return {
            'total_changes': len(changes),
            'changes_by_type': self.group_by_type(changes),
            'changes_by_severity': self.group_by_severity(changes),
            'peak_activity_times': self.find_peak_times(changes),
            'change_frequency_trend': self.calculate_trend(changes),
            'unusual_patterns': self.detect_anomalies(changes)
        }
    
    def predict_next_change(self, site_id):
        """Predict when next change might occur"""
        history = self.get_change_history(site_id)
        intervals = [h.interval for h in history]
        
        # Simple prediction based on average interval
        avg_interval = statistics.mean(intervals)
        last_change = history[-1].timestamp
        
        return last_change + timedelta(seconds=avg_interval)
```

### 10.4 Change Timeline Visualization

```python
class TimelineGenerator:
    """
    Generate visual timeline of changes
    """
    
    def generate_timeline(self, site_id, start_date, end_date):
        changes = self.get_changes_in_range(site_id, start_date, end_date)
        
        timeline_data = []
        for change in changes:
            timeline_data.append({
                'date': change.detected_at.isoformat(),
                'type': change.change_type,
                'severity': change.severity,
                'title': self.get_change_title(change),
                'description': change.description,
                'link': f"/changes/{change.id}"
            })
            
        return timeline_data
```

---

## 11. Implementation Code

### 11.1 Core Monitor Engine

```python
# monitor_engine.py
import asyncio
import hashlib
import json
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
from playwright.async_api import async_playwright

class ChangeType(Enum):
    DOM = "dom"
    VISUAL = "visual"
    CONTENT = "content"

class SeverityLevel(Enum):
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    INFO = 1

@dataclass
class MonitorConfig:
    url: str
    name: str
    check_interval: int = 3600  # seconds
    dom_monitoring: bool = True
    visual_monitoring: bool = True
    content_monitoring: bool = True
    selectors: List[str] = field(default_factory=list)
    alert_threshold: float = 0.1
    notification_channels: List[str] = field(default_factory=lambda: ["email"])

@dataclass
class Snapshot:
    site_id: str
    timestamp: datetime
    dom_hash: str
    content_hash: str
    visual_hash: str
    screenshot_path: Optional[str] = None
    dom_content: Optional[str] = None
    text_content: Optional[str] = None

@dataclass
class Change:
    site_id: str
    change_type: ChangeType
    severity: SeverityLevel
    detected_at: datetime
    description: str
    diff_data: Dict
    snapshot_before: Snapshot
    snapshot_after: Snapshot

class DOMWatcher:
    """Monitors DOM changes using multiple strategies"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.ignore_elements = config.get('ignore_elements', [])
        self.ignore_attributes = config.get('ignore_attributes', [])
    
    async def capture_dom(self, page) -> Dict:
        """Capture DOM state from page"""
        dom_data = await page.evaluate("""
            () => {
                const getElementData = (el) => ({
                    tag: el.tagName.toLowerCase(),
                    id: el.id,
                    class: el.className,
                    text: el.innerText?.substring(0, 500),
                    attributes: Array.from(el.attributes)
                        .filter(a => !['data-timestamp', 'data-session'].includes(a.name))
                        .reduce((acc, a) => ({...acc, [a.name]: a.value}), {})
                });
                
                return {
                    title: document.title,
                    url: window.location.href,
                    elements: Array.from(document.querySelectorAll('body *'))
                        .filter(el => !['script', 'style', 'noscript'].includes(el.tagName.toLowerCase()))
                        .map(getElementData)
                };
            }
        """)
        return dom_data
    
    def compute_hash(self, dom_data: Dict) -> str:
        """Compute SHA-256 hash of DOM content"""
        content = json.dumps(dom_data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def detect_changes(self, before: Dict, after: Dict) -> List[Dict]:
        """Detect DOM changes between two states"""
        changes = []
        
        # Check title change
        if before.get('title') != after.get('title'):
            changes.append({
                'type': 'TITLE_CHANGED',
                'before': before.get('title'),
                'after': after.get('title')
            })
        
        # Compare elements
        before_elements = {self._element_key(e): e for e in before.get('elements', [])}
        after_elements = {self._element_key(e): e for e in after.get('elements', [])}
        
        # Find added elements
        for key, element in after_elements.items():
            if key not in before_elements:
                changes.append({
                    'type': 'ELEMENT_ADDED',
                    'element': element
                })
        
        # Find removed elements
        for key, element in before_elements.items():
            if key not in after_elements:
                changes.append({
                    'type': 'ELEMENT_REMOVED',
                    'element': element
                })
        
        # Find modified elements
        for key in set(before_elements.keys()) & set(after_elements.keys()):
            before_el = before_elements[key]
            after_el = after_elements[key]
            
            if before_el.get('text') != after_el.get('text'):
                changes.append({
                    'type': 'TEXT_CHANGED',
                    'element': after_el,
                    'before_text': before_el.get('text'),
                    'after_text': after_el.get('text')
                })
            
            if before_el.get('attributes') != after_el.get('attributes'):
                changes.append({
                    'type': 'ATTRIBUTES_CHANGED',
                    'element': after_el,
                    'before_attrs': before_el.get('attributes'),
                    'after_attrs': after_el.get('attributes')
                })
        
        return changes
    
    def _element_key(self, element: Dict) -> str:
        """Generate unique key for element"""
        return f"{element.get('tag')}#{element.get('id')}.{element.get('class')}"


class VisualComparator:
    """Compares screenshots for visual changes"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.threshold = config.get('threshold', 0.1)
        self.algorithm = config.get('algorithm', 'pixel')
    
    async def capture_screenshot(self, page, full_page: bool = True) -> bytes:
        """Capture screenshot from page"""
        screenshot = await page.screenshot(
            full_page=full_page,
            type='png'
        )
        return screenshot
    
    def compare_images(self, baseline: bytes, current: bytes) -> Dict:
        """Compare two images and return diff metrics"""
        from PIL import Image
        import io
        
        # Load images
        img1 = Image.open(io.BytesIO(baseline))
        img2 = Image.open(io.BytesIO(current))
        
        # Ensure same size
        if img1.size != img2.size:
            img2 = img2.resize(img1.size)
        
        # Pixel-by-pixel comparison
        pixels1 = list(img1.getdata())
        pixels2 = list(img2.getdata())
        
        diff_pixels = sum(
            1 for p1, p2 in zip(pixels1, pixels2)
            if self._pixel_diff(p1, p2) > self.threshold
        )
        
        total_pixels = len(pixels1)
        diff_percentage = (diff_pixels / total_pixels) * 100
        
        return {
            'diff_pixels': diff_pixels,
            'total_pixels': total_pixels,
            'diff_percentage': diff_percentage,
            'similarity_score': 100 - diff_percentage,
            'threshold_exceeded': diff_percentage > (self.threshold * 100)
        }
    
    def _pixel_diff(self, p1, p2) -> float:
        """Calculate difference between two pixels"""
        if len(p1) == 4 and len(p2) == 4:
            # RGBA
            r1, g1, b1, a1 = p1
            r2, g2, b2, a2 = p2
        else:
            # RGB
            r1, g1, b1 = p1[:3]
            r2, g2, b2 = p2[:3]
            a1 = a2 = 255
        
        # Weighted RGB difference
        diff = (
            abs(r1 - r2) * 0.299 +
            abs(g1 - g2) * 0.587 +
            abs(b1 - b2) * 0.114
        ) / 255.0
        
        return diff


class ContentAnalyzer:
    """Analyzes text content for changes"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.selectors = config.get('selectors', ['body'])
    
    async def extract_content(self, page) -> Dict:
        """Extract text content from page"""
        content = await page.evaluate("""
            (selectors) => {
                const results = {};
                selectors.forEach(selector => {
                    const elements = document.querySelectorAll(selector);
                    results[selector] = Array.from(elements)
                        .map(el => el.innerText)
                        .join('\\n');
                });
                return results;
            }
        """, self.selectors)
        
        # Extract metadata
        metadata = await page.evaluate("""
            () => ({
                title: document.title,
                description: document.querySelector('meta[name="description"]')?.content,
                keywords: document.querySelector('meta[name="keywords"]')?.content,
                author: document.querySelector('meta[name="author"]')?.content,
                og_title: document.querySelector('meta[property="og:title"]')?.content,
                og_description: document.querySelector('meta[property="og:description"]')?.content
            })
        """)
        
        return {
            'content': content,
            'metadata': metadata
        }
    
    def compute_hash(self, content_data: Dict) -> str:
        """Compute hash of content"""
        content = json.dumps(content_data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def detect_content_changes(self, before: Dict, after: Dict) -> List[Dict]:
        """Detect content changes"""
        changes = []
        
        # Compare metadata
        before_meta = before.get('metadata', {})
        after_meta = after.get('metadata', {})
        
        for key in set(before_meta.keys()) | set(after_meta.keys()):
            if before_meta.get(key) != after_meta.get(key):
                changes.append({
                    'type': f'META_{key.upper()}_CHANGED',
                    'before': before_meta.get(key),
                    'after': after_meta.get(key)
                })
        
        # Compare content sections
        before_content = before.get('content', {})
        after_content = after.get('content', {})
        
        for selector in set(before_content.keys()) | set(after_content.keys()):
            before_text = before_content.get(selector, '')
            after_text = after_content.get(selector, '')
            
            if before_text != after_text:
                changes.append({
                    'type': 'CONTENT_CHANGED',
                    'selector': selector,
                    'diff': self._generate_text_diff(before_text, after_text)
                })
        
        return changes
    
    def _generate_text_diff(self, before: str, after: str) -> Dict:
        """Generate text diff"""
        import difflib
        
        before_lines = before.splitlines()
        after_lines = after.splitlines()
        
        diff = list(difflib.unified_diff(
            before_lines, after_lines,
            lineterm='',
            n=3
        ))
        
        return {
            'unified_diff': '\\n'.join(diff),
            'added_lines': len([l for l in diff if l.startswith('+')]),
            'removed_lines': len([l for l in diff if l.startswith('-')])
        }


class AlertManager:
    """Manages alert generation and notifications"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.channels = {}
        self._setup_channels()
    
    def _setup_channels(self):
        """Initialize notification channels"""
        if 'email' in self.config.get('channels', []):
            self.channels['email'] = EmailNotifier(self.config.get('email', {}))
        if 'sms' in self.config.get('channels', []):
            self.channels['sms'] = SMSNotifier(self.config.get('sms', {}))
        if 'voice' in self.config.get('channels', []):
            self.channels['voice'] = VoiceNotifier(self.config.get('voice', {}))
        if 'tts' in self.config.get('channels', []):
            self.channels['tts'] = TTSNotifier(self.config.get('tts', {}))
    
    def determine_severity(self, change: Change) -> SeverityLevel:
        """Determine alert severity"""
        score = 0
        
        # Base score from change type
        type_scores = {
            'TITLE_CHANGED': 30,
            'PRICE_CHANGED': 40,
            'AVAILABILITY_CHANGED': 35,
            'ELEMENT_REMOVED': 25,
            'TEXT_CHANGED': 20,
            'ATTRIBUTES_CHANGED': 15,
            'ELEMENT_ADDED': 10
        }
        
        for change_detail in change.diff_data.get('changes', []):
            score += type_scores.get(change_detail['type'], 10)
        
        # Adjust based on magnitude
        if change.change_type == ChangeType.VISUAL:
            diff_pct = change.diff_data.get('diff_percentage', 0)
            score += min(diff_pct * 0.5, 30)
        
        # Map to severity
        if score >= 70:
            return SeverityLevel.CRITICAL
        elif score >= 50:
            return SeverityLevel.HIGH
        elif score >= 30:
            return SeverityLevel.MEDIUM
        elif score >= 10:
            return SeverityLevel.LOW
        else:
            return SeverityLevel.INFO
    
    async def send_alert(self, change: Change):
        """Send alert through configured channels"""
        severity = self.determine_severity(change)
        
        # Determine which channels to use
        channel_map = {
            SeverityLevel.CRITICAL: ['email', 'sms', 'voice', 'tts'],
            SeverityLevel.HIGH: ['email', 'sms', 'tts'],
            SeverityLevel.MEDIUM: ['email'],
            SeverityLevel.LOW: ['email'],
            SeverityLevel.INFO: []
        }
        
        channels_to_use = channel_map.get(severity, [])
        
        for channel_name in channels_to_use:
            if channel_name in self.channels:
                try:
                    await self.channels[channel_name].send(change, severity)
                except Exception as e:
                    print(f"Failed to send {channel_name} alert: {e}")


class EmailNotifier:
    """Email notification handler"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.smtp_server = config.get('smtp_server', 'smtp.gmail.com')
        self.smtp_port = config.get('smtp_port', 587)
        self.username = config.get('username')
        self.password = config.get('password')
        self.recipients = config.get('recipients', [])
    
    async def send(self, change: Change, severity: SeverityLevel):
        """Send email notification"""
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"[{severity.name}] Change Detected: {change.description[:50]}"
        msg['From'] = self.username
        msg['To'] = ', '.join(self.recipients)
        
        # HTML content
        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <div style="background: {'#dc3545' if severity == SeverityLevel.CRITICAL else '#fd7e14' if severity == SeverityLevel.HIGH else '#6c757d'}; 
                        color: white; padding: 20px;">
                <h1>{severity.name} Alert</h1>
                <p>Change detected at {change.detected_at}</p>
            </div>
            <div style="padding: 20px;">
                <h2>{change.description}</h2>
                <p><strong>Type:</strong> {change.change_type.value}</p>
                <p><strong>Details:</strong></p>
                <pre>{json.dumps(change.diff_data, indent=2)}</pre>
            </div>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(html, 'html'))
        
        with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)


class SMSNotifier:
    """Twilio SMS notification handler"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.account_sid = config.get('account_sid')
        self.auth_token = config.get('auth_token')
        self.from_number = config.get('from_number')
        self.to_numbers = config.get('to_numbers', [])
    
    async def send(self, change: Change, severity: SeverityLevel):
        """Send SMS notification"""
        from twilio.rest import Client
        
        client = Client(self.account_sid, self.auth_token)
        
        message = f"ALERT [{severity.name}]: {change.description[:100]}..."
        
        for number in self.to_numbers:
            client.messages.create(
                body=message,
                from_=self.from_number,
                to=number
            )


class VoiceNotifier:
    """Twilio voice call notification handler"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.account_sid = config.get('account_sid')
        self.auth_token = config.get('auth_token')
        self.from_number = config.get('from_number')
        self.to_numbers = config.get('to_numbers', [])
    
    async def send(self, change: Change, severity: SeverityLevel):
        """Make voice call"""
        from twilio.rest import Client
        
        client = Client(self.account_sid, self.auth_token)
        
        # Create TwiML for voice message
        message = f"Alert from OpenClaw monitoring. A {severity.name.lower()} priority change has been detected. Please check your dashboard."
        
        twiml = f"""
        <?xml version="1.0" encoding="UTF-8"?>
        <Response>
            <Say voice="Polly.Joanna">{message}</Say>
        </Response>
        """
        
        for number in self.to_numbers:
            client.calls.create(
                twiml=twiml,
                to=number,
                from_=self.from_number
            )


class TTSNotifier:
    """Text-to-speech notification handler"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.voice_id = config.get('voice_id', 'default')
    
    async def send(self, change: Change, severity: SeverityLevel):
        """Generate and play TTS notification"""
        # This would integrate with the agent's TTS system
        message = f"Attention: {severity.name} level change detected. {change.description}"
        
        # Placeholder for TTS integration
        print(f"[TTS] {message}")
        # await agent.tts.speak(message, voice_id=self.voice_id)


class MonitorEngine:
    """Main monitoring engine"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.monitors: Dict[str, MonitorConfig] = {}
        self.snapshots: Dict[str, Snapshot] = {}
        self.dom_watcher = DOMWatcher(config.get('dom', {}))
        self.visual_comparator = VisualComparator(config.get('visual', {}))
        self.content_analyzer = ContentAnalyzer(config.get('content', {}))
        self.alert_manager = AlertManager(config.get('alerts', {}))
        self.running = False
    
    def add_monitor(self, config: MonitorConfig):
        """Add a new monitor"""
        monitor_id = hashlib.md5(config.url.encode()).hexdigest()[:12]
        self.monitors[monitor_id] = config
        return monitor_id
    
    async def run_check(self, monitor_id: str) -> Optional[Change]:
        """Run a single monitoring check"""
        config = self.monitors.get(monitor_id)
        if not config:
            return None
        
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            try:
                await page.goto(config.url, wait_until='networkidle')
                
                # Capture current state
                current_snapshot = await self._capture_snapshot(
                    monitor_id, page, config
                )
                
                # Get baseline
                baseline = self.snapshots.get(monitor_id)
                
                if baseline:
                    # Detect changes
                    change = await self._detect_changes(
                        monitor_id, baseline, current_snapshot, config
                    )
                    
                    if change:
                        # Send alert
                        await self.alert_manager.send_alert(change)
                        return change
                
                # Update baseline
                self.snapshots[monitor_id] = current_snapshot
                
            finally:
                await browser.close()
        
        return None
    
    async def _capture_snapshot(self, monitor_id: str, page, config: MonitorConfig) -> Snapshot:
        """Capture current state snapshot"""
        timestamp = datetime.now()
        
        dom_hash = ''
        content_hash = ''
        visual_hash = ''
        screenshot_path = None
        dom_content = None
        text_content = None
        
        if config.dom_monitoring:
            dom_data = await self.dom_watcher.capture_dom(page)
            dom_hash = self.dom_watcher.compute_hash(dom_data)
            dom_content = json.dumps(dom_data)
        
        if config.content_monitoring:
            content_data = await self.content_analyzer.extract_content(page)
            content_hash = self.content_analyzer.compute_hash(content_data)
            text_content = json.dumps(content_data)
        
        if config.visual_monitoring:
            screenshot = await self.visual_comparator.capture_screenshot(page)
            visual_hash = hashlib.sha256(screenshot).hexdigest()
            # Save screenshot to file
            screenshot_path = f"./snapshots/{monitor_id}_{timestamp.isoformat()}.png"
            import os
            os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
            with open(screenshot_path, 'wb') as f:
                f.write(screenshot)
        
        return Snapshot(
            site_id=monitor_id,
            timestamp=timestamp,
            dom_hash=dom_hash,
            content_hash=content_hash,
            visual_hash=visual_hash,
            screenshot_path=screenshot_path,
            dom_content=dom_content,
            text_content=text_content
        )
    
    async def _detect_changes(self, monitor_id: str, baseline: Snapshot, 
                              current: Snapshot, config: MonitorConfig) -> Optional[Change]:
        """Detect changes between snapshots"""
        changes_detected = []
        diff_data = {}
        
        # DOM changes
        if config.dom_monitoring and baseline.dom_hash != current.dom_hash:
            before_dom = json.loads(baseline.dom_content or '{}')
            after_dom = json.loads(current.dom_content or '{}')
            dom_changes = self.dom_watcher.detect_changes(before_dom, after_dom)
            if dom_changes:
                changes_detected.append(ChangeType.DOM)
                diff_data['dom_changes'] = dom_changes
        
        # Content changes
        if config.content_monitoring and baseline.content_hash != current.content_hash:
            before_content = json.loads(baseline.text_content or '{}')
            after_content = json.loads(current.text_content or '{}')
            content_changes = self.content_analyzer.detect_content_changes(before_content, after_content)
            if content_changes:
                changes_detected.append(ChangeType.CONTENT)
                diff_data['content_changes'] = content_changes
        
        # Visual changes
        if config.visual_monitoring and baseline.visual_hash != current.visual_hash:
            with open(baseline.screenshot_path, 'rb') as f:
                baseline_img = f.read()
            with open(current.screenshot_path, 'rb') as f:
                current_img = f.read()
            
            visual_diff = self.visual_comparator.compare_images(baseline_img, current_img)
            if visual_diff['threshold_exceeded']:
                changes_detected.append(ChangeType.VISUAL)
                diff_data['visual_diff'] = visual_diff
        
        if changes_detected:
            # Determine primary change type
            primary_type = changes_detected[0]
            
            # Generate description
            description = self._generate_description(changes_detected, diff_data)
            
            return Change(
                site_id=monitor_id,
                change_type=primary_type,
                severity=SeverityLevel.MEDIUM,  # Will be adjusted by alert manager
                detected_at=current.timestamp,
                description=description,
                diff_data=diff_data,
                snapshot_before=baseline,
                snapshot_after=current
            )
        
        return None
    
    def _generate_description(self, change_types: List[ChangeType], diff_data: Dict) -> str:
        """Generate human-readable change description"""
        descriptions = []
        
        if ChangeType.DOM in change_types:
            dom_changes = diff_data.get('dom_changes', [])
            descriptions.append(f"{len(dom_changes)} DOM changes detected")
        
        if ChangeType.CONTENT in change_types:
            content_changes = diff_data.get('content_changes', [])
            descriptions.append(f"{len(content_changes)} content changes detected")
        
        if ChangeType.VISUAL in change_types:
            visual_diff = diff_data.get('visual_diff', {})
            diff_pct = visual_diff.get('diff_percentage', 0)
            descriptions.append(f"Visual change: {diff_pct:.2f}% difference")
        
        return '. '.join(descriptions)
    
    async def run_continuous(self):
        """Run monitoring continuously"""
        self.running = True
        
        while self.running:
            tasks = []
            for monitor_id, config in self.monitors.items():
                task = asyncio.create_task(self.run_check(monitor_id))
                tasks.append((monitor_id, task))
            
            # Wait for all checks to complete
            for monitor_id, task in tasks:
                try:
                    change = await task
                    if change:
                        print(f"Change detected for {monitor_id}: {change.description}")
                except Exception as e:
                    print(f"Error checking {monitor_id}: {e}")
            
            # Wait before next cycle
            await asyncio.sleep(60)  # Check every minute
    
    def stop(self):
        """Stop monitoring"""
        self.running = False


# Example usage
async def main():
    # Configuration
    config = {
        'dom': {
            'ignore_elements': ['script', 'style'],
            'ignore_attributes': ['data-timestamp']
        },
        'visual': {
            'threshold': 0.1,
            'algorithm': 'pixel'
        },
        'content': {
            'selectors': ['article', 'main', '.content']
        },
        'alerts': {
            'channels': ['email', 'tts'],
            'email': {
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'username': 'your-email@gmail.com',
                'password': 'your-app-password',
                'recipients': ['alert@example.com']
            },
            'tts': {
                'voice_id': 'default'
            }
        }
    }
    
    # Create engine
    engine = MonitorEngine(config)
    
    # Add monitors
    engine.add_monitor(MonitorConfig(
        url='https://example.com',
        name='Example Site',
        check_interval=300,  # 5 minutes
        dom_monitoring=True,
        visual_monitoring=True,
        content_monitoring=True,
        selectors=['#main-content', '.article'],
        alert_threshold=0.05,
        notification_channels=['email', 'tts']
    ))
    
    # Run monitoring
    await engine.run_continuous()

if __name__ == '__main__':
    asyncio.run(main())
```

### 11.2 Cron Scheduler Integration

```python
# cron_scheduler.py
from croniter import croniter
from datetime import datetime
import asyncio

class CronScheduler:
    """Cron-based monitoring scheduler"""
    
    def __init__(self, engine: MonitorEngine):
        self.engine = engine
        self.jobs = []
        self.running = False
    
    def add_job(self, monitor_id: str, cron_expression: str):
        """Add a scheduled monitoring job"""
        self.jobs.append({
            'monitor_id': monitor_id,
            'cron': cron_expression,
            'iterator': croniter(cron_expression, datetime.now())
        })
    
    async def run(self):
        """Run scheduler loop"""
        self.running = True
        
        while self.running:
            now = datetime.now()
            
            for job in self.jobs:
                next_run = job['iterator'].get_next(datetime)
                
                if next_run <= now:
                    # Run the check
                    asyncio.create_task(
                        self.engine.run_check(job['monitor_id'])
                    )
                    # Reset iterator
                    job['iterator'] = croniter(job['cron'], now)
            
            await asyncio.sleep(1)
    
    def stop(self):
        """Stop scheduler"""
        self.running = False


# Preset schedules
SCHEDULE_PRESETS = {
    'realtime': '*/30 * * * * *',  # Every 30 seconds
    'frequent': '*/5 * * * *',      # Every 5 minutes
    'regular': '0 * * * *',         # Every hour
    'daily': '0 9 * * *',           # Daily at 9 AM
    'weekly': '0 9 * * 1',          # Weekly on Monday 9 AM
}
```

### 11.3 Dashboard Server

```python
# dashboard_server.py
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import json

app = FastAPI(title="OpenClaw Web Monitor Dashboard")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class DashboardAPI:
    """Dashboard API endpoints"""
    
    def __init__(self, engine: MonitorEngine):
        self.engine = engine
        self.setup_routes()
    
    def setup_routes(self):
        
        @app.get("/", response_class=HTMLResponse)
        async def dashboard():
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>OpenClaw Web Monitor</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
                    .header { background: #333; color: white; padding: 20px; margin: -20px -20px 20px -20px; }
                    .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 20px; }
                    .stat-card { background: #f5f5f5; padding: 20px; border-radius: 8px; }
                    .stat-value { font-size: 32px; font-weight: bold; color: #333; }
                    .alert-critical { background: #dc3545; color: white; }
                    .alert-high { background: #fd7e14; color: white; }
                    table { width: 100%; border-collapse: collapse; }
                    th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
                    th { background: #f5f5f5; }
                    .status-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; }
                    .status-active { background: #28a745; }
                    .status-inactive { background: #dc3545; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>OpenClaw Web Monitor Dashboard</h1>
                </div>
                
                <div class="stats">
                    <div class="stat-card">
                        <div>Active Monitors</div>
                        <div class="stat-value" id="monitor-count">0</div>
                    </div>
                    <div class="stat-card alert-critical">
                        <div>Critical Alerts</div>
                        <div class="stat-value" id="critical-count">0</div>
                    </div>
                    <div class="stat-card alert-high">
                        <div>High Alerts</div>
                        <div class="stat-value" id="high-count">0</div>
                    </div>
                    <div class="stat-card">
                        <div>Changes Today</div>
                        <div class="stat-value" id="changes-count">0</div>
                    </div>
                </div>
                
                <h2>Monitored Sites</h2>
                <table id="sites-table">
                    <thead>
                        <tr>
                            <th>Site</th>
                            <th>Status</th>
                            <th>Last Check</th>
                            <th>Last Change</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody></tbody>
                </table>
                
                <script>
                    const ws = new WebSocket('ws://localhost:8000/ws');
                    
                    ws.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        updateDashboard(data);
                    };
                    
                    function updateDashboard(data) {
                        document.getElementById('monitor-count').textContent = data.monitor_count;
                        document.getElementById('critical-count').textContent = data.critical_alerts;
                        document.getElementById('high-count').textContent = data.high_alerts;
                        document.getElementById('changes-count').textContent = data.changes_today;
                        
                        const tbody = document.querySelector('#sites-table tbody');
                        tbody.innerHTML = data.sites.map(site => `
                            <tr>
                                <td>${site.name}<br><small>${site.url}</small></td>
                                <td><span class="status-dot status-${site.status}"></span> ${site.status}</td>
                                <td>${site.last_check}</td>
                                <td>${site.last_change || 'Never'}</td>
                                <td>
                                    <button onclick="checkNow('${site.id}')">Check Now</button>
                                    <button onclick="viewHistory('${site.id}')">History</button>
                                </td>
                            </tr>
                        `).join('');
                    }
                    
                    function checkNow(siteId) {
                        fetch(`/api/sites/${siteId}/check`, {method: 'POST'});
                    }
                    
                    function viewHistory(siteId) {
                        window.location.href = `/history/${siteId}`;
                    }
                </script>
            </body>
            </html>
            """
        
        @app.get("/api/overview")
        async def get_overview():
            return {
                'monitor_count': len(self.engine.monitors),
                'critical_alerts': 0,  # From database
                'high_alerts': 0,
                'changes_today': 0,
                'system_status': 'healthy'
            }
        
        @app.get("/api/sites")
        async def get_sites():
            sites = []
            for monitor_id, config in self.engine.monitors.items():
                snapshot = self.engine.snapshots.get(monitor_id)
                sites.append({
                    'id': monitor_id,
                    'name': config.name,
                    'url': config.url,
                    'status': 'active',
                    'last_check': snapshot.timestamp.isoformat() if snapshot else None,
                    'last_change': None  # From database
                })
            return sites
        
        @app.post("/api/sites/{site_id}/check")
        async def check_site(site_id: str):
            change = await self.engine.run_check(site_id)
            return {'change_detected': change is not None}
        
        @app.websocket("/ws")
        async def websocket(websocket: WebSocket):
            await websocket.accept()
            while True:
                # Send periodic updates
                data = await self.get_dashboard_data()
                await websocket.send_json(data)
                await asyncio.sleep(5)
    
    async def get_dashboard_data(self):
        """Get current dashboard data"""
        return {
            'monitor_count': len(self.engine.monitors),
            'critical_alerts': 0,
            'high_alerts': 0,
            'changes_today': 0,
            'sites': [
                {
                    'id': mid,
                    'name': config.name,
                    'url': config.url,
                    'status': 'active',
                    'last_check': self.engine.snapshots.get(mid, {}).timestamp.isoformat() if self.engine.snapshots.get(mid) else None,
                    'last_change': None
                }
                for mid, config in self.engine.monitors.items()
            ]
        }


# Run server
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 11.4 Configuration File

```yaml
# config/web_monitoring.yaml
web_monitoring:
  # System settings
  system:
    max_concurrent_checks: 5
    request_timeout: 30
    user_agent: "OpenClaw-WebMonitor/1.0"
    
  # Browser settings
  browser:
    headless: true
    viewport:
      width: 1920
      height: 1080
    
  # DOM monitoring
  dom:
    enabled: true
    hash_algorithm: "sha256"
    ignore_elements:
      - "script"
      - "style"
      - "noscript"
      - ".advertisement"
    ignore_attributes:
      - "data-timestamp"
      - "data-session-id"
      
  # Visual monitoring
  visual:
    enabled: true
    format: "png"
    quality: 90
    comparison_algorithm: "pixel"
    threshold: 0.1
    capture_full_page: true
    
  # Content monitoring
  content:
    enabled: true
    selectors:
      - "article"
      - "main"
      - "[role='main']"
      - ".content"
    extract_metadata: true
    
  # Alert configuration
  alerts:
    enabled: true
    channels:
      - email
      - sms
      - voice
      - tts
      
    email:
      smtp_server: "smtp.gmail.com"
      smtp_port: 587
      use_tls: true
      username: "${EMAIL_USERNAME}"
      password: "${EMAIL_PASSWORD}"
      recipients:
        - "${ALERT_EMAIL}"
      templates:
        critical: "templates/alert_critical.html"
        standard: "templates/alert_standard.html"
        
    sms:
      provider: "twilio"
      account_sid: "${TWILIO_ACCOUNT_SID}"
      auth_token: "${TWILIO_AUTH_TOKEN}"
      from_number: "${TWILIO_PHONE_NUMBER}"
      to_numbers:
        - "${ALERT_PHONE}"
        
    voice:
      provider: "twilio"
      account_sid: "${TWILIO_ACCOUNT_SID}"
      auth_token: "${TWILIO_AUTH_TOKEN}"
      from_number: "${TWILIO_PHONE_NUMBER}"
      to_numbers:
        - "${ALERT_PHONE}"
      voice: "Polly.Joanna"
      
    tts:
      enabled: true
      voice_id: "default"
      speak_on_severity:
        - CRITICAL
        - HIGH
        
  # Scheduling
  scheduling:
    default_interval: 3600  # 1 hour
    
    presets:
      realtime:
        interval: 30
        unit: "seconds"
      frequent:
        interval: 5
        unit: "minutes"
      regular:
        interval: 1
        unit: "hours"
      daily:
        cron: "0 9 * * *"
      weekly:
        cron: "0 9 * * 1"
        
  # Storage
  storage:
    snapshots_dir: "./data/snapshots"
    database:
      type: "sqlite"
      path: "./data/monitoring.db"
      
    retention:
      snapshots:
        keep_last: 100
        keep_days: 90
      changes:
        keep_days: 365
      alerts:
        keep_days: 90
        
  # Dashboard
  dashboard:
    enabled: true
    host: "0.0.0.0"
    port: 8000
    refresh_interval: 5
    
  # Monitored sites (can be loaded from separate file)
  sites:
    - name: "Example News"
      url: "https://example.com/news"
      schedule: "frequent"
      dom: true
      visual: true
      content: true
      selectors:
        - ".headlines"
        - ".breaking-news"
      alert_threshold: 0.05
      
    - name: "Product Page"
      url: "https://example.com/product"
      schedule: "realtime"
      dom: true
      visual: true
      content: true
      selectors:
        - ".price"
        - ".availability"
        - ".stock-status"
      alert_threshold: 0.01
      notification_channels:
        - email
        - sms
        - voice
```

---

## 12. Integration with OpenClaw Agent

### 12.1 Agent Loop Integration

```python
# agent_integration.py

class WebMonitoringAgentLoop:
    """
    Integration of web monitoring into OpenClaw's 15 agentic loops
    """
    
    LOOP_ID = "web_monitor"
    LOOP_NAME = "Web Monitoring & Change Detection"
    PRIORITY = 8  # High priority
    
    def __init__(self, agent_core):
        self.agent = agent_core
        self.monitor_engine = None
        self.scheduler = None
        
    async def initialize(self):
        """Initialize the monitoring loop"""
        config = self.load_config()
        self.monitor_engine = MonitorEngine(config)
        self.scheduler = CronScheduler(self.monitor_engine)
        
        # Register with agent's heartbeat
        self.agent.heartbeat.register_component('web_monitor', self.get_status)
        
    async def run(self):
        """Main loop execution"""
        await self.agent.log("Starting web monitoring loop", level="INFO")
        
        # Load monitored sites from agent memory
        sites = await self.agent.memory.get('monitored_sites', [])
        
        for site in sites:
            monitor_id = self.monitor_engine.add_monitor(
                MonitorConfig(**site)
            )
            self.scheduler.add_job(monitor_id, site['schedule'])
        
        # Start scheduler
        await self.scheduler.run()
    
    async def on_change_detected(self, change: Change):
        """Handle detected change"""
        # Log to agent
        await self.agent.log(
            f"Change detected: {change.description}",
            level="WARNING" if change.severity.value >= 4 else "INFO"
        )
        
        # Store in agent memory
        await self.agent.memory.store('recent_changes', change, ttl=86400)
        
        # Notify through agent's notification system
        await self.agent.notify({
            'type': 'web_change',
            'severity': change.severity.name,
            'message': change.description,
            'data': change.diff_data
        })
        
        # Use GPT-5.2 for intelligent analysis
        analysis = await self.agent.gpt52.generate(
            prompt=f"""
            Analyze this website change:
            Site: {change.site_id}
            Type: {change.change_type.value}
            Description: {change.description}
            Diff Data: {json.dumps(change.diff_data, indent=2)}
            
            Provide:
            1. Summary of what changed
            2. Potential impact
            3. Recommended action
            4. Urgency assessment
            """,
            temperature=0.3
        )
        
        await self.agent.log(f"AI Analysis: {analysis}", level="INFO")
    
    def get_status(self) -> Dict:
        """Return current status for heartbeat"""
        return {
            'status': 'active' if self.monitor_engine and self.monitor_engine.running else 'inactive',
            'monitors': len(self.monitor_engine.monitors) if self.monitor_engine else 0,
            'last_check': None  # From database
        }
```

---

## 13. Summary

This comprehensive web monitoring and change detection system provides:

### Core Features
1. **Multi-layered Detection**: DOM, visual, and content monitoring
2. **Intelligent Alerting**: Severity-based multi-channel notifications
3. **Flexible Scheduling**: Cron-based, adaptive, and event-triggered
4. **Rich Visualization**: Dashboard with diff viewing and timeline
5. **Historical Tracking**: Complete audit trail with trend analysis

### Integration Points
- Gmail for email notifications
- Twilio for SMS and voice alerts
- TTS for agent voice notifications
- GPT-5.2 for intelligent change analysis
- Agent memory for persistence
- Heartbeat for health monitoring

### Scalability
- Async architecture for concurrent monitoring
- Configurable resource limits
- Efficient storage with retention policies
- Modular design for easy extension

---

*Document Version: 1.0*  
*Generated for OpenClaw Windows 10 AI Agent Framework*
