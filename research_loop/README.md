# Research Loop - Autonomous Information Gathering System

## Overview

The Research Loop is a core component of the OpenClaw-inspired AI Agent Framework, providing autonomous information gathering, synthesis, and knowledge management capabilities.

## Features

### 1. Multi-Modal Research Triggers
- **Scheduled Research**: Cron-based automated research tasks
- **Event-Driven Research**: Reactive research based on system events
- **Curiosity-Driven Research**: Self-directed learning based on knowledge gaps

### 2. Intelligent Query Generation
- GPT-5.2 powered concept extraction
- Template-based query generation
- Multi-engine query optimization
- Query deduplication and ranking

### 3. Multi-Source Discovery
- Parallel search across multiple engines
- Source quality ranking
- Credibility scoring
- Duplicate detection

### 4. Web Crawling & Content Extraction
- Respects robots.txt
- Configurable crawl depth
- Content type detection
- Media extraction

### 5. Information Extraction
- Named Entity Recognition (NER)
- Fact extraction
- Relationship extraction
- Statistics and quotes extraction

### 6. Multi-Source Synthesis
- Conflict resolution
- Consensus identification
- Multi-level summarization
- Confidence scoring

### 7. Knowledge Consolidation
- MEMORY.md integration
- Knowledge graph updates
- Cross-reference management
- Update tracking

### 8. Source Tracking & Citations
- Multiple citation formats (APA, MLA, Chicago, IEEE, Harvard)
- Source reliability tracking
- Citation verification
- Usage analytics

### 9. Research Depth Control
- 5 depth levels (surface, shallow, medium, deep, exhaustive)
- Adaptive research based on findings
- Time budget management
- Resource monitoring

## Installation

```bash
# Install dependencies
pip install aiohttp beautifulsoup4 lxml pydantic croniter pyyaml

# Optional: For Windows notifications
pip install win10toast
```

## Quick Start

```python
import asyncio
from research_loop import ResearchLoop
from research_loop.config import ResearchLoopConfig

# Load configuration
config = ResearchLoopConfig.from_yaml("research_loop_config.yaml")

# Create and run research loop
research_loop = ResearchLoop(config)

# Run continuously
asyncio.run(research_loop.run())
```

## Configuration

### Basic Configuration

```yaml
research_loop:
  enabled: true
  poll_interval: 30
  
  triggers:
    scheduled:
      morning_briefing:
        cron: "0 8 * * *"
        topics: ["tech_news"]
        depth: shallow
```

### Environment Variables

```bash
export GOOGLE_SEARCH_API_KEY="your_key"
export BING_SEARCH_API_KEY="your_key"
export RESEARCH_LOG_LEVEL="INFO"
```

## Usage Examples

### Submit a Research Request

```python
# Submit user research request
task_id = await research_loop.submit_research_request(
    topic="quantum computing",
    depth="deep",
    context="Focus on recent breakthroughs"
)

# Check status
status = await research_loop.get_task_status(task_id)
print(status)
```

### Manual Trigger

```python
# Force a curiosity-driven research
await research_loop.force_trigger("curiosity")

# Force scheduled research
await research_loop.force_trigger("scheduled")
```

### Event-Driven Research

```python
from research_loop.triggers import TriggerEngine

trigger_engine = TriggerEngine()

# Queue an event
trigger_engine.on_unknown_topic(
    topic="neuromorphic computing",
    context="User asked about this concept"
)

trigger_engine.on_user_request(
    topic="climate change mitigation",
    depth="deep"
)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      RESEARCH LOOP                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   TRIGGER    │───▶│   QUERY      │───▶│   SOURCE     │  │
│  │   ENGINE     │    │   GENERATOR  │    │   DISCOVERY  │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                   │           │
│         ▼                   ▼                   ▼           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              INFORMATION PROCESSING                   │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐           │  │
│  │  │ Extraction│─▶│ Synthesis│─▶│  Memory  │           │  │
│  │  └──────────┘  └──────────┘  └──────────┘           │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Module Structure

```
research_loop/
├── __init__.py              # Module exports
├── research_loop.py         # Main orchestrator
├── triggers.py              # Trigger mechanisms
├── query_generator.py       # Query generation
├── source_discovery.py      # Source discovery & crawling
├── extractor.py             # Information extraction
├── synthesizer.py           # Synthesis & summarization
├── memory_writer.py         # Memory consolidation
├── citation_tracker.py      # Source tracking
├── depth_controller.py      # Depth & resource control
├── models.py                # Data models
└── config.py                # Configuration
```

## Research Depth Levels

| Level | Sources | Depth | Time Budget | Use Case |
|-------|---------|-------|-------------|----------|
| Surface | 3 | 1 | 60s | Quick checks |
| Shallow | 5 | 1 | 180s | Brief overviews |
| Medium | 10 | 2 | 600s | Standard research |
| Deep | 20 | 3 | 1800s | Comprehensive |
| Exhaustive | 50 | 5 | 3600s | Complete coverage |

## API Reference

### ResearchLoop

Main orchestrator class for the research system.

```python
class ResearchLoop:
    async def run()                          # Main loop
    async def stop()                         # Stop gracefully
    async def submit_research_request()      # Submit user request
    async def get_task_status(task_id)       # Get task status
    async def get_statistics()               # Get loop statistics
    async def force_trigger(trigger_name)    # Force trigger
```

### TriggerEngine

Manages all trigger types.

```python
class TriggerEngine:
    async def check_scheduled_triggers()     # Check scheduled
    async def check_event_triggers()         # Check events
    async def check_curiosity_triggers()     # Check curiosity
    def on_unknown_topic(topic)              # Queue unknown topic
    def on_user_request(topic)               # Queue user request
```

### QueryGenerator

Generates optimized search queries.

```python
class QueryGenerator:
    async def generate_queries(task, context)  # Generate queries
```

### SourceDiscoveryEngine

Discovers and ranks sources.

```python
class SourceDiscoveryEngine:
    async def discover_sources(queries, config)  # Discover sources
```

### SynthesisEngine

Synthesizes information from multiple sources.

```python
class SynthesisEngine:
    async def synthesize(extracted_info, task)  # Synthesize
```

## Performance Metrics

| Metric | Target |
|--------|--------|
| Query Generation | < 2s |
| Source Discovery | < 10s |
| Content Extraction | < 5s/page |
| Synthesis | < 30s |
| Memory Write | < 2s |

## Error Handling

The Research Loop includes comprehensive error handling:

- **Retry Logic**: Automatic retries for failed operations
- **Fallback Strategies**: Alternative approaches when primary fails
- **Graceful Degradation**: Continue with reduced functionality
- **Error Reporting**: Detailed error logging and reporting

## Security Features

- **Domain Blocking**: Configurable blocked domains list
- **Content Filtering**: File extension filtering
- **Robots.txt Respect**: Honors crawler directives
- **Secure Storage**: Encrypted credential storage
- **No Personal Data**: Configurable PII exclusion

## Windows 10 Integration

### Task Scheduler

```python
# Schedule research tasks using Windows Task Scheduler
await research_loop.schedule_windows_task(task, schedule)
```

### Notifications

```python
# Send Windows toast notifications
from research_loop.windows_integration import send_notification

send_notification(
    title="Research Complete",
    message=f"Found {facts_count} facts from {sources_count} sources"
)
```

## Troubleshooting

### Common Issues

1. **Rate Limiting**
   - Reduce concurrent requests
   - Add delays between requests
   - Use multiple API keys

2. **Memory Issues**
   - Reduce max_sources
   - Lower crawl depth
   - Enable caching

3. **Timeout Errors**
   - Increase timeout values
   - Reduce content size limits
   - Check network connectivity

### Debug Mode

```python
import logging
logging.getLogger('research_loop').setLevel(logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - See LICENSE file for details

## Credits

- OpenClaw Framework
- GPT-5.2 Integration
- Windows 10 Agent Platform
