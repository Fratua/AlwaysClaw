"""
OpenClaw Web Monitoring Engine
Core monitoring system for DOM, visual, and content change detection
"""

import asyncio
import hashlib
import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import aiohttp
from playwright.async_api import async_playwright, Page


class ChangeType(Enum):
    """Types of changes that can be detected"""
    DOM = "dom"
    VISUAL = "visual"
    CONTENT = "content"
    META = "meta"
    PRICE = "price"
    AVAILABILITY = "availability"
    SECURITY = "security"


class SeverityLevel(Enum):
    """Alert severity levels"""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    INFO = 1


@dataclass
class MonitorConfig:
    """Configuration for a single monitor"""
    url: str
    name: str
    check_interval: int = 3600
    dom_monitoring: bool = True
    visual_monitoring: bool = True
    content_monitoring: bool = True
    selectors: List[str] = field(default_factory=list)
    alert_threshold: float = 0.1
    notification_channels: List[str] = field(default_factory=lambda: ["email"])
    ignore_elements: List[str] = field(default_factory=lambda: ["script", "style"])
    ignore_attributes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MonitorConfig':
        return cls(**data)


@dataclass
class Snapshot:
    """Snapshot of website state at a point in time"""
    site_id: str
    timestamp: datetime
    dom_hash: str
    content_hash: str
    visual_hash: str
    screenshot_path: Optional[str] = None
    dom_content: Optional[str] = None
    text_content: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'site_id': self.site_id,
            'timestamp': self.timestamp.isoformat(),
            'dom_hash': self.dom_hash,
            'content_hash': self.content_hash,
            'visual_hash': self.visual_hash,
            'screenshot_path': self.screenshot_path,
            'dom_content': self.dom_content,
            'text_content': self.text_content,
            'metadata': self.metadata
        }


@dataclass
class Change:
    """Represents a detected change"""
    site_id: str
    site_name: str
    site_url: str
    change_type: ChangeType
    severity: SeverityLevel
    detected_at: datetime
    description: str
    diff_data: Dict
    snapshot_before: Snapshot
    snapshot_after: Snapshot
    category: str = "general"
    significance_score: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'site_id': self.site_id,
            'site_name': self.site_name,
            'site_url': self.site_url,
            'change_type': self.change_type.value,
            'severity': self.severity.name,
            'detected_at': self.detected_at.isoformat(),
            'description': self.description,
            'diff_data': self.diff_data,
            'category': self.category,
            'significance_score': self.significance_score
        }


class DOMWatcher:
    """
    Monitors DOM changes using multiple detection strategies:
    - Full DOM hash comparison
    - Selective element monitoring
    - Attribute change tracking
    - Text content monitoring
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.ignore_elements = config.get('ignore_elements', ['script', 'style', 'noscript'])
        self.ignore_attributes = config.get('ignore_attributes', [])
    
    async def capture_dom(self, page: Page) -> Dict:
        """Capture DOM state from page"""
        dom_data = await page.evaluate("""
            (ignoreElements) => {
                const getElementData = (el) => {
                    const attrs = {};
                    for (const attr of el.attributes) {
                        if (!ignoreElements.includes(attr.name)) {
                            attrs[attr.name] = attr.value;
                        }
                    }
                    return {
                        tag: el.tagName.toLowerCase(),
                        id: el.id,
                        class: el.className,
                        text: el.innerText?.substring(0, 1000) || '',
                        attributes: attrs,
                        childCount: el.children.length
                    };
                };
                
                const shouldInclude = (el) => {
                    const tag = el.tagName.toLowerCase();
                    return !ignoreElements.includes(tag);
                };
                
                return {
                    title: document.title,
                    url: window.location.href,
                    doctype: document.doctype?.name,
                    viewport: {
                        width: window.innerWidth,
                        height: window.innerHeight
                    },
                    elements: Array.from(document.querySelectorAll('body *'))
                        .filter(shouldInclude)
                        .map(getElementData)
                };
            }
        """, self.ignore_elements)
        return dom_data
    
    def compute_hash(self, dom_data: Dict) -> str:
        """Compute SHA-256 hash of DOM content"""
        content = json.dumps(dom_data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def detect_changes(self, before: Dict, after: Dict) -> List[Dict]:
        """Detect DOM changes between two states"""
        changes = []
        
        # Check title change
        if before.get('title') != after.get('title'):
            changes.append({
                'type': 'TITLE_CHANGED',
                'before': before.get('title'),
                'after': after.get('title'),
                'impact': 'high'
            })
        
        # Compare elements
        before_elements = {self._element_key(e): e for e in before.get('elements', [])}
        after_elements = {self._element_key(e): e for e in after.get('elements', [])}
        
        # Find added elements
        for key, element in after_elements.items():
            if key not in before_elements:
                changes.append({
                    'type': 'ELEMENT_ADDED',
                    'element': element,
                    'impact': 'medium'
                })
        
        # Find removed elements
        for key, element in before_elements.items():
            if key not in after_elements:
                changes.append({
                    'type': 'ELEMENT_REMOVED',
                    'element': element,
                    'impact': 'medium'
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
                    'after_text': after_el.get('text'),
                    'impact': 'medium'
                })
            
            if before_el.get('attributes') != after_el.get('attributes'):
                changes.append({
                    'type': 'ATTRIBUTES_CHANGED',
                    'element': after_el,
                    'before_attrs': before_el.get('attributes'),
                    'after_attrs': after_el.get('attributes'),
                    'impact': 'low'
                })
            
            if before_el.get('childCount') != after_el.get('childCount'):
                changes.append({
                    'type': 'STRUCTURE_CHANGED',
                    'element': after_el,
                    'before_count': before_el.get('childCount'),
                    'after_count': after_el.get('childCount'),
                    'impact': 'medium'
                })
        
        return changes
    
    def _element_key(self, element: Dict) -> str:
        """Generate unique key for element"""
        tag = element.get('tag', '')
        id_val = element.get('id', '')
        class_val = element.get('class', '')
        text_preview = element.get('text', '')[:50]
        return f"{tag}#{id_val}.{class_val}:{text_preview}"


class VisualComparator:
    """Compares screenshots for visual changes"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.threshold = config.get('threshold', 0.1)
        self.algorithm = config.get('algorithm', 'pixel')
        self.ignore_regions = config.get('ignore_regions', [])
    
    async def capture_screenshot(self, page: Page, full_page: bool = True) -> bytes:
        """Capture screenshot from page"""
        screenshot = await page.screenshot(
            full_page=full_page,
            type='png',
            animations='disabled'
        )
        return screenshot
    
    def compare_images(self, baseline: bytes, current: bytes) -> Dict:
        """Compare two images and return diff metrics"""
        try:
            from PIL import Image
            import io
        except ImportError:
            return {
                'error': 'PIL not installed',
                'diff_pixels': 0,
                'total_pixels': 0,
                'diff_percentage': 0,
                'similarity_score': 100,
                'threshold_exceeded': False
            }
        
        # Load images
        img1 = Image.open(io.BytesIO(baseline))
        img2 = Image.open(io.BytesIO(current))
        
        # Ensure same size
        if img1.size != img2.size:
            img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)
        
        # Pixel-by-pixel comparison
        pixels1 = list(img1.convert('RGB').getdata())
        pixels2 = list(img2.convert('RGB').getdata())
        
        diff_pixels = 0
        diff_regions = []
        
        width, height = img1.size
        
        for i, (p1, p2) in enumerate(zip(pixels1, pixels2)):
            diff = self._pixel_diff(p1, p2)
            if diff > self.threshold:
                diff_pixels += 1
                x = i % width
                y = i // width
                diff_regions.append({'x': x, 'y': y, 'diff': diff})
        
        total_pixels = len(pixels1)
        diff_percentage = (diff_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        
        # Cluster diff regions
        clustered_regions = self._cluster_regions(diff_regions, width, height)
        
        return {
            'diff_pixels': diff_pixels,
            'total_pixels': total_pixels,
            'diff_percentage': round(diff_percentage, 4),
            'similarity_score': round(100 - diff_percentage, 4),
            'threshold_exceeded': diff_percentage > (self.threshold * 100),
            'changed_regions': clustered_regions,
            'algorithm': self.algorithm
        }
    
    def _pixel_diff(self, p1, p2) -> float:
        """Calculate difference between two pixels"""
        r1, g1, b1 = p1
        r2, g2, b2 = p2
        
        # Weighted RGB difference (perceptual)
        diff = (
            abs(r1 - r2) * 0.299 +
            abs(g1 - g2) * 0.587 +
            abs(b1 - b2) * 0.114
        ) / 255.0
        
        return diff
    
    def _cluster_regions(self, regions: List[Dict], width: int, height: int, 
                         cluster_size: int = 50) -> List[Dict]:
        """Cluster nearby diff regions into bounding boxes"""
        if not regions:
            return []
        
        # Simple clustering - group by proximity
        clusters = []
        used = set()
        
        for i, region in enumerate(regions):
            if i in used:
                continue
            
            cluster = [region]
            used.add(i)
            
            for j, other in enumerate(regions[i+1:], i+1):
                if j in used:
                    continue
                
                # Check if within cluster range
                dist = ((region['x'] - other['x']) ** 2 + 
                       (region['y'] - other['y']) ** 2) ** 0.5
                
                if dist < cluster_size:
                    cluster.append(other)
                    used.add(j)
            
            # Calculate bounding box
            xs = [r['x'] for r in cluster]
            ys = [r['y'] for r in cluster]
            
            clusters.append({
                'x': min(xs),
                'y': min(ys),
                'width': max(xs) - min(xs) + 1,
                'height': max(ys) - min(ys) + 1,
                'pixel_count': len(cluster)
            })
        
        return clusters


class ContentAnalyzer:
    """Analyzes text content for changes"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.selectors = config.get('selectors', ['body'])
        self.extract_metadata = config.get('extract_metadata', True)
    
    async def extract_content(self, page: Page) -> Dict:
        """Extract text content from page"""
        content = await page.evaluate("""
            (selectors) => {
                const results = {};
                selectors.forEach(selector => {
                    const elements = document.querySelectorAll(selector);
                    results[selector] = Array.from(elements)
                        .map(el => ({
                            text: el.innerText?.trim() || '',
                            html: el.innerHTML?.substring(0, 5000) || ''
                        }));
                });
                return results;
            }
        """, self.selectors)
        
        # Extract metadata
        metadata = {}
        if self.extract_metadata:
            metadata = await page.evaluate("""
                () => ({
                    title: document.title,
                    description: document.querySelector('meta[name="description"]')?.content,
                    keywords: document.querySelector('meta[name="keywords"]')?.content,
                    author: document.querySelector('meta[name="author"]')?.content,
                    og_title: document.querySelector('meta[property="og:title"]')?.content,
                    og_description: document.querySelector('meta[property="og:description"]')?.content,
                    canonical: document.querySelector('link[rel="canonical"]')?.href,
                    last_modified: document.lastModified
                })
            """)
        
        return {
            'content': content,
            'metadata': metadata
        }
    
    def compute_hash(self, content_data: Dict) -> str:
        """Compute hash of content"""
        content = json.dumps(content_data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
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
                    'field': key,
                    'before': before_meta.get(key),
                    'after': after_meta.get(key),
                    'impact': 'high' if key in ['title', 'description'] else 'low'
                })
        
        # Compare content sections
        before_content = before.get('content', {})
        after_content = after.get('content', {})
        
        for selector in set(before_content.keys()) | set(after_content.keys()):
            before_sections = before_content.get(selector, [])
            after_sections = after_content.get(selector, [])
            
            if len(before_sections) != len(after_sections):
                changes.append({
                    'type': 'CONTENT_STRUCTURE_CHANGED',
                    'selector': selector,
                    'before_count': len(before_sections),
                    'after_count': len(after_sections),
                    'impact': 'medium'
                })
            
            for i, (before_sec, after_sec) in enumerate(zip(before_sections, after_sections)):
                before_text = before_sec.get('text', '')
                after_text = after_sec.get('text', '')
                
                if before_text != after_text:
                    changes.append({
                        'type': 'CONTENT_CHANGED',
                        'selector': selector,
                        'index': i,
                        'diff': self._generate_text_diff(before_text, after_text),
                        'impact': 'medium'
                    })
        
        return changes
    
    def _generate_text_diff(self, before: str, after: str) -> Dict:
        """Generate text diff"""
        try:
            import difflib
        except ImportError:
            return {
                'before_length': len(before),
                'after_length': len(after),
                'change_ratio': len(set(after.split()) - set(before.split()))
            }
        
        before_lines = before.splitlines()
        after_lines = after.splitlines()
        
        diff = list(difflib.unified_diff(
            before_lines, after_lines,
            lineterm='',
            n=3
        ))
        
        return {
            'unified_diff': '\n'.join(diff),
            'added_lines': len([l for l in diff if l.startswith('+') and not l.startswith('+++')]),
            'removed_lines': len([l for l in diff if l.startswith('-') and not l.startswith('---')]),
            'before_length': len(before),
            'after_length': len(after)
        }


class ChangeClassifier:
    """Classifies changes into categories"""
    
    CATEGORIES = {
        'price': {
            'patterns': ['price', 'cost', '$', '€', '£', 'usd', 'eur'],
            'selectors': ['[data-track="price"]', '.price', '.cost', '[class*="price"]'],
            'severity': SeverityLevel.HIGH
        },
        'availability': {
            'patterns': ['in stock', 'out of stock', 'available', 'sold out', 'unavailable'],
            'selectors': ['.stock', '.availability', '[data-availability]'],
            'severity': SeverityLevel.HIGH
        },
        'security': {
            'patterns': ['security', 'privacy', 'terms', 'policy', 'gdpr', 'cookie'],
            'selectors': ['.security', '.privacy-policy', '.terms'],
            'severity': SeverityLevel.CRITICAL
        },
        'content': {
            'patterns': ['article', 'blog', 'news', 'post'],
            'selectors': ['article', '.blog-post', '.news-item'],
            'severity': SeverityLevel.MEDIUM
        },
        'status': {
            'patterns': ['status', 'health', 'up', 'down', 'maintenance'],
            'selectors': ['.status', '.health', '[data-status]'],
            'severity': SeverityLevel.HIGH
        }
    }
    
    def classify(self, change: Change) -> str:
        """Classify a change into a category"""
        description = change.description.lower()
        diff_data = json.dumps(change.diff_data).lower()
        
        scores = {}
        
        for category, config in self.CATEGORIES.items():
            score = 0
            
            # Check patterns in description
            for pattern in config['patterns']:
                if pattern in description:
                    score += 2
                if pattern in diff_data:
                    score += 1
            
            # Check selectors
            for selector in config['selectors']:
                if selector in diff_data:
                    score += 3
            
            scores[category] = score
        
        # Return highest scoring category
        if scores:
            best_category = max(scores, key=scores.get)
            if scores[best_category] > 0:
                return best_category
        
        return 'general'
    
    def get_severity_for_category(self, category: str) -> SeverityLevel:
        """Get default severity for a category"""
        return self.CATEGORIES.get(category, {}).get('severity', SeverityLevel.MEDIUM)


class MonitorEngine:
    """Main monitoring engine"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.monitors: Dict[str, MonitorConfig] = {}
        self.snapshots: Dict[str, Snapshot] = {}
        self.change_history: List[Change] = []
        
        # Initialize components
        self.dom_watcher = DOMWatcher(config.get('dom', {}))
        self.visual_comparator = VisualComparator(config.get('visual', {}))
        self.content_analyzer = ContentAnalyzer(config.get('content', {}))
        self.change_classifier = ChangeClassifier()
        
        # State
        self.running = False
        self.check_count = 0
        self.change_count = 0
        
        # Setup storage
        self._setup_storage()
    
    def _setup_storage(self):
        """Setup storage directories"""
        storage_config = self.config.get('storage', {})
        self.snapshots_dir = Path(storage_config.get('snapshots_dir', './data/snapshots'))
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
    
    def add_monitor(self, config: MonitorConfig) -> str:
        """Add a new monitor"""
        monitor_id = hashlib.md5(config.url.encode()).hexdigest()[:12]
        self.monitors[monitor_id] = config
        return monitor_id
    
    def remove_monitor(self, monitor_id: str) -> bool:
        """Remove a monitor"""
        if monitor_id in self.monitors:
            del self.monitors[monitor_id]
            if monitor_id in self.snapshots:
                del self.snapshots[monitor_id]
            return True
        return False
    
    async def run_check(self, monitor_id: str) -> Optional[Change]:
        """Run a single monitoring check"""
        config = self.monitors.get(monitor_id)
        if not config:
            return None
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent=self.config.get('user_agent', 'OpenClaw-WebMonitor/1.0')
            )
            page = await context.new_page()
            
            try:
                # Navigate to page
                await page.goto(
                    config.url, 
                    wait_until='networkidle',
                    timeout=self.config.get('request_timeout', 30) * 1000
                )
                
                # Wait for stability
                await asyncio.sleep(2)
                
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
                        self.change_count += 1
                        self.change_history.append(change)
                        return change
                
                # Update baseline
                self.snapshots[monitor_id] = current_snapshot
                
            except (RuntimeError, ValueError, TypeError) as e:
                print(f"Error checking {monitor_id}: {e}")
                
            finally:
                await browser.close()
        
        self.check_count += 1
        return None
    
    async def _capture_snapshot(self, monitor_id: str, page: Page, 
                                 config: MonitorConfig) -> Snapshot:
        """Capture current state snapshot"""
        timestamp = datetime.now()
        
        dom_hash = ''
        content_hash = ''
        visual_hash = ''
        screenshot_path = None
        dom_content = None
        text_content = None
        metadata = {}
        
        if config.dom_monitoring:
            dom_data = await self.dom_watcher.capture_dom(page)
            dom_hash = self.dom_watcher.compute_hash(dom_data)
            dom_content = json.dumps(dom_data)
        
        if config.content_monitoring:
            content_data = await self.content_analyzer.extract_content(page)
            content_hash = self.content_analyzer.compute_hash(content_data)
            text_content = json.dumps(content_data)
            metadata = content_data.get('metadata', {})
        
        if config.visual_monitoring:
            screenshot = await self.visual_comparator.capture_screenshot(page)
            visual_hash = hashlib.sha256(screenshot).hexdigest()
            
            # Save screenshot
            timestamp_str = timestamp.strftime('%Y%m%d_%H%M%S')
            screenshot_path = str(self.snapshots_dir / f"{monitor_id}_{timestamp_str}.png")
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
            text_content=text_content,
            metadata=metadata
        )
    
    async def _detect_changes(self, monitor_id: str, baseline: Snapshot, 
                              current: Snapshot, config: MonitorConfig) -> Optional[Change]:
        """Detect changes between snapshots"""
        changes_detected = []
        diff_data = {}
        
        # DOM changes
        if config.dom_monitoring and baseline.dom_hash != current.dom_hash:
            try:
                before_dom = json.loads(baseline.dom_content or '{}')
                after_dom = json.loads(current.dom_content or '{}')
                dom_changes = self.dom_watcher.detect_changes(before_dom, after_dom)
                if dom_changes:
                    changes_detected.append(ChangeType.DOM)
                    diff_data['dom_changes'] = dom_changes
            except json.JSONDecodeError:
                pass
        
        # Content changes
        if config.content_monitoring and baseline.content_hash != current.content_hash:
            try:
                before_content = json.loads(baseline.text_content or '{}')
                after_content = json.loads(current.text_content or '{}')
                content_changes = self.content_analyzer.detect_content_changes(before_content, after_content)
                if content_changes:
                    changes_detected.append(ChangeType.CONTENT)
                    diff_data['content_changes'] = content_changes
            except json.JSONDecodeError:
                pass
        
        # Visual changes
        if config.visual_monitoring and baseline.visual_hash != current.visual_hash:
            if baseline.screenshot_path and os.path.exists(baseline.screenshot_path):
                with open(baseline.screenshot_path, 'rb') as f:
                    baseline_img = f.read()
                with open(current.screenshot_path, 'rb') as f:
                    current_img = f.read()
                
                visual_diff = self.visual_comparator.compare_images(baseline_img, current_img)
                if visual_diff.get('threshold_exceeded'):
                    changes_detected.append(ChangeType.VISUAL)
                    diff_data['visual_diff'] = visual_diff
        
        if changes_detected:
            # Determine primary change type
            primary_type = changes_detected[0]
            
            # Generate description
            description = self._generate_description(changes_detected, diff_data)
            
            # Create change object
            change = Change(
                site_id=monitor_id,
                site_name=config.name,
                site_url=config.url,
                change_type=primary_type,
                severity=SeverityLevel.MEDIUM,
                detected_at=current.timestamp,
                description=description,
                diff_data=diff_data,
                snapshot_before=baseline,
                snapshot_after=current
            )
            
            # Classify and update
            change.category = self.change_classifier.classify(change)
            change.severity = self.change_classifier.get_severity_for_category(change.category)
            
            return change
        
        return None
    
    def _generate_description(self, change_types: List[ChangeType], diff_data: Dict) -> str:
        """Generate human-readable change description"""
        descriptions = []
        
        if ChangeType.DOM in change_types:
            dom_changes = diff_data.get('dom_changes', [])
            dom_summary = f"{len(dom_changes)} DOM changes"
            
            # Count by type
            type_counts = {}
            for change in dom_changes:
                change_type = change.get('type', 'UNKNOWN')
                type_counts[change_type] = type_counts.get(change_type, 0) + 1
            
            if type_counts:
                type_summary = ', '.join([f"{k}: {v}" for k, v in type_counts.items()])
                dom_summary += f" ({type_summary})"
            
            descriptions.append(dom_summary)
        
        if ChangeType.CONTENT in change_types:
            content_changes = diff_data.get('content_changes', [])
            content_summary = f"{len(content_changes)} content changes"
            descriptions.append(content_summary)
        
        if ChangeType.VISUAL in change_types:
            visual_diff = diff_data.get('visual_diff', {})
            diff_pct = visual_diff.get('diff_percentage', 0)
            descriptions.append(f"Visual change: {diff_pct:.2f}% difference")
        
        return '; '.join(descriptions)
    
    async def run_all_checks(self) -> List[Change]:
        """Run checks for all monitors"""
        changes = []
        
        for monitor_id in self.monitors:
            change = await self.run_check(monitor_id)
            if change:
                changes.append(change)
        
        return changes
    
    async def run_continuous(self, interval: int = 60):
        """Run monitoring continuously"""
        self.running = True
        
        while self.running:
            await self.run_all_checks()
            
            # Wait before next cycle
            for _ in range(interval):
                if not self.running:
                    break
                await asyncio.sleep(1)
    
    def stop(self):
        """Stop monitoring"""
        self.running = False
    
    def get_status(self) -> Dict:
        """Get current engine status"""
        return {
            'running': self.running,
            'monitors': len(self.monitors),
            'checks_run': self.check_count,
            'changes_detected': self.change_count,
            'snapshots_stored': len(self.snapshots)
        }
    
    def get_monitor_status(self, monitor_id: str) -> Optional[Dict]:
        """Get status for a specific monitor"""
        config = self.monitors.get(monitor_id)
        snapshot = self.snapshots.get(monitor_id)
        
        if not config:
            return None
        
        return {
            'id': monitor_id,
            'name': config.name,
            'url': config.url,
            'active': True,
            'last_check': snapshot.timestamp.isoformat() if snapshot else None,
            'dom_hash': snapshot.dom_hash if snapshot else None,
            'content_hash': snapshot.content_hash if snapshot else None,
            'visual_hash': snapshot.visual_hash if snapshot else None
        }


# Example usage
async def main():
    """Example usage of the monitoring engine"""
    config = {
        'dom': {
            'ignore_elements': ['script', 'style', 'noscript'],
            'ignore_attributes': ['data-timestamp']
        },
        'visual': {
            'threshold': 0.1,
            'algorithm': 'pixel'
        },
        'content': {
            'selectors': ['article', 'main', '.content'],
            'extract_metadata': True
        },
        'storage': {
            'snapshots_dir': './data/snapshots'
        },
        'user_agent': 'OpenClaw-WebMonitor/1.0',
        'request_timeout': 30
    }
    
    # Create engine
    engine = MonitorEngine(config)
    
    # Add a monitor
    monitor_id = engine.add_monitor(MonitorConfig(
        url='https://example.com',
        name='Example Site',
        check_interval=300,
        dom_monitoring=True,
        visual_monitoring=True,
        content_monitoring=True,
        selectors=['#main-content', '.article'],
        alert_threshold=0.05,
        notification_channels=['email']
    ))
    
    print(f"Added monitor: {monitor_id}")
    print(f"Status: {engine.get_status()}")
    
    # Run a check
    change = await engine.run_check(monitor_id)
    
    if change:
        print(f"Change detected: {change.description}")
        print(f"Category: {change.category}")
        print(f"Severity: {change.severity.name}")
    else:
        print("No changes detected")
    
    print(f"Final status: {engine.get_status()}")


if __name__ == '__main__':
    asyncio.run(main())
