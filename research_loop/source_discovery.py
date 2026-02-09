"""
Source Discovery and Web Crawling System
Multi-source discovery, crawling, and content extraction.
"""

import asyncio
import logging
import os
import urllib.robotparser
from typing import List, Dict, Optional, Any
from datetime import datetime
from urllib.parse import urlparse, urljoin
import aiohttp
from bs4 import BeautifulSoup

from .models import (
    ResearchTask, SearchQuery, DiscoveredSource, RankedSource,
    CrawledContent, ExtractedContent, CrawlConfig, ResearchConfig,
    Media
)
from .config import ResearchLoopConfig, CREDIBILITY_DOMAINS


logger = logging.getLogger(__name__)


class SourceDiscoveryEngine:
    """
    Multi-source discovery engine.
    
    Discovers sources from multiple search engines,
    ranks them by quality, and manages the crawling process.
    """
    
    def __init__(self, config: Optional[ResearchLoopConfig] = None):
        self.config = config or ResearchLoopConfig()
        self.search_engines = SearchEngineManager()
        self.crawler = WebCrawler(self.config)
        self.source_ranker = SourceRanker()
    
    async def discover_sources(
        self,
        queries: List[SearchQuery],
        config: ResearchConfig
    ) -> List[DiscoveredSource]:
        """
        Discover relevant sources for research queries.
        
        Args:
            queries: List of search queries
            config: Research configuration
            
        Returns:
            List of discovered and ranked sources
        """
        logger.info(f"Discovering sources for {len(queries)} queries")
        
        all_sources = []
        
        # Parallel search across engines
        search_tasks = [
            self._search_engine_query(query, config)
            for query in queries
        ]
        
        search_results = await asyncio.gather(*search_tasks)
        
        # Aggregate results
        for results in search_results:
            all_sources.extend(results)
        
        logger.info(f"Discovered {len(all_sources)} total sources")
        
        # Remove duplicates
        unique_sources = self._deduplicate_sources(all_sources)
        
        logger.info(f"{len(unique_sources)} unique sources after deduplication")
        
        # Rank sources
        if unique_sources:
            ranked_sources = await self.source_ranker.rank(
                unique_sources,
                queries[0] if queries else None
            )
            
            # Return top sources based on config
            return [rs.source for rs in ranked_sources[:config.max_sources]]
        
        return []
    
    async def _search_engine_query(
        self,
        query: SearchQuery,
        config: ResearchConfig
    ) -> List[DiscoveredSource]:
        """Execute search on appropriate engines"""
        sources = []
        
        for engine_name in query.target_engines:
            try:
                engine = self.search_engines.get(engine_name)
                
                results = await engine.search(
                    query=query.text,
                    num_results=config.results_per_engine
                )
                
                for rank, result in enumerate(results, 1):
                    source = DiscoveredSource(
                        url=result["url"],
                        title=result.get("title", ""),
                        snippet=result.get("snippet"),
                        engine=engine_name,
                        query=query.text,
                        rank=rank,
                        credibility_score=self._estimate_credibility(result["url"])
                    )
                    sources.append(source)
                    
            except (OSError, ValueError, KeyError) as e:
                logger.warning(f"Search failed on {engine_name}: {e}")
                continue
        
        return sources
    
    def _estimate_credibility(self, url: str) -> float:
        """Estimate source credibility based on domain"""
        domain = urlparse(url).netloc.lower()
        
        for cred_domain, score in CREDIBILITY_DOMAINS.items():
            if cred_domain in domain:
                return score
        
        return 0.5  # Default for unknown domains
    
    def _deduplicate_sources(
        self,
        sources: List[DiscoveredSource]
    ) -> List[DiscoveredSource]:
        """Remove duplicate sources based on URL"""
        seen_urls = set()
        unique = []
        
        for source in sorted(sources, key=lambda s: s.credibility_score, reverse=True):
            # Normalize URL
            normalized = source.url.rstrip('/').lower()
            
            if normalized not in seen_urls:
                seen_urls.add(normalized)
                unique.append(source)
        
        return unique


class WebCrawler:
    """
    Intelligent web crawler with content extraction.
    
    Features:
    - Respects robots.txt
    - Configurable crawl depth
    - Content type detection
    - Media extraction
    """
    
    def __init__(self, config: Optional[ResearchLoopConfig] = None):
        self.config = config or ResearchLoopConfig()
        self.session: Optional[aiohttp.ClientSession] = None
        self.robots_cache: Dict[str, Any] = {}
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                headers={
                    "User-Agent": self.config.search.user_agent,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                    "Accept-Encoding": "gzip, deflate",
                    "Connection": "keep-alive",
                },
                timeout=aiohttp.ClientTimeout(total=self.config.crawling.timeout_seconds)
            )
        return self.session
    
    async def crawl_source(
        self,
        source: DiscoveredSource,
        config: CrawlConfig
    ) -> Optional[CrawledContent]:
        """
        Crawl and extract content from a source.
        
        Args:
            source: Source to crawl
            config: Crawl configuration
            
        Returns:
            Crawled content or None if failed
        """
        # Check robots.txt if enabled
        if config.respect_robots_txt and not await self._can_fetch(source.url):
            logger.debug(f"Skipping {source.url} - blocked by robots.txt")
            return None
        
        try:
            session = await self._get_session()
            
            async with session.get(source.url) as response:
                if response.status != 200:
                    logger.warning(f"HTTP {response.status} for {source.url}")
                    return None
                
                # Check content type
                content_type = response.headers.get("Content-Type", "")
                
                if "text/html" not in content_type:
                    logger.debug(f"Skipping non-HTML content: {source.url}")
                    return None
                
                # Fetch content
                html = await response.text()
                
                # Extract content
                content = await self._extract_content(
                    html=html,
                    url=source.url,
                    extract_type=config.extract_type
                )
                
                # Follow links if depth allows
                linked_content = []
                if config.follow_links and config.current_depth < config.max_depth:
                    linked_content = await self._crawl_linked_pages(
                        content.links,
                        config
                    )
                
                return CrawledContent(
                    source=source,
                    content=content,
                    crawl_timestamp=datetime.now(),
                    crawl_depth=config.current_depth,
                    linked_content=linked_content
                )
                
        except asyncio.TimeoutError:
            logger.warning(f"Timeout crawling {source.url}")
            return None
        except (OSError, ValueError) as e:
            logger.warning(f"Crawl failed for {source.url}: {e}")
            return None
    
    async def _can_fetch(self, url: str) -> bool:
        """Check if URL can be fetched according to robots.txt"""
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

        if robots_url in self.robots_cache:
            rp = self.robots_cache[robots_url]
        else:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(robots_url)
            try:
                await asyncio.to_thread(rp.read)
            except (OSError, ValueError) as e:
                logger.debug(f"Could not fetch robots.txt for {parsed.netloc}: {e}")
                return True  # Allow if robots.txt is unreachable
            self.robots_cache[robots_url] = rp

        user_agent = self.config.search.user_agent
        return rp.can_fetch(user_agent, url)
    
    async def _extract_content(
        self,
        html: str,
        url: str,
        extract_type: str
    ) -> ExtractedContent:
        """Extract structured content from HTML"""
        soup = BeautifulSoup(html, 'lxml')
        
        # Detect content type if not specified
        if extract_type == "general":
            extract_type = self._detect_content_type(soup, url)
        
        # Extract based on type
        extractors = {
            "article": self._extract_article,
            "documentation": self._extract_documentation,
            "forum": self._extract_forum_post,
            "product": self._extract_product_info,
            "research": self._extract_research_paper,
            "general": self._extract_general
        }
        
        extractor = extractors.get(extract_type, self._extract_general)
        content = await extractor(soup, url)
        
        # Extract metadata
        content.metadata = self._extract_metadata(soup, url)
        
        # Extract media
        content.media = await self._extract_media(soup, url)
        
        # Extract links
        content.links = self._extract_links(soup, url)
        
        return content
    
    def _detect_content_type(self, soup: BeautifulSoup, url: str) -> str:
        """Detect the type of content on the page"""
        # Check URL patterns
        if any(x in url for x in ["/docs/", "/documentation", "/api/"]):
            return "documentation"
        
        if any(x in url for x in ["/forum/", "/community/", "reddit.com", "stackoverflow.com"]):
            return "forum"
        
        # Check meta tags
        og_type = soup.find("meta", property="og:type")
        if og_type:
            type_value = og_type.get("content", "")
            if "article" in type_value:
                return "article"
        
        # Check for academic indicators
        if soup.find("abstract") or soup.find("div", class_="abstract"):
            return "research"
        
        # Default to article if it looks like content
        article_tags = soup.find_all(['article', 'main'])
        if article_tags:
            return "article"
        
        return "general"
    
    async def _extract_article(self, soup: BeautifulSoup, url: str) -> ExtractedContent:
        """Extract article content"""
        content = ExtractedContent()
        
        # Extract title
        title_tag = soup.find('title') or soup.find('h1')
        content.title = title_tag.get_text(strip=True) if title_tag else None
        
        # Extract author
        author_meta = soup.find('meta', attrs={'name': 'author'})
        if author_meta:
            content.author = author_meta.get('content')
        
        # Extract publish date
        date_meta = soup.find('meta', property='article:published_time')
        if date_meta:
            try:
                content.publish_date = datetime.fromisoformat(
                    date_meta.get('content', '').replace('Z', '+00:00')
                )
            except (ValueError, TypeError) as e:
                logger.debug(f"Failed to parse publish date: {e}")
        
        # Extract main content
        article = soup.find('article') or soup.find('main') or soup.find('div', class_='content')
        if article:
            # Remove script and style elements
            for script in article.find_all(['script', 'style', 'nav', 'header', 'footer']):
                script.decompose()
            
            content.content = article.get_text(separator='\n', strip=True)
            content.html_content = str(article)
        
        # Calculate word count and reading time
        if content.content:
            content.word_count = len(content.content.split())
            content.reading_time = max(1, content.word_count // 200)  # 200 WPM
        
        content.content_type = "article"
        
        return content
    
    async def _extract_documentation(self, soup: BeautifulSoup, url: str) -> ExtractedContent:
        """Extract documentation content"""
        content = ExtractedContent()
        
        # Documentation often has specific structure
        content.title = soup.find('title').get_text(strip=True) if soup.find('title') else None
        
        # Look for main documentation area
        doc_area = (
            soup.find('div', class_='documentation') or
            soup.find('div', class_='docs') or
            soup.find('main') or
            soup.find('article')
        )
        
        if doc_area:
            content.content = doc_area.get_text(separator='\n', strip=True)
            content.html_content = str(doc_area)
        
        # Extract code examples
        code_blocks = soup.find_all('pre') + soup.find_all('code')
        content.metadata['code_examples'] = len(code_blocks)
        
        content.content_type = "documentation"
        
        return content
    
    async def _extract_forum_post(self, soup: BeautifulSoup, url: str) -> ExtractedContent:
        """Extract forum post content"""
        content = ExtractedContent()
        
        content.title = soup.find('title').get_text(strip=True) if soup.find('title') else None
        
        # Look for post content
        post_area = (
            soup.find('div', class_='post') or
            soup.find('div', class_='comment') or
            soup.find('article')
        )
        
        if post_area:
            content.content = post_area.get_text(separator='\n', strip=True)
        
        content.content_type = "forum"
        
        return content
    
    async def _extract_product_info(self, soup: BeautifulSoup, url: str) -> ExtractedContent:
        """Extract product information"""
        content = ExtractedContent()
        
        content.title = soup.find('title').get_text(strip=True) if soup.find('title') else None
        
        # Look for product description
        product_area = (
            soup.find('div', class_='product-description') or
            soup.find('div', class_='description') or
            soup.find('main')
        )
        
        if product_area:
            content.content = product_area.get_text(separator='\n', strip=True)
        
        content.content_type = "product"
        
        return content
    
    async def _extract_research_paper(self, soup: BeautifulSoup, url: str) -> ExtractedContent:
        """Extract research paper content"""
        content = ExtractedContent()
        
        content.title = soup.find('title').get_text(strip=True) if soup.find('title') else None
        
        # Extract abstract
        abstract = soup.find('abstract') or soup.find('div', class_='abstract')
        if abstract:
            content.abstract = abstract.get_text(strip=True)
        
        # Extract main content
        paper_area = soup.find('div', class_='paper') or soup.find('main') or soup.find('article')
        if paper_area:
            content.content = paper_area.get_text(separator='\n', strip=True)
        
        # Extract sections
        content.sections = []
        for heading in soup.find_all(['h1', 'h2', 'h3']):
            section = {
                'heading': heading.get_text(strip=True),
                'level': int(heading.name[1]),
                'content': ''
            }
            
            # Get content until next heading
            next_node = heading.find_next_sibling()
            while next_node and next_node.name not in ['h1', 'h2', 'h3']:
                if next_node.get_text(strip=True):
                    section['content'] += next_node.get_text(strip=True) + '\n'
                next_node = next_node.find_next_sibling()
            
            content.sections.append(section)
        
        content.content_type = "research"
        
        return content
    
    async def _extract_general(self, soup: BeautifulSoup, url: str) -> ExtractedContent:
        """Extract general page content"""
        content = ExtractedContent()
        
        content.title = soup.find('title').get_text(strip=True) if soup.find('title') else None
        
        # Try to find main content area
        main = (
            soup.find('main') or
            soup.find('article') or
            soup.find('div', class_='content') or
            soup.find('div', class_='main') or
            soup.find('body')
        )
        
        if main:
            # Remove navigation, ads, etc.
            for elem in main.find_all(['nav', 'aside', 'footer', 'header', 'script', 'style']):
                elem.decompose()
            
            content.content = main.get_text(separator='\n', strip=True)
        
        content.content_type = "general"
        
        return content
    
    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract page metadata"""
        metadata = {}
        
        # Open Graph metadata
        for meta in soup.find_all('meta', property=True):
            prop = meta.get('property', '')
            if prop.startswith('og:'):
                metadata[prop] = meta.get('content', '')
        
        # Twitter Card metadata
        for meta in soup.find_all('meta', attrs={'name': True}):
            name = meta.get('name', '')
            if name.startswith('twitter:'):
                metadata[name] = meta.get('content', '')
        
        # Description
        desc = soup.find('meta', attrs={'name': 'description'})
        if desc:
            metadata['description'] = desc.get('content', '')
        
        # Keywords
        keywords = soup.find('meta', attrs={'name': 'keywords'})
        if keywords:
            metadata['keywords'] = keywords.get('content', '').split(',')
        
        return metadata
    
    async def _extract_media(self, soup: BeautifulSoup, url: str) -> List[Media]:
        """Extract media from page"""
        media = []
        
        # Images
        for img in soup.find_all('img'):
            src = img.get('src')
            if src:
                media.append(Media(
                    type='image',
                    url=urljoin(url, src),
                    alt_text=img.get('alt'),
                    caption=img.get('title')
                ))
        
        # Videos
        for video in soup.find_all('video'):
            src = video.get('src')
            if src:
                media.append(Media(
                    type='video',
                    url=urljoin(url, src)
                ))
        
        return media
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract links from page"""
        links = []
        
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            if href:
                absolute_url = urljoin(base_url, href)
                # Filter to same domain or related
                links.append(absolute_url)
        
        return links[:50]  # Limit links
    
    async def _crawl_linked_pages(
        self,
        links: List[str],
        config: CrawlConfig
    ) -> List[CrawledContent]:
        """Crawl linked pages up to configured depth"""
        # Filter to relevant links
        relevant = self._filter_relevant_links(links, config.topic)
        
        # Limit concurrent crawls
        semaphore = asyncio.Semaphore(self.config.crawling.max_concurrent)
        
        async def crawl_with_limit(url: str) -> Optional[CrawledContent]:
            async with semaphore:
                new_config = CrawlConfig(
                    extract_type=config.extract_type,
                    follow_links=config.follow_links,
                    max_depth=config.max_depth,
                    current_depth=config.current_depth + 1,
                    max_links=config.max_links,
                    max_concurrent=config.max_concurrent,
                    timeout_seconds=config.timeout_seconds,
                    respect_robots_txt=config.respect_robots_txt,
                    topic=config.topic
                )
                
                source = DiscoveredSource(url=url, title="", engine="crawl", query="", rank=0)
                return await self.crawl_source(source, new_config)
        
        # Crawl in parallel
        tasks = [crawl_with_limit(url) for url in relevant[:config.max_links]]
        results = await asyncio.gather(*tasks)
        
        return [r for r in results if r is not None]
    
    def _filter_relevant_links(self, links: List[str], topic: Optional[str]) -> List[str]:
        """Filter links to relevant ones"""
        # Simple filtering - could be more sophisticated
        filtered = []
        
        for link in links:
            # Skip non-HTTP links
            if not link.startswith(('http://', 'https://')):
                continue
            
            # Skip common non-content URLs
            skip_patterns = ['#', 'javascript:', 'mailto:', '.pdf', '.zip', '.exe']
            if any(p in link for p in skip_patterns):
                continue
            
            filtered.append(link)
        
        return filtered
    
    async def close(self):
        """Close crawler resources"""
        if self.session and not self.session.closed:
            await self.session.close()


class SourceRanker:
    """
    Rank sources by quality, relevance, and credibility.
    """
    
    async def rank(
        self,
        sources: List[DiscoveredSource],
        query: Optional[SearchQuery]
    ) -> List[RankedSource]:
        """
        Rank sources by multiple quality factors.
        
        Args:
            sources: List of discovered sources
            query: Original search query for relevance scoring
            
        Returns:
            List of ranked sources
        """
        ranked = []
        
        for source in sources:
            scores = {
                "credibility": self._score_credibility(source),
                "relevance": self._score_relevance(source, query),
                "freshness": self._score_freshness(source),
                "authority": self._score_authority(source),
                "position": self._score_position(source)
            }
            
            # Calculate weighted overall score
            weights = {
                "credibility": 0.3,
                "relevance": 0.25,
                "freshness": 0.15,
                "authority": 0.2,
                "position": 0.1
            }
            
            overall_score = sum(
                scores[key] * weights[key]
                for key in weights
            )
            
            ranked.append(RankedSource(
                source=source,
                scores=scores,
                overall_score=overall_score,
                confidence=min(1.0, overall_score + 0.1)
            ))
        
        # Sort by overall score
        ranked.sort(key=lambda x: x.overall_score, reverse=True)
        
        return ranked
    
    def _score_credibility(self, source: DiscoveredSource) -> float:
        """Score source credibility"""
        return source.credibility_score
    
    def _score_relevance(
        self,
        source: DiscoveredSource,
        query: Optional[SearchQuery]
    ) -> float:
        """Score relevance to query"""
        if not query:
            return 0.5
        
        # Simple text matching
        query_terms = set(query.text.lower().split())
        title_terms = set(source.title.lower().split())
        snippet_terms = set((source.snippet or "").lower().split())
        
        title_match = len(query_terms & title_terms) / len(query_terms) if query_terms else 0
        snippet_match = len(query_terms & snippet_terms) / len(query_terms) if query_terms else 0
        
        return (title_match * 0.6) + (snippet_match * 0.4)
    
    def _score_freshness(self, source: DiscoveredSource) -> float:
        """Score content freshness"""
        # For now, use a neutral score
        # Production would check publication dates
        return 0.7
    
    def _score_authority(self, source: DiscoveredSource) -> float:
        """Score source authority"""
        return source.credibility_score
    
    def _score_position(self, source: DiscoveredSource) -> float:
        """Score based on search result position"""
        # Higher rank (lower number) = better score
        return max(0, 1.0 - (source.rank - 1) * 0.1)


class SearchEngineManager:
    """
    Manager for multiple search engines.
    """
    
    def __init__(self):
        self.engines: Dict[str, 'SearchEngine'] = {}
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize available search engines"""
        # These would be actual implementations
        self.engines = {
            "google": GoogleSearchEngine(),
            "bing": BingSearchEngine(),
            "duckduckgo": DuckDuckGoSearchEngine(),
            "google_scholar": GoogleScholarEngine(),
            "arxiv": ArXivEngine()
        }
    
    def get(self, name: str) -> 'SearchEngine':
        """Get a search engine by name"""
        if name not in self.engines:
            raise ValueError(f"Unknown search engine: {name}")
        return self.engines[name]


class SearchEngine:
    """Base class for search engines"""
    
    async def search(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """Execute search and return results"""
        raise NotImplementedError


class GoogleSearchEngine(SearchEngine):
    """Google Custom Search API implementation"""

    def __init__(self):
        self.api_key = os.environ.get('GOOGLE_SEARCH_API_KEY', '')
        self.cx = os.environ.get('GOOGLE_SEARCH_CX', '')

    async def search(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        if not self.api_key or not self.cx:
            logger.debug("Google Search: GOOGLE_SEARCH_API_KEY or GOOGLE_SEARCH_CX not set")
            return []

        results = []
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    'key': self.api_key,
                    'cx': self.cx,
                    'q': query,
                    'num': min(num_results, 10),
                }
                async with session.get(
                    'https://www.googleapis.com/customsearch/v1',
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"Google Search API returned {resp.status}")
                        return []
                    data = await resp.json()
                    for item in data.get('items', []):
                        results.append({
                            'url': item.get('link', ''),
                            'title': item.get('title', ''),
                            'snippet': item.get('snippet', ''),
                        })
        except (aiohttp.ClientError, KeyError, ValueError) as e:
            logger.warning(f"Google Search error: {e}")
        return results


class BingSearchEngine(SearchEngine):
    """Bing Web Search API v7 implementation"""

    def __init__(self):
        self.api_key = os.environ.get('BING_SEARCH_API_KEY', '')

    async def search(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        if not self.api_key:
            logger.debug("Bing Search: BING_SEARCH_API_KEY not set")
            return []

        results = []
        try:
            headers = {'Ocp-Apim-Subscription-Key': self.api_key}
            params = {'q': query, 'count': min(num_results, 50)}
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    'https://api.bing.microsoft.com/v7.0/search',
                    headers=headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"Bing Search API returned {resp.status}")
                        return []
                    data = await resp.json()
                    for item in data.get('webPages', {}).get('value', []):
                        results.append({
                            'url': item.get('url', ''),
                            'title': item.get('name', ''),
                            'snippet': item.get('snippet', ''),
                        })
        except (aiohttp.ClientError, KeyError, ValueError) as e:
            logger.warning(f"Bing Search error: {e}")
        return results


class DuckDuckGoSearchEngine(SearchEngine):
    """DuckDuckGo Search via duckduckgo-search package (no API key needed)"""

    async def search(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        results = []
        try:
            from duckduckgo_search import DDGS

            def _search_sync():
                with DDGS() as ddgs:
                    return list(ddgs.text(query, max_results=num_results))

            raw = await asyncio.get_event_loop().run_in_executor(None, _search_sync)
            for item in raw:
                results.append({
                    'url': item.get('href', item.get('link', '')),
                    'title': item.get('title', ''),
                    'snippet': item.get('body', item.get('snippet', '')),
                })
        except ImportError:
            logger.warning("duckduckgo-search not installed (pip install duckduckgo-search)")
        except (OSError, RuntimeError) as e:
            logger.warning(f"DuckDuckGo search error: {e}")
        return results


class GoogleScholarEngine(SearchEngine):
    """Google Scholar via scholarly package (no API key, rate-limited)"""

    async def search(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        results = []
        try:
            from scholarly import scholarly

            def _search_sync():
                found = []
                search_query = scholarly.search_pubs(query)
                for _ in range(num_results):
                    try:
                        pub = next(search_query)
                        bib = pub.get('bib', {})
                        found.append({
                            'url': pub.get('pub_url', pub.get('eprint_url', '')),
                            'title': bib.get('title', ''),
                            'snippet': bib.get('abstract', '')[:300] if bib.get('abstract') else '',
                        })
                    except StopIteration:
                        break
                return found

            results = await asyncio.get_event_loop().run_in_executor(None, _search_sync)
        except ImportError:
            logger.debug("scholarly not installed - Google Scholar disabled")
        except (OSError, RuntimeError) as e:
            logger.warning(f"Google Scholar error: {e}")
        return results


class ArXivEngine(SearchEngine):
    """ArXiv search via arxiv package (free API, no key needed)"""

    async def search(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        results = []
        try:
            import arxiv

            def _search_sync():
                found = []
                search = arxiv.Search(query=query, max_results=num_results)
                for paper in search.results():
                    found.append({
                        'url': paper.entry_id,
                        'title': paper.title,
                        'snippet': (paper.summary or '')[:300],
                    })
                return found

            results = await asyncio.get_event_loop().run_in_executor(None, _search_sync)
        except ImportError:
            logger.warning("arxiv package not installed (pip install arxiv)")
        except (OSError, RuntimeError) as e:
            logger.warning(f"ArXiv search error: {e}")
        return results
