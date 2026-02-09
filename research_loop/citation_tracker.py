"""
Citation Tracking and Source Reliability System
Manages source citations and tracks source reliability over time.
"""

import json
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
import aiohttp

from .models import (
    DiscoveredSource, Citation, VerificationResult, UsageContext
)
from .config import CREDIBILITY_DOMAINS


logger = logging.getLogger(__name__)


class CitationTracker:
    """
    Track and manage source citations.
    
    Features:
    - Citation generation in multiple formats
    - Source usage tracking
    - Citation verification
    - Citation database management
    """
    
    def __init__(self, citations_path: str = "citations.json"):
        self.citations_path = Path(citations_path)
        self.citations_db: Dict[str, Citation] = {}
        self._load_citations()
    
    def _load_citations(self):
        """Load citations from database"""
        if self.citations_path.exists():
            try:
                with open(self.citations_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for cid, cdata in data.items():
                        self.citations_db[cid] = Citation(**cdata)
            except (OSError, json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f"Failed to load citations: {e}")
    
    async def _save_citations(self):
        """Save citations to database"""
        try:
            self.citations_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                cid: citation.model_dump()
                for cid, citation in self.citations_db.items()
            }
            with open(self.citations_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
        except (OSError, TypeError, ValueError) as e:
            logger.error(f"Failed to save citations: {e}")
    
    async def track_source(
        self,
        source: DiscoveredSource,
        usage_context: UsageContext
    ) -> Citation:
        """
        Track a source usage and generate citation.
        
        Args:
            source: Source to track
            usage_context: Context of source usage
            
        Returns:
            Generated citation
        """
        # Generate citation ID
        citation_id = self._generate_citation_id(source)
        
        # Check if already tracked
        if citation_id in self.citations_db:
            citation = self.citations_db[citation_id]
            # Update usage count
            citation.usage_context["usage_count"] = citation.usage_context.get("usage_count", 0) + 1
            citation.usage_context["last_used"] = datetime.now().isoformat()
        else:
            # Create new citation
            citation = Citation(
                id=citation_id,
                source_url=source.url,
                source_title=source.title or "Untitled",
                access_date=datetime.now(),
                usage_context={
                    **usage_context.model_dump(),
                    "usage_count": 1
                },
                reliability_score=source.credibility_score,
                citation_format=self._generate_formats(source)
            )
            
            self.citations_db[citation_id] = citation
            logger.debug(f"New citation tracked: {citation_id}")
        
        # Save to database
        await self._save_citations()
        
        return citation
    
    def _generate_citation_id(self, source: DiscoveredSource) -> str:
        """Generate unique citation ID"""
        import hashlib
        
        # Create ID from URL and access date
        url_hash = hashlib.md5(source.url.encode()).hexdigest()[:12]
        date_str = datetime.now().strftime("%Y%m%d")
        
        return f"cite_{date_str}_{url_hash}"
    
    def _generate_formats(self, source: DiscoveredSource) -> Dict[str, str]:
        """Generate citation in multiple formats"""
        return {
            "apa": self._apa_format(source),
            "mla": self._mla_format(source),
            "chicago": self._chicago_format(source),
            "ieee": self._ieee_format(source),
            "harvard": self._harvard_format(source),
            "simple": f"{source.title or 'Source'} - {source.url}"
        }
    
    def _apa_format(self, source: DiscoveredSource) -> str:
        """Generate APA citation"""
        date = datetime.now().strftime("%Y")
        
        if source.content_type == "article":
            return f"{source.title}. ({date}). Retrieved {datetime.now().strftime('%B %d, %Y')}, from {source.url}"
        
        return f"{source.title or 'Unknown'}. ({date}). Retrieved from {source.url}"
    
    def _mla_format(self, source: DiscoveredSource) -> str:
        """Generate MLA citation"""
        date = datetime.now().strftime("%d %b. %Y")
        
        return f'"{source.title or "Untitled"}." Web. {date} <{source.url}>.'
    
    def _chicago_format(self, source: DiscoveredSource) -> str:
        """Generate Chicago citation"""
        date = datetime.now().strftime("%B %d, %Y")
        
        return f'"{source.title or "Untitled"}." Accessed {date}. {source.url}.'
    
    def _ieee_format(self, source: DiscoveredSource) -> str:
        """Generate IEEE citation"""
        return f'[1] "{source.title or "Untitled"}," {source.url}, accessed {datetime.now().strftime("%b. %d, %Y")}.'
    
    def _harvard_format(self, source: DiscoveredSource) -> str:
        """Generate Harvard citation"""
        date = datetime.now().strftime("%Y")
        
        return f'{source.title or "Untitled"} ({date}) Available at: {source.url} (Accessed: {datetime.now().strftime("%d %B %Y")}).'
    
    async def verify_citation(self, citation_id: str) -> VerificationResult:
        """
        Verify if a citation is still valid.
        
        Args:
            citation_id: Citation to verify
            
        Returns:
            Verification result
        """
        citation = self.citations_db.get(citation_id)
        
        if not citation:
            return VerificationResult(
                valid=False,
                reason="Citation not found"
            )
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.head(
                    citation.source_url,
                    timeout=10,
                    allow_redirects=True
                ) as response:
                    
                    if response.status == 200:
                        citation.verification_status = "verified"
                        citation.last_verified = datetime.now()
                        await self._save_citations()
                        
                        return VerificationResult(
                            valid=True,
                            last_verified=datetime.now()
                        )
                    else:
                        return VerificationResult(
                            valid=False,
                            reason=f"HTTP {response.status}",
                            http_status=response.status,
                            last_verified=datetime.now()
                        )
                        
        except asyncio.TimeoutError:
            return VerificationResult(
                valid=False,
                reason="Timeout",
                last_verified=datetime.now()
            )
        except (aiohttp.ClientError, OSError, ConnectionError) as e:
            return VerificationResult(
                valid=False,
                reason=str(e),
                last_verified=datetime.now()
            )
    
    async def verify_all_citations(self) -> Dict[str, VerificationResult]:
        """Verify all citations in database"""
        results = {}
        
        for citation_id in self.citations_db:
            results[citation_id] = await self.verify_citation(citation_id)
        
        return results
    
    def get_citation(self, citation_id: str) -> Optional[Citation]:
        """Get a citation by ID"""
        return self.citations_db.get(citation_id)
    
    def get_citations_for_source(self, url: str) -> List[Citation]:
        """Get all citations for a source URL"""
        return [
            c for c in self.citations_db.values()
            if c.source_url == url
        ]
    
    def get_citations_by_date(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> List[Citation]:
        """Get citations within date range"""
        citations = []
        
        for citation in self.citations_db.values():
            access_date = citation.access_date
            
            if start and access_date < start:
                continue
            if end and access_date > end:
                continue
            
            citations.append(citation)
        
        return citations
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get citation statistics"""
        total = len(self.citations_db)
        verified = sum(
            1 for c in self.citations_db.values()
            if c.verification_status == "verified"
        )
        
        return {
            "total_citations": total,
            "verified": verified,
            "unverified": total - verified,
            "verification_rate": verified / total if total > 0 else 0
        }


class SourceReliabilityTracker:
    """
    Track and score source reliability over time.
    
    Maintains reliability scores for domains based on:
    - Historical accuracy
    - Verification success rate
    - User feedback
    - Content quality signals
    """
    
    def __init__(self, reliability_path: str = "source_reliability.json"):
        self.reliability_path = Path(reliability_path)
        self.reliability_db: Dict[str, Dict[str, Any]] = {}
        self._load_reliability()
    
    def _load_reliability(self):
        """Load reliability database"""
        if self.reliability_path.exists():
            try:
                with open(self.reliability_path, 'r', encoding='utf-8') as f:
                    self.reliability_db = json.load(f)
            except (OSError, json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f"Failed to load reliability data: {e}")
        
        # Initialize with known credible domains
        for domain, score in CREDIBILITY_DOMAINS.items():
            if domain not in self.reliability_db:
                self.reliability_db[domain] = {
                    "score": score,
                    "total_verifications": 0,
                    "successful_verifications": 0,
                    "failed_verifications": 0,
                    "last_verified": None,
                    "first_seen": datetime.now().isoformat()
                }
    
    async def _save_reliability(self):
        """Save reliability database"""
        try:
            self.reliability_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.reliability_path, 'w', encoding='utf-8') as f:
                json.dump(self.reliability_db, f, indent=2, default=str)
        except (OSError, TypeError, ValueError) as e:
            logger.error(f"Failed to save reliability data: {e}")
    
    async def update_reliability(
        self,
        source: DiscoveredSource,
        verification_result: VerificationResult
    ):
        """
        Update source reliability based on verification.
        
        Args:
            source: Source to update
            verification_result: Verification result
        """
        domain = self._extract_domain(source.url)
        
        # Get or create entry
        if domain not in self.reliability_db:
            self.reliability_db[domain] = {
                "score": 0.5,
                "total_verifications": 0,
                "successful_verifications": 0,
                "failed_verifications": 0,
                "last_verified": None,
                "first_seen": datetime.now().isoformat()
            }
        
        entry = self.reliability_db[domain]
        
        # Update based on verification
        if verification_result.valid:
            # Increase reliability
            entry["successful_verifications"] += 1
            entry["score"] = min(1.0, entry["score"] + 0.05)
        else:
            # Decrease reliability
            entry["failed_verifications"] += 1
            entry["score"] = max(0.0, entry["score"] - 0.1)
        
        entry["total_verifications"] += 1
        entry["last_verified"] = datetime.now().isoformat()
        
        # Save
        await self._save_reliability()
        
        logger.debug(f"Updated reliability for {domain}: {entry['score']:.2f}")
    
    def get_reliability_score(self, url: str) -> float:
        """
        Get reliability score for a URL.
        
        Args:
            url: URL to check
            
        Returns:
            Reliability score (0-1)
        """
        domain = self._extract_domain(url)
        
        # Check exact domain match
        if domain in self.reliability_db:
            return self.reliability_db[domain]["score"]
        
        # Check for partial matches
        for known_domain, data in self.reliability_db.items():
            if known_domain in domain or domain in known_domain:
                return data["score"]
        
        # Return neutral score for unknown domains
        return 0.5
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        parsed = urlparse(url)
        return parsed.netloc.lower()
    
    def get_reliable_sources(
        self,
        min_score: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Get list of reliable sources"""
        reliable = []
        
        for domain, data in self.reliability_db.items():
            if data["score"] >= min_score:
                reliable.append({
                    "domain": domain,
                    "score": data["score"],
                    "verifications": data["total_verifications"]
                })
        
        # Sort by score
        reliable.sort(key=lambda x: x["score"], reverse=True)
        
        return reliable
    
    def get_unreliable_sources(
        self,
        max_score: float = 0.4
    ) -> List[Dict[str, Any]]:
        """Get list of unreliable sources"""
        unreliable = []
        
        for domain, data in self.reliability_db.items():
            if data["score"] <= max_score:
                unreliable.append({
                    "domain": domain,
                    "score": data["score"],
                    "verifications": data["total_verifications"]
                })
        
        return unreliable
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get reliability statistics"""
        total_domains = len(self.reliability_db)
        
        if total_domains == 0:
            return {
                "total_domains": 0,
                "average_score": 0,
                "high_reliability": 0,
                "low_reliability": 0
            }
        
        scores = [d["score"] for d in self.reliability_db.values()]
        
        return {
            "total_domains": total_domains,
            "average_score": sum(scores) / len(scores),
            "high_reliability": sum(1 for s in scores if s >= 0.7),
            "medium_reliability": sum(1 for s in scores if 0.4 <= s < 0.7),
            "low_reliability": sum(1 for s in scores if s < 0.4),
            "total_verifications": sum(
                d["total_verifications"]
                for d in self.reliability_db.values()
            )
        }


# Import asyncio for timeout
import asyncio
