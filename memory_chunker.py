"""
Markdown Chunking Module for Memory System
Implements intelligent content chunking with overlap preservation
"""

import re
import hashlib
from typing import List, Iterator, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

from memory_models import MemoryChunk, ChunkingConfig


@dataclass
class HeaderInfo:
    """Information about a markdown header."""
    level: int
    text: str
    line_no: int


class MarkdownChunker:
    """
    Intelligent markdown chunker that respects document structure.
    
    Features:
    - Respects header boundaries when possible
    - Preserves paragraph boundaries
    - Uses sliding window with configurable overlap
    - Line-aware chunking with accurate line numbers
    - Hash-based deduplication support
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
    
    def chunk_file(self, file_path: Path) -> List[MemoryChunk]:
        """
        Chunk an entire markdown file.
        
        Args:
            file_path: Path to markdown file
            
        Returns:
            List of memory chunks
        """
        content = file_path.read_text(encoding='utf-8')
        return self.chunk(content, source_path=str(file_path))
    
    def chunk(
        self,
        content: str,
        source_path: str = ""
    ) -> List[MemoryChunk]:
        """
        Chunk markdown content with intelligent boundaries.
        
        Strategy:
        1. Parse document structure (headers, sections)
        2. Respect header boundaries when possible
        3. Respect paragraph boundaries
        4. Use sliding window with overlap for remaining content
        
        Args:
            content: Markdown content to chunk
            source_path: Optional source file path for metadata
            
        Returns:
            List of memory chunks with metadata
        """
        lines = content.split('\n')
        
        # Parse document structure
        headers = self._extract_headers(lines)
        sections = self._identify_sections(lines, headers)
        
        # Chunk each section
        all_chunks = []
        for section in sections:
            section_chunks = self._chunk_section(lines, section)
            all_chunks.extend(section_chunks)
        
        # Update total chunk counts
        for i, chunk in enumerate(all_chunks):
            chunk.total_chunks = len(all_chunks)
            chunk.chunk_index = i
        
        return all_chunks
    
    def _extract_headers(self, lines: List[str]) -> List[HeaderInfo]:
        """Extract all headers from document."""
        headers = []
        for i, line in enumerate(lines):
            match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if match:
                headers.append(HeaderInfo(
                    level=len(match.group(1)),
                    text=match.group(2).strip(),
                    line_no=i + 1
                ))
        return headers
    
    def _identify_sections(
        self,
        lines: List[str],
        headers: List[HeaderInfo]
    ) -> List[Tuple[int, int, Optional[HeaderInfo]]]:
        """
        Identify document sections based on headers.
        
        Returns:
            List of (start_line, end_line, header_info) tuples
        """
        if not headers:
            # No headers - treat entire document as one section
            return [(0, len(lines), None)]
        
        sections = []
        for i, header in enumerate(headers):
            start = header.line_no - 1  # 0-indexed
            
            # Find end of section (next header at same or higher level)
            if i + 1 < len(headers):
                end = headers[i + 1].line_no - 1
            else:
                end = len(lines)
            
            sections.append((start, end, header))
        
        return sections
    
    def _chunk_section(
        self,
        lines: List[str],
        section: Tuple[int, int, Optional[HeaderInfo]]
    ) -> List[MemoryChunk]:
        """
        Chunk a single document section.
        
        Args:
            lines: All document lines
            section: (start, end, header) tuple
            
        Returns:
            List of chunks for this section
        """
        start, end, header = section
        section_lines = lines[start:end]
        
        # If section fits in one chunk, return as-is
        section_content = '\n'.join(section_lines)
        if len(section_content) <= self.config.chunk_size:
            return [self._create_chunk(
                content=section_content,
                line_start=start + 1,
                line_end=end,
                chunk_index=0,
                total_chunks=1
            )]
        
        # Section needs multiple chunks
        chunks = []
        current_lines = []
        current_chars = 0
        chunk_index = 0
        
        # Always include header in first chunk of section
        if header:
            header_line = lines[header.line_no - 1]
            current_lines.append({'line': header_line, 'line_no': header.line_no})
            current_chars += len(header_line) + 1
        
        # Process remaining lines
        content_start = start + 1 if header else start
        
        for i in range(content_start, end):
            line = lines[i]
            line_length = len(line) + 1  # +1 for newline
            
            # Check if adding this line would exceed chunk size
            if current_chars + line_length > self.config.chunk_size and current_lines:
                # Save current chunk
                chunk_content = '\n'.join(l['line'] for l in current_lines)
                chunks.append(self._create_chunk(
                    content=chunk_content,
                    line_start=current_lines[0]['line_no'],
                    line_end=current_lines[-1]['line_no'],
                    chunk_index=chunk_index,
                    total_chunks=0  # Updated later
                ))
                
                # Carry over overlap
                current_lines = self._get_overlap_lines(current_lines)
                current_chars = sum(len(l['line']) + 1 for l in current_lines)
                chunk_index += 1
            
            # Add current line
            current_lines.append({'line': line, 'line_no': i + 1})
            current_chars += line_length
        
        # Don't forget the last chunk
        if current_lines:
            chunk_content = '\n'.join(l['line'] for l in current_lines)
            chunks.append(self._create_chunk(
                content=chunk_content,
                line_start=current_lines[0]['line_no'],
                line_end=current_lines[-1]['line_no'],
                chunk_index=chunk_index,
                total_chunks=0  # Updated later
            ))
        
        return chunks
    
    def _create_chunk(
        self,
        content: str,
        line_start: int,
        line_end: int,
        chunk_index: int,
        total_chunks: int
    ) -> MemoryChunk:
        """Create a MemoryChunk from accumulated lines."""
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        
        return MemoryChunk(
            content=content,
            line_start=line_start,
            line_end=line_end,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            content_hash=content_hash
        )
    
    def _get_overlap_lines(self, lines: List[dict]) -> List[dict]:
        """
        Get lines to carry over for overlap.
        
        Takes lines from the end of the current chunk to include
        in the next chunk for context continuity.
        """
        if self.config.overlap_size <= 0:
            return []
        
        overlap_chars = 0
        overlap_lines = []
        
        # Work backwards from end
        for line_info in reversed(lines):
            line = line_info['line']
            overlap_lines.insert(0, line_info)
            overlap_chars += len(line) + 1
            
            if overlap_chars >= self.config.overlap_size:
                break
        
        return overlap_lines
    
    def chunk_streaming(
        self,
        content: str,
        source_path: str = ""
    ) -> Iterator[MemoryChunk]:
        """
        Stream chunks one at a time (generator version).
        
        Useful for processing large documents without loading all chunks into memory.
        """
        chunks = self.chunk(content, source_path)
        for chunk in chunks:
            yield chunk
    
    def estimate_chunks(self, content: str) -> int:
        """
        Estimate the number of chunks for given content.
        
        This is a rough estimate for planning purposes.
        """
        content_length = len(content)
        effective_chunk_size = self.config.chunk_size - self.config.overlap_size
        
        if effective_chunk_size <= 0:
            return 1
        
        return max(1, (content_length + effective_chunk_size - 1) // effective_chunk_size)


class SmartChunker(MarkdownChunker):
    """
    Enhanced chunker with content-type awareness.
    
    Applies different chunking strategies based on content type:
    - Code blocks: Keep intact when possible
    - Lists: Keep list items together
    - Tables: Keep rows together
    """
    
    def _chunk_section(
        self,
        lines: List[str],
        section: Tuple[int, int, Optional[HeaderInfo]]
    ) -> List[MemoryChunk]:
        """
        Chunk section with content-type awareness.
        """
        start, end, header = section
        
        # Identify special blocks (code, tables, lists)
        blocks = self._identify_blocks(lines, start, end)
        
        # Chunk respecting block boundaries
        chunks = []
        current_lines = []
        current_chars = 0
        chunk_index = 0
        
        # Include header in first chunk
        if header:
            header_line = lines[header.line_no - 1]
            current_lines.append({'line': header_line, 'line_no': header.line_no})
            current_chars += len(header_line) + 1
        
        for block in blocks:
            block_lines = block['lines']
            block_chars = sum(len(l['line']) + 1 for l in block_lines)
            
            # Check if block fits in current chunk
            if current_chars + block_chars <= self.config.chunk_size:
                # Add block to current chunk
                current_lines.extend(block_lines)
                current_chars += block_chars
            else:
                # Block doesn't fit - finalize current chunk if any
                if current_lines:
                    chunk_content = '\n'.join(l['line'] for l in current_lines)
                    chunks.append(self._create_chunk(
                        content=chunk_content,
                        line_start=current_lines[0]['line_no'],
                        line_end=current_lines[-1]['line_no'],
                        chunk_index=chunk_index,
                        total_chunks=0
                    ))
                    chunk_index += 1
                
                # Check if block itself fits in a chunk
                if block_chars <= self.config.chunk_size:
                    # Block fits in its own chunk
                    chunk_content = '\n'.join(l['line'] for l in block_lines)
                    chunks.append(self._create_chunk(
                        content=chunk_content,
                        line_start=block_lines[0]['line_no'],
                        line_end=block_lines[-1]['line_no'],
                        chunk_index=chunk_index,
                        total_chunks=0
                    ))
                    chunk_index += 1
                    current_lines = []
                    current_chars = 0
                else:
                    # Block too big - need to split it
                    # For now, fall back to regular chunking
                    current_lines = self._get_overlap_lines(current_lines)
                    current_lines.extend(block_lines)
                    current_chars = sum(len(l['line']) + 1 for l in current_lines)
        
        # Finalize last chunk
        if current_lines:
            chunk_content = '\n'.join(l['line'] for l in current_lines)
            chunks.append(self._create_chunk(
                content=chunk_content,
                line_start=current_lines[0]['line_no'],
                line_end=current_lines[-1]['line_no'],
                chunk_index=chunk_index,
                total_chunks=0
            ))
        
        return chunks
    
    def _identify_blocks(
        self,
        lines: List[str],
        start: int,
        end: int
    ) -> List[dict]:
        """
        Identify special content blocks in the section.
        
        Returns list of blocks with type and lines.
        """
        blocks = []
        i = start
        
        while i < end:
            line = lines[i]
            
            # Check for code block
            if line.strip().startswith('```'):
                block_lines = [{'line': line, 'line_no': i + 1}]
                i += 1
                while i < end and not lines[i].strip().startswith('```'):
                    block_lines.append({'line': lines[i], 'line_no': i + 1})
                    i += 1
                if i < end:
                    block_lines.append({'line': lines[i], 'line_no': i + 1})
                    i += 1
                blocks.append({'type': 'code', 'lines': block_lines})
                continue
            
            # Check for table
            if '|' in line and i + 1 < end and '|' in lines[i + 1]:
                block_lines = [{'line': line, 'line_no': i + 1}]
                i += 1
                while i < end and '|' in lines[i]:
                    block_lines.append({'line': lines[i], 'line_no': i + 1})
                    i += 1
                blocks.append({'type': 'table', 'lines': block_lines})
                continue
            
            # Check for list
            if re.match(r'^[\s]*[-*+\d]+[.)]?\s', line):
                block_lines = [{'line': line, 'line_no': i + 1}]
                i += 1
                while i < end and (lines[i].strip() == '' or 
                                   lines[i].startswith(' ') or
                                   re.match(r'^[\s]*[-*+\d]+[.)]?\s', lines[i])):
                    block_lines.append({'line': lines[i], 'line_no': i + 1})
                    i += 1
                blocks.append({'type': 'list', 'lines': block_lines})
                continue
            
            # Regular paragraph
            block_lines = [{'line': line, 'line_no': i + 1}]
            i += 1
            blocks.append({'type': 'paragraph', 'lines': block_lines})
        
        return blocks


# Utility functions

def chunk_simple(
    content: str,
    chunk_size: int = 1600,  # ~400 tokens
    overlap: int = 320       # ~80 tokens
) -> List[MemoryChunk]:
    """
    Simple chunking without markdown awareness.
    
    Useful for non-markdown content or quick chunking.
    """
    config = ChunkingConfig(
        chunk_tokens=chunk_size // 4,
        overlap_tokens=overlap // 4
    )
    chunker = MarkdownChunker(config)
    return chunker.chunk(content)


def get_chunk_token_estimate(content: str) -> int:
    """Estimate token count for content."""
    return len(content) // 4


def merge_small_chunks(
    chunks: List[MemoryChunk],
    min_tokens: int = 100
) -> List[MemoryChunk]:
    """
    Merge chunks that are too small.
    
    Useful for post-processing to reduce chunk count.
    """
    if not chunks:
        return chunks
    
    merged = []
    current = chunks[0]
    
    for next_chunk in chunks[1:]:
        current_tokens = current.token_estimate
        next_tokens = next_chunk.token_estimate
        
        if current_tokens < min_tokens and current_tokens + next_tokens < 400:
            # Merge chunks
            current.content += '\n' + next_chunk.content
            current.line_end = next_chunk.line_end
            current.content_hash = hashlib.sha256(
                current.content.encode('utf-8')
            ).hexdigest()
        else:
            merged.append(current)
            current = next_chunk
    
    merged.append(current)
    
    # Update indices
    for i, chunk in enumerate(merged):
        chunk.chunk_index = i
        chunk.total_chunks = len(merged)
    
    return merged
