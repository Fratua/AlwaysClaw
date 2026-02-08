# Email Parsing and Content Extraction System
## Technical Specification for OpenClaw Windows 10 AI Agent

---

## Executive Summary

This document provides a comprehensive technical specification for an enterprise-grade email parsing and content extraction system designed for the Windows 10 OpenClaw-inspired AI agent framework. The system handles MIME message parsing, HTML-to-text conversion, attachment processing, email normalization, and structured data extraction.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [MIME Message Parsing](#mime-message-parsing)
4. [HTML to Text Conversion](#html-to-text-conversion)
5. [Attachment Handling](#attachment-handling)
6. [Email Header Analysis](#email-header-analysis)
7. [Character Encoding Handling](#character-encoding-handling)
8. [Email Normalization](#email-normalization)
9. [Structured Data Extraction](#structured-data-extraction)
10. [Processing Pipelines](#processing-pipelines)
11. [Implementation Code](#implementation-code)
12. [Integration Points](#integration-points)

---

## Architecture Overview

### System Design Principles

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    EMAIL PARSING SYSTEM ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │   Input      │───▶│   Parsing    │───▶│  Extraction  │               │
│  │   Sources    │    │   Engine     │    │   Pipeline   │               │
│  └──────────────┘    └──────────────┘    └──────────────┘               │
│         │                   │                   │                       │
│         ▼                   ▼                   ▼                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │ Gmail API    │    │ MIME Parser  │    │ Content      │               │
│  │ IMAP/ POP3   │    │ HTML Parser  │    │ Normalizer   │               │
│  │ File (.eml)  │    │ Attachment   │    │ Signature    │               │
│  │ MSG Files    │    │ Extractor    │    │ Extractor    │               │
│  └──────────────┘    └──────────────┘    └──────────────┘               │
│                                                  │                       │
│                                                  ▼                       │
│                                         ┌──────────────┐                │
│                                         │   Output     │                │
│                                         │   Formats    │                │
│                                         │              │                │
│                                         │ • JSON       │                │
│                                         │ • Plain Text │                │
│                                         │ • Structured │                │
│                                         │ • Attachments│                │
│                                         └──────────────┘                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Component | Library/Tool | Purpose |
|-----------|--------------|---------|
| MIME Parsing | `email` (stdlib) + `policy.default` | RFC-compliant email parsing |
| HTML Parsing | `BeautifulSoup4` + `lxml` | DOM manipulation |
| HTML→Text | `html2text` | Markdown-formatted text |
| Signature Extraction | `talon` (Mailgun) | ML-based signature detection |
| Quote Extraction | `talon.quotations` | Email thread parsing |
| Encoding | `chardet` + `charset-normalizer` | Character encoding detection |
| Validation | `pydantic` + `email-validator` | Data validation |
| MSG Files | `extract-msg` | Outlook .msg support |

---

## Core Components

### 1. EmailMessage Model (Pydantic)

```python
from pydantic import BaseModel, EmailStr, Field
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from enum import Enum

class AttachmentType(str, Enum):
    DOCUMENT = "document"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    ARCHIVE = "archive"
    CODE = "code"
    UNKNOWN = "unknown"

class EmailAttachment(BaseModel):
    filename: str
    content_type: str
    size_bytes: int
    attachment_type: AttachmentType
    content_id: Optional[str] = None
    is_inline: bool = False
    content: Optional[bytes] = None
    extracted_text: Optional[str] = None  # For searchable attachments
    
class EmailAddress(BaseModel):
    email: EmailStr
    name: Optional[str] = None
    
    def __str__(self) -> str:
        if self.name:
            return f"{self.name} <{self.email}>"
        return self.email

class EmailSignature(BaseModel):
    raw_text: str
    name: Optional[str] = None
    title: Optional[str] = None
    company: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    social_links: Dict[str, str] = Field(default_factory=dict)

class EmailQuote(BaseModel):
    quoted_text: str
    author: Optional[str] = None
    date: Optional[datetime] = None
    is_forwarded: bool = False

class ParsedEmail(BaseModel):
    # Identification
    message_id: str
    thread_id: Optional[str] = None
    
    # Headers
    subject: str
    from_address: EmailAddress
    to_addresses: List[EmailAddress]
    cc_addresses: List[EmailAddress] = Field(default_factory=list)
    bcc_addresses: List[EmailAddress] = Field(default_factory=list)
    reply_to: Optional[EmailAddress] = None
    
    # Timing
    date: datetime
    received_date: Optional[datetime] = None
    
    # Content
    body_plain: Optional[str] = None
    body_html: Optional[str] = None
    body_clean: str  # Normalized, cleaned text
    
    # Structured Content
    signature: Optional[EmailSignature] = None
    quotes: List[EmailQuote] = Field(default_factory=list)
    main_content: str  # Body without signatures/quotes
    
    # Attachments
    attachments: List[EmailAttachment] = Field(default_factory=list)
    has_attachments: bool = False
    
    # Metadata
    headers: Dict[str, str] = Field(default_factory=dict)
    priority: Optional[str] = None
    in_reply_to: Optional[str] = None
    references: List[str] = Field(default_factory=list)
    
    # Processing metadata
    encoding_detected: str = "utf-8"
    processing_timestamp: datetime = Field(default_factory=datetime.utcnow)
    parsing_version: str = "1.0.0"
```

---

## MIME Message Parsing

### Parsing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     MIME PARSING PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Raw Email (bytes/str)                                          │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────┐                                    │
│  │ BytesParser with        │◄── policy=email.policy.default    │
│  │ policy.default          │    (Modern Python 3.6+ API)        │
│  └─────────────────────────┘                                    │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────┐                                    │
│  │ EmailMessage Object     │                                    │
│  │ - Headers accessible    │                                    │
│  │ - is_multipart() check  │                                    │
│  └─────────────────────────┘                                    │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────┐    ┌─────────────────────────┐     │
│  │ Multipart?              │───▶│ Iterate parts with      │     │
│  │                         │    │ walk() or iter_parts()  │     │
│  └─────────────────────────┘    └─────────────────────────┘     │
│       │                                    │                    │
│       │ No                                 ▼                    │
│       │                           ┌─────────────────────────┐   │
│       │                           │ For each part:          │   │
│       │                           │ - get_content_type()    │   │
│       │                           │ - get_payload(decode)   │   │
│       │                           │ - get_filename()        │   │
│       │                           └─────────────────────────┘   │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────┐                                    │
│  │ Extract body with       │◄── preferencelist=('plain', 'html')│
│  │ get_body()              │                                    │
│  └─────────────────────────┘                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
import email
from email import policy
from email.parser import BytesParser
from email.message import EmailMessage
from typing import Union, Optional

class MIMEParser:
    """RFC-compliant MIME email parser with modern Python email API."""
    
    def __init__(self):
        self.policy = policy.default
    
    def parse(self, email_data: Union[str, bytes]) -> EmailMessage:
        """
        Parse email from string or bytes.
        
        Args:
            email_data: Raw email content as string or bytes
            
        Returns:
            EmailMessage object with parsed content
        """
        if isinstance(email_data, bytes):
            return BytesParser(policy=self.policy).parsebytes(email_data)
        else:
            return email.message_from_string(email_data, policy=self.policy)
    
    def parse_from_file(self, filepath: str) -> EmailMessage:
        """Parse email from file path."""
        with open(filepath, 'rb') as f:
            return BytesParser(policy=self.policy).parse(f)
    
    def get_body_content(self, msg: EmailMessage, prefer_html: bool = False) -> dict:
        """
        Extract body content from email.
        
        Returns dict with 'plain' and 'html' keys.
        """
        result = {'plain': None, 'html': None}
        
        # Use modern get_body() API (Python 3.6+)
        if prefer_html:
            body = msg.get_body(preferencelist=('html', 'plain'))
        else:
            body = msg.get_body(preferencelist=('plain', 'html'))
        
        if body:
            content_type = body.get_content_type()
            content = body.get_content()
            
            if content_type == 'text/plain':
                result['plain'] = content
            elif content_type == 'text/html':
                result['html'] = content
        
        # Fallback: manual iteration for complex multipart
        if not result['plain'] and not result['html']:
            for part in msg.walk():
                content_type = part.get_content_type()
                
                if content_type == 'text/plain' and not result['plain']:
                    try:
                        result['plain'] = part.get_payload(decode=True).decode(
                            part.get_content_charset() or 'utf-8', errors='replace'
                        )
                    except Exception:
                        pass
                        
                elif content_type == 'text/html' and not result['html']:
                    try:
                        result['html'] = part.get_payload(decode=True).decode(
                            part.get_content_charset() or 'utf-8', errors='replace'
                        )
                    except Exception:
                        pass
        
        return result
```

---

## HTML to Text Conversion

### Conversion Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                 HTML TO TEXT CONVERSION PIPELINE                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  HTML Content                                                   │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────┐                                    │
│  │ BeautifulSoup Parser    │◄── lxml or html5lib parser        │
│  │ (lxml is fastest)       │                                    │
│  └─────────────────────────┘                                    │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────┐                                    │
│  │ Preprocessing           │                                    │
│  │ - Remove script/style   │                                    │
│  │ - Handle inline images  │                                    │
│  │ - Normalize links       │                                    │
│  └─────────────────────────┘                                    │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────┐    ┌─────────────────────────┐     │
│  │ html2text Converter     │───▶│ Markdown-formatted      │     │
│  │ (preserves structure)   │    │ plain text output       │     │
│  └─────────────────────────┘    └─────────────────────────┘     │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────┐                                    │
│  │ Post-processing         │                                    │
│  │ - Normalize whitespace  │                                    │
│  │ - Decode HTML entities  │                                    │
│  │ - Handle line breaks    │                                    │
│  └─────────────────────────┘                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
from bs4 import BeautifulSoup
import html2text
import html
from typing import Optional

class HTMLToTextConverter:
    """Convert HTML emails to clean plain text."""
    
    def __init__(self):
        self.converter = html2text.HTML2Text()
        self._configure_converter()
    
    def _configure_converter(self):
        """Configure html2text for email processing."""
        # Body width: 0 = no wrapping
        self.converter.body_width = 0
        
        # Include links in format: [text](url)
        self.converter.ignore_links = False
        
        # Don't ignore images (alt text will be shown)
        self.converter.ignore_images = False
        
        # Use reference-style links
        self.converter.inline_links = True
        
        # Protect links (don't convert to markdown)
        self.converter.protect_links = False
        
        # Include tables
        self.converter.ignore_tables = False
        
        # Pad tables with spaces
        self.converter.pad_tables = True
        
        # Skip internal links
        self.converter.skip_internal_links = True
        
        # Treat <div> as paragraph
        self.converter.use_automatic_links = True
    
    def convert(self, html_content: str, clean_whitespace: bool = True) -> str:
        """
        Convert HTML to plain text.
        
        Args:
            html_content: Raw HTML string
            clean_whitespace: Whether to normalize whitespace
            
        Returns:
            Plain text representation
        """
        if not html_content:
            return ""
        
        # Preprocess HTML
        html_content = self._preprocess_html(html_content)
        
        # Convert to markdown-style text
        text = self.converter.handle(html_content)
        
        # Post-process
        if clean_whitespace:
            text = self._clean_whitespace(text)
        
        # Decode HTML entities
        text = html.unescape(text)
        
        return text.strip()
    
    def _preprocess_html(self, html_content: str) -> str:
        """Clean HTML before conversion."""
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Remove script and style elements
        for element in soup(['script', 'style', 'noscript']):
            element.decompose()
        
        # Handle inline images - keep alt text
        for img in soup.find_all('img'):
            alt = img.get('alt', '')
            if alt:
                img.replace_with(f" [Image: {alt}] ")
            else:
                img.decompose()
        
        # Convert blockquotes to quoted text
        for blockquote in soup.find_all('blockquote'):
            text = blockquote.get_text()
            quoted = '\n'.join(f'> {line}' for line in text.split('\n'))
            blockquote.replace_with(quoted)
        
        return str(soup)
    
    def _clean_whitespace(self, text: str) -> str:
        """Normalize whitespace in converted text."""
        import re
        
        # Replace multiple newlines with max 2
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        
        return '\n'.join(lines)


class AlternativeHTMLExtractor:
    """Alternative extraction using BeautifulSoup directly."""
    
    @staticmethod
    def extract_text(html_content: str, separator: str = '\n') -> str:
        """
        Extract text using BeautifulSoup's get_text().
        Faster but less structure preservation.
        """
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Remove script and style
        for element in soup(['script', 'style']):
            element.decompose()
        
        # Get text with separator
        text = soup.get_text(separator=separator, strip=True)
        
        # Clean up
        import re
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return html.unescape(text)
```

---

## Attachment Handling

### Attachment Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                  ATTACHMENT HANDLING PIPELINE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  MIME Parts                                                     │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────┐                                    │
│  │ Detect Attachments      │◄── Check Content-Disposition     │
│  │                         │    or get_filename()              │
│  └─────────────────────────┘                                    │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────┐                                    │
│  │ Classify Attachment     │◄── By MIME type & extension      │
│  │                         │                                    │
│  └─────────────────────────┘                                    │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────┐    ┌─────────────────────────┐     │
│  │ Extract Content         │───▶│ Text Extractable?       │     │
│  │ (decode base64/qp)      │    │ (PDF, DOCX, etc.)       │     │
│  └─────────────────────────┘    └─────────────────────────┘     │
│       │                                    │                    │
│       │                                    │ Yes                │
│       │                                    ▼                    │
│       │                           ┌─────────────────────────┐   │
│       │                           │ Extract Text Content    │   │
│       │                           │ for search/indexing     │   │
│       │                           └─────────────────────────┘   │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────┐                                    │
│  │ Store/Save Attachment   │◄── To filesystem or database     │
│  │                         │                                    │
│  └─────────────────────────┘                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
import mimetypes
import magic  # python-magic for file type detection
from pathlib import Path
from typing import List, Optional, BinaryIO
import base64
import quopri

class AttachmentHandler:
    """Handle email attachment extraction and processing."""
    
    # MIME type to category mapping
    MIME_CATEGORIES = {
        # Documents
        'application/pdf': AttachmentType.DOCUMENT,
        'application/msword': AttachmentType.DOCUMENT,
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': AttachmentType.DOCUMENT,
        'application/vnd.ms-excel': AttachmentType.DOCUMENT,
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': AttachmentType.DOCUMENT,
        'application/vnd.ms-powerpoint': AttachmentType.DOCUMENT,
        'application/vnd.openxmlformats-officedocument.presentationml.presentation': AttachmentType.DOCUMENT,
        'text/plain': AttachmentType.DOCUMENT,
        'text/csv': AttachmentType.DOCUMENT,
        'application/rtf': AttachmentType.DOCUMENT,
        
        # Images
        'image/jpeg': AttachmentType.IMAGE,
        'image/png': AttachmentType.IMAGE,
        'image/gif': AttachmentType.IMAGE,
        'image/bmp': AttachmentType.IMAGE,
        'image/webp': AttachmentType.IMAGE,
        'image/svg+xml': AttachmentType.IMAGE,
        'image/tiff': AttachmentType.IMAGE,
        
        # Audio
        'audio/mpeg': AttachmentType.AUDIO,
        'audio/wav': AttachmentType.AUDIO,
        'audio/ogg': AttachmentType.AUDIO,
        'audio/mp4': AttachmentType.AUDIO,
        
        # Video
        'video/mp4': AttachmentType.VIDEO,
        'video/mpeg': AttachmentType.VIDEO,
        'video/quicktime': AttachmentType.VIDEO,
        'video/webm': AttachmentType.VIDEO,
        
        # Archives
        'application/zip': AttachmentType.ARCHIVE,
        'application/x-rar-compressed': AttachmentType.ARCHIVE,
        'application/x-7z-compressed': AttachmentType.ARCHIVE,
        'application/gzip': AttachmentType.ARCHIVE,
        'application/x-tar': AttachmentType.ARCHIVE,
        
        # Code
        'text/html': AttachmentType.CODE,
        'text/css': AttachmentType.CODE,
        'application/javascript': AttachmentType.CODE,
        'application/json': AttachmentType.CODE,
        'text/xml': AttachmentType.CODE,
        'application/xml': AttachmentType.CODE,
    }
    
    # Extensions for text extraction
    TEXT_EXTRACTABLE = {'.pdf', '.docx', '.doc', '.txt', '.csv', '.json', '.xml', '.html'}
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else None
        self.magic = magic.Magic(mime=True)
    
    def extract_attachments(self, msg: EmailMessage) -> List[EmailAttachment]:
        """Extract all attachments from email message."""
        attachments = []
        
        for part in msg.walk():
            # Check if this part is an attachment
            if self._is_attachment(part):
                attachment = self._process_attachment(part)
                if attachment:
                    attachments.append(attachment)
        
        return attachments
    
    def _is_attachment(self, part) -> bool:
        """Determine if MIME part is an attachment."""
        # Check Content-Disposition header
        content_disposition = part.get('Content-Disposition', '')
        if 'attachment' in content_disposition.lower():
            return True
        
        # Check if has filename
        if part.get_filename():
            return True
        
        # Inline images with Content-ID
        content_id = part.get('Content-ID')
        content_type = part.get_content_type()
        if content_id and content_type.startswith('image/'):
            return True
        
        return False
    
    def _process_attachment(self, part) -> Optional[EmailAttachment]:
        """Process a single attachment part."""
        try:
            filename = part.get_filename() or 'unnamed'
            content_type = part.get_content_type()
            content_id = part.get('Content-ID', '').strip('<>')
            
            # Get decoded payload
            payload = part.get_payload(decode=True)
            if not payload:
                return None
            
            # Detect MIME type if generic
            if content_type in ('application/octet-stream', 'application/binary'):
                content_type = self.magic.from_buffer(payload)
            
            # Classify attachment
            attachment_type = self.MIME_CATEGORIES.get(
                content_type, AttachmentType.UNKNOWN
            )
            
            # Check if inline
            content_disposition = part.get('Content-Disposition', '')
            is_inline = 'inline' in content_disposition.lower()
            
            attachment = EmailAttachment(
                filename=filename,
                content_type=content_type,
                size_bytes=len(payload),
                attachment_type=attachment_type,
                content_id=content_id or None,
                is_inline=is_inline,
                content=payload
            )
            
            # Extract text if possible
            if self._is_text_extractable(filename, content_type):
                attachment.extracted_text = self._extract_text(attachment)
            
            return attachment
            
        except Exception as e:
            # Log error but don't fail entire parsing
            print(f"Error processing attachment: {e}")
            return None
    
    def _is_text_extractable(self, filename: str, content_type: str) -> bool:
        """Check if attachment supports text extraction."""
        ext = Path(filename).suffix.lower()
        return ext in self.TEXT_EXTRACTABLE or content_type.startswith('text/')
    
    def _extract_text(self, attachment: EmailAttachment) -> Optional[str]:
        """Extract searchable text from attachment."""
        ext = Path(attachment.filename).suffix.lower()
        
        try:
            if ext == '.pdf':
                return self._extract_pdf_text(attachment.content)
            elif ext in ('.docx', '.doc'):
                return self._extract_docx_text(attachment.content)
            elif attachment.content_type.startswith('text/'):
                return attachment.content.decode('utf-8', errors='replace')
            else:
                return None
        except Exception as e:
            print(f"Text extraction failed for {attachment.filename}: {e}")
            return None
    
    def _extract_pdf_text(self, content: bytes) -> Optional[str]:
        """Extract text from PDF using PyPDF2 or pdfplumber."""
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                return '\n'.join(page.extract_text() or '' for page in pdf.pages)
        except ImportError:
            # Fallback to PyPDF2
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(content))
            return '\n'.join(page.extract_text() or '' for page in reader.pages)
    
    def _extract_docx_text(self, content: bytes) -> Optional[str]:
        """Extract text from DOCX using python-docx."""
        import docx
        doc = docx.Document(io.BytesIO(content))
        return '\n'.join(paragraph.text for paragraph in doc.paragraphs)
    
    def save_attachment(self, attachment: EmailAttachment, 
                        output_dir: Optional[str] = None) -> str:
        """Save attachment to filesystem."""
        output_path = Path(output_dir) if output_dir else self.storage_path
        if not output_path:
            raise ValueError("No output directory specified")
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Sanitize filename
        safe_filename = self._sanitize_filename(attachment.filename)
        filepath = output_path / safe_filename
        
        # Handle duplicates
        counter = 1
        original_filepath = filepath
        while filepath.exists():
            stem = original_filepath.stem
            suffix = original_filepath.suffix
            filepath = output_path / f"{stem}_{counter}{suffix}"
            counter += 1
        
        with open(filepath, 'wb') as f:
            f.write(attachment.content)
        
        return str(filepath)
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for filesystem safety."""
        import re
        # Remove path separators and dangerous characters
        filename = re.sub(r'[\\/:*?"<>|]', '_', filename)
        # Limit length
        if len(filename) > 255:
            name, ext = Path(filename).stem, Path(filename).suffix
            filename = name[:255 - len(ext)] + ext
        return filename
```

---

## Email Header Analysis

### Header Processing

```python
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import re

class HeaderAnalyzer:
    """Analyze and extract information from email headers."""
    
    # Priority mappings
    PRIORITY_MAP = {
        '1': 'high',
        '2': 'high',
        '3': 'normal',
        '4': 'low',
        '5': 'low',
    }
    
    def __init__(self):
        self.received_pattern = re.compile(
            r'from\s+(?P<from_host>\S+)\s+\(.*?\[(?P<ip>[\d.]+)\]\)'
            r'.*?(?P<date>\w{3},\s+\d{1,2}\s+\w{3}.*?\d{4}.*?[+-]\d{4})',
            re.DOTALL | re.IGNORECASE
        )
    
    def analyze(self, msg: EmailMessage) -> Dict:
        """Analyze all headers from email message."""
        return {
            'message_id': self._get_message_id(msg),
            'thread_id': self._get_thread_id(msg),
            'subject': self._get_subject(msg),
            'from_address': self._parse_address(msg.get('From', '')),
            'to_addresses': self._parse_address_list(msg.get('To', '')),
            'cc_addresses': self._parse_address_list(msg.get('Cc', '')),
            'bcc_addresses': self._parse_address_list(msg.get('Bcc', '')),
            'reply_to': self._parse_address(msg.get('Reply-To', '')) if msg.get('Reply-To') else None,
            'date': self._parse_date(msg.get('Date')),
            'received_date': self._parse_received_date(msg),
            'priority': self._get_priority(msg),
            'in_reply_to': msg.get('In-Reply-To', '').strip('<>') or None,
            'references': self._parse_references(msg.get('References', '')),
            'headers': dict(msg.items()),  # All raw headers
        }
    
    def _get_message_id(self, msg: EmailMessage) -> str:
        """Extract message ID."""
        msg_id = msg.get('Message-ID', '')
        return msg_id.strip('<>') if msg_id else ''
    
    def _get_thread_id(self, msg: EmailMessage) -> Optional[str]:
        """Extract thread ID (Gmail specific or from References)."""
        # Gmail thread ID
        thread_id = msg.get('X-Gm-Thread-ID') or msg.get('X-GM-THRID')
        if thread_id:
            return thread_id
        
        # Use first reference as thread indicator
        references = self._parse_references(msg.get('References', ''))
        if references:
            return references[0]
        
        # Use In-Reply-To
        in_reply_to = msg.get('In-Reply-To', '').strip('<>')
        if in_reply_to:
            return in_reply_to
        
        return None
    
    def _get_subject(self, msg: EmailMessage) -> str:
        """Extract and clean subject."""
        subject = msg.get('Subject', '')
        # Remove Re: and Fwd: prefixes for grouping
        return subject.strip()
    
    def _parse_address(self, address_str: str) -> Optional[EmailAddress]:
        """Parse single email address."""
        if not address_str:
            return None
        
        from email.utils import parseaddr
        name, email_addr = parseaddr(address_str)
        
        if not email_addr:
            return None
        
        return EmailAddress(
            email=email_addr.lower(),
            name=name.strip() if name else None
        )
    
    def _parse_address_list(self, addresses_str: str) -> List[EmailAddress]:
        """Parse comma-separated list of addresses."""
        if not addresses_str:
            return []
        
        from email.utils import getaddresses
        addresses = getaddresses([addresses_str])
        
        result = []
        for name, email_addr in addresses:
            if email_addr:
                result.append(EmailAddress(
                    email=email_addr.lower(),
                    name=name.strip() if name else None
                ))
        
        return result
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date header to datetime."""
        if not date_str:
            return None
        
        from email.utils import parsedate_to_datetime
        try:
            return parsedate_to_datetime(date_str)
        except Exception:
            return None
    
    def _parse_received_date(self, msg: EmailMessage) -> Optional[datetime]:
        """Extract received date from Received headers."""
        received_headers = msg.get_all('Received', [])
        
        for header in received_headers:
            match = self.received_pattern.search(header)
            if match:
                date_str = match.group('date')
                return self._parse_date(date_str)
        
        return None
    
    def _get_priority(self, msg: EmailMessage) -> Optional[str]:
        """Determine email priority."""
        # Check X-Priority
        x_priority = msg.get('X-Priority', '')
        if x_priority:
            priority_num = x_priority.split('.')[0]
            return self.PRIORITY_MAP.get(priority_num)
        
        # Check Importance
        importance = msg.get('Importance', '').lower()
        if importance in ('high', 'low'):
            return importance
        
        # Check X-MSMail-Priority (Outlook)
        ms_priority = msg.get('X-MSMail-Priority', '').lower()
        if ms_priority in ('high', 'low'):
            return ms_priority
        
        return 'normal'
    
    def _parse_references(self, references_str: str) -> List[str]:
        """Parse References header."""
        if not references_str:
            return []
        
        # Split by whitespace and clean
        refs = references_str.split()
        return [ref.strip('<>') for ref in refs if ref]
    
    def get_delivery_path(self, msg: EmailMessage) -> List[Dict]:
        """Trace email delivery path from Received headers."""
        received_headers = msg.get_all('Received', [])
        path = []
        
        for header in reversed(received_headers):  # Oldest first
            info = self._parse_received_header(header)
            if info:
                path.append(info)
        
        return path
    
    def _parse_received_header(self, header: str) -> Optional[Dict]:
        """Parse individual Received header."""
        match = self.received_pattern.search(header)
        if match:
            return {
                'from_host': match.group('from_host'),
                'ip_address': match.group('ip'),
                'date': match.group('date'),
            }
        return None
```

---

## Character Encoding Handling

### Encoding Detection & Conversion

```python
import chardet
from charset_normalizer import detect as charset_detect
from typing import Union, Optional

class EncodingHandler:
    """Handle character encoding detection and conversion."""
    
    # Common encodings to try
    FALLBACK_ENCODINGS = [
        'utf-8',
        'latin-1',
        'cp1252',
        'iso-8859-1',
        'windows-1252',
        'ascii',
    ]
    
    def __init__(self):
        self.preferred_detector = 'chardet'  # or 'charset_normalizer'
    
    def decode(self, data: bytes, 
               declared_encoding: Optional[str] = None) -> Tuple[str, str]:
        """
        Decode bytes to string with encoding detection.
        
        Returns:
            Tuple of (decoded_string, detected_encoding)
        """
        # Try declared encoding first
        if declared_encoding:
            try:
                return data.decode(declared_encoding), declared_encoding
            except (UnicodeDecodeError, LookupError):
                pass
        
        # Detect encoding
        detected = self.detect_encoding(data)
        
        # Try detected encoding
        if detected and detected != declared_encoding:
            try:
                return data.decode(detected), detected
            except UnicodeDecodeError:
                pass
        
        # Try fallback encodings
        for encoding in self.FALLBACK_ENCODINGS:
            try:
                return data.decode(encoding), encoding
            except UnicodeDecodeError:
                continue
        
        # Last resort: decode with replacement
        return data.decode('utf-8', errors='replace'), 'utf-8-replace'
    
    def detect_encoding(self, data: bytes) -> Optional[str]:
        """Detect character encoding of byte data."""
        if not data:
            return 'utf-8'
        
        # Use chardet
        if self.preferred_detector == 'chardet':
            result = chardet.detect(data)
            if result and result['confidence'] > 0.5:
                return result['encoding']
        
        # Fallback to charset_normalizer
        try:
            result = charset_detect(data)
            if result:
                return result['encoding']
        except Exception:
            pass
        
        return None
    
    def normalize_encoding_name(self, encoding: str) -> str:
        """Normalize encoding name for consistency."""
        encoding = encoding.lower().replace('_', '-')
        
        # Common aliases
        aliases = {
            'utf8': 'utf-8',
            'utf-8-sig': 'utf-8',
            'latin1': 'latin-1',
            'iso-8859-1': 'latin-1',
            'windows-1252': 'cp1252',
            'us-ascii': 'ascii',
        }
        
        return aliases.get(encoding, encoding)
    
    def encode_for_storage(self, text: str, 
                           encoding: str = 'utf-8') -> bytes:
        """Encode text for storage with error handling."""
        return text.encode(encoding, errors='replace')
    
    def clean_utf8(self, text: str) -> str:
        """
        Clean text to ensure valid UTF-8.
        Removes or replaces invalid characters.
        """
        # Encode to bytes and decode with replacement
        return text.encode('utf-8', errors='ignore').decode('utf-8')
    
    def handle_mime_encoded_words(self, text: str) -> str:
        """
        Decode MIME encoded-words (e.g., =?UTF-8?B?...?=).
        """
        import email.header
        
        try:
            decoded_header = email.header.decode_header(text)
            parts = []
            
            for content, charset in decoded_header:
                if isinstance(content, bytes):
                    if charset:
                        parts.append(content.decode(charset, errors='replace'))
                    else:
                        parts.append(content.decode('utf-8', errors='replace'))
                else:
                    parts.append(content)
            
            return ''.join(parts)
        except Exception:
            return text
```

---

## Email Normalization

### Normalization Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                   EMAIL NORMALIZATION PIPELINE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Raw Email Content                                              │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────┐                                    │
│  │ Whitespace Normalization│                                    │
│  │ - Normalize newlines    │                                    │
│  │ - Remove trailing spaces│                                    │
│  └─────────────────────────┘                                    │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────┐                                    │
│  │ Quote Normalization     │                                    │
│  │ - Standardize quote     │                                    │
│  │   markers (> | >>)      │                                    │
│  └─────────────────────────┘                                    │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────┐                                    │
│  │ Signature Removal       │◄── Using talon library           │
│  │                         │                                    │
│  └─────────────────────────┘                                    │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────┐                                    │
│  │ Quote Removal           │◄── Extract main content          │
│  │                         │                                    │
│  └─────────────────────────┘                                    │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────┐                                    │
│  │ Content Cleaning        │                                    │
│  │ - Remove boilerplate    │                                    │
│  │ - Fix spacing           │                                    │
│  └─────────────────────────┘                                    │
│       │                                                         │
│       ▼                                                         │
│  Clean, Normalized Text                                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
import re
from typing import Optional, List
import talon
from talon import quotations
from talon.signature import extract as extract_signature

class EmailNormalizer:
    """Normalize and clean email content."""
    
    def __init__(self):
        # Initialize talon for signature/quote extraction
        talon.init()
    
    def normalize(self, text: str, 
                  remove_signature: bool = True,
                  remove_quotes: bool = True) -> Dict[str, any]:
        """
        Normalize email text.
        
        Returns dict with:
            - clean_text: Main content
            - signature: Extracted signature (if any)
            - quotes: Extracted quotes (if any)
            - original: Original text
        """
        result = {
            'original': text,
            'clean_text': text,
            'signature': None,
            'quotes': []
        }
        
        if not text:
            return result
        
        # Step 1: Basic whitespace normalization
        text = self._normalize_whitespace(text)
        
        # Step 2: Normalize quote markers
        text = self._normalize_quotes(text)
        
        # Step 3: Extract signature
        if remove_signature:
            text, signature = self._extract_signature(text)
            if signature:
                result['signature'] = signature
        
        # Step 4: Extract quotes
        if remove_quotes:
            text, quotes = self._extract_quotes(text)
            if quotes:
                result['quotes'] = quotes
        
        # Step 5: Final cleaning
        text = self._final_clean(text)
        
        result['clean_text'] = text
        return result
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace characters."""
        # Normalize newlines
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove trailing whitespace from each line
        lines = [line.rstrip() for line in text.split('\n')]
        
        # Remove excessive blank lines (max 2 consecutive)
        result = []
        blank_count = 0
        for line in lines:
            if line == '':
                blank_count += 1
                if blank_count <= 2:
                    result.append(line)
            else:
                blank_count = 0
                result.append(line)
        
        return '\n'.join(result)
    
    def _normalize_quotes(self, text: str) -> str:
        """Normalize quote markers to standard format."""
        lines = text.split('\n')
        normalized = []
        
        for line in lines:
            # Count leading quote markers
            quote_match = re.match(r'^(\s*)([>|]+)\s?(.*)', line)
            if quote_match:
                indent = quote_match.group(1)
                markers = quote_match.group(2)
                content = quote_match.group(3)
                
                # Normalize to > style
                depth = len(markers.replace(' ', ''))
                normalized_marker = '> ' * depth
                normalized.append(f"{indent}{normalized_marker}{content}")
            else:
                normalized.append(line)
        
        return '\n'.join(normalized)
    
    def _extract_signature(self, text: str) -> Tuple[str, Optional[str]]:
        """Extract signature using talon."""
        try:
            # Use talon's signature extraction
            body, signature = extract_signature(text)
            return body.strip(), signature.strip() if signature else None
        except Exception:
            return text, None
    
    def _extract_quotes(self, text: str) -> Tuple[str, List[str]]:
        """Extract quoted content using talon."""
        try:
            # Extract reply from plain text
            reply = quotations.extract_from_plain(text)
            
            # The difference is the quoted content
            if reply and len(reply) < len(text):
                # Find quoted sections
                quotes = self._find_quoted_sections(text, reply)
                return reply.strip(), quotes
            
            return text.strip(), []
        except Exception:
            return text.strip(), []
    
    def _find_quoted_sections(self, original: str, reply: str) -> List[str]:
        """Identify quoted sections from original vs reply."""
        quotes = []
        
        # Simple approach: find lines not in reply
        original_lines = original.split('\n')
        reply_lines = set(reply.split('\n'))
        
        current_quote = []
        for line in original_lines:
            if line not in reply_lines or line.startswith('>'):
                current_quote.append(line)
            elif current_quote:
                quotes.append('\n'.join(current_quote))
                current_quote = []
        
        if current_quote:
            quotes.append('\n'.join(current_quote))
        
        return quotes
    
    def _final_clean(self, text: str) -> str:
        """Final cleaning steps."""
        # Remove common email separators
        separators = [
            r'--+\s*Original message\s*--+',
            r'--+\s*Forwarded message\s*--+',
            r'On .+ wrote:',
            r'From: .+',
            r'Sent: .+',
            r'To: .+',
            r'Subject: .+',
        ]
        
        for pattern in separators:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Clean up remaining whitespace
        text = self._normalize_whitespace(text)
        
        return text.strip()
    
    def normalize_html_email(self, html_content: str) -> Dict[str, any]:
        """Normalize HTML email content."""
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Remove script and style
        for element in soup(['script', 'style', 'noscript']):
            element.decompose()
        
        # Convert to text
        converter = HTMLToTextConverter()
        text = converter.convert(str(soup))
        
        # Apply normal text normalization
        return self.normalize(text)
```

---

## Structured Data Extraction

### Signature Extraction

```python
import re
from typing import Dict, Optional, List

class SignatureExtractor:
    """Extract structured data from email signatures."""
    
    # Common signature patterns
    SIGNATURE_PATTERNS = {
        'phone': [
            r'(?:[Tt]el|[Pp]hone|[Mm]obile|[Cc]ell|[Ff]ax)[.:]?\s*([\d\s\-\(\)\+\.]+)',
            r'\b(\+?\d[\d\s\-\(\)]{7,20})\b',
        ],
        'email': [
            r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
        ],
        'url': [
            r'(https?://[^\s<>"{}|\\^`\[\]]+)',
            r'www\.[^\s<>"{}|\\^`\[\]]+',
        ],
        'title': [
            r'\b([A-Z][a-z]+\s+(?:Manager|Director|Engineer|Developer|Designer|Consultant|Analyst|Coordinator|Specialist|Lead|Head|VP|CEO|CTO|CFO|COO|President|Founder))\b',
        ],
    }
    
    def extract(self, signature_text: str) -> EmailSignature:
        """Extract structured data from signature text."""
        if not signature_text:
            return EmailSignature(raw_text='')
        
        # Extract name (first line is often the name)
        lines = [l.strip() for l in signature_text.split('\n') if l.strip()]
        name = self._extract_name(lines)
        
        # Extract phone numbers
        phones = self._extract_phones(signature_text)
        
        # Extract email
        emails = self._extract_emails(signature_text)
        
        # Extract URLs
        urls = self._extract_urls(signature_text)
        
        # Extract title
        title = self._extract_title(signature_text)
        
        # Extract company
        company = self._extract_company(signature_text, lines)
        
        # Extract social links
        social_links = self._extract_social_links(urls)
        
        return EmailSignature(
            raw_text=signature_text,
            name=name,
            title=title,
            company=company,
            phone=phones[0] if phones else None,
            email=emails[0] if emails else None,
            social_links=social_links
        )
    
    def _extract_name(self, lines: List[str]) -> Optional[str]:
        """Extract name from signature lines."""
        if not lines:
            return None
        
        # First non-empty line that looks like a name
        for line in lines[:3]:  # Check first 3 lines
            # Skip lines that look like contact info
            if any(marker in line.lower() for marker in ['@', 'tel', 'phone', 'www', 'http']):
                continue
            
            # Skip lines that are just separators
            if re.match(r'^[-=_\*]+$', line):
                continue
            
            # Looks like a name (2-3 words, no special chars)
            if re.match(r'^[A-Za-z\s\.\-]{2,50}$', line):
                words = line.split()
                if 1 <= len(words) <= 4:
                    return line.strip()
        
        return None
    
    def _extract_phones(self, text: str) -> List[str]:
        """Extract phone numbers from text."""
        phones = []
        for pattern in self.SIGNATURE_PATTERNS['phone']:
            matches = re.findall(pattern, text)
            phones.extend(matches)
        return list(set(phones))  # Remove duplicates
    
    def _extract_emails(self, text: str) -> List[str]:
        """Extract email addresses from text."""
        emails = []
        for pattern in self.SIGNATURE_PATTERNS['email']:
            matches = re.findall(pattern, text)
            emails.extend(matches)
        return list(set(emails))
    
    def _extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text."""
        urls = []
        for pattern in self.SIGNATURE_PATTERNS['url']:
            matches = re.findall(pattern, text)
            urls.extend(matches)
        return list(set(urls))
    
    def _extract_title(self, text: str) -> Optional[str]:
        """Extract job title from text."""
        for pattern in self.SIGNATURE_PATTERNS['title']:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None
    
    def _extract_company(self, text: str, lines: List[str]) -> Optional[str]:
        """Extract company name from signature."""
        # Look for common company indicators
        company_patterns = [
            r'(?:at|with)\s+([A-Z][A-Za-z0-9\s&]+(?:Inc\.?|LLC|Ltd\.?|Corp\.?|Company|Co\.?)?)',
            r'^([A-Z][A-Za-z0-9\s&]+(?:Inc\.?|LLC|Ltd\.?|Corp\.?|GmbH|S\.A\.?))$',
        ]
        
        for pattern in company_patterns:
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_social_links(self, urls: List[str]) -> Dict[str, str]:
        """Extract social media links from URLs."""
        social_patterns = {
            'linkedin': r'linkedin\.com',
            'twitter': r'twitter\.com|x\.com',
            'facebook': r'facebook\.com',
            'instagram': r'instagram\.com',
            'github': r'github\.com',
        }
        
        social_links = {}
        for url in urls:
            for platform, pattern in social_patterns.items():
                if re.search(pattern, url, re.IGNORECASE):
                    social_links[platform] = url
                    break
        
        return social_links
```

---

## Processing Pipelines

### Main Email Processing Pipeline

```python
from typing import Callable, List, Optional
from dataclasses import dataclass

@dataclass
class ProcessingContext:
    """Context passed through processing pipeline."""
    email_data: Union[str, bytes]
    parsed_email: Optional[ParsedEmail] = None
    metadata: Dict[str, any] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.errors is None:
            self.errors = []

class PipelineStage:
    """Base class for pipeline stages."""
    
    def __init__(self, name: str):
        self.name = name
    
    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Process the context. Override in subclasses."""
        raise NotImplementedError

class EmailProcessingPipeline:
    """Main email processing pipeline."""
    
    def __init__(self):
        self.stages: List[PipelineStage] = []
        self.middleware: List[Callable] = []
    
    def add_stage(self, stage: PipelineStage):
        """Add a processing stage."""
        self.stages.append(stage)
        return self
    
    def add_middleware(self, middleware: Callable):
        """Add middleware that wraps all stages."""
        self.middleware.append(middleware)
        return self
    
    def process(self, email_data: Union[str, bytes]) -> ProcessingContext:
        """Run email through processing pipeline."""
        context = ProcessingContext(email_data=email_data)
        
        for stage in self.stages:
            try:
                # Apply middleware
                process_fn = stage.process
                for mw in reversed(self.middleware):
                    process_fn = mw(process_fn)
                
                context = process_fn(context)
                
                if context.errors and self._should_stop_on_error(stage):
                    break
                    
            except Exception as e:
                context.errors.append(f"{stage.name}: {str(e)}")
                if self._should_stop_on_error(stage):
                    break
        
        return context
    
    def _should_stop_on_error(self, stage: PipelineStage) -> bool:
        """Determine if pipeline should stop on error."""
        # Critical stages stop pipeline
        critical_stages = ['parse', 'decode']
        return stage.name.lower() in critical_stages

# Built-in Pipeline Stages

class ParseStage(PipelineStage):
    """Parse raw email to EmailMessage."""
    
    def __init__(self):
        super().__init__('parse')
        self.parser = MIMEParser()
    
    def process(self, context: ProcessingContext) -> ProcessingContext:
        try:
            msg = self.parser.parse(context.email_data)
            context.metadata['email_message'] = msg
        except Exception as e:
            context.errors.append(f"Parse error: {e}")
        return context

class HeaderAnalysisStage(PipelineStage):
    """Analyze email headers."""
    
    def __init__(self):
        super().__init__('headers')
        self.analyzer = HeaderAnalyzer()
    
    def process(self, context: ProcessingContext) -> ProcessingContext:
        msg = context.metadata.get('email_message')
        if not msg:
            context.errors.append("No email message to analyze")
            return context
        
        try:
            headers = self.analyzer.analyze(msg)
            context.metadata['headers'] = headers
        except Exception as e:
            context.errors.append(f"Header analysis error: {e}")
        
        return context

class BodyExtractionStage(PipelineStage):
    """Extract body content from email."""
    
    def __init__(self, prefer_html: bool = False):
        super().__init__('body')
        self.prefer_html = prefer_html
        self.parser = MIMEParser()
        self.html_converter = HTMLToTextConverter()
    
    def process(self, context: ProcessingContext) -> ProcessingContext:
        msg = context.metadata.get('email_message')
        if not msg:
            return context
        
        try:
            bodies = self.parser.get_body_content(msg, self.prefer_html)
            
            # Convert HTML to text if needed
            if bodies['html'] and not bodies['plain']:
                bodies['plain'] = self.html_converter.convert(bodies['html'])
            
            context.metadata['bodies'] = bodies
        except Exception as e:
            context.errors.append(f"Body extraction error: {e}")
        
        return context

class AttachmentExtractionStage(PipelineStage):
    """Extract attachments from email."""
    
    def __init__(self, storage_path: Optional[str] = None):
        super().__init__('attachments')
        self.handler = AttachmentHandler(storage_path)
    
    def process(self, context: ProcessingContext) -> ProcessingContext:
        msg = context.metadata.get('email_message')
        if not msg:
            return context
        
        try:
            attachments = self.handler.extract_attachments(msg)
            context.metadata['attachments'] = attachments
        except Exception as e:
            context.errors.append(f"Attachment extraction error: {e}")
        
        return context

class NormalizationStage(PipelineStage):
    """Normalize email content."""
    
    def __init__(self):
        super().__init__('normalize')
        self.normalizer = EmailNormalizer()
    
    def process(self, context: ProcessingContext) -> ProcessingContext:
        bodies = context.metadata.get('bodies', {})
        plain = bodies.get('plain', '')
        
        if not plain:
            return context
        
        try:
            normalized = self.normalizer.normalize(plain)
            context.metadata['normalized'] = normalized
        except Exception as e:
            context.errors.append(f"Normalization error: {e}")
        
        return context

class SignatureExtractionStage(PipelineStage):
    """Extract structured signature data."""
    
    def __init__(self):
        super().__init__('signature')
        self.extractor = SignatureExtractor()
    
    def process(self, context: ProcessingContext) -> ProcessingContext:
        normalized = context.metadata.get('normalized', {})
        signature_text = normalized.get('signature')
        
        if signature_text:
            try:
                signature = self.extractor.extract(signature_text)
                context.metadata['signature_structured'] = signature
            except Exception as e:
                context.errors.append(f"Signature extraction error: {e}")
        
        return context

class FinalAssemblyStage(PipelineStage):
    """Assemble final ParsedEmail object."""
    
    def __init__(self):
        super().__init__('assemble')
    
    def process(self, context: ProcessingContext) -> ProcessingContext:
        try:
            headers = context.metadata.get('headers', {})
            bodies = context.metadata.get('bodies', {})
            normalized = context.metadata.get('normalized', {})
            attachments = context.metadata.get('attachments', [])
            signature = context.metadata.get('signature_structured')
            
            parsed = ParsedEmail(
                message_id=headers.get('message_id', ''),
                thread_id=headers.get('thread_id'),
                subject=headers.get('subject', ''),
                from_address=headers.get('from_address'),
                to_addresses=headers.get('to_addresses', []),
                cc_addresses=headers.get('cc_addresses', []),
                bcc_addresses=headers.get('bcc_addresses', []),
                reply_to=headers.get('reply_to'),
                date=headers.get('date'),
                received_date=headers.get('received_date'),
                body_plain=bodies.get('plain'),
                body_html=bodies.get('html'),
                body_clean=normalized.get('clean_text', ''),
                signature=signature,
                quotes=normalized.get('quotes', []),
                main_content=normalized.get('clean_text', ''),
                attachments=attachments,
                has_attachments=len(attachments) > 0,
                headers=headers.get('headers', {}),
                priority=headers.get('priority'),
                in_reply_to=headers.get('in_reply_to'),
                references=headers.get('references', []),
            )
            
            context.parsed_email = parsed
            
        except Exception as e:
            context.errors.append(f"Assembly error: {e}")
        
        return context
```

---

## Implementation Code

### Complete Email Parser Class

```python
"""
Email Parsing and Content Extraction System
For OpenClaw Windows 10 AI Agent

Usage:
    parser = EmailParser()
    parsed = parser.parse_email(raw_email_bytes)
    print(parsed.body_clean)
    print(parsed.attachments)
"""

import email
import io
from datetime import datetime
from pathlib import Path
from typing import Union, Optional, List, Dict, BinaryIO
from email import policy
from email.parser import BytesParser
from email.message import EmailMessage

from pydantic import BaseModel, EmailStr, Field


class EmailParser:
    """
    Main email parser class that orchestrates all parsing operations.
    
    This class provides a high-level interface for parsing emails from
    various sources (bytes, files, Gmail API responses) and returns
    a structured ParsedEmail object.
    """
    
    def __init__(self, 
                 attachment_storage: Optional[str] = None,
                 extract_attachment_text: bool = True):
        """
        Initialize email parser.
        
        Args:
            attachment_storage: Path to store extracted attachments
            extract_attachment_text: Whether to extract text from attachments
        """
        self.mime_parser = MIMEParser()
        self.header_analyzer = HeaderAnalyzer()
        self.html_converter = HTMLToTextConverter()
        self.attachment_handler = AttachmentHandler(attachment_storage)
        self.encoding_handler = EncodingHandler()
        self.normalizer = EmailNormalizer()
        self.signature_extractor = SignatureExtractor()
        self.extract_attachment_text = extract_attachment_text
        
        # Build processing pipeline
        self.pipeline = self._build_pipeline()
    
    def _build_pipeline(self) -> EmailProcessingPipeline:
        """Build default processing pipeline."""
        pipeline = EmailProcessingPipeline()
        
        pipeline.add_stage(ParseStage())
        pipeline.add_stage(HeaderAnalysisStage())
        pipeline.add_stage(BodyExtractionStage(prefer_html=False))
        pipeline.add_stage(AttachmentExtractionStage())
        pipeline.add_stage(NormalizationStage())
        pipeline.add_stage(SignatureExtractionStage())
        pipeline.add_stage(FinalAssemblyStage())
        
        return pipeline
    
    def parse_email(self, email_data: Union[str, bytes]) -> ParsedEmail:
        """
        Parse email from raw data.
        
        Args:
            email_data: Raw email content as string or bytes
            
        Returns:
            ParsedEmail object with all extracted information
        """
        context = self.pipeline.process(email_data)
        
        if context.errors:
            # Log errors but still return what we have
            print(f"Parsing completed with errors: {context.errors}")
        
        if not context.parsed_email:
            raise EmailParseError("Failed to parse email: " + 
                                  "; ".join(context.errors))
        
        return context.parsed_email
    
    def parse_from_file(self, filepath: Union[str, Path]) -> ParsedEmail:
        """Parse email from file."""
        filepath = Path(filepath)
        
        with open(filepath, 'rb') as f:
            return self.parse_email(f.read())
    
    def parse_gmail_message(self, gmail_message: Dict) -> ParsedEmail:
        """
        Parse email from Gmail API message format.
        
        Args:
            gmail_message: Gmail API message resource
            
        Returns:
            ParsedEmail object
        """
        import base64
        
        # Get raw message
        if 'raw' in gmail_message:
            raw_data = base64.urlsafe_b64decode(gmail_message['raw'])
            return self.parse_email(raw_data)
        
        # Parse from payload structure
        return self._parse_gmail_payload(gmail_message)
    
    def _parse_gmail_payload(self, message: Dict) -> ParsedEmail:
        """Parse Gmail message from payload structure."""
        import base64
        
        payload = message.get('payload', {})
        headers = {h['name']: h['value'] for h in payload.get('headers', [])}
        
        # Extract body
        body_plain, body_html = self._extract_gmail_body(payload)
        
        # Process attachments
        attachments = self._extract_gmail_attachments(payload)
        
        # Normalize
        if body_html and not body_plain:
            body_plain = self.html_converter.convert(body_html)
        
        normalized = self.normalizer.normalize(body_plain or '')
        
        # Extract signature
        signature = None
        if normalized.get('signature'):
            signature = self.signature_extractor.extract(normalized['signature'])
        
        # Parse addresses
        from_addr = self._parse_address(headers.get('From', ''))
        to_addrs = self._parse_address_list(headers.get('To', ''))
        cc_addrs = self._parse_address_list(headers.get('Cc', ''))
        
        return ParsedEmail(
            message_id=headers.get('Message-ID', message.get('id', '')),
            thread_id=message.get('threadId'),
            subject=headers.get('Subject', ''),
            from_address=from_addr,
            to_addresses=to_addrs,
            cc_addresses=cc_addrs,
            date=self._parse_date(headers.get('Date')),
            body_plain=body_plain,
            body_html=body_html,
            body_clean=normalized.get('clean_text', ''),
            signature=signature,
            quotes=normalized.get('quotes', []),
            main_content=normalized.get('clean_text', ''),
            attachments=attachments,
            has_attachments=len(attachments) > 0,
            headers=headers,
            in_reply_to=headers.get('In-Reply-To', '').strip('<>') or None,
        )
    
    def _extract_gmail_body(self, payload: Dict) -> tuple:
        """Extract body from Gmail payload."""
        import base64
        
        plain = None
        html = None
        
        mime_type = payload.get('mimeType', '')
        
        if mime_type == 'text/plain':
            data = payload.get('body', {}).get('data', '')
            if data:
                plain = base64.urlsafe_b64decode(data).decode('utf-8', errors='replace')
        
        elif mime_type == 'text/html':
            data = payload.get('body', {}).get('data', '')
            if data:
                html = base64.urlsafe_b64decode(data).decode('utf-8', errors='replace')
        
        elif 'parts' in payload:
            for part in payload['parts']:
                p_type = part.get('mimeType', '')
                data = part.get('body', {}).get('data', '')
                
                if data:
                    content = base64.urlsafe_b64decode(data).decode('utf-8', errors='replace')
                    if p_type == 'text/plain' and not plain:
                        plain = content
                    elif p_type == 'text/html' and not html:
                        html = content
        
        return plain, html
    
    def _extract_gmail_attachments(self, payload: Dict) -> List[EmailAttachment]:
        """Extract attachments from Gmail payload."""
        import base64
        
        attachments = []
        
        if 'parts' not in payload:
            return attachments
        
        for part in payload['parts']:
            filename = part.get('filename', '')
            body = part.get('body', {})
            
            if filename and 'attachmentId' in body:
                attachment = EmailAttachment(
                    filename=filename,
                    content_type=part.get('mimeType', 'application/octet-stream'),
                    size_bytes=body.get('size', 0),
                    attachment_type=self._classify_attachment_type(
                        part.get('mimeType', '')
                    ),
                    content_id=part.get('headers', [{}])[0].get('value') if part.get('headers') else None,
                    is_inline='Content-ID' in [h.get('name', '') for h in part.get('headers', [])],
                )
                attachments.append(attachment)
        
        return attachments
    
    def _classify_attachment_type(self, mime_type: str) -> AttachmentType:
        """Classify attachment by MIME type."""
        return AttachmentHandler.MIME_CATEGORIES.get(
            mime_type, AttachmentType.UNKNOWN
        )
    
    def _parse_address(self, address_str: str) -> Optional[EmailAddress]:
        """Parse email address string."""
        from email.utils import parseaddr
        name, email_addr = parseaddr(address_str)
        if email_addr:
            return EmailAddress(email=email_addr.lower(), name=name or None)
        return None
    
    def _parse_address_list(self, addresses_str: str) -> List[EmailAddress]:
        """Parse list of email addresses."""
        from email.utils import getaddresses
        
        if not addresses_str:
            return []
        
        result = []
        for name, email_addr in getaddresses([addresses_str]):
            if email_addr:
                result.append(EmailAddress(
                    email=email_addr.lower(),
                    name=name or None
                ))
        return result
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date string to datetime."""
        if not date_str:
            return None
        
        from email.utils import parsedate_to_datetime
        try:
            return parsedate_to_datetime(date_str)
        except Exception:
            return None


class EmailParseError(Exception):
    """Exception raised when email parsing fails."""
    pass
```

---

## Integration Points

### Gmail API Integration

```python
class GmailEmailParser:
    """Parse emails from Gmail API."""
    
    def __init__(self, credentials: Dict):
        self.credentials = credentials
        self.parser = EmailParser()
    
    def parse_message(self, message_id: str) -> ParsedEmail:
        """Fetch and parse Gmail message."""
        # Use Gmail API to fetch message
        from googleapiclient.discovery import build
        
        service = build('gmail', 'v1', credentials=self.credentials)
        message = service.users().messages().get(
            userId='me', 
            id=message_id,
            format='raw'
        ).execute()
        
        return self.parser.parse_gmail_message(message)
    
    def parse_messages(self, query: str = '', max_results: int = 100) -> List[ParsedEmail]:
        """Parse multiple messages matching query."""
        from googleapiclient.discovery import build
        
        service = build('gmail', 'v1', credentials=self.credentials)
        
        results = service.users().messages().list(
            userId='me',
            q=query,
            maxResults=max_results
        ).execute()
        
        messages = results.get('messages', [])
        parsed_emails = []
        
        for msg_meta in messages:
            try:
                message = service.users().messages().get(
                    userId='me',
                    id=msg_meta['id'],
                    format='raw'
                ).execute()
                
                parsed = self.parser.parse_gmail_message(message)
                parsed_emails.append(parsed)
                
            except Exception as e:
                print(f"Error parsing message {msg_meta['id']}: {e}")
        
        return parsed_emails
```

### IMAP Integration

```python
import imaplib
import email

class IMAPEmailParser:
    """Parse emails from IMAP server."""
    
    def __init__(self, host: str, username: str, password: str,
                 use_ssl: bool = True):
        self.host = host
        self.username = username
        self.password = password
        self.use_ssl = use_ssl
        self.parser = EmailParser()
    
    def fetch_emails(self, folder: str = 'INBOX',
                     criteria: str = 'ALL') -> List[ParsedEmail]:
        """Fetch and parse emails from IMAP folder."""
        
        # Connect to IMAP server
        if self.use_ssl:
            mail = imaplib.IMAP4_SSL(self.host)
        else:
            mail = imaplib.IMAP4(self.host)
        
        mail.login(self.username, self.password)
        mail.select(folder)
        
        # Search for messages
        _, message_numbers = mail.search(None, criteria)
        
        parsed_emails = []
        
        for num in message_numbers[0].split():
            try:
                _, msg_data = mail.fetch(num, '(RFC822)')
                raw_email = msg_data[0][1]
                
                parsed = self.parser.parse_email(raw_email)
                parsed_emails.append(parsed)
                
            except Exception as e:
                print(f"Error parsing message {num}: {e}")
        
        mail.close()
        mail.logout()
        
        return parsed_emails
```

---

## Dependencies

```
# requirements.txt for Email Parsing System

# Core email processing
email-validator>=2.0.0

# HTML processing
beautifulsoup4>=4.12.0
lxml>=4.9.0
html2text>=2020.1.16

# Signature/quote extraction
talon>=1.4.0

# Encoding detection
chardet>=5.0.0
charset-normalizer>=3.0.0

# File type detection
python-magic>=0.4.27

# Document text extraction
pdfplumber>=0.9.0
python-docx>=0.8.11
PyPDF2>=3.0.0

# Data validation
pydantic>=2.0.0

# Outlook .msg files (optional)
extract-msg>=0.41.0

# Gmail API (optional)
google-api-python-client>=2.0.0
google-auth>=2.0.0
```

---

## Performance Considerations

| Operation | Time Complexity | Memory | Notes |
|-----------|-----------------|--------|-------|
| MIME Parsing | O(n) | O(n) | n = email size |
| HTML→Text | O(n*m) | O(n) | m = HTML complexity |
| Attachment Extraction | O(n) | O(n) | Streams large files |
| Signature Extraction | O(n) | O(1) | ML model loaded once |
| Quote Extraction | O(n) | O(1) | Pattern matching |

### Optimization Strategies

1. **Lazy Loading**: Don't extract attachment content until needed
2. **Streaming**: Process large attachments as streams
3. **Caching**: Cache parsed emails by message ID
4. **Parallel Processing**: Process multiple emails concurrently
5. **Incremental Parsing**: For email threads, parse only new messages

---

## Error Handling

```python
class EmailParsingError(Exception):
    """Base exception for email parsing errors."""
    pass

class MIMEParsingError(EmailParsingError):
    """Error parsing MIME structure."""
    pass

class EncodingError(EmailParsingError):
    """Error handling character encoding."""
    pass

class AttachmentError(EmailParsingError):
    """Error processing attachment."""
    pass

class ValidationError(EmailParsingError):
    """Error validating parsed email."""
    pass
```

---

## Testing

```python
# Example test cases

def test_parse_simple_email():
    """Test parsing a simple plain text email."""
    email_text = """From: sender@example.com
To: recipient@example.com
Subject: Test Email
Date: Mon, 01 Jan 2024 12:00:00 +0000
Message-ID: <test123@example.com>

This is a test email.
"""
    parser = EmailParser()
    parsed = parser.parse_email(email_text)
    
    assert parsed.subject == "Test Email"
    assert parsed.from_address.email == "sender@example.com"
    assert parsed.body_clean == "This is a test email."

def test_parse_html_email():
    """Test parsing HTML email."""
    email_text = """From: sender@example.com
To: recipient@example.com
Subject: HTML Test
Content-Type: text/html; charset=utf-8

<html><body><h1>Hello</h1><p>World</p></body></html>
"""
    parser = EmailParser()
    parsed = parser.parse_email(email_text)
    
    assert "Hello" in parsed.body_clean
    assert "World" in parsed.body_clean

def test_extract_signature():
    """Test signature extraction."""
    email_text = """Thanks for your help!

--
John Doe
Software Engineer
john@example.com
"""
    extractor = SignatureExtractor()
    sig = extractor.extract("--\nJohn Doe\nSoftware Engineer\njohn@example.com")
    
    assert sig.name == "John Doe"
    assert sig.title == "Software Engineer"
    assert sig.email == "john@example.com"
```

---

## Summary

This email parsing and content extraction system provides:

1. **Complete MIME Parsing**: RFC-compliant parsing with modern Python email API
2. **HTML to Text Conversion**: Clean text extraction preserving structure
3. **Attachment Handling**: Multi-format support with text extraction
4. **Header Analysis**: Comprehensive header parsing and analysis
5. **Encoding Support**: Automatic encoding detection and conversion
6. **Email Normalization**: Signature/quote extraction and content cleaning
7. **Structured Extraction**: Contact info, social links from signatures
8. **Pipeline Architecture**: Modular, extensible processing
9. **Multiple Integrations**: Gmail API, IMAP, file-based parsing
10. **Production Ready**: Error handling, validation, and performance optimization

---

*Document Version: 1.0*
*Last Updated: 2024*
*For: OpenClaw Windows 10 AI Agent System*
