"""
Email Parsing and Content Extraction System
For OpenClaw Windows 10 AI Agent

A comprehensive email parsing library that handles:
- MIME message parsing
- HTML to text conversion
- Attachment detection and extraction
- Email header analysis
- Character encoding handling
- Email normalization and cleaning
- Structured data extraction (signatures, quotes)

Usage:
    from email_parser import EmailParser, ParsedEmail
    
    parser = EmailParser()
    parsed = parser.parse_email(raw_email_bytes)
    print(parsed.body_clean)
    print(parsed.attachments)
"""

import email
import io
import re
import base64
import quopri
import mimetypes
from datetime import datetime
from pathlib import Path
from enum import Enum
from typing import Union, Optional, List, Dict, Any, Tuple, BinaryIO
from dataclasses import dataclass, field
from email import policy
from email.parser import BytesParser
from email.message import EmailMessage

# Optional imports with fallbacks
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

try:
    import html2text
    HAS_HTML2TEXT = True
except ImportError:
    HAS_HTML2TEXT = False

try:
    import talon
    from talon import quotations
    from talon.signature import extract as extract_signature
    HAS_TALON = True
except ImportError:
    HAS_TALON = False

try:
    import chardet
    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False

try:
    import magic
    HAS_MAGIC = True
except ImportError:
    HAS_MAGIC = False


# ============================================================================
# ENUMS AND MODELS
# ============================================================================

class AttachmentType(str, Enum):
    """Classification of attachment types."""
    DOCUMENT = "document"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    ARCHIVE = "archive"
    CODE = "code"
    UNKNOWN = "unknown"


@dataclass
class EmailAddress:
    """Represents an email address with optional display name."""
    email: str
    name: Optional[str] = None
    
    def __str__(self) -> str:
        if self.name:
            return f"{self.name} <{self.email}>"
        return self.email


@dataclass
class EmailAttachment:
    """Represents an email attachment."""
    filename: str
    content_type: str
    size_bytes: int
    attachment_type: AttachmentType = AttachmentType.UNKNOWN
    content_id: Optional[str] = None
    is_inline: bool = False
    content: Optional[bytes] = None
    extracted_text: Optional[str] = None


@dataclass
class EmailSignature:
    """Structured signature data extracted from email."""
    raw_text: str
    name: Optional[str] = None
    title: Optional[str] = None
    company: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    social_links: Dict[str, str] = field(default_factory=dict)


@dataclass
class EmailQuote:
    """Represents a quoted section in an email."""
    quoted_text: str
    author: Optional[str] = None
    date: Optional[datetime] = None
    is_forwarded: bool = False


@dataclass
class ParsedEmail:
    """Complete parsed email with all extracted information."""
    # Identification
    message_id: str = ""
    thread_id: Optional[str] = None
    
    # Headers
    subject: str = ""
    from_address: Optional[EmailAddress] = None
    to_addresses: List[EmailAddress] = field(default_factory=list)
    cc_addresses: List[EmailAddress] = field(default_factory=list)
    bcc_addresses: List[EmailAddress] = field(default_factory=list)
    reply_to: Optional[EmailAddress] = None
    
    # Timing
    date: Optional[datetime] = None
    received_date: Optional[datetime] = None
    
    # Content
    body_plain: Optional[str] = None
    body_html: Optional[str] = None
    body_clean: str = ""
    
    # Structured Content
    signature: Optional[EmailSignature] = None
    quotes: List[EmailQuote] = field(default_factory=list)
    main_content: str = ""
    
    # Attachments
    attachments: List[EmailAttachment] = field(default_factory=list)
    has_attachments: bool = False
    
    # Metadata
    headers: Dict[str, str] = field(default_factory=dict)
    priority: Optional[str] = None
    in_reply_to: Optional[str] = None
    references: List[str] = field(default_factory=list)
    
    # Processing metadata
    encoding_detected: str = "utf-8"
    processing_timestamp: datetime = field(default_factory=datetime.utcnow)
    parsing_version: str = "1.0.0"


# ============================================================================
# EXCEPTIONS
# ============================================================================

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


# ============================================================================
# MIME PARSER
# ============================================================================

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
    
    def get_body_content(self, msg: EmailMessage, prefer_html: bool = False) -> Dict[str, Optional[str]]:
        """
        Extract body content from email.
        
        Returns dict with 'plain' and 'html' keys.
        """
        result: Dict[str, Optional[str]] = {'plain': None, 'html': None}
        
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
                        payload = part.get_payload(decode=True)
                        if payload:
                            result['plain'] = payload.decode(
                                part.get_content_charset() or 'utf-8', errors='replace'
                            )
                    except Exception:
                        pass
                        
                elif content_type == 'text/html' and not result['html']:
                    try:
                        payload = part.get_payload(decode=True)
                        if payload:
                            result['html'] = payload.decode(
                                part.get_content_charset() or 'utf-8', errors='replace'
                            )
                    except Exception:
                        pass
        
        return result


# ============================================================================
# HTML TO TEXT CONVERTER
# ============================================================================

class HTMLToTextConverter:
    """Convert HTML emails to clean plain text."""
    
    def __init__(self):
        if HAS_HTML2TEXT:
            self.converter = html2text.HTML2Text()
            self._configure_converter()
        else:
            self.converter = None
    
    def _configure_converter(self):
        """Configure html2text for email processing."""
        if not self.converter:
            return
            
        # Body width: 0 = no wrapping
        self.converter.body_width = 0
        
        # Include links in format: [text](url)
        self.converter.ignore_links = False
        
        # Don't ignore images (alt text will be shown)
        self.converter.ignore_images = False
        
        # Use inline links
        self.converter.inline_links = True
        
        # Protect links
        self.converter.protect_links = False
        
        # Include tables
        self.converter.ignore_tables = False
        
        # Pad tables with spaces
        self.converter.pad_tables = True
        
        # Skip internal links
        self.converter.skip_internal_links = True
    
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
        
        # Convert to markdown-style text or plain text
        if self.converter:
            text = self.converter.handle(html_content)
        elif HAS_BS4:
            text = self._bs4_convert(html_content)
        else:
            # Fallback: simple regex
            text = self._simple_convert(html_content)
        
        # Post-process
        if clean_whitespace:
            text = self._clean_whitespace(text)
        
        # Decode HTML entities
        import html
        text = html.unescape(text)
        
        return text.strip()
    
    def _preprocess_html(self, html_content: str) -> str:
        """Clean HTML before conversion."""
        if not HAS_BS4:
            return html_content
            
        soup = BeautifulSoup(html_content, 'html.parser')
        
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
    
    def _bs4_convert(self, html_content: str) -> str:
        """Convert using BeautifulSoup."""
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup.get_text(separator='\n', strip=True)
    
    def _simple_convert(self, html_content: str) -> str:
        """Simple regex-based conversion as fallback."""
        # Remove script and style tags
        text = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        # Replace <br>, <p> with newlines
        text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</p>', '\n\n', text, flags=re.IGNORECASE)
        # Remove all other tags
        text = re.sub(r'<[^>]+>', '', text)
        return text
    
    def _clean_whitespace(self, text: str) -> str:
        """Normalize whitespace in converted text."""
        # Replace multiple newlines with max 2
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        
        return '\n'.join(lines)


# ============================================================================
# ATTACHMENT HANDLER
# ============================================================================

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
        self.magic = None
        if HAS_MAGIC:
            try:
                self.magic = magic.Magic(mime=True)
            except Exception:
                pass
    
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
                if self.magic:
                    try:
                        content_type = self.magic.from_buffer(payload)
                    except Exception:
                        pass
            
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
            
            return attachment
            
        except Exception as e:
            # Log error but don't fail entire parsing
            print(f"Error processing attachment: {e}")
            return None
    
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
        # Remove path separators and dangerous characters
        filename = re.sub(r'[\\/:*?"<>|]', '_', filename)
        # Limit length
        if len(filename) > 255:
            name, ext = Path(filename).stem, Path(filename).suffix
            filename = name[:255 - len(ext)] + ext
        return filename


# ============================================================================
# HEADER ANALYZER
# ============================================================================

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
    
    def analyze(self, msg: EmailMessage) -> Dict[str, Any]:
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


# ============================================================================
# ENCODING HANDLER
# ============================================================================

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
        self.preferred_detector = 'chardet' if HAS_CHARDET else 'none'
    
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
        if self.preferred_detector == 'chardet' and HAS_CHARDET:
            result = chardet.detect(data)
            if result and result['confidence'] > 0.5:
                return result['encoding']
        
        return None
    
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


# ============================================================================
# EMAIL NORMALIZER
# ============================================================================

class EmailNormalizer:
    """Normalize and clean email content."""
    
    def __init__(self):
        # Initialize talon for signature/quote extraction
        self.has_talon = HAS_TALON
    
    def normalize(self, text: str, 
                  remove_signature: bool = True,
                  remove_quotes: bool = True) -> Dict[str, Any]:
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
        if remove_signature and self.has_talon:
            text, signature = self._extract_signature(text)
            if signature:
                result['signature'] = signature
        
        # Step 4: Extract quotes
        if remove_quotes and self.has_talon:
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


# ============================================================================
# SIGNATURE EXTRACTOR
# ============================================================================

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


# ============================================================================
# MAIN EMAIL PARSER
# ============================================================================

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
    
    def parse_email(self, email_data: Union[str, bytes]) -> ParsedEmail:
        """
        Parse email from raw data.
        
        Args:
            email_data: Raw email content as string or bytes
            
        Returns:
            ParsedEmail object with all extracted information
        """
        errors = []
        
        try:
            # Parse MIME structure
            msg = self.mime_parser.parse(email_data)
            
            # Analyze headers
            headers = self.header_analyzer.analyze(msg)
            
            # Extract body content
            bodies = self.mime_parser.get_body_content(msg, prefer_html=False)
            
            # Convert HTML to text if needed
            if bodies['html'] and not bodies['plain']:
                bodies['plain'] = self.html_converter.convert(bodies['html'])
            
            # Extract attachments
            attachments = self.attachment_handler.extract_attachments(msg)
            
            # Normalize content
            normalized = self.normalizer.normalize(bodies['plain'] or '')
            
            # Extract signature data
            signature = None
            if normalized.get('signature'):
                signature = self.signature_extractor.extract(normalized['signature'])
            
            # Build final object
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
            
            return parsed
            
        except Exception as e:
            errors.append(str(e))
            raise EmailParsingError(f"Failed to parse email: {'; '.join(errors)}")
    
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
        # Get raw message
        if 'raw' in gmail_message:
            raw_data = base64.urlsafe_b64decode(gmail_message['raw'])
            return self.parse_email(raw_data)
        
        # Parse from payload structure
        return self._parse_gmail_payload(gmail_message)
    
    def _parse_gmail_payload(self, message: Dict) -> ParsedEmail:
        """Parse Gmail message from payload structure."""
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
    
    def _extract_gmail_body(self, payload: Dict) -> Tuple[Optional[str], Optional[str]]:
        """Extract body from Gmail payload."""
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


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def quick_parse(email_data: Union[str, bytes]) -> ParsedEmail:
    """Quick parse function for simple use cases."""
    parser = EmailParser()
    return parser.parse_email(email_data)


def parse_file(filepath: Union[str, Path]) -> ParsedEmail:
    """Parse email from file."""
    parser = EmailParser()
    return parser.parse_from_file(filepath)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Parse a simple email
    sample_email = """From: sender@example.com
To: recipient@example.com
Subject: Test Email
Date: Mon, 01 Jan 2024 12:00:00 +0000
Message-ID: <test123@example.com>
Content-Type: text/plain; charset=utf-8

Hello,

This is a test email.

Best regards,
John Doe
Software Engineer
john@example.com
"""
    
    parser = EmailParser()
    parsed = parser.parse_email(sample_email)
    
    print("Subject:", parsed.subject)
    print("From:", parsed.from_address)
    print("Body:", parsed.body_clean)
    if parsed.signature:
        print("Signature:", parsed.signature.raw_text)
