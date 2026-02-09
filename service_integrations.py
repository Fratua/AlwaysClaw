"""
Service Integration Layer
Windows 10 OpenClaw-Inspired AI Agent System

This module implements integrations with external services:
- Gmail API
- Twilio Voice/SMS
- Browser Control (Playwright/Selenium)
- Windows System Integration
"""

import asyncio
import json
import base64
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class EmailResult:
    """Result of email operation"""
    success: bool
    message_id: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CallResult:
    """Result of voice call operation"""
    success: bool
    call_sid: Optional[str] = None
    status: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SMSResult:
    """Result of SMS operation"""
    success: bool
    message_sid: Optional[str] = None
    status: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BrowserResult:
    """Result of browser operation"""
    success: bool
    page: Optional[Any] = None
    title: Optional[str] = None
    content: Optional[str] = None
    screenshot: Optional[bytes] = None
    error: Optional[str] = None


@dataclass
class SystemResult:
    """Result of system operation"""
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None
    exit_code: int = 0


@dataclass
class Email:
    """Email data structure"""
    id: str
    thread_id: str
    subject: str
    sender: str
    recipients: List[str]
    body: str
    snippet: str
    labels: List[str]
    is_unread: bool
    received_at: datetime
    attachments: List[Dict] = field(default_factory=list)


@dataclass
class BrowserAction:
    """Browser action specification"""
    type: str  # click, type, screenshot, extract, navigate, scroll
    selector: Optional[str] = None
    text: Optional[str] = None
    url: Optional[str] = None
    coordinates: Optional[tuple] = None


# =============================================================================
# GMAIL INTEGRATION
# =============================================================================

class GmailIntegration:
    """
    Gmail API integration for email operations.
    Supports sending, receiving, and managing emails.
    """
    
    def __init__(self, credentials_path: Optional[str] = None):
        self.credentials_path = credentials_path
        self.service = None
        self.authenticated = False
        
    async def initialize(self) -> bool:
        """Initialize Gmail API connection"""
        try:
            # In production, this would use Google API client
            # from googleapiclient.discovery import build
            # from google.oauth2.credentials import Credentials
            
            logger.info("Initializing Gmail integration...")
            
            # Placeholder for actual implementation
            self.authenticated = True
            self.service = {"status": "connected"}
            
            logger.info("Gmail integration initialized successfully")
            return True
            
        except (OSError, ImportError, ValueError) as e:
            logger.error(f"Failed to initialize Gmail: {e}")
            return False
    
    async def send_email(
        self,
        to: Union[str, List[str]],
        subject: str,
        body: str,
        cc: Optional[Union[str, List[str]]] = None,
        bcc: Optional[Union[str, List[str]]] = None,
        attachments: Optional[List[str]] = None,
        html_body: Optional[str] = None
    ) -> EmailResult:
        """
        Send email via Gmail API.
        
        Args:
            to: Recipient email(s)
            subject: Email subject
            body: Plain text body
            cc: CC recipient(s)
            bcc: BCC recipient(s)
            attachments: List of file paths to attach
            html_body: HTML version of body
            
        Returns:
            EmailResult: Result of send operation
        """
        if not self.authenticated:
            return EmailResult(success=False, error="Not authenticated")
        
        try:
            logger.info(f"Sending email to: {to}")
            
            # Convert single recipient to list
            if isinstance(to, str):
                to = [to]
            
            # Build message
            message = {
                'to': ', '.join(to),
                'subject': subject,
                'body': body
            }
            
            if cc:
                message['cc'] = cc if isinstance(cc, str) else ', '.join(cc)
            if bcc:
                message['bcc'] = bcc if isinstance(bcc, str) else ', '.join(bcc)
            
            # In production:
            # result = self.service.users().messages().send(
            #     userId='me',
            #     body={'raw': base64.urlsafe_b64encode(message.as_bytes()).decode()}
            # ).execute()
            
            # Placeholder
            result = {'id': f'msg_{datetime.now().timestamp()}'}
            
            logger.info(f"Email sent successfully: {result['id']}")
            
            return EmailResult(
                success=True,
                message_id=result['id'],
                timestamp=datetime.now()
            )
            
        except (OSError, ConnectionError, ValueError) as e:
            logger.error(f"Failed to send email: {e}")
            return EmailResult(success=False, error=str(e))
    
    async def check_new_emails(
        self,
        query: str = "is:unread",
        max_results: int = 10
    ) -> List[Email]:
        """
        Check for new emails matching query.
        
        Args:
            query: Gmail search query
            max_results: Maximum number of results
            
        Returns:
            List[Email]: List of matching emails
        """
        if not self.authenticated:
            logger.error("Not authenticated")
            return []
        
        try:
            logger.info(f"Checking emails with query: {query}")
            
            # In production:
            # results = self.service.users().messages().list(
            #     userId='me',
            #     q=query,
            #     maxResults=max_results
            # ).execute()
            
            # Placeholder implementation
            emails = []
            
            # Simulated email
            emails.append(Email(
                id="msg_123",
                thread_id="thread_123",
                subject="Test Email",
                sender="test@example.com",
                recipients=["user@example.com"],
                body="This is a test email body.",
                snippet="This is a test...",
                labels=["INBOX", "UNREAD"],
                is_unread=True,
                received_at=datetime.now()
            ))
            
            return emails
            
        except (OSError, ConnectionError, ValueError) as e:
            logger.error(f"Failed to check emails: {e}")
            return []
    
    async def get_email(self, message_id: str) -> Optional[Email]:
        """Get full email by ID"""
        if not self.authenticated:
            return None
        
        try:
            logger.info(f"Getting email: {message_id}")
            
            # In production:
            # message = self.service.users().messages().get(
            #     userId='me',
            #     id=message_id
            # ).execute()
            
            # Placeholder
            return Email(
                id=message_id,
                thread_id="thread_123",
                subject="Retrieved Email",
                sender="sender@example.com",
                recipients=["user@example.com"],
                body="Email body content",
                snippet="Snippet...",
                labels=["INBOX"],
                is_unread=False,
                received_at=datetime.now()
            )
            
        except (OSError, ConnectionError, ValueError) as e:
            logger.error(f"Failed to get email: {e}")
            return None
    
    async def mark_as_read(self, message_id: str) -> bool:
        """Mark email as read"""
        if not self.authenticated:
            return False
        
        try:
            logger.info(f"Marking email as read: {message_id}")
            
            # In production:
            # self.service.users().messages().modify(
            #     userId='me',
            #     id=message_id,
            #     body={'removeLabelIds': ['UNREAD']}
            # ).execute()
            
            return True
            
        except (OSError, ConnectionError, ValueError) as e:
            logger.error(f"Failed to mark as read: {e}")
            return False
    
    async def delete_email(self, message_id: str) -> bool:
        """Delete email"""
        if not self.authenticated:
            return False
        
        try:
            logger.info(f"Deleting email: {message_id}")
            
            # In production:
            # self.service.users().messages().trash(
            #     userId='me',
            #     id=message_id
            # ).execute()
            
            return True
            
        except (OSError, ConnectionError, ValueError) as e:
            logger.error(f"Failed to delete email: {e}")
            return False


# =============================================================================
# TWILIO INTEGRATION
# =============================================================================

class TwilioIntegration:
    """
    Twilio integration for voice calls and SMS.
    Supports making calls, sending texts, and receiving communications.
    """
    
    def __init__(
        self,
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None,
        phone_number: Optional[str] = None
    ):
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.phone_number = phone_number
        self.client = None
        self.authenticated = False
        
    async def initialize(self) -> bool:
        """Initialize Twilio client"""
        try:
            # In production:
            # from twilio.rest import Client
            # self.client = Client(self.account_sid, self.auth_token)
            
            logger.info("Initializing Twilio integration...")
            
            # Placeholder
            self.authenticated = True
            self.client = {"status": "connected"}
            
            logger.info("Twilio integration initialized successfully")
            return True
            
        except (OSError, ImportError, ValueError) as e:
            logger.error(f"Failed to initialize Twilio: {e}")
            return False
    
    async def make_call(
        self,
        to_number: str,
        message: str,
        voice: str = 'Polly.Joanna',
        language: str = 'en-US',
        record: bool = False
    ) -> CallResult:
        """
        Make voice call via Twilio.
        
        Args:
            to_number: Phone number to call
            message: Message to speak
            voice: Voice to use
            language: Language code
            record: Whether to record the call
            
        Returns:
            CallResult: Result of call operation
        """
        if not self.authenticated:
            return CallResult(success=False, error="Not authenticated")
        
        try:
            logger.info(f"Making call to: {to_number}")
            
            # Build TwiML
            twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="{voice}" language="{language}">{message}</Say>
</Response>"""
            
            # In production:
            # call = self.client.calls.create(
            #     twiml=twiml,
            #     to=to_number,
            #     from_=self.phone_number,
            #     record=record
            # )
            
            # Placeholder
            call = type('obj', (object,), {
                'sid': f'CA{datetime.now().timestamp()}',
                'status': 'queued'
            })()
            
            logger.info(f"Call initiated: {call.sid}")
            
            return CallResult(
                success=True,
                call_sid=call.sid,
                status=call.status
            )
            
        except (ConnectionError, TimeoutError, ValueError) as e:
            logger.error(f"Failed to make call: {e}")
            return CallResult(success=False, error=str(e))
    
    async def send_sms(
        self,
        to_number: str,
        message: str,
        media_urls: Optional[List[str]] = None
    ) -> SMSResult:
        """
        Send SMS via Twilio.
        
        Args:
            to_number: Phone number to send to
            message: Message text
            media_urls: Optional list of media URLs to attach
            
        Returns:
            SMSResult: Result of SMS operation
        """
        if not self.authenticated:
            return SMSResult(success=False, error="Not authenticated")
        
        try:
            logger.info(f"Sending SMS to: {to_number}")
            
            # In production:
            # sms = self.client.messages.create(
            #     body=message,
            #     to=to_number,
            #     from_=self.phone_number,
            #     media_url=media_urls
            # )
            
            # Placeholder
            sms = type('obj', (object,), {
                'sid': f'SM{datetime.now().timestamp()}',
                'status': 'queued'
            })()
            
            logger.info(f"SMS sent: {sms.sid}")
            
            return SMSResult(
                success=True,
                message_sid=sms.sid,
                status=sms.status
            )
            
        except (ConnectionError, TimeoutError, ValueError) as e:
            logger.error(f"Failed to send SMS: {e}")
            return SMSResult(success=False, error=str(e))
    
    async def send_mms(
        self,
        to_number: str,
        message: str,
        media_path: str
    ) -> SMSResult:
        """
        Send MMS with media attachment.
        
        Args:
            to_number: Phone number to send to
            message: Message text
            media_path: Path to media file
            
        Returns:
            SMSResult: Result of MMS operation
        """
        if not self.authenticated:
            return SMSResult(success=False, error="Not authenticated")
        
        try:
            logger.info(f"Sending MMS to: {to_number}")
            
            # Upload media to accessible URL (in production)
            # media_url = await self._upload_media(media_path)
            
            # In production:
            # sms = self.client.messages.create(
            #     body=message,
            #     to=to_number,
            #     from_=self.phone_number,
            #     media_url=[media_url]
            # )
            
            # Placeholder
            sms = type('obj', (object,), {
                'sid': f'MM{datetime.now().timestamp()}',
                'status': 'queued'
            })()
            
            logger.info(f"MMS sent: {sms.sid}")
            
            return SMSResult(
                success=True,
                message_sid=sms.sid,
                status=sms.status
            )
            
        except (ConnectionError, TimeoutError, ValueError) as e:
            logger.error(f"Failed to send MMS: {e}")
            return SMSResult(success=False, error=str(e))
    
    async def get_call_status(self, call_sid: str) -> Dict:
        """Get status of a call"""
        if not self.authenticated:
            return {'error': 'Not authenticated'}
        
        try:
            # In production:
            # call = self.client.calls(call_sid).fetch()
            # return {
            #     'sid': call.sid,
            #     'status': call.status,
            #     'duration': call.duration,
            #     'price': call.price
            # }
            
            return {
                'sid': call_sid,
                'status': 'completed',
                'duration': '60',
                'price': '0.50'
            }
            
        except (ConnectionError, TimeoutError, ValueError) as e:
            logger.error(f"Failed to get call status: {e}")
            return {'error': str(e)}


# =============================================================================
# BROWSER CONTROL INTEGRATION
# =============================================================================

class BrowserControlIntegration:
    """
    Browser automation via Playwright.
    Supports navigation, interaction, and data extraction.
    """
    
    def __init__(self, headless: bool = False):
        self.headless = headless
        self.browser = None
        self.context = None
        self.pages: Dict[str, Any] = {}
        self.active_page_id: Optional[str] = None
        
    async def initialize(self) -> bool:
        """Initialize browser"""
        try:
            # In production:
            # from playwright.async_api import async_playwright
            # self.playwright = await async_playwright().start()
            # self.browser = await self.playwright.chromium.launch(headless=self.headless)
            # self.context = await self.browser.new_context()
            
            logger.info("Initializing browser control...")
            
            # Placeholder
            self.browser = {"status": "connected"}
            
            logger.info("Browser control initialized successfully")
            return True
            
        except (OSError, ImportError, RuntimeError) as e:
            logger.error(f"Failed to initialize browser: {e}")
            return False
    
    async def navigate(
        self,
        url: str,
        wait_until: str = 'networkidle',
        timeout: int = 30000
    ) -> BrowserResult:
        """
        Navigate to URL.
        
        Args:
            url: URL to navigate to
            wait_until: When to consider navigation complete
            timeout: Navigation timeout in ms
            
        Returns:
            BrowserResult: Result of navigation
        """
        if not self.browser:
            return BrowserResult(success=False, error="Browser not initialized")
        
        try:
            logger.info(f"Navigating to: {url}")
            
            # In production:
            # page = await self.context.new_page()
            # await page.goto(url, wait_until=wait_until, timeout=timeout)
            # page_id = f"page_{len(self.pages)}"
            # self.pages[page_id] = page
            # self.active_page_id = page_id
            
            # Placeholder
            page_id = f"page_{len(self.pages)}"
            self.pages[page_id] = {"url": url, "title": "Loaded Page"}
            self.active_page_id = page_id
            
            return BrowserResult(
                success=True,
                page=page_id,
                title="Loaded Page"
            )
            
        except (OSError, RuntimeError, TimeoutError, ValueError) as e:
            logger.error(f"Failed to navigate: {e}")
            return BrowserResult(success=False, error=str(e))
    
    async def execute_action(self, action: BrowserAction) -> BrowserResult:
        """
        Execute browser action.
        
        Args:
            action: BrowserAction to execute
            
        Returns:
            BrowserResult: Result of action
        """
        if not self.browser:
            return BrowserResult(success=False, error="Browser not initialized")
        
        page = self.get_active_page()
        if not page:
            return BrowserResult(success=False, error="No active page")
        
        try:
            logger.info(f"Executing action: {action.type}")
            
            # In production:
            # if action.type == 'click':
            #     await page.click(action.selector)
            # elif action.type == 'type':
            #     await page.fill(action.selector, action.text)
            # elif action.type == 'screenshot':
            #     screenshot = await page.screenshot()
            #     return BrowserResult(success=True, screenshot=screenshot)
            # elif action.type == 'extract':
            #     content = await page.eval_on_selector(action.selector, 'el => el.textContent')
            #     return BrowserResult(success=True, content=content)
            # elif action.type == 'navigate':
            #     await page.goto(action.url)
            # elif action.type == 'scroll':
            #     await page.evaluate('window.scrollBy(0, 500)')
            
            # Placeholder
            return BrowserResult(success=True)
            
        except (OSError, RuntimeError, TimeoutError, ValueError) as e:
            logger.error(f"Failed to execute action: {e}")
            return BrowserResult(success=False, error=str(e))
    
    def get_active_page(self) -> Optional[Any]:
        """Get currently active page"""
        if self.active_page_id and self.active_page_id in self.pages:
            return self.pages[self.active_page_id]
        return None
    
    async def take_screenshot(
        self,
        full_page: bool = False,
        selector: Optional[str] = None
    ) -> Optional[bytes]:
        """
        Take screenshot of current page.
        
        Args:
            full_page: Capture full page or viewport
            selector: Optional element to capture
            
        Returns:
            Screenshot bytes or None
        """
        if not self.browser:
            return None
        
        try:
            logger.info("Taking screenshot...")
            
            # In production:
            # page = self.get_active_page()
            # if selector:
            #     element = await page.query_selector(selector)
            #     screenshot = await element.screenshot()
            # else:
            #     screenshot = await page.screenshot(full_page=full_page)
            # return screenshot
            
            # Placeholder - return empty bytes
            return b''
            
        except (OSError, RuntimeError) as e:
            logger.error(f"Failed to take screenshot: {e}")
            return None

    async def extract_data(
        self,
        selector: str,
        attribute: Optional[str] = None
    ) -> List[str]:
        """
        Extract data from page elements.
        
        Args:
            selector: CSS selector for elements
            attribute: Optional attribute to extract
            
        Returns:
            List of extracted values
        """
        if not self.browser:
            return []
        
        try:
            logger.info(f"Extracting data: {selector}")
            
            # In production:
            # page = self.get_active_page()
            # elements = await page.query_selector_all(selector)
            # results = []
            # for element in elements:
            #     if attribute:
            #         value = await element.get_attribute(attribute)
            #     else:
            #         value = await element.text_content()
            #     results.append(value)
            # return results
            
            # Placeholder
            return ["extracted_data_1", "extracted_data_2"]
            
        except (OSError, RuntimeError, ValueError) as e:
            logger.error(f"Failed to extract data: {e}")
            return []
    
    async def close(self) -> None:
        """Close browser"""
        try:
            # In production:
            # await self.context.close()
            # await self.browser.close()
            # await self.playwright.stop()
            
            logger.info("Browser closed")
            
        except (OSError, RuntimeError) as e:
            logger.error(f"Error closing browser: {e}")


# =============================================================================
# WINDOWS SYSTEM INTEGRATION
# =============================================================================

class WindowsSystemIntegration:
    """
    Windows system integration for system-level operations.
    Supports process management, file operations, and system queries.
    """
    
    def __init__(self):
        self.initialized = False
        
    async def initialize(self) -> bool:
        """Initialize system integration"""
        try:
            logger.info("Initializing Windows system integration...")
            
            # In production, would use:
            # - ctypes for Windows API calls
            # - psutil for process management
            # - WMI for system queries
            # - pywin32 for COM integration
            
            self.initialized = True
            logger.info("Windows system integration initialized")
            return True
            
        except (OSError, ImportError, RuntimeError) as e:
            logger.error(f"Failed to initialize system integration: {e}")
            return False
    
    async def execute_command(
        self,
        command: str,
        shell: bool = True,
        timeout: int = 30
    ) -> SystemResult:
        """
        Execute system command.
        
        Args:
            command: Command to execute
            shell: Use shell execution
            timeout: Timeout in seconds
            
        Returns:
            SystemResult: Result of command execution
        """
        try:
            logger.info(f"Executing command: {command}")
            
            # In production:
            # import subprocess
            # result = subprocess.run(
            #     command,
            #     shell=shell,
            #     capture_output=True,
            #     text=True,
            #     timeout=timeout
            # )
            
            # Placeholder
            return SystemResult(
                success=True,
                output="Command executed successfully",
                exit_code=0
            )
            
        except (OSError, RuntimeError, TimeoutError, PermissionError) as e:
            logger.error(f"Command execution failed: {e}")
            return SystemResult(
                success=False,
                error=str(e),
                exit_code=1
            )
    
    async def launch_application(
        self,
        app_path: str,
        arguments: Optional[List[str]] = None
    ) -> SystemResult:
        """
        Launch application.
        
        Args:
            app_path: Path to application
            arguments: Command line arguments
            
        Returns:
            SystemResult: Result of launch
        """
        try:
            logger.info(f"Launching application: {app_path}")
            
            # In production:
            # import subprocess
            # subprocess.Popen([app_path] + (arguments or []))
            
            return SystemResult(
                success=True,
                output=f"Launched: {app_path}"
            )
            
        except (OSError, PermissionError, FileNotFoundError) as e:
            logger.error(f"Failed to launch application: {e}")
            return SystemResult(success=False, error=str(e))
    
    async def get_system_info(self) -> Dict:
        """Get system information"""
        try:
            # In production:
            # import psutil
            # import platform
            
            info = {
                'platform': 'Windows-10',
                'processor': 'Unknown',
                'memory_total': '16 GB',
                'memory_available': '8 GB',
                'disk_usage': '50%',
                'uptime': '24 hours'
            }
            
            return info
            
        except (OSError, ImportError, RuntimeError) as e:
            logger.error(f"Failed to get system info: {e}")
            return {'error': str(e)}
    
    async def get_running_processes(self) -> List[Dict]:
        """Get list of running processes"""
        try:
            # In production:
            # import psutil
            # processes = []
            # for proc in psutil.process_iter(['pid', 'name', 'status']):
            #     processes.append(proc.info)
            # return processes
            
            # Placeholder
            return [
                {'pid': 1234, 'name': 'chrome.exe', 'status': 'running'},
                {'pid': 5678, 'name': 'notepad.exe', 'status': 'running'}
            ]
            
        except (OSError, ImportError, RuntimeError) as e:
            logger.error(f"Failed to get processes: {e}")
            return []
    
    async def kill_process(self, pid: int) -> bool:
        """Kill process by PID"""
        try:
            logger.info(f"Killing process: {pid}")
            
            # In production:
            # import psutil
            # process = psutil.Process(pid)
            # process.terminate()
            
            return True
            
        except (OSError, PermissionError, ProcessLookupError) as e:
            logger.error(f"Failed to kill process: {e}")
            return False
    
    async def take_screenshot(self) -> Optional[bytes]:
        """Take system screenshot"""
        try:
            logger.info("Taking system screenshot...")
            
            # In production:
            # from PIL import ImageGrab
            # screenshot = ImageGrab.grab()
            # buffer = io.BytesIO()
            # screenshot.save(buffer, format='PNG')
            # return buffer.getvalue()
            
            # Placeholder
            return b''
            
        except (OSError, ImportError, RuntimeError) as e:
            logger.error(f"Failed to take screenshot: {e}")
            return None

    async def set_volume(self, level: float) -> bool:
        """Set system volume (0.0 - 1.0)"""
        try:
            logger.info(f"Setting volume to: {level}")
            
            # In production:
            # from ctypes import cast, POINTER
            # from comtypes import CLSCTX_ALL
            # from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
            # devices = AudioUtilities.GetSpeakers()
            # interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            # volume = cast(interface, POINTER(IAudioEndpointVolume))
            # volume.SetMasterVolumeLevelScalar(level, None)
            
            return True
            
        except (OSError, ImportError, RuntimeError) as e:
            logger.error(f"Failed to set volume: {e}")
            return False
    
    async def get_active_window(self) -> Dict:
        """Get information about active window"""
        try:
            # In production:
            # import win32gui
            # hwnd = win32gui.GetForegroundWindow()
            # title = win32gui.GetWindowText(hwnd)
            # return {'hwnd': hwnd, 'title': title}
            
            return {'hwnd': 12345, 'title': 'Active Window'}
            
        except (OSError, ImportError, RuntimeError) as e:
            logger.error(f"Failed to get active window: {e}")
            return {}


# =============================================================================
# SERVICE INTEGRATION MANAGER
# =============================================================================

class ServiceIntegrationManager:
    """
    Manages all service integrations.
    Provides unified interface for external service operations.
    """
    
    def __init__(self):
        self.gmail = GmailIntegration()
        self.twilio = TwilioIntegration()
        self.browser = BrowserControlIntegration()
        self.system = WindowsSystemIntegration()
        
        self.services = {
            'gmail': self.gmail,
            'twilio': self.twilio,
            'browser': self.browser,
            'system': self.system
        }
        
    async def initialize_all(self) -> Dict[str, bool]:
        """Initialize all service integrations"""
        results = {}
        
        for name, service in self.services.items():
            try:
                results[name] = await service.initialize()
            except (OSError, ImportError, RuntimeError, ValueError) as e:
                logger.error(f"Failed to initialize {name}: {e}")
                results[name] = False
        
        return results
    
    async def execute(
        self,
        service: str,
        action: str,
        **kwargs
    ) -> Any:
        """
        Execute action on specified service.
        
        Args:
            service: Service name
            action: Action to execute
            **kwargs: Action parameters
            
        Returns:
            Action result
        """
        svc = self.services.get(service)
        if not svc:
            raise ValueError(f"Unknown service: {service}")
        
        method = getattr(svc, action, None)
        if not method:
            raise ValueError(f"Unknown action: {action} for service {service}")
        
        return await method(**kwargs)
    
    def get_service_status(self) -> Dict[str, Dict]:
        """Get status of all services"""
        status = {}
        
        for name, service in self.services.items():
            status[name] = {
                'initialized': getattr(service, 'authenticated', False) or 
                              getattr(service, 'initialized', False) or
                              getattr(service, 'browser', None) is not None,
                'type': type(service).__name__
            }
        
        return status
