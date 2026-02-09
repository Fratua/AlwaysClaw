"""
Service Integration Layer - Python-native API for direct Python callers.

This module provides async Python classes for integrating with external services:
- Gmail API (via gmail_client_implementation)
- Twilio Voice/SMS (via twilio_voice_integration)
- Browser Control (via Playwright)
- Windows System Integration

For the Node.js-to-Python JSON-RPC interface, see python_bridge.py instead.
"""

import asyncio
import json
import base64
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ServiceNotInitializedError(RuntimeError):
    """Raised when a service method is called but the service failed to initialize."""

    def __init__(self, service_name: str, original_error: Optional[str] = None):
        msg = f"Service '{service_name}' is not initialized"
        if original_error:
            msg += f" (init error: {original_error})"
        super().__init__(msg)
        self.service_name = service_name
        self.original_error = original_error


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
            logger.info("Initializing Gmail integration...")

            from gmail_client_implementation import GmailClient
            self.service = GmailClient()
            self.authenticated = True
            
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
            
            result = self.service.messages.send_message(
                to=message['to'],
                subject=message['subject'],
                body=message['body'],
            )
            
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
            
            raw_msgs = self.service.messages.list_messages(
                query=query, max_results=max_results
            )
            emails = []
            for msg in (raw_msgs or []):
                emails.append(Email(
                    id=msg.get('id', ''),
                    thread_id=msg.get('threadId', ''),
                    subject=msg.get('subject', ''),
                    sender=msg.get('from', ''),
                    recipients=msg.get('to', '').split(',') if msg.get('to') else [],
                    body=msg.get('body', ''),
                    snippet=msg.get('snippet', ''),
                    labels=msg.get('labelIds', []),
                    is_unread='UNREAD' in msg.get('labelIds', []),
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
            
            msg = self.service.messages.get_message(message_id)
            if not msg:
                return None
            return Email(
                id=msg.get('id', message_id),
                thread_id=msg.get('threadId', ''),
                subject=msg.get('subject', ''),
                sender=msg.get('from', ''),
                recipients=msg.get('to', '').split(',') if msg.get('to') else [],
                body=msg.get('body', ''),
                snippet=msg.get('snippet', ''),
                labels=msg.get('labelIds', []),
                is_unread='UNREAD' in msg.get('labelIds', []),
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
            
            self.service.messages.modify_labels(
                message_id=message_id,
                remove_label_ids=['UNREAD'],
            )
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
            
            self.service.messages.trash_message(message_id)
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
            logger.info("Initializing Twilio integration...")
            from twilio_voice_integration import TwilioVoiceManager, TwilioConfig
            twilio_config = TwilioConfig(
                account_sid=self.account_sid or '',
                auth_token=self.auth_token or '',
                phone_number=self.phone_number or '',
            )
            self.client = TwilioVoiceManager(twilio_config)
            self.authenticated = True
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

            call = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.client.make_call(to_number, message)
            )
            call_sid = getattr(call, 'sid', f'CA{datetime.now().timestamp()}')
            call_status = getattr(call, 'status', 'queued')

            logger.info(f"Call initiated: {call_sid}")

            return CallResult(
                success=True,
                call_sid=call_sid,
                status=call_status
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
            
            sms = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.client.send_sms(to_number, message)
            )
            sms_sid = getattr(sms, 'sid', f'SM{datetime.now().timestamp()}')
            sms_status = getattr(sms, 'status', 'queued')

            logger.info(f"SMS sent: {sms_sid}")

            return SMSResult(
                success=True,
                message_sid=sms_sid,
                status=sms_status
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
            
            mms = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.client.send_mms(to_number, message, media_path)
            )
            mms_sid = getattr(mms, 'sid', f'MM{datetime.now().timestamp()}')
            mms_status = getattr(mms, 'status', 'queued')

            logger.info(f"MMS sent: {mms_sid}")

            return SMSResult(
                success=True,
                message_sid=mms_sid,
                status=mms_status
            )
            
        except (ConnectionError, TimeoutError, ValueError) as e:
            logger.error(f"Failed to send MMS: {e}")
            return SMSResult(success=False, error=str(e))
    
    async def get_call_status(self, call_sid: str) -> Dict:
        """Get status of a call"""
        if not self.authenticated:
            return {'error': 'Not authenticated'}
        
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.client.get_call_status(call_sid)
            )
            return result if isinstance(result, dict) else {
                'sid': call_sid,
                'status': getattr(result, 'status', 'unknown'),
                'duration': getattr(result, 'duration', '0'),
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
            logger.info("Initializing browser control...")
            from playwright.async_api import async_playwright
            self._playwright = await async_playwright().start()
            self.browser = await self._playwright.chromium.launch(headless=self.headless)
            self.context = await self.browser.new_context()
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
            
            page = await self.context.new_page()
            await page.goto(url, wait_until=wait_until, timeout=timeout)
            page_id = f"page_{len(self.pages)}"
            self.pages[page_id] = page
            self.active_page_id = page_id
            title = await page.title()

            return BrowserResult(
                success=True,
                page=page_id,
                title=title
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
            
            if action.type == 'click':
                await page.click(action.selector)
            elif action.type == 'type':
                await page.fill(action.selector, action.text or '')
            elif action.type == 'screenshot':
                screenshot = await page.screenshot()
                return BrowserResult(success=True, screenshot=screenshot)
            elif action.type == 'extract':
                content = await page.eval_on_selector(action.selector, 'el => el.textContent')
                return BrowserResult(success=True, content=content)
            elif action.type == 'navigate':
                await page.goto(action.url)
            elif action.type == 'scroll':
                await page.evaluate('window.scrollBy(0, 500)')

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
            
            page = self.get_active_page()
            if not page:
                return None
            if selector:
                element = await page.query_selector(selector)
                if element:
                    screenshot = await element.screenshot()
                else:
                    return None
            else:
                screenshot = await page.screenshot(full_page=full_page)
            return screenshot
            
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
            
            page = self.get_active_page()
            if not page:
                return []
            elements = await page.query_selector_all(selector)
            results = []
            for element in elements:
                if attribute:
                    value = await element.get_attribute(attribute)
                else:
                    value = await element.text_content()
                if value is not None:
                    results.append(value)
            return results
            
        except (OSError, RuntimeError, ValueError) as e:
            logger.error(f"Failed to extract data: {e}")
            return []
    
    async def close(self) -> None:
        """Close browser"""
        try:
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if hasattr(self, '_playwright') and self._playwright:
                await self._playwright.stop()
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
            import psutil
            import platform
            self._psutil = psutil
            self._platform = platform
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
            import subprocess
            logger.info(f"Executing command: {command}")
            result = subprocess.run(
                command,
                shell=shell,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return SystemResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr if result.returncode != 0 else None,
                exit_code=result.returncode,
            )
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out after {timeout}s: {command}")
            return SystemResult(success=False, error="Command timed out", exit_code=1)
        except (OSError, RuntimeError, PermissionError) as e:
            logger.error(f"Command execution failed: {e}")
            return SystemResult(success=False, error=str(e), exit_code=1)

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
            import subprocess
            logger.info(f"Launching application: {app_path}")
            proc = subprocess.Popen([app_path] + (arguments or []))
            return SystemResult(
                success=True,
                output=f"Launched: {app_path} (PID {proc.pid})"
            )
        except (OSError, PermissionError, FileNotFoundError) as e:
            logger.error(f"Failed to launch application: {e}")
            return SystemResult(success=False, error=str(e))

    async def get_system_info(self) -> Dict:
        """Get system information"""
        try:
            import psutil
            import platform
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            boot = datetime.fromtimestamp(psutil.boot_time())
            uptime = datetime.now() - boot
            return {
                'platform': platform.platform(),
                'processor': platform.processor() or platform.machine(),
                'memory_total': f"{mem.total / (1024**3):.1f} GB",
                'memory_available': f"{mem.available / (1024**3):.1f} GB",
                'disk_usage': f"{disk.percent}%",
                'uptime': str(uptime).split('.')[0],
            }
        except (OSError, ImportError, RuntimeError) as e:
            logger.error(f"Failed to get system info: {e}")
            return {'error': str(e)}

    async def get_running_processes(self) -> List[Dict]:
        """Get list of running processes"""
        try:
            import psutil
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'status']):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return processes
        except (OSError, ImportError, RuntimeError) as e:
            logger.error(f"Failed to get processes: {e}")
            return []

    async def kill_process(self, pid: int) -> bool:
        """Kill process by PID"""
        try:
            import psutil
            logger.info(f"Killing process: {pid}")
            process = psutil.Process(pid)
            process.terminate()
            return True
        except (OSError, PermissionError, ProcessLookupError) as e:
            logger.error(f"Failed to kill process: {e}")
            return False
        except ImportError:
            logger.error("psutil not available for kill_process")
            return False

    async def take_screenshot(self) -> Optional[bytes]:
        """Take system screenshot"""
        try:
            import io
            logger.info("Taking system screenshot...")
            from PIL import ImageGrab
            screenshot = ImageGrab.grab()
            buffer = io.BytesIO()
            screenshot.save(buffer, format='PNG')
            return buffer.getvalue()
        except (OSError, ImportError, RuntimeError) as e:
            logger.error(f"Failed to take screenshot: {e}")
            return None

    async def set_volume(self, level: float) -> bool:
        """Set system volume (0.0 - 1.0)"""
        try:
            logger.info(f"Setting volume to: {level}")
            from ctypes import cast, POINTER
            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = cast(interface, POINTER(IAudioEndpointVolume))
            volume.SetMasterVolumeLevelScalar(level, None)
            return True
        except (OSError, ImportError, RuntimeError) as e:
            logger.error(f"Failed to set volume: {e}")
            return False

    async def get_active_window(self) -> Dict:
        """Get information about active window"""
        try:
            import win32gui
            hwnd = win32gui.GetForegroundWindow()
            title = win32gui.GetWindowText(hwnd)
            return {'hwnd': hwnd, 'title': title}
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

        # Per-service initialization tracking
        self._service_status: Dict[str, dict] = {}
        for name in self.services:
            self._service_status[name] = {
                'initialized': False,
                'error': None,
                'timestamp': None
            }

    async def initialize_all(self) -> Dict[str, bool]:
        """Initialize all service integrations, tracking status per service."""
        results = {}

        for name, service in self.services.items():
            try:
                success = await service.initialize()
                results[name] = success
                self._service_status[name] = {
                    'initialized': success,
                    'error': None if success else 'initialize() returned False',
                    'timestamp': datetime.now()
                }
                if not success:
                    logger.warning(f"Service '{name}' initialize() returned False")
            except (OSError, ImportError, RuntimeError, ValueError) as e:
                logger.error(f"Failed to initialize {name}: {e}")
                results[name] = False
                self._service_status[name] = {
                    'initialized': False,
                    'error': str(e),
                    'timestamp': datetime.now()
                }

        return results

    def is_service_ready(self, service_name: str) -> bool:
        """Check whether a service is initialized and ready."""
        status = self._service_status.get(service_name)
        if not status:
            return False
        return status['initialized']

    def _require_service(self, service_name: str) -> None:
        """Raise ServiceNotInitializedError if service is not ready."""
        if not self.is_service_ready(service_name):
            status = self._service_status.get(service_name, {})
            raise ServiceNotInitializedError(
                service_name,
                original_error=status.get('error')
            )

    # --- Guarded convenience methods ---

    async def send_email(self, **kwargs) -> 'EmailResult':
        """Send email via Gmail, raising if not initialized."""
        self._require_service('gmail')
        return await self.gmail.send_email(**kwargs)

    async def check_new_emails(self, **kwargs) -> list:
        """Check new emails via Gmail, raising if not initialized."""
        self._require_service('gmail')
        return await self.gmail.check_new_emails(**kwargs)

    async def send_sms(self, **kwargs) -> 'SMSResult':
        """Send SMS via Twilio, raising if not initialized."""
        self._require_service('twilio')
        return await self.twilio.send_sms(**kwargs)

    async def make_call(self, **kwargs) -> 'CallResult':
        """Make voice call via Twilio, raising if not initialized."""
        self._require_service('twilio')
        return await self.twilio.make_call(**kwargs)

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

        self._require_service(service)

        method = getattr(svc, action, None)
        if not method:
            raise ValueError(f"Unknown action: {action} for service {service}")

        return await method(**kwargs)

    def get_service_status(self) -> Dict[str, Dict]:
        """Get status of all services"""
        status = {}

        for name, service in self.services.items():
            svc_status = self._service_status.get(name, {})
            status[name] = {
                'initialized': svc_status.get('initialized', False),
                'error': svc_status.get('error'),
                'timestamp': svc_status.get('timestamp').isoformat() if svc_status.get('timestamp') else None,
                'type': type(service).__name__
            }

        return status

    def get_health(self) -> Dict[str, Any]:
        """Get health status of all services."""
        all_ok = all(s.get('initialized', False) for s in self._service_status.values())
        return {
            'healthy': all_ok,
            'services': self.get_service_status()
        }
