"""
Multi-Factor Authentication Handling System
For Windows 10 OpenClaw AI Agent Framework
"""

import os
import re
import time
import asyncio
import logging
import pyotp
from enum import Enum, auto
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable

logger = logging.getLogger(__name__)


class MFAType(Enum):
    """Supported MFA types."""
    TOTP = auto()           # Time-based One-Time Password
    HOTP = auto()           # HMAC-based One-Time Password
    SMS = auto()            # SMS-delivered code
    EMAIL = auto()          # Email-delivered code
    PUSH = auto()           # Push notification
    WEBAUTHN = auto()       # FIDO2/WebAuthn
    SECURITY_KEY = auto()   # Hardware security key
    BACKUP_CODE = auto()    # Recovery/backup codes
    UNKNOWN = auto()        # Unknown type


@dataclass
class MFACredentials:
    """MFA credentials for a service."""
    service_name: str
    username: str
    mfa_type: MFAType = MFAType.UNKNOWN
    
    # TOTP/HOTP
    totp_secret: Optional[str] = None
    
    # SMS
    phone_number: Optional[str] = None
    
    # Email
    email_address: Optional[str] = None
    
    # Backup codes
    backup_codes: List[str] = field(default_factory=list)
    used_backup_codes: List[str] = field(default_factory=list)
    
    # WebAuthn
    webauthn_credential_id: Optional[str] = None
    
    # Additional metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None


@dataclass
class MFAResult:
    """MFA handling result."""
    success: bool
    error: Optional[str] = None
    code_used: Optional[str] = None
    mfa_type: Optional[MFAType] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OTPMessage:
    """OTP message structure."""
    code: str
    source: str  # 'sms', 'email', etc.
    received_at: datetime
    expires_at: Optional[datetime] = None


class TOTPGenerator:
    """
    TOTP code generation for MFA automation.
    """
    
    def __init__(self, default_digits: int = 6, default_interval: int = 30):
        self.default_digits = default_digits
        self.default_interval = default_interval
    
    def generate_code(
        self, 
        secret: str, 
        offset: int = 0,
        digits: Optional[int] = None,
        interval: Optional[int] = None
    ) -> str:
        """
        Generate TOTP code from secret.
        
        Args:
            secret: Base32-encoded TOTP secret
            offset: Time offset in seconds
            digits: Number of digits (default: 6)
            interval: Time window in seconds (default: 30)
            
        Returns:
            TOTP code as string
        """
        # Clean secret (remove spaces and convert to uppercase)
        secret = secret.replace(' ', '').upper()
        
        # Create TOTP object
        totp = pyotp.TOTP(
            secret,
            digits=digits or self.default_digits,
            interval=interval or self.default_interval
        )
        
        # Generate code with optional time offset
        if offset != 0:
            current_time = time.time() + offset
            return totp.at(current_time)
        
        return totp.now()
    
    def verify_code(
        self,
        secret: str,
        code: str,
        valid_window: int = 1,
        digits: Optional[int] = None,
        interval: Optional[int] = None
    ) -> bool:
        """
        Verify TOTP code.
        
        Args:
            secret: Base32-encoded TOTP secret
            code: Code to verify
            valid_window: Number of intervals before/after to check
            digits: Number of digits
            interval: Time window in seconds
            
        Returns:
            True if code is valid
        """
        secret = secret.replace(' ', '').upper()
        
        totp = pyotp.TOTP(
            secret,
            digits=digits or self.default_digits,
            interval=interval or self.default_interval
        )
        
        return totp.verify(code, valid_window=valid_window)
    
    def get_remaining_seconds(
        self, 
        secret: str,
        interval: Optional[int] = None
    ) -> int:
        """
        Get seconds remaining in current TOTP window.
        
        Args:
            secret: TOTP secret (for interval detection)
            interval: Time window in seconds
            
        Returns:
            Seconds remaining
        """
        interval = interval or self.default_interval
        return interval - (int(time.time()) % interval)
    
    def should_wait_for_next(
        self, 
        secret: str,
        threshold: int = 5,
        interval: Optional[int] = None
    ) -> bool:
        """
        Check if we should wait for next code window.
        
        Args:
            secret: TOTP secret
            threshold: Minimum seconds remaining before waiting
            interval: Time window in seconds
            
        Returns:
            True if should wait for next window
        """
        remaining = self.get_remaining_seconds(secret, interval)
        return remaining < threshold
    
    def get_provisioning_uri(
        self,
        secret: str,
        account_name: str,
        issuer_name: str,
        digits: Optional[int] = None,
        interval: Optional[int] = None
    ) -> str:
        """
        Generate provisioning URI for QR code.
        
        Args:
            secret: TOTP secret
            account_name: User account name
            issuer_name: Service/issuer name
            digits: Number of digits
            interval: Time window in seconds
            
        Returns:
            Provisioning URI
        """
        secret = secret.replace(' ', '').upper()
        
        totp = pyotp.TOTP(
            secret,
            digits=digits or self.default_digits,
            interval=interval or self.default_interval
        )
        
        return totp.provisioning_uri(
            name=account_name,
            issuer_name=issuer_name
        )


class TOTPMFAHandler:
    """
    Automated TOTP MFA handling.
    """
    
    # Common selectors for TOTP input fields
    TOTP_INPUT_SELECTORS = [
        'input[name*="totp" i]',
        'input[name*="code" i]',
        'input[name*="otp" i]',
        'input[name*="token" i]',
        'input[name*="authenticator" i]',
        'input[autocomplete="one-time-code"]',
        'input[type="number"][maxlength="6"]',
        'input[type="tel"][maxlength="6"]',
        'input[placeholder*="code" i]',
        'input[placeholder*="6" i]',
    ]
    
    # Submit button selectors
    SUBMIT_SELECTORS = [
        'button[type="submit"]',
        'input[type="submit"]',
        'button:has-text("Verify")',
        'button:has-text("Submit")',
        'button:has-text("Continue")',
        'a:has-text("Verify")',
    ]
    
    def __init__(self, totp_generator: Optional[TOTPGenerator] = None):
        self.totp = totp_generator or TOTPGenerator()
    
    async def handle(
        self,
        page,
        mfa_creds: MFACredentials,
        auto_submit: bool = True
    ) -> MFAResult:
        """
        Handle TOTP MFA challenge.
        
        Args:
            page: Playwright page object
            mfa_creds: MFA credentials
            auto_submit: Whether to auto-submit the form
            
        Returns:
            MFAResult with success status
        """
        if not mfa_creds.totp_secret:
            return MFAResult(
                success=False,
                error="TOTP secret not configured"
            )
        
        # Wait for TOTP input field
        code_input = await self._find_code_input(page)
        if not code_input:
            return MFAResult(
                success=False,
                error="TOTP input field not found"
            )
        
        # Check if we should wait for next window
        if self.totp.should_wait_for_next(mfa_creds.totp_secret):
            wait_time = self.totp.get_remaining_seconds(mfa_creds.totp_secret)
            logger.info(f"Waiting {wait_time}s for next TOTP window")
            await asyncio.sleep(wait_time + 1)
        
        # Generate TOTP code
        code = self.totp.generate_code(mfa_creds.totp_secret)
        logger.debug(f"Generated TOTP code for {mfa_creds.service_name}")
        
        # Fill code
        await code_input.fill(code)
        
        # Submit if requested
        if auto_submit:
            await self._submit_code(page)
        
        # Wait for verification
        success = await self._wait_for_verification(page)
        
        if success:
            return MFAResult(
                success=True,
                code_used=code,
                mfa_type=MFAType.TOTP
            )
        else:
            return MFAResult(
                success=False,
                error="TOTP verification failed",
                code_used=code,
                mfa_type=MFAType.TOTP
            )
    
    async def _find_code_input(self, page) -> Optional[Any]:
        """Find TOTP code input field."""
        for selector in self.TOTP_INPUT_SELECTORS:
            try:
                element = await page.wait_for_selector(
                    selector,
                    timeout=2000,
                    state='visible'
                )
                if element:
                    return element
            except (OSError, ValueError, TimeoutError) as e:
                logger.warning(f"MFA browser op failed: {e}")
                continue
        return None

    async def _submit_code(self, page) -> bool:
        """Submit the MFA code."""
        for selector in self.SUBMIT_SELECTORS:
            try:
                button = await page.query_selector(selector)
                if button and await button.is_visible():
                    await button.click()
                    return True
            except (OSError, ValueError, TimeoutError) as e:
                logger.warning(f"MFA browser op failed: {e}")
                continue

        # Try form submission
        try:
            await page.keyboard.press('Enter')
            return True
        except (OSError, ValueError, TimeoutError) as e:
            logger.warning(f"MFA browser op failed: {e}")
        
        return False
    
    async def _wait_for_verification(self, page, timeout: int = 30) -> bool:
        """Wait for MFA verification to complete."""
        # Success indicators
        success_selectors = [
            'text=/success/i',
            'text=/verified/i',
            'text=/welcome/i',
            '.dashboard',
            '[data-testid="dashboard"]',
        ]
        
        # Error indicators
        error_selectors = [
            'text=/invalid/i',
            'text=/incorrect/i',
            'text=/failed/i',
            '.error',
            '.alert-error',
            '[role="alert"]',
        ]
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check for success
            for selector in success_selectors:
                try:
                    element = await page.query_selector(selector)
                    if element and await element.is_visible():
                        return True
                except (OSError, ValueError, TimeoutError) as e:
                    logger.warning(f"MFA browser op failed: {e}")

            # Check for error
            for selector in error_selectors:
                try:
                    element = await page.query_selector(selector)
                    if element and await element.is_visible():
                        return False
                except (OSError, ValueError, TimeoutError) as e:
                    logger.warning(f"MFA browser op failed: {e}")
            
            await asyncio.sleep(0.5)
        
        # Timeout - assume success if no error
        return True


class SMSOTPHandler:
    """
    Automated SMS OTP handling.
    """
    
    # Code extraction patterns
    CODE_PATTERNS = [
        r'\b(\d{6})\b',  # 6-digit code
        r'\b(\d{4})\b',  # 4-digit code
        r'code[\s:is]+(\d+)',  # "code is 123456"
        r'code[\s:is]+([a-zA-Z0-9]+)',  # alphanumeric
        r'OTP[\s:is]+(\d+)',  # "OTP is 123456"
        r'verification[\s:]+(\d+)',  # "verification: 123456"
        r'password[\s:is]+(\d+)',  # "password is 123456"
        r'pin[\s:is]+(\d+)',  # "PIN is 1234"
    ]
    
    def __init__(self, sms_provider):
        """
        Initialize SMS OTP handler.
        
        Args:
            sms_provider: SMS provider instance (e.g., Twilio client)
        """
        self.sms_provider = sms_provider
        self.message_cache: Dict[str, List[Dict]] = {}
    
    async def handle(
        self,
        page,
        mfa_creds: MFACredentials,
        timeout: int = 60,
        auto_submit: bool = True
    ) -> MFAResult:
        """
        Handle SMS OTP challenge.
        
        Args:
            page: Playwright page object
            mfa_creds: MFA credentials with phone number
            timeout: Maximum wait time in seconds
            auto_submit: Whether to auto-submit the form
            
        Returns:
            MFAResult with success status
        """
        if not mfa_creds.phone_number:
            return MFAResult(
                success=False,
                error="Phone number not configured"
            )
        
        # Record message timestamp before requesting code
        before_time = datetime.now()
        
        # Trigger SMS code (usually by clicking "Send code")
        await self._trigger_code_send(page)
        
        # Wait for SMS
        code = await self._wait_for_sms_code(
            mfa_creds.phone_number,
            before_time,
            timeout
        )
        
        if not code:
            return MFAResult(
                success=False,
                error="SMS code not received within timeout"
            )
        
        # Fill code
        code_input = await self._find_code_input(page)
        if code_input:
            await code_input.fill(code)
            
            if auto_submit:
                await self._submit_code(page)
        else:
            return MFAResult(
                success=False,
                error="Code input field not found"
            )
        
        # Verify
        success = await self._wait_for_verification(page)
        
        return MFAResult(
            success=success,
            code_used=code,
            mfa_type=MFAType.SMS
        )
    
    async def _trigger_code_send(self, page) -> bool:
        """Trigger SMS code sending."""
        send_button_selectors = [
            'button:has-text("Send code")',
            'button:has-text("Text me")',
            'button:has-text("Send SMS")',
            'a:has-text("Send code")',
            'button[name="send_code"]',
        ]
        
        for selector in send_button_selectors:
            try:
                button = await page.query_selector(selector)
                if button and await button.is_visible():
                    await button.click()
                    logger.debug("Triggered SMS code send")
                    await asyncio.sleep(2)  # Wait for SMS to be sent
                    return True
            except (OSError, ValueError, TimeoutError) as e:
                logger.warning(f"MFA browser op failed: {e}")

        return False

    async def _wait_for_sms_code(
        self,
        phone_number: str,
        after_time: datetime,
        timeout: int
    ) -> Optional[str]:
        """
        Wait for and extract SMS code.
        
        Args:
            phone_number: Phone number to check
            after_time: Only check messages after this time
            timeout: Maximum wait time
            
        Returns:
            Extracted code or None
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Fetch messages from provider
            messages = await self._fetch_messages(phone_number, after_time)
            
            for message in messages:
                # Extract code using patterns
                code = self._extract_code_from_text(message.get('body', ''))
                if code:
                    logger.debug(f"Extracted SMS code: {code}")
                    return code
            
            await asyncio.sleep(2)
        
        return None
    
    async def _fetch_messages(
        self,
        phone_number: str,
        after_time: datetime
    ) -> List[Dict]:
        """
        Fetch SMS messages from provider.

        Override this method for specific SMS provider.
        """
        try:
            from twilio.rest import Client as TwilioClient
            account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
            auth_token = os.environ.get('TWILIO_AUTH_TOKEN')
            if not account_sid or not auth_token:
                logger.warning("Twilio credentials not configured")
                return []
            client = TwilioClient(account_sid, auth_token)
            messages = client.messages.list(to=phone_number, limit=5)
            codes = []
            for msg in messages:
                match = re.search(r'\b\d{4,8}\b', msg.body)
                if match:
                    codes.append({'body': msg.body, 'code': match.group()})
            return codes
        except ImportError:
            logger.warning("twilio not installed")
            return []
        except (ConnectionError, TimeoutError, ValueError, AttributeError) as e:
            logger.warning(f"Failed to fetch SMS messages: {e}")
            return []
    
    def _extract_code_from_text(self, text: str) -> Optional[str]:
        """
        Extract OTP code from message text.
        
        Args:
            text: SMS message text
            
        Returns:
            Extracted code or None
        """
        for pattern in self.CODE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    async def _find_code_input(self, page) -> Optional[Any]:
        """Find code input field."""
        selectors = [
            'input[name*="code" i]',
            'input[name*="otp" i]',
            'input[autocomplete="one-time-code"]',
            'input[type="number"]',
            'input[type="tel"]',
        ]
        
        for selector in selectors:
            try:
                element = await page.wait_for_selector(selector, timeout=2000)
                if element:
                    return element
            except (OSError, ValueError, TimeoutError) as e:
                logger.warning(f"MFA browser op failed: {e}")

        return None

    async def _submit_code(self, page) -> bool:
        """Submit the code."""
        selectors = [
            'button[type="submit"]',
            'button:has-text("Verify")',
            'button:has-text("Submit")',
        ]

        for selector in selectors:
            try:
                button = await page.query_selector(selector)
                if button:
                    await button.click()
                    return True
            except (OSError, ValueError, TimeoutError) as e:
                logger.warning(f"MFA browser op failed: {e}")

        return False

    async def _wait_for_verification(self, page, timeout: int = 30) -> bool:
        """Wait for verification to complete."""
        await asyncio.sleep(2)
        return True


class TwilioSMSHandler(SMSOTPHandler):
    """
    SMS OTP handler using Twilio.
    """
    
    def __init__(self, twilio_client):
        """
        Initialize Twilio SMS handler.
        
        Args:
            twilio_client: Twilio client instance
        """
        super().__init__(twilio_client)
        self.twilio = twilio_client
    
    async def _fetch_messages(
        self,
        phone_number: str,
        after_time: datetime
    ) -> List[Dict]:
        """
        Fetch messages from Twilio.
        """
        try:
            # Twilio API call
            messages = self.twilio.messages.list(
                to=phone_number,
                date_sent_after=after_time
            )
            
            return [
                {
                    'sid': msg.sid,
                    'body': msg.body,
                    'from': msg.from_,
                    'to': msg.to,
                    'date_sent': msg.date_sent
                }
                for msg in messages
            ]
        except (ConnectionError, TimeoutError, ValueError, AttributeError) as e:
            logger.error(f"Failed to fetch Twilio messages: {e}")
            return []


class BackupCodeHandler:
    """
    Backup code MFA handler.
    """
    
    def __init__(self, credential_vault):
        self.vault = credential_vault
    
    async def handle(
        self,
        page,
        mfa_creds: MFACredentials,
        auto_submit: bool = True
    ) -> MFAResult:
        """
        Handle MFA using backup code.
        
        Args:
            page: Playwright page object
            mfa_creds: MFA credentials with backup codes
            auto_submit: Whether to auto-submit
            
        Returns:
            MFAResult with success status
        """
        # Get unused backup code
        available_codes = [
            code for code in mfa_creds.backup_codes
            if code not in mfa_creds.used_backup_codes
        ]
        
        if not available_codes:
            return MFAResult(
                success=False,
                error="No unused backup codes available"
            )
        
        # Use first available code
        code = available_codes[0]
        
        # Find input field
        code_input = await self._find_code_input(page)
        if not code_input:
            return MFAResult(
                success=False,
                error="Backup code input field not found"
            )
        
        # Fill code
        await code_input.fill(code)
        
        if auto_submit:
            await self._submit_code(page)
        
        # Mark code as used
        mfa_creds.used_backup_codes.append(code)
        await self._save_used_code(mfa_creds, code)
        
        # Wait for verification
        success = await self._wait_for_verification(page)
        
        return MFAResult(
            success=success,
            code_used=code,
            mfa_type=MFAType.BACKUP_CODE
        )
    
    async def _save_used_code(
        self,
        mfa_creds: MFACredentials,
        code: str
    ) -> None:
        """Save used backup code to vault."""
        # Update credentials in vault
        await self.vault.update_mfa_credentials(mfa_creds)
    
    async def _find_code_input(self, page) -> Optional[Any]:
        """Find backup code input field."""
        selectors = [
            'input[name*="backup" i]',
            'input[name*="recovery" i]',
            'input[placeholder*="backup" i]',
            'input[placeholder*="recovery" i]',
        ]
        
        for selector in selectors:
            try:
                element = await page.wait_for_selector(selector, timeout=2000)
                if element:
                    return element
            except (OSError, ValueError, TimeoutError) as e:
                logger.warning(f"MFA browser op failed: {e}")

        return None

    async def _submit_code(self, page) -> bool:
        """Submit the code."""
        selectors = [
            'button[type="submit"]',
            'button:has-text("Verify")',
        ]

        for selector in selectors:
            try:
                button = await page.query_selector(selector)
                if button:
                    await button.click()
                    return True
            except (OSError, ValueError, TimeoutError) as e:
                logger.warning(f"MFA browser op failed: {e}")

        return False

    async def _wait_for_verification(self, page, timeout: int = 30) -> bool:
        """Wait for verification."""
        await asyncio.sleep(2)
        return True


class MFAHandler:
    """
    Main MFA handler coordinating all MFA types.
    """
    
    # MFA type detection patterns
    MFA_DETECTION_PATTERNS = {
        MFAType.TOTP: [
            'text=/authenticator/i',
            'text=/authenticator app/i',
            'text=/TOTP/i',
            'text=/time-based/i',
            'input[name*="totp" i]',
            'input[name*="authenticator" i]',
        ],
        MFAType.SMS: [
            'text=/SMS/i',
            'text=/text message/i',
            'text=/phone/i',
            'button:has-text("Text me")',
            'button:has-text("Send code")',
        ],
        MFAType.EMAIL: [
            'text=/email/i',
            'text=/e-mail/i',
            'button:has-text("Email me")',
        ],
        MFAType.PUSH: [
            'text=/push/i',
            'text=/notification/i',
            'text=/approve/i',
            'text=/mobile app/i',
        ],
        MFAType.BACKUP_CODE: [
            'text=/backup/i',
            'text=/recovery/i',
            'text=/backup code/i',
        ],
    }
    
    def __init__(
        self,
        credential_vault,
        sms_handler: Optional[SMSOTPHandler] = None,
        email_handler: Optional[Any] = None,
        push_handler: Optional[Any] = None
    ):
        self.vault = credential_vault
        self.sms = sms_handler
        self.email = email_handler
        self.push = push_handler
        
        self.totp_handler = TOTPMFAHandler()
        self.backup_handler = BackupCodeHandler(credential_vault)
    
    async def detect_mfa_type(self, page) -> MFAType:
        """
        Detect MFA type from page content.
        
        Args:
            page: Playwright page object
            
        Returns:
            Detected MFA type
        """
        for mfa_type, patterns in self.MFA_DETECTION_PATTERNS.items():
            for pattern in patterns:
                try:
                    element = await page.query_selector(pattern)
                    if element and await element.is_visible():
                        logger.debug(f"Detected MFA type: {mfa_type.name}")
                        return mfa_type
                except (OSError, ValueError, TimeoutError) as e:
                    logger.warning(f"MFA browser op failed: {e}")

        # Default to TOTP if input field found
        try:
            code_input = await page.query_selector('input[type="number"], input[type="tel"]')
            if code_input:
                return MFAType.TOTP
        except (OSError, ValueError, TimeoutError) as e:
            logger.warning(f"MFA browser op failed: {e}")
        
        return MFAType.UNKNOWN
    
    async def handle_mfa(
        self,
        page,
        service_name: str,
        username: str,
        preferred_type: Optional[MFAType] = None
    ) -> MFAResult:
        """
        Handle MFA challenge automatically.
        
        Args:
            page: Playwright page object
            service_name: Service identifier
            username: Username
            preferred_type: Preferred MFA type to use
            
        Returns:
            MFAResult with success status
        """
        # Detect MFA type if not specified
        if preferred_type is None or preferred_type == MFAType.UNKNOWN:
            detected_type = await self.detect_mfa_type(page)
            if detected_type != MFAType.UNKNOWN:
                preferred_type = detected_type
        
        # Get MFA credentials
        mfa_creds = await self.vault.get_mfa_credentials(service_name, username)
        
        if not mfa_creds:
            return MFAResult(
                success=False,
                error="MFA credentials not found"
            )
        
        # Determine which handler to use
        handler_map = {
            MFAType.TOTP: self.totp_handler,
            MFAType.SMS: self.sms,
            MFAType.EMAIL: self.email,
            MFAType.BACKUP_CODE: self.backup_handler,
        }
        
        handler = handler_map.get(preferred_type or mfa_creds.mfa_type)
        
        if not handler:
            return MFAResult(
                success=False,
                error=f"No handler available for MFA type: {preferred_type or mfa_creds.mfa_type}"
            )
        
        # Execute handler
        return await handler.handle(page, mfa_creds)
    
    async def generate_totp_code(
        self,
        service_name: str,
        username: str
    ) -> Optional[str]:
        """
        Generate TOTP code for service.
        
        Args:
            service_name: Service identifier
            username: Username
            
        Returns:
            TOTP code or None
        """
        mfa_creds = await self.vault.get_mfa_credentials(service_name, username)
        
        if not mfa_creds or not mfa_creds.totp_secret:
            return None
        
        return self.totp_handler.totp.generate_code(mfa_creds.totp_secret)


# Convenience functions
def create_totp_generator() -> TOTPGenerator:
    """Create TOTP generator."""
    return TOTPGenerator()


def generate_totp_secret() -> str:
    """Generate new TOTP secret."""
    return pyotp.random_base32()


def verify_totp_code(secret: str, code: str, valid_window: int = 1) -> bool:
    """Verify TOTP code."""
    totp = TOTPGenerator()
    return totp.verify_code(secret, code, valid_window)
