"""
Form-Based Authentication Automation
For Windows 10 OpenClaw AI Agent Framework
"""

import re
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class LoginForm:
    """Detected login form structure."""
    form_element: Any  # Playwright ElementHandle
    username_field: Optional[Any] = None
    password_field: Optional[Any] = None
    submit_button: Optional[Any] = None
    captcha_field: Optional[Any] = None
    mfa_field: Optional[Any] = None
    remember_checkbox: Optional[Any] = None
    
    # Form metadata
    action_url: str = ""
    method: str = "POST"
    has_csrf_token: bool = False
    csrf_token_field: Optional[str] = None
    
    # Security indicators
    is_https: bool = True
    has_password_field: bool = False
    autocomplete_enabled: bool = True
    
    # Detection confidence
    confidence_score: float = 0.0


@dataclass
class AuthenticationResult:
    """Authentication attempt result."""
    success: bool
    error: Optional[str] = None
    session_cookies: List[Dict] = field(default_factory=list)
    login_time: float = 0.0
    mfa_required: bool = False
    captcha_required: bool = False
    redirect_url: Optional[str] = None
    user_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CaptchaResult:
    """CAPTCHA solving result."""
    success: bool
    solution: Optional[str] = None
    error: Optional[str] = None
    captcha_type: str = "unknown"


class FormDetector:
    """
    Intelligent login form detection.
    """
    
    # Username field selectors (in order of preference)
    USERNAME_SELECTORS = [
        'input[type="email"]',
        'input[name="email" i]',
        'input[name="username" i]',
        'input[name="user" i]',
        'input[name="login" i]',
        'input[name="userid" i]',
        'input[id="email" i]',
        'input[id="username" i]',
        'input[id="user" i]',
        'input[id="login" i]',
        'input[autocomplete="username"]',
        'input[autocomplete="email"]',
        'input[placeholder*="email" i]',
        'input[placeholder*="username" i]',
        'input[placeholder*="user" i]',
    ]
    
    # Password field selectors
    PASSWORD_SELECTORS = [
        'input[type="password"]',
        'input[name*="pass" i]',
        'input[name*="pwd" i]',
        'input[id*="pass" i]',
        'input[id*="pwd" i]',
        'input[autocomplete="current-password"]',
        'input[autocomplete="password"]',
        'input[placeholder*="password" i]',
        'input[placeholder*="pass" i]',
    ]
    
    # Submit button selectors
    SUBMIT_SELECTORS = [
        'button[type="submit"]',
        'input[type="submit"]',
        'button:has-text("Sign in")',
        'button:has-text("Log in")',
        'button:has-text("Login")',
        'button:has-text("Signin")',
        'button:has-text("Submit")',
        'input[value*="Sign in" i]',
        'input[value*="Log in" i]',
        'input[value*="Login" i]',
        'a:has-text("Sign in")',
        'a:has-text("Log in")',
        '.btn-login',
        '.login-button',
        '#login-button',
        '[data-testid="login-button"]',
    ]
    
    # CAPTCHA field selectors
    CAPTCHA_SELECTORS = [
        'input[name*="captcha" i]',
        'input[id*="captcha" i]',
        'input[name*="g-recaptcha" i]',
        '.g-recaptcha',
        '[class*="captcha" i]',
        'img[src*="captcha" i]',
        '#recaptcha',
        '.h-captcha',
    ]
    
    # Remember me checkbox selectors
    REMEMBER_SELECTORS = [
        'input[type="checkbox"][name*="remember" i]',
        'input[type="checkbox"][id*="remember" i]',
        'input[type="checkbox"][name*="stay" i]',
        'input[type="checkbox"][name*="keep" i]',
        'label:has-text("Remember") input[type="checkbox"]',
        'label:has-text("Stay signed") input[type="checkbox"]',
    ]
    
    # MFA field selectors
    MFA_SELECTORS = [
        'input[name*="otp" i]',
        'input[name*="code" i]',
        'input[name*="token" i]',
        'input[name*="totp" i]',
        'input[name*="2fa" i]',
        'input[name*="mfa" i]',
        'input[autocomplete="one-time-code"]',
    ]
    
    # Form indicators (to identify login forms)
    LOGIN_FORM_INDICATORS = [
        'input[type="password"]',
        'input[name*="pass" i]',
        'input[name*="login" i]',
        'form[action*="login" i]',
        'form[action*="signin" i]',
        'form[action*="auth" i]',
        'text=/sign in/i',
        'text=/log in/i',
        'text=/login/i',
    ]
    
    async def detect_login_form(self, page) -> Optional[LoginForm]:
        """
        Detect and analyze login form on page.
        
        Args:
            page: Playwright page object
            
        Returns:
            LoginForm object or None
        """
        # Get all forms on the page
        forms = await page.query_selector_all('form')
        
        best_form = None
        best_score = 0.0
        
        for form in forms:
            # Check if form is visible
            try:
                if not await form.is_visible():
                    continue
            except:
                continue
            
            # Analyze form
            form_info = await self._analyze_form(page, form)
            
            if form_info and form_info.confidence_score > best_score:
                best_score = form_info.confidence_score
                best_form = form_info
        
        # Also check for non-form login patterns (div-based logins)
        if not best_form or best_score < 0.5:
            div_form = await self._detect_div_based_login(page)
            if div_form and div_form.confidence_score > best_score:
                best_form = div_form
        
        return best_form
    
    async def _analyze_form(self, page, form) -> Optional[LoginForm]:
        """Analyze a form for login characteristics."""
        score = 0.0
        
        # Check for password field (strong indicator)
        password_field = await self._find_field(form, self.PASSWORD_SELECTORS)
        if not password_field:
            return None
        
        score += 0.5  # Strong indicator
        
        # Find username field
        username_field = await self._find_field(form, self.USERNAME_SELECTORS)
        if username_field:
            score += 0.2
        
        # Find submit button
        submit_button = await self._find_field(form, self.SUBMIT_SELECTORS)
        if submit_button:
            score += 0.1
        
        # Get form attributes
        action = await form.get_attribute('action') or ''
        method = await form.get_attribute('method') or 'GET'
        
        # Check for CSRF token
        csrf_field = await form.query_selector(
            'input[name*="csrf" i], input[name*="token" i], '
            'input[name="_token"], input[name="authenticity_token"]'
        )
        
        # Check for CAPTCHA
        captcha_field = await self._find_field(form, self.CAPTCHA_SELECTORS)
        if captcha_field:
            score += 0.05
        
        # Check for remember checkbox
        remember_checkbox = await self._find_field(form, self.REMEMBER_SELECTORS)
        if remember_checkbox:
            score += 0.05
        
        # Check for MFA field
        mfa_field = await self._find_field(form, self.MFA_SELECTORS)
        
        # Check if action URL contains login indicators
        if any(indicator in action.lower() for indicator in ['login', 'signin', 'auth', 'authenticate']):
            score += 0.1
        
        # Build form object
        is_https = page.url.startswith('https')
        
        return LoginForm(
            form_element=form,
            username_field=username_field,
            password_field=password_field,
            submit_button=submit_button,
            captcha_field=captcha_field,
            mfa_field=mfa_field,
            remember_checkbox=remember_checkbox,
            action_url=action,
            method=method.upper(),
            has_csrf_token=csrf_field is not None,
            csrf_token_field=await csrf_field.get_attribute('name') if csrf_field else None,
            is_https=is_https,
            has_password_field=True,
            autocomplete_enabled=True,
            confidence_score=min(score, 1.0)
        )
    
    async def _detect_div_based_login(self, page) -> Optional[LoginForm]:
        """Detect login patterns not wrapped in form tags."""
        # Look for password input
        password_field = await page.query_selector('input[type="password"]')
        if not password_field:
            return None
        
        # Check if it's in a likely login container
        username_field = None
        submit_button = None
        
        # Try to find username field nearby
        for selector in self.USERNAME_SELECTORS:
            try:
                username_field = await page.query_selector(selector)
                if username_field:
                    break
            except:
                pass
        
        # Try to find submit button
        for selector in self.SUBMIT_SELECTORS:
            try:
                submit_button = await page.query_selector(selector)
                if submit_button:
                    break
            except:
                pass
        
        if username_field and submit_button:
            return LoginForm(
                form_element=password_field,  # Use password field as reference
                username_field=username_field,
                password_field=password_field,
                submit_button=submit_button,
                action_url=page.url,
                method="POST",
                is_https=page.url.startswith('https'),
                has_password_field=True,
                confidence_score=0.6
            )
        
        return None
    
    async def _find_field(self, container, selectors: List[str]) -> Optional[Any]:
        """Find field matching any selector."""
        for selector in selectors:
            try:
                field = await container.query_selector(selector)
                if field:
                    # Check visibility
                    try:
                        if await field.is_visible():
                            return field
                    except:
                        return field
            except:
                continue
        return None


class CaptchaSolver:
    """
    CAPTCHA solving interface.
    """
    
    CAPTCHA_TYPES = {
        'recaptcha_v2': 'Google reCAPTCHA v2',
        'recaptcha_v3': 'Google reCAPTCHA v3',
        'hcaptcha': 'hCaptcha',
        'image_captcha': 'Image-based CAPTCHA',
        'text_captcha': 'Text-based CAPTCHA',
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize CAPTCHA solver.
        
        Args:
            api_key: API key for external solving service (2captcha, Anti-Captcha, etc.)
        """
        self.api_key = api_key
    
    async def detect_captcha_type(self, page) -> Optional[str]:
        """
        Detect type of CAPTCHA on page.
        
        Args:
            page: Playwright page object
            
        Returns:
            CAPTCHA type or None
        """
        # Check for reCAPTCHA v2
        recaptcha_v2 = await page.query_selector('.g-recaptcha')
        if recaptcha_v2:
            return 'recaptcha_v2'
        
        # Check for reCAPTCHA v3 (invisible)
        recaptcha_v3 = await page.query_selector('script[src*="recaptcha/api.js"]')
        if recaptcha_v3:
            return 'recaptcha_v3'
        
        # Check for hCaptcha
        hcaptcha = await page.query_selector('.h-captcha')
        if hcaptcha:
            return 'hcaptcha'
        
        # Check for image CAPTCHA
        img_captcha = await page.query_selector('img[src*="captcha"], img[alt*="captcha" i]')
        if img_captcha:
            return 'image_captcha'
        
        return None
    
    async def solve(self, page, captcha_type: str) -> CaptchaResult:
        """
        Solve CAPTCHA.
        
        Args:
            page: Playwright page object
            captcha_type: Type of CAPTCHA
            
        Returns:
            CaptchaResult with solution
        """
        if captcha_type == 'recaptcha_v2':
            return await self._solve_recaptcha_v2(page)
        elif captcha_type == 'hcaptcha':
            return await self._solve_hcaptcha(page)
        elif captcha_type == 'image_captcha':
            return await self._solve_image_captcha(page)
        else:
            return CaptchaResult(
                success=False,
                error=f"Unsupported CAPTCHA type: {captcha_type}",
                captcha_type=captcha_type
            )
    
    async def _solve_recaptcha_v2(self, page) -> CaptchaResult:
        """Solve reCAPTCHA v2."""
        # This would integrate with external solving service
        # For now, return failure
        return CaptchaResult(
            success=False,
            error="reCAPTCHA v2 solving requires external service",
            captcha_type='recaptcha_v2'
        )
    
    async def _solve_hcaptcha(self, page) -> CaptchaResult:
        """Solve hCaptcha."""
        return CaptchaResult(
            success=False,
            error="hCaptcha solving requires external service",
            captcha_type='hcaptcha'
        )
    
    async def _solve_image_captcha(self, page) -> CaptchaResult:
        """Solve image-based CAPTCHA."""
        return CaptchaResult(
            success=False,
            error="Image CAPTCHA solving requires OCR service",
            captcha_type='image_captcha'
        )


class LoginVerifier:
    """
    Verifies successful login through multiple indicators.
    """
    
    # Success indicators
    SUCCESS_INDICATORS = [
        # URL patterns
        '/dashboard', '/home', '/welcome', '/account',
        '/profile', '/main', '/app', '/overview',
        '/inbox', '/mailbox', '/messages',
        
        # Page elements
        'text=/welcome/i',
        'text=/dashboard/i',
        'text=/my account/i',
        'text=/profile/i',
        'text=/logout/i',
        'text=/sign out/i',
        'text=/signout/i',
        
        # UI elements
        '.user-menu',
        '.account-dropdown',
        '.profile-image',
        '.user-avatar',
        '[data-testid="user-menu"]',
        '[data-testid="dashboard"]',
        '.logged-in',
        '.authenticated',
    ]
    
    # Failure indicators
    FAILURE_INDICATORS = [
        # Error messages
        'text=/invalid/i',
        'text=/incorrect/i',
        'text=/failed/i',
        'text=/error/i',
        'text=/wrong password/i',
        'text=/authentication failed/i',
        'text=/login failed/i',
        'text=/access denied/i',
        
        # Error elements
        '.error',
        '.alert-error',
        '.form-error',
        '.login-error',
        '[role="alert"]',
        '.has-error',
        '.is-invalid',
    ]
    
    # MFA indicators
    MFA_INDICATORS = [
        'text=/two-factor/i',
        'text=/2fa/i',
        'text=/mfa/i',
        'text=/verification code/i',
        'text=/enter code/i',
        'text=/authenticator/i',
        'input[name*="otp" i]',
        'input[name*="code" i]',
        'input[autocomplete="one-time-code"]',
    ]
    
    async def verify_login(self, page, timeout: int = 10) -> Tuple[bool, Optional[str]]:
        """
        Verify if login was successful.
        
        Args:
            page: Playwright page object
            timeout: Maximum wait time in seconds
            
        Returns:
            Tuple of (success, error_message)
        """
        start_time = asyncio.get_event_loop().time()
        
        while (asyncio.get_event_loop().time() - start_time) < timeout:
            # Check for failure indicators first
            for indicator in self.FAILURE_INDICATORS:
                try:
                    element = await page.query_selector(indicator)
                    if element and await element.is_visible():
                        error_text = await element.text_content()
                        return False, error_text or "Login failed"
                except:
                    pass
            
            # Check for success indicators
            for indicator in self.SUCCESS_INDICATORS:
                try:
                    element = await page.query_selector(indicator)
                    if element and await element.is_visible():
                        return True, None
                except:
                    pass
            
            # Check for MFA requirement
            for indicator in self.MFA_INDICATORS:
                try:
                    element = await page.query_selector(indicator)
                    if element and await element.is_visible():
                        return False, "MFA_REQUIRED"
                except:
                    pass
            
            # Check URL change
            current_url = page.url
            if any(indicator in current_url.lower() for indicator in ['dashboard', 'home', 'welcome', 'account']):
                return True, None
            
            await asyncio.sleep(0.5)
        
        # Timeout - check for session cookies as fallback
        try:
            cookies = await page.context.cookies()
            session_cookies = [c for c in cookies if self._is_session_cookie(c)]
            
            if session_cookies:
                return True, None
        except:
            pass
        
        return False, "Could not verify login status"
    
    def _is_session_cookie(self, cookie: Dict) -> bool:
        """Check if cookie is likely a session cookie."""
        session_names = ['session', 'sid', 'auth', 'token', 'login', 'user']
        cookie_name = cookie.get('name', '').lower()
        return any(name in cookie_name for name in session_names)
    
    async def check_mfa_required(self, page) -> bool:
        """Check if MFA is required."""
        for indicator in self.MFA_INDICATORS:
            try:
                element = await page.query_selector(indicator)
                if element and await element.is_visible():
                    return True
            except:
                pass
        return False


class FormAuthenticator:
    """
    Automated form-based authentication.
    """
    
    def __init__(
        self,
        credential_vault,
        mfa_handler=None,
        captcha_solver: Optional[CaptchaSolver] = None
    ):
        """
        Initialize form authenticator.
        
        Args:
            credential_vault: Credential vault instance
            mfa_handler: MFA handler instance
            captcha_solver: CAPTCHA solver instance
        """
        self.vault = credential_vault
        self.mfa = mfa_handler
        self.captcha = captcha_solver
        self.form_detector = FormDetector()
        self.login_verifier = LoginVerifier()
    
    async def authenticate(
        self,
        page,
        service_name: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        wait_for_mfa: bool = True,
        timeout: int = 60,
        enable_remember: bool = False
    ) -> AuthenticationResult:
        """
        Perform automated form authentication.
        
        Args:
            page: Playwright page object
            service_name: Service identifier
            username: Username (if None, uses default)
            password: Password (if None, retrieved from vault)
            wait_for_mfa: Whether to handle MFA if required
            timeout: Maximum authentication time
            enable_remember: Whether to check "remember me"
            
        Returns:
            AuthenticationResult
        """
        start_time = asyncio.get_event_loop().time()
        
        # Get credentials if not provided
        if not username or not password:
            creds = await self.vault.get_credentials(service_name, username)
            if not creds:
                return AuthenticationResult(
                    success=False,
                    error="Credentials not found"
                )
            username = creds.username
            password = creds.password
        
        # Detect login form
        login_form = await self.form_detector.detect_login_form(page)
        if not login_form:
            return AuthenticationResult(
                success=False,
                error="Login form not detected"
            )
        
        logger.debug(f"Detected login form with confidence {login_form.confidence_score}")
        
        # Fill username
        if login_form.username_field:
            await login_form.username_field.fill(username)
            await asyncio.sleep(self._random_delay(0.1, 0.3))
        
        # Fill password
        await login_form.password_field.fill(password)
        await asyncio.sleep(self._random_delay(0.1, 0.3))
        
        # Handle CAPTCHA if present
        if login_form.captcha_field and self.captcha:
            captcha_type = await self.captcha.detect_captcha_type(page)
            if captcha_type:
                captcha_result = await self.captcha.solve(page, captcha_type)
                if captcha_result.success:
                    await login_form.captcha_field.fill(captcha_result.solution)
                else:
                    return AuthenticationResult(
                        success=False,
                        error=f"CAPTCHA handling failed: {captcha_result.error}",
                        captcha_required=True
                    )
        
        # Check remember me
        if enable_remember and login_form.remember_checkbox:
            await login_form.remember_checkbox.check()
        
        # Submit form
        if login_form.submit_button:
            await login_form.submit_button.click()
        else:
            # Try pressing Enter
            await login_form.password_field.press('Enter')
        
        # Wait for navigation
        try:
            await page.wait_for_load_state('networkidle', timeout=timeout * 1000)
        except:
            pass
        
        # Check for MFA requirement
        if await self.login_verifier.check_mfa_required(page):
            if not wait_for_mfa:
                return AuthenticationResult(
                    success=False,
                    error="MFA required but not handled",
                    mfa_required=True
                )
            
            if self.mfa:
                mfa_result = await self.mfa.handle_mfa(page, service_name, username)
                if not mfa_result.success:
                    return AuthenticationResult(
                        success=False,
                        error=f"MFA handling failed: {mfa_result.error}",
                        mfa_required=True
                    )
            else:
                return AuthenticationResult(
                    success=False,
                    error="MFA required but no handler configured",
                    mfa_required=True
                )
        
        # Verify successful login
        success, error = await self.login_verifier.verify_login(page)
        
        if success:
            # Update credential usage stats
            await self._update_credential_stats(service_name, username)
            
            return AuthenticationResult(
                success=True,
                session_cookies=await page.context.cookies(),
                login_time=asyncio.get_event_loop().time() - start_time,
                redirect_url=page.url
            )
        else:
            if error == "MFA_REQUIRED":
                return AuthenticationResult(
                    success=False,
                    error="MFA required",
                    mfa_required=True
                )
            
            return AuthenticationResult(
                success=False,
                error=error or "Login verification failed"
            )
    
    async def _update_credential_stats(self, service_name: str, username: str) -> None:
        """Update credential usage statistics."""
        try:
            creds = await self.vault.get_credentials(service_name, username)
            if creds:
                creds.use_count += 1
                creds.last_used = datetime.now()
                # Note: This would need a save method in the vault
        except:
            pass
    
    def _random_delay(self, min_delay: float, max_delay: float) -> float:
        """Generate random delay for human-like behavior."""
        import random
        return random.uniform(min_delay, max_delay)
    
    async def is_logged_in(self, page, service_name: str) -> bool:
        """
        Check if already logged in to service.
        
        Args:
            page: Playwright page object
            service_name: Service identifier
            
        Returns:
            True if logged in
        """
        success, _ = await self.login_verifier.verify_login(page)
        return success


# Convenience functions
async def detect_login_form(page) -> Optional[LoginForm]:
    """Detect login form on page."""
    detector = FormDetector()
    return await detector.detect_login_form(page)


async def verify_login(page, timeout: int = 10) -> Tuple[bool, Optional[str]]:
    """Verify if login was successful."""
    verifier = LoginVerifier()
    return await verifier.verify_login(page, timeout)
