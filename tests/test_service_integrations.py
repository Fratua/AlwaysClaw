"""Tests for service_integrations with mocked services."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime


class TestGmailIntegration:
    """Test GmailIntegration."""

    @pytest.mark.asyncio
    async def test_init_unauthenticated(self):
        from service_integrations import GmailIntegration
        gmail = GmailIntegration()
        assert gmail.authenticated is False

    @pytest.mark.asyncio
    async def test_send_email_requires_auth(self):
        from service_integrations import GmailIntegration
        gmail = GmailIntegration()
        result = await gmail.send_email("to@example.com", "Subject", "Body")
        assert result.success is False
        assert "Not authenticated" in result.error

    @pytest.mark.asyncio
    async def test_check_emails_requires_auth(self):
        from service_integrations import GmailIntegration
        gmail = GmailIntegration()
        result = await gmail.check_new_emails()
        assert result == []


class TestTwilioIntegration:
    """Test TwilioIntegration."""

    @pytest.mark.asyncio
    async def test_init(self):
        from service_integrations import TwilioIntegration
        twilio = TwilioIntegration(
            account_sid="test_sid",
            auth_token="test_token",
            phone_number="+1234567890"
        )
        assert twilio.authenticated is False
        assert twilio.account_sid == "test_sid"

    @pytest.mark.asyncio
    async def test_make_call_requires_auth(self):
        from service_integrations import TwilioIntegration
        twilio = TwilioIntegration()
        result = await twilio.make_call("+1234567890", "Hello")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_send_sms_requires_auth(self):
        from service_integrations import TwilioIntegration
        twilio = TwilioIntegration()
        result = await twilio.send_sms("+1234567890", "Test message")
        assert result.success is False


class TestBrowserControlIntegration:
    """Test BrowserControlIntegration."""

    def test_init(self):
        from service_integrations import BrowserControlIntegration
        browser = BrowserControlIntegration(headless=True)
        assert browser.headless is True
        assert browser.browser is None
        assert browser.pages == {}

    @pytest.mark.asyncio
    async def test_navigate_requires_init(self):
        from service_integrations import BrowserControlIntegration
        browser = BrowserControlIntegration()
        result = await browser.navigate("https://example.com")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_execute_action_requires_init(self):
        from service_integrations import BrowserControlIntegration, BrowserAction
        browser = BrowserControlIntegration()
        action = BrowserAction(type="click", selector="button")
        result = await browser.execute_action(action)
        assert result.success is False

    def test_get_active_page_none(self):
        from service_integrations import BrowserControlIntegration
        browser = BrowserControlIntegration()
        assert browser.get_active_page() is None


class TestServiceIntegrationManager:
    """Test the unified manager."""

    def test_init(self):
        from service_integrations import ServiceIntegrationManager
        manager = ServiceIntegrationManager()
        assert 'gmail' in manager.services
        assert 'twilio' in manager.services
        assert 'browser' in manager.services
        assert 'system' in manager.services

    def test_get_service_status(self):
        from service_integrations import ServiceIntegrationManager
        manager = ServiceIntegrationManager()
        status = manager.get_service_status()
        assert 'gmail' in status
        assert 'twilio' in status
        for name, info in status.items():
            assert 'type' in info

    @pytest.mark.asyncio
    async def test_execute_unknown_service(self):
        from service_integrations import ServiceIntegrationManager
        manager = ServiceIntegrationManager()
        with pytest.raises(ValueError, match="Unknown service"):
            await manager.execute("nonexistent", "action")

    @pytest.mark.asyncio
    async def test_execute_unknown_action(self):
        from service_integrations import ServiceIntegrationManager
        manager = ServiceIntegrationManager()
        # Mark gmail as initialized so we can test the unknown action path
        manager._service_status['gmail'] = {'initialized': True, 'error': None}
        # Gmail is a valid service but 'nonexistent_action' is not
        with pytest.raises(ValueError, match="Unknown action"):
            await manager.execute("gmail", "nonexistent_action")
