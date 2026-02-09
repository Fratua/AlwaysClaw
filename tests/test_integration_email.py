"""Integration tests for email workflow system."""

import pytest
import os
import sys
from unittest.mock import MagicMock, patch, AsyncMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestEmailWorkflowImport:
    """Test email workflow engine can be imported."""

    def test_import(self):
        import email_workflow_engine
        assert email_workflow_engine is not None


class TestGmailClientImport:
    """Test Gmail client can be imported."""

    def test_import(self):
        import gmail_client_implementation
        assert gmail_client_implementation is not None


class TestNotificationDelivery:
    """Test notification delivery routing."""

    def test_email_delivery_method_exists(self):
        """Verify _deliver_email method exists in UserNotificationSystem."""
        from notification_system import UserNotificationSystem
        assert hasattr(UserNotificationSystem, '_deliver_email')
        assert callable(getattr(UserNotificationSystem, '_deliver_email'))

    def test_sms_delivery_method_exists(self):
        """Verify _deliver_sms method exists."""
        from notification_system import UserNotificationSystem
        assert hasattr(UserNotificationSystem, '_deliver_sms')

    def test_voice_delivery_method_exists(self):
        """Verify _deliver_voice method exists."""
        from notification_system import UserNotificationSystem
        assert hasattr(UserNotificationSystem, '_deliver_voice')

    def test_notification_to_dict(self):
        from notification_system import UserNotification, NotificationChannel
        notif = UserNotification(
            title="Test Email",
            message="This is a test",
            channel=NotificationChannel.EMAIL,
        )
        d = notif.to_dict()
        assert d['title'] == "Test Email"
        assert d['channel'] == "email"

    def test_send_notification_dashboard_channel(self):
        """Test sending a notification via dashboard channel (in-memory, no external deps)."""
        from notification_system import UserNotificationSystem, UserNotification, NotificationChannel
        system = UserNotificationSystem()
        notif = UserNotification(
            title="Dashboard Test",
            message="Testing dashboard delivery",
            channel=NotificationChannel.DASHBOARD,
        )
        result = system.send_notification(notif)
        assert isinstance(result, bool)
