"""Integration tests for the dashboard server API endpoints."""

import pytest
import os
import sys
from unittest.mock import MagicMock, patch, AsyncMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDashboardImport:
    """Test dashboard server can be imported."""

    def test_import_dashboard(self):
        import dashboard_server
        assert hasattr(dashboard_server, 'DashboardAPI')

    def test_dashboard_api_class(self):
        from dashboard_server import DashboardAPI
        # DashboardAPI requires engine, scheduler, alert_manager
        mock_engine = MagicMock()
        mock_scheduler = MagicMock()
        mock_alert_mgr = MagicMock()
        api = DashboardAPI(mock_engine, mock_scheduler, mock_alert_mgr)
        assert api is not None
        assert api.engine is mock_engine

    def test_dashboard_api_has_fastapi_app(self):
        from dashboard_server import DashboardAPI, HAS_FASTAPI
        if HAS_FASTAPI:
            mock_engine = MagicMock()
            mock_scheduler = MagicMock()
            mock_alert_mgr = MagicMock()
            api = DashboardAPI(mock_engine, mock_scheduler, mock_alert_mgr)
            assert hasattr(api, 'app')
            assert api.app is not None
        else:
            assert True  # FastAPI not installed, but module still importable


class TestDashboardEndpoints:
    """Test FastAPI endpoints with TestClient."""

    @pytest.fixture
    def client(self):
        from dashboard_server import DashboardAPI, HAS_FASTAPI
        if not HAS_FASTAPI:
            pytest.fail("FastAPI not installed - required for endpoint tests")
        from fastapi.testclient import TestClient
        mock_engine = MagicMock()
        mock_engine.run_check = AsyncMock(return_value=None)
        mock_scheduler = MagicMock()
        mock_scheduler.get_statistics = MagicMock(return_value={})
        mock_alert_mgr = MagicMock()
        api = DashboardAPI(mock_engine, mock_scheduler, mock_alert_mgr)
        return TestClient(api.app)

    def test_root_endpoint(self, client):
        """Test / returns HTML dashboard."""
        response = client.get("/")
        assert response.status_code == 200

    def test_api_overview(self, client):
        """Test /api/overview endpoint."""
        response = client.get("/api/overview")
        assert response.status_code == 200


class TestNotificationSystem:
    """Test notification system delivery routing."""

    def test_import(self):
        from notification_system import UserNotificationSystem, NotificationChannel
        assert NotificationChannel.EMAIL.value == "email"

    def test_notification_channels(self):
        from notification_system import NotificationChannel
        channels = list(NotificationChannel)
        assert len(channels) >= 4  # EMAIL, SMS, VOICE, DASHBOARD, ALL

    def test_notification_creation(self):
        from notification_system import UserNotification, NotificationChannel
        notif = UserNotification(
            title="Test",
            message="Test message",
            channel=NotificationChannel.DASHBOARD
        )
        assert notif.title == "Test"
        d = notif.to_dict()
        assert d['channel'] == 'dashboard'

    def test_notification_system_has_delivery_methods(self):
        from notification_system import UserNotificationSystem
        # UserNotificationSystem should have delivery methods
        assert hasattr(UserNotificationSystem, '_deliver_email')
        assert hasattr(UserNotificationSystem, '_deliver_sms')
        assert hasattr(UserNotificationSystem, '_deliver_voice')
        assert hasattr(UserNotificationSystem, 'send_notification')
