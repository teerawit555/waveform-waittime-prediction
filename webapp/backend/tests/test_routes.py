import os
import sys
from unittest.mock import MagicMock

# 1. Set environment variables before import to configure ADMIN_TOKEN, MLflow status, and training settings
os.environ["NEUROSETTLE_ADMIN_TOKEN"] = "test-admin-token"
os.environ["DEFAULT_MODEL_NAME"] = "TCN_aug_weighted_v1"
os.environ["ENABLE_TRAINING"] = "0"
os.environ["MLFLOW_DISABLED"] = "1"

# 2. Mock heavy dependencies
sys.modules['torch'] = MagicMock()
sys.modules['autogluon'] = MagicMock()
sys.modules['autogluon.tabular'] = MagicMock()
sys.modules['mlflow'] = MagicMock()
sys.modules['mlflow.tracking'] = MagicMock()

# Mock inference_service module so it does not load actual models or spawn threads
mock_inference_service_module = MagicMock()
mock_inference_service = MagicMock()
mock_inference_service_module.inference_service = mock_inference_service
sys.modules['app.inference_service'] = mock_inference_service_module

# Now import unittest and test-client targets
import unittest
from run import create_app
import app.routes as routes

class TestRoutes(unittest.TestCase):
    def setUp(self):
        # Override ADMIN_TOKEN on routes explicitly to be absolutely sure
        routes.ADMIN_TOKEN = "test-admin-token"
        self.app = create_app()
        self.client = self.app.test_client()

    def test_health_check(self):
        """Verify GET /api/health returns 200 and {"ok": True}."""
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"ok": True})

    def test_api_root(self):
        """Verify GET /api/ returns 200 and running message."""
        response = self.client.get('/api/')
        self.assertEqual(response.status_code, 200)
        self.assertIn("message", response.get_json())

    def test_security_headers(self):
        """Verify HTTP Response has all security headers applied."""
        response = self.client.get('/api/health')
        self.assertEqual(response.headers.get("X-Content-Type-Options"), "nosniff")
        self.assertEqual(response.headers.get("X-Frame-Options"), "DENY")
        self.assertEqual(response.headers.get("Referrer-Policy"), "no-referrer")
        self.assertEqual(response.headers.get("Cache-Control"), "no-store")

    def test_admin_routes_unauthorized(self):
        """Verify admin routes return 401 Unauthorized without X-Admin-Token."""
        endpoints = [
            ('/api/jobs', 'GET'),
            ('/api/jobs/queue', 'GET'),
            ('/api/mlflow/model-registry', 'GET'),
            ('/api/tcn-models', 'GET'),
        ]
        for url, method in endpoints:
            response = self.client.get(url)
            self.assertEqual(response.status_code, 401, f"URL {url} did not return 401")

    def test_admin_routes_authorized_header(self):
        """Verify admin routes are accessible with correct X-Admin-Token header."""
        response = self.client.get('/api/jobs', headers={"X-Admin-Token": "test-admin-token"})
        self.assertEqual(response.status_code, 200)

    def test_admin_routes_authorized_bearer(self):
        """Verify admin routes are accessible with correct Authorization Bearer token."""
        response = self.client.get('/api/jobs', headers={"Authorization": "Bearer test-admin-token"})
        self.assertEqual(response.status_code, 200)

    def test_admin_routes_incorrect_token(self):
        """Verify admin routes deny access when incorrect token is provided."""
        response = self.client.get('/api/jobs', headers={"X-Admin-Token": "wrong-token"})
        self.assertEqual(response.status_code, 401)
