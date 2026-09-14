"""
Locust performance test script.

Usage:
    locust -f tests/performance/locustfile.py --host=http://localhost:8000
"""

from locust import HttpUser, task, between, events
import json


class RAGUser(HttpUser):
    """Simulated user for performance testing."""

    wait_time = between(1, 3)
    host = "http://localhost:8000"

    def on_start(self):
        """Authenticate on start."""
        response = self.client.post(
            "/api/v1/auth/login",
            json={"username": "testuser", "password": "testpass"},
        )
        if response.status_code == 200:
            data = response.json()
            self.token = data.get("data", {}).get("access_token", "")
            self.headers = {"Authorization": f"Bearer {self.token}"}
        else:
            self.token = ""
            self.headers = {}

    @task(5)
    def rag_query(self):
        """Simulate RAG queries (highest weight)."""
        queries = [
            "What is the company's growth strategy?",
            "Summarize the Q3 financial report",
            "What are the key risks identified?",
            "Explain the new product roadmap",
            "What is the market analysis?",
        ]
        self.client.post(
            "/api/v1/query/",
            json={"query": queries[self.environment.runner.user_count % 5], "top_k": 5},
            headers=self.headers,
            name="/api/v1/query/",
        )

    @task(3)
    def list_documents(self):
        """Simulate document listing."""
        self.client.get(
            "/api/v1/documents/",
            headers=self.headers,
            name="/api/v1/documents/",
        )

    @task(1)
    def health_check(self):
        """Simulate health checks."""
        self.client.get("/health")

    @task(1)
    def query_history(self):
        """Simulate checking query history."""
        self.client.get(
            "/api/v1/query/history",
            headers=self.headers,
            name="/api/v1/query/history",
        )