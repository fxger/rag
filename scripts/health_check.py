"""
Health check script.

Probes all system components and reports status.
Run: python scripts/health_check.py
"""

import asyncio
import sys
import os
import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_URL = os.getenv("RAG_BASE_URL", "http://localhost:8000")


async def check_health():
    """Check all system health endpoints."""
    results = {}

    async with httpx.AsyncClient(timeout=10.0) as client:
        # Liveness
        try:
            r = await client.get(f"{BASE_URL}/health")
            results["liveness"] = "✅ UP" if r.status_code == 200 else f"❌ DOWN ({r.status_code})"
        except Exception as e:
            results["liveness"] = f"❌ DOWN ({e})"

        # Readiness
        try:
            r = await client.get(f"{BASE_URL}/health/ready")
            if r.status_code == 200:
                data = r.json()
                services = data.get("data", data)
                results["readiness"] = "✅ READY"
                for svc, status in services.items() if isinstance(services, dict) else []:
                    results[f"  {svc}"] = "✅" if status else "❌"
            else:
                results["readiness"] = f"❌ NOT READY ({r.status_code})"
        except Exception as e:
            results["readiness"] = f"❌ DOWN ({e})"

    print("\n" + "=" * 50)
    print("  Enterprise RAG System — Health Report")
    print("=" * 50)
    for check, status in results.items():
        print(f"  {check}: {status}")
    print("=" * 50 + "\n")

    return all("✅" in v for v in results.values())


if __name__ == "__main__":
    success = asyncio.run(check_health())
    sys.exit(0 if success else 1)