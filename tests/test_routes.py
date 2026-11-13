"""Route and API foundation tests for NIVARAN."""
import io
import pytest
from app import create_app
from config import Config


@pytest.fixture
def client():
    """Create test client with isolated configuration."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_route(client):
    """Verify home page loads successfully with HTML content."""
    res = client.get("/")
    assert res.status_code == 200
    assert b"NIVARAN" in res.data
    assert b"Scholarship Grievance Redressal" in res.data
    assert b"screen-home" in res.data
    assert b"screen-analysis" in res.data
    assert b"screen-ask" in res.data
    assert b"screen-steps" in res.data


def test_four_screen_routes_accessible(client):
    """Verify all four Civic Hearth screen routes load successfully with HTTP 200."""
    for path in ["/", "/welcome", "/analysis", "/document-check", "/ask", "/ask-questions", "/steps", "/next-steps"]:
        res = client.get(path)
        assert res.status_code == 200
        assert b"NIVARAN" in res.data
        assert b"FICTIONAL DEMONSTRATION SYSTEM" in res.data


def test_security_headers(client):
    """Verify security headers are applied to HTTP responses."""
    res = client.get("/")
    assert res.headers.get("X-Content-Type-Options") == "nosniff"
    assert res.headers.get("X-Frame-Options") == "DENY"
    assert res.headers.get("X-XSS-Protection") == "1; mode=block"


def test_health_check_endpoint(client):
    """Verify health check returns valid JSON without leaking secrets."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "healthy"
    assert "vector_store_docs_count" in data
    assert "gemini_model" in data
    # Ensure raw secret is not returned
    assert "SECRET_KEY" not in data


def test_kb_stats_endpoint(client):
    """Verify knowledge base stats endpoint returns metadata."""
    res = client.get("/api/knowledge-base/stats")
    assert res.status_code == 200
    data = res.get_json()
    assert "collection_name" in data
    assert "total_chunks" in data


def test_upload_validation(client):
    """Verify file upload endpoint handles missing, invalid and valid files."""
    # 1. No file
    res = client.post("/api/upload")
    assert res.status_code == 400

    # 2. Unsupported extension (e.g. .exe)
    res = client.post(
        "/api/upload",
        data={"file": (io.BytesIO(b"malicious content"), "exploit.exe")},
        content_type="multipart/form-data"
    )
    assert res.status_code == 400
    assert "Unsupported file type" in res.get_json()["error"]

    # 3. Valid text file upload
    test_content = b"Rejection Notice: Application rejected due to unverified income certificate."
    res = client.post(
        "/api/upload",
        data={"file": (io.BytesIO(test_content), "rejection_notice.txt")},
        content_type="multipart/form-data"
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["chunks_extracted"] >= 1
    assert "unverified income" in data["preview"]


def test_query_validation(client):
    """Verify query endpoint validation."""
    # 1. Empty query
    res = client.post("/api/query", json={"query": ""})
    assert res.status_code == 400

    # 2. Query with missing Gemini key returns structured response without crash
    res = client.post("/api/query", json={"query": "What is the income criteria?"})
    assert res.status_code == 200
    data = res.get_json()
    assert "citations" in data
    assert "answer" in data
