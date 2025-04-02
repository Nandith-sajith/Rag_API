import pytest
from fastapi.testclient import TestClient
from src.app import app

@pytest.fixture
def client():
    return TestClient(app)

def test_process_prompt(client):
    response = client.post(
        "/rag_query",
        json={"query": "What are the rules of Monopoly?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "confidence" in data
    assert "evaluation" in data
    assert isinstance(data["confidence"], float)
    assert 0 <= data["confidence"] <= 1