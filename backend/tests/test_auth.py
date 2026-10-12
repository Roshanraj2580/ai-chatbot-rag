import pytest
from fastapi.testclient import TestClient
from backend.app import app
from backend.auth import create_access_token

client = TestClient(app)

def test_auth_registration_and_login():
    username = f"testuser_{pytest.__version__.replace('.', '_')}"
    password = "password123"

    # Register
    reg_resp = client.post("/auth/register", json={"username": username, "password": password})
    assert reg_resp.status_code in [201, 400]  # If already exists or created
    
    # Login
    login_resp = client.post("/auth/login", json={"username": username, "password": password})
    assert login_resp.status_code == 200
    data = login_resp.json()
    assert "access_token" in data
    assert data["username"] == username
    assert data["token_type"] == "bearer"

    token = data["access_token"]

    # Verify /auth/me with Bearer token
    me_resp = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_resp.status_code == 200
    me_data = me_resp.json()
    assert me_data["username"] == username

    # Verify /auth/me without token fails with 401
    unauth_resp = client.get("/auth/me")
    assert unauth_resp.status_code == 401


def test_auth_invalid_credentials():
    resp = client.post("/auth/login", json={"username": "non_existent_user_999", "password": "wrong"})
    assert resp.status_code == 401
