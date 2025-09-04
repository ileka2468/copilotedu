"""Tests for Google OAuth direct integration."""
import os
import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.auth.models import Role, Permission, RolePermission


@pytest.fixture(scope="session")
def engine():
    # Use a file-based SQLite database
    test_db_path = "test_oauth.db"
    if os.path.exists(test_db_path):
        try:
            os.remove(test_db_path)
        except Exception:
            pass
    return create_engine(f"sqlite:///{test_db_path}", connect_args={"check_same_thread": False})


@pytest.fixture(scope="session", autouse=True)
def setup_database(engine):
    Base.metadata.create_all(bind=engine)

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    try:
        if not db.query(Role).filter(Role.id == 1).first():
            db.add(Role(id=1, name="teacher", description="Test Role"))
        if not db.query(Permission).filter(Permission.id == 1).first():
            db.add(Permission(id=1, name="test_permission", description="Test Permission"))
        if not db.query(RolePermission).filter_by(role_id=1, permission_id=1).first():
            db.add(RolePermission(role_id=1, permission_id=1))
        db.commit()
    finally:
        db.close()

    yield

    Base.metadata.drop_all(bind=engine)
    try:
        engine.dispose()
    except Exception:
        pass
    if os.path.exists("test_oauth.db"):
        try:
            os.remove("test_oauth.db")
        except Exception:
            pass


@pytest.fixture
def client(engine, monkeypatch):
    from app.main import app
    from app.database import get_db

    # Set required env vars
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "dummy-client-id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "dummy-secret")
    monkeypatch.setenv("GOOGLE_REDIRECT_URI", "http://testserver/auth/oauth/google/callback")

    Base.metadata.create_all(bind=engine)

    def override_get_db():
        try:
            db = sessionmaker(autocommit=False, autoflush=False, bind=engine)()
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def test_google_login_redirect_sets_state_cookie(client):
    resp = client.get("/auth/oauth/google/login", allow_redirects=False)
    assert resp.status_code in (status.HTTP_307_TEMPORARY_REDIRECT, status.HTTP_302_FOUND)
    # Location should include client_id
    location = resp.headers.get("location")
    assert "client_id=dummy-client-id" in location
    # Cookie should include oauth_state
    cookies = resp.cookies
    assert "oauth_state" in cookies


def test_google_callback_success(client, monkeypatch):
    # Prepare state
    login_resp = client.get("/auth/oauth/google/login", allow_redirects=False)
    state_cookie = login_resp.cookies.get("oauth_state")
    assert state_cookie

    # Mock token exchange
    class DummyResp:
        def __init__(self, status_code, data):
            self.status_code = status_code
            self._data = data
        def json(self):
            return self._data

    def fake_post(url, data=None, timeout=None):
        assert "authorization_code" in (data or {}).get("grant_type", "")
        return DummyResp(200, {"access_token": "ya29.fake"})

    def fake_get(url, headers=None, timeout=None):
        assert headers and headers.get("Authorization", "").startswith("Bearer ")
        return DummyResp(200, {"email": "oauth.user@example.com", "name": "OAuth User"})

    monkeypatch.setattr("app.auth.router.requests.post", fake_post)
    monkeypatch.setattr("app.auth.router.requests.get", fake_get)

    # Callback
    cb_resp = client.get(
        "/auth/oauth/google/callback",
        params={"code": "dummy-code", "state": state_cookie},
        cookies={"oauth_state": state_cookie},
    )
    assert cb_resp.status_code == status.HTTP_200_OK
    data = cb_resp.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"
    assert data["email"] == "oauth.user@example.com"
