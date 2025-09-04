"""Tests for the authentication system."""
import os
import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

# Import app after setting up the test database to avoid table redefinition
from app.database import Base
from app.auth.models import User, RefreshToken, Role, Permission, RolePermission, UserCreate
from app.auth.service import AuthService

# Create a fresh SQLAlchemy engine and session for testing
@pytest.fixture(scope="session")
def engine():
    # Use a file-based SQLite database to avoid in-memory issues
    test_db_path = "test.db"
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    return create_engine(f"sqlite:///{test_db_path}", connect_args={"check_same_thread": False})

# Create all tables once
@pytest.fixture(scope="session", autouse=True)
def setup_database(engine):
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    # Add initial test data
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    
    try:
        # Add test role if not exists
        if not db.query(Role).filter(Role.id == 1).first():
            role = Role(id=1, name="teacher", description="Test Role")
            db.add(role)
        
        # Add test permission if not exists
        if not db.query(Permission).filter(Permission.id == 1).first():
            permission = Permission(id=1, name="test_permission", description="Test Permission")
            db.add(permission)
            
        # Add role-permission association if not exists
        if not db.query(RolePermission).filter_by(role_id=1, permission_id=1).first():
            role_permission = RolePermission(role_id=1, permission_id=1)
            db.add(role_permission)
            
        db.commit()
    finally:
        db.close()
    
    yield  # Testing happens here
    
    # Clean up after all tests
    Base.metadata.drop_all(bind=engine)
    # Dispose connections before removing file (Windows file lock)
    try:
        engine.dispose()
    except Exception:
        pass
    if os.path.exists("test.db"):
        try:
            os.remove("test.db")
        except PermissionError:
            # Retry after a short delay if needed
            import time
            time.sleep(0.2)
            try:
                os.remove("test.db")
            except Exception:
                pass

# Create a session for testing
@pytest.fixture
def db_session(engine):
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

# Override the get_db dependency for FastAPI app
@pytest.fixture
def client(engine):
    from app.main import app
    from app.database import get_db
    
    # Create test database and tables
    Base.metadata.create_all(bind=engine)
    
    # Override get_db dependency
    def override_get_db():
        try:
            db = sessionmaker(autocommit=False, autoflush=False, bind=engine)()
            yield db
        finally:
            db.close()
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client
    
    # Clean up
    app.dependency_overrides.clear()

# Test data
TEST_USER = {
    "email": "test@example.com",
    "password": "TestPass123!",
    "full_name": "Test User",
    "role_id": 1
}

@pytest.fixture
def auth_service(db_session):
    return AuthService(db_session)

@pytest.fixture
def test_user(db_session, auth_service):
    # Create a test user
    existing = db_session.query(User).filter(User.email == TEST_USER["email"]).first()
    if existing:
        return existing
    user = auth_service.register_user(
        UserCreate(
            email=TEST_USER["email"],
            password=TEST_USER["password"],
            full_name=TEST_USER["full_name"],
            role_id=TEST_USER["role_id"]
        )
    )
    db_session.commit()
    return user

# Test cases
def test_register_user(client, auth_service):
    """Test user registration."""
    # Test successful registration
    response = client.post(
        "/auth/register",
        json={
            "email": "newuser@example.com",
            "password": "NewPass123!",
            "full_name": "New User",
            "role_id": 1
        }
    )
    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert "id" in data
    assert data["email"] == "newuser@example.com"
    assert "hashed_password" not in data  # Password should not be returned
    
    # Test duplicate email
    response = client.post(
        "/auth/register",
        json={
            "email": "newuser@example.com",
            "password": "AnotherPass123!",
            "full_name": "Another User",
            "role_id": 1
        }
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST

def test_login(client, test_user):
    """Test user login and token generation."""
    # Test successful login
    response = client.post(
        "/auth/login",
        data={
            "username": TEST_USER["email"],
            "password": TEST_USER["password"],
        },
        headers={"content-type": "application/x-www-form-urlencoded"}
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"
    
    # Test invalid credentials
    response = client.post(
        "/auth/login",
        data={
            "username": TEST_USER["email"],
            "password": "wrongpassword",
        },
        headers={"content-type": "application/x-www-form-urlencoded"}
    )
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

def test_refresh_token(client, test_user):
    """Test token refresh functionality."""
    # First login to get refresh token
    login_response = client.post(
        "/auth/login",
        data={
            "username": TEST_USER["email"],
            "password": TEST_USER["password"],
        },
        headers={"content-type": "application/x-www-form-urlencoded"}
    )
    refresh_token = login_response.json()["refresh_token"]
    
    # Test refresh token
    response = client.post(
        "/auth/refresh",
        json={"refresh_token": refresh_token}
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"
    
    # Test invalid refresh token
    response = client.post(
        "/auth/refresh",
        json={"refresh_token": "invalid_token"}
    )
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

def test_protected_endpoint(client, test_user):
    """Test access to protected endpoints."""
    # Get access token
    login_response = client.post(
        "/auth/login",
        data={
            "username": TEST_USER["email"],
            "password": TEST_USER["password"],
        },
        headers={"content-type": "application/x-www-form-urlencoded"}
    )
    access_token = login_response.json()["access_token"]
    
    # Test access with valid token
    response = client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {access_token}"}
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["email"] == TEST_USER["email"]
    
    # Test access with invalid token
    response = client.get(
        "/auth/me",
        headers={"Authorization": "Bearer invalid_token"}
    )
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

def test_password_reset(client, test_user):
    """Test password reset flow."""
    # Request password reset
    response = client.post(
        "/auth/request-password-reset",
        json={"email": TEST_USER["email"]}
    )
    assert response.status_code == status.HTTP_200_OK
    
    # In a real test, we would check the email and get the reset token
    # For now, we'll test with an invalid token
    response = client.post(
        "/auth/reset-password",
        json={
            "token": "invalid_token",
            "new_password": "NewPass123!"
        }
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST

def test_change_password(client, test_user):
    """Test changing password."""
    # Login first
    login_response = client.post(
        "/auth/login",
        data={
            "username": TEST_USER["email"],
            "password": TEST_USER["password"],
        },
        headers={"content-type": "application/x-www-form-urlencoded"}
    )
    access_token = login_response.json()["access_token"]
    
    # Change password
    response = client.post(
        "/auth/change-password",
        json={
            "current_password": TEST_USER["password"],
            "new_password": "NewSecurePass123!"
        },
        headers={"Authorization": f"Bearer {access_token}"}
    )
    assert response.status_code == status.HTTP_204_NO_CONTENT
    
    # Verify new password works
    login_response = client.post(
        "/auth/login",
        data={
            "username": TEST_USER["email"],
            "password": "NewSecurePass123!",
        },
        headers={"content-type": "application/x-www-form-urlencoded"}
    )
    assert login_response.status_code == status.HTTP_200_OK
    
    # Change password back for other tests
    client.post(
        "/auth/change-password",
        json={
            "current_password": "NewSecurePass123!",
            "new_password": TEST_USER["password"]
        },
        headers={"Authorization": f"Bearer {login_response.json()['access_token']}"}
    )

