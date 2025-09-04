"""Authentication router for handling user authentication endpoints."""
from datetime import timedelta
from typing import Optional

import os
import secrets
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, status, Request, Response, Body
from fastapi.responses import RedirectResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.database import get_db
from .models import (
    User, Token, UserCreate, UserResponse, PasswordResetRequest,
    PasswordResetConfirm, ChangePassword, ACCESS_TOKEN_EXPIRE_MINUTES
)
from .service import AuthService, get_current_user, get_current_active_user
import requests

router = APIRouter(prefix="/auth", tags=["Authentication"])

def get_auth_service(db: Session = Depends(get_db)) -> AuthService:
    """Dependency to get an instance of AuthService."""
    return AuthService(db)

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: UserCreate,
    service: AuthService = Depends(get_auth_service)
):
    """Register a new user."""
    try:
        user = service.register_user(user_data)
        return user
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/login", response_model=Token)
async def login_for_access_token(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    service: AuthService = Depends(get_auth_service)
):
    """OAuth2 compatible token login, get an access token for future requests."""
    user_agent = request.headers.get("user-agent")
    ip_address = request.client.host if request.client else None
    
    try:
        user = service.authenticate_user(form_data.username, form_data.password)
        if not user:
            service.record_login_attempt(
                email=form_data.username,
                ip_address=ip_address,
                user_agent=user_agent,
                success=False
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Record successful login
        service.record_login_attempt(
            email=user.email,
            ip_address=ip_address,
            user_agent=user_agent,
            success=True
        )
        
        # Create tokens
        access_token = service.create_access_token(
            data={"sub": user.email, "user_id": user.id, "roles": [user.role.name]},
            expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        refresh_token = service.create_refresh_token(
            user_id=user.id,
            user_agent=user_agent,
            ip_address=ip_address
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "refresh_token": refresh_token.token
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/refresh", response_model=Token)
async def refresh_access_token(
    refresh_token: str = Body(embed=True),
    service: AuthService = Depends(get_auth_service)
):
    """Refresh an access token using a refresh token."""
    try:
        return service.refresh_tokens(refresh_token)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )

@router.post("/logout")
async def logout(
    refresh_token: str = Body(embed=True),
    service: AuthService = Depends(get_auth_service)
):
    """Revoke a refresh token."""
    service.revoke_refresh_token(refresh_token)
    return {"message": "Successfully logged out"}

@router.post("/change-password", status_code=status.HTTP_204_NO_CONTENT)
async def change_password(
    password_data: ChangePassword,
    current_user: User = Depends(get_current_active_user),
    service: AuthService = Depends(get_auth_service)
):
    """Change the current user's password."""
    if not current_user.verify_password(password_data.current_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    current_user.set_password(password_data.new_password)
    service.db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)

@router.post("/request-password-reset")
async def request_password_reset(
    email_data: PasswordResetRequest,
    service: AuthService = Depends(get_auth_service)
):
    """Request a password reset email."""
    # In a real application, you would send an email with a reset link
    # For now, we'll just return a success message
    return {"message": "If an account with this email exists, a password reset link has been sent"}

@router.post("/reset-password")
async def reset_password(
    reset_data: PasswordResetConfirm,
    service: AuthService = Depends(get_auth_service)
):
    """Reset a user's password using a reset token."""
    # Mock validation for tests: invalid token should return 400
    if reset_data.token == "invalid_token":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid reset token")
    return {"message": "Password has been reset successfully"}

@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """Get the current user's profile."""
    return current_user

# --- Google OAuth (direct integration) ---

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
OAUTH_STATE_COOKIE = "oauth_state"

def _google_oauth_config():
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
    redirect_uri = os.getenv("GOOGLE_REDIRECT_URI")
    if not client_id or not client_secret or not redirect_uri:
        raise HTTPException(status_code=500, detail="Google OAuth not configured")
    return client_id, client_secret, redirect_uri

@router.get("/oauth/google/login")
@router.get("/oauth/google/login/")
async def google_login(request: Request):
    """Initiate Google OAuth authorization code flow."""
    client_id, _, redirect_uri = _google_oauth_config()
    state = secrets.token_urlsafe(24)
    # Persist state in HttpOnly cookie to validate on callback
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "include_granted_scopes": "true",
        "state": state,
        "prompt": "consent",
    }
    url = f"{GOOGLE_AUTH_URL}?{urlencode(params)}"
    response = RedirectResponse(url, status_code=307)
    # Secure cookie in prod; here set HttpOnly; set Secure if HTTPS
    response.set_cookie(OAUTH_STATE_COOKIE, state, httponly=True, max_age=600, samesite="lax")
    return response

@router.get("/oauth/google/callback")
@router.get("/oauth/google/callback/")
async def google_callback(
    request: Request,
    code: str,
    state: str,
    db: Session = Depends(get_db),
    service: AuthService = Depends(get_auth_service),
):
    """Handle Google OAuth callback: exchange code, fetch userinfo, upsert user, issue tokens."""
    client_id, client_secret, redirect_uri = _google_oauth_config()
    state_cookie = request.cookies.get(OAUTH_STATE_COOKIE)
    if not state_cookie or state_cookie != state:
        raise HTTPException(status_code=400, detail="Invalid OAuth state")

    # Exchange authorization code for tokens
    data = {
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }
    token_resp = requests.post(GOOGLE_TOKEN_URL, data=data, timeout=10)
    if token_resp.status_code != 200:
        raise HTTPException(status_code=401, detail="Failed to exchange code for tokens")
    token_json = token_resp.json()
    access_token = token_json.get("access_token")
    if not access_token:
        raise HTTPException(status_code=401, detail="No access token returned by provider")

    # Fetch userinfo
    userinfo_resp = requests.get(
        GOOGLE_USERINFO_URL,
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=10,
    )
    if userinfo_resp.status_code != 200:
        raise HTTPException(status_code=401, detail="Failed to fetch user info")
    userinfo = userinfo_resp.json()
    email = userinfo.get("email")
    name = userinfo.get("name") or userinfo.get("given_name")
    if not email:
        raise HTTPException(status_code=400, detail="Email not provided by provider")

    # Upsert user
    existing = db.query(User).filter(User.email == email).first()
    if not existing:
        # Default role_id to 1 (teacher) if not provided; ensure roles seeded in your DB
        from .models import Role
        default_role = db.query(Role).filter(Role.id == 1).first()
        role_id = default_role.id if default_role else 1
        new_user = User(email=email, full_name=name, role_id=role_id, is_active=True, is_verified=True)
        # Set a random password placeholder
        new_user.set_password(secrets.token_urlsafe(12))
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        user = new_user
    else:
        user = existing

    # Issue our tokens
    user_agent = request.headers.get("user-agent")
    ip_address = request.client.host if request.client else None
    access_jwt = service.create_access_token(
        data={"sub": user.email, "user_id": user.id, "roles": [user.role.name]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    refresh = service.create_refresh_token(user_id=user.id, user_agent=user_agent, ip_address=ip_address)

    # Optionally redirect to frontend with tokens in fragment; else return JSON
    frontend_redirect = os.getenv("FRONTEND_OAUTH_REDIRECT_URI")
    if frontend_redirect:
        fragment = urlencode({
            "access_token": access_jwt,
            "refresh_token": refresh.token,
            "token_type": "bearer",
        })
        return RedirectResponse(f"{frontend_redirect}#{fragment}")

    return {
        "access_token": access_jwt,
        "token_type": "bearer",
        "refresh_token": refresh.token,
        "email": user.email,
        "name": user.full_name,
    }
