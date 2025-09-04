"""Authentication service for handling user authentication and token management."""
from datetime import datetime, timedelta, UTC
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from app.database import get_db
from .models import (
    User, Token, TokenData, UserCreate, UserResponse,
    RefreshToken, LoginAttempt, Role, Permission, RolePermission,
    SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES, REFRESH_TOKEN_EXPIRE_DAYS
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

class AuthService:
    def __init__(self, db: Session):
        self.db = db

    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create a new access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.now(UTC) + expires_delta
        else:
            expire = datetime.now(UTC) + timedelta(minutes=15)
        to_encode.update({"exp": expire, "type": "access"})
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    def create_refresh_token(self, user_id: int, user_agent: str = None, ip_address: str = None) -> RefreshToken:
        """Create and store a new refresh token."""
        token = RefreshToken.generate_token()
        expires_at = datetime.now(UTC) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        
        db_token = RefreshToken(
            token=token,
            user_id=user_id,
            expires_at=expires_at,
            user_agent=user_agent,
            ip_address=ip_address
        )
        self.db.add(db_token)
        self.db.commit()
        self.db.refresh(db_token)
        return db_token

    def verify_token(self, token: str) -> TokenData:
        """Verify and decode a JWT token."""
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            email: str = payload.get("sub")
            user_id: int = payload.get("user_id")
            roles: list = payload.get("roles", [])
            if email is None or user_id is None:
                raise credentials_exception
            return TokenData(email=email, user_id=user_id, roles=roles)
        except JWTError:
            raise credentials_exception

    def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """Authenticate a user with email and password."""
        user = self.db.query(User).filter(User.email == email).first()
        if not user or not user.verify_password(password):
            return None
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Inactive user"
            )
        if user.is_locked():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is temporarily locked due to too many failed login attempts"
            )
        return user

    def register_user(self, user_data: UserCreate) -> User:
        """Register a new user."""
        # Check if user already exists
        if self.db.query(User).filter(User.email == user_data.email).first():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create new user
        user = User(
            email=user_data.email,
            full_name=user_data.full_name,
            role_id=user_data.role_id
        )
        user.set_password(user_data.password)
        
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    def refresh_tokens(self, refresh_token: str) -> Token:
        """Refresh access token using a valid refresh token."""
        db_token = self.db.query(RefreshToken).filter(
            RefreshToken.token == refresh_token,
            RefreshToken.revoked == False,
            RefreshToken.expires_at > datetime.now(UTC)
        ).first()
        
        if not db_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired refresh token"
            )
        
        user = self.db.query(User).filter(User.id == db_token.user_id).first()
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        
        # Create new access token
        access_token = self.create_access_token(
            data={"sub": user.email, "user_id": user.id, "roles": [user.role.name]},
            expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        
        return Token(
            access_token=access_token,
            refresh_token=db_token.token
        )

    def revoke_refresh_token(self, token: str) -> None:
        """Revoke a refresh token."""
        db_token = self.db.query(RefreshToken).filter(RefreshToken.token == token).first()
        if db_token:
            db_token.revoked = True
            self.db.commit()

    def record_login_attempt(self, email: str, ip_address: str, user_agent: str, success: bool) -> None:
        """Record a login attempt."""
        user = self.db.query(User).filter(User.email == email).first()
        if not user:
            return
            
        # Record the attempt
        attempt = LoginAttempt(
            user_id=user.id,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success
        )
        self.db.add(attempt)
        
        if success:
            user.failed_login_attempts = 0
            user.locked_until = None
            user.last_login = datetime.now(UTC)
        else:
            user.failed_login_attempts += 1
            # Lock account after 5 failed attempts for 30 minutes
            if user.failed_login_attempts >= 5:
                user.locked_until = datetime.now(UTC) + timedelta(minutes=30)
        
        self.db.commit()

def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """Dependency to get the current user from the JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("user_id")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise credentials_exception
    return user

def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Dependency to get the current active user."""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user
