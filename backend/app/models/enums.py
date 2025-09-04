"""Shared enums for models and auth."""
import enum

class UserRole(enum.Enum):
    teacher = "teacher"
    admin = "admin"
    district = "district"
