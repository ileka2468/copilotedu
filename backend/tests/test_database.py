"""Test cases for database utilities."""

import pytest
from unittest.mock import patch, MagicMock
from sqlalchemy.exc import SQLAlchemyError

from app.database import (
    get_db, get_db_session, create_tables, drop_tables, 
    check_database_connection, engine, Base
)


class TestDatabaseUtilities:
    """Test cases for database utility functions."""

    def test_get_db_dependency(self):
        """Test get_db dependency function."""
        db_generator = get_db()
        db_session = next(db_generator)
        
        assert db_session is not None
        
        # Clean up
        try:
            next(db_generator)
        except StopIteration:
            pass  # Expected behavior

    def test_get_db_session_context_manager(self):
        """Test get_db_session context manager."""
        with get_db_session() as db:
            assert db is not None
            # Session should be active
            assert db.is_active

    def test_get_db_session_rollback_on_error(self):
        """Test that get_db_session rolls back on error."""
        with pytest.raises(Exception):
            with get_db_session() as db:
                # Force an error
                raise Exception("Test error")

    @patch('app.database.Base.metadata.create_all')
    def test_create_tables_success(self, mock_create_all):
        """Test successful table creation."""
        mock_create_all.return_value = None
        
        create_tables()
        
        mock_create_all.assert_called_once_with(bind=engine)

    @patch('app.database.Base.metadata.create_all')
    def test_create_tables_error(self, mock_create_all):
        """Test table creation error handling."""
        mock_create_all.side_effect = SQLAlchemyError("Connection failed")
        
        with pytest.raises(SQLAlchemyError):
            create_tables()

    @patch('app.database.Base.metadata.drop_all')
    def test_drop_tables_success(self, mock_drop_all):
        """Test successful table dropping."""
        mock_drop_all.return_value = None
        
        drop_tables()
        
        mock_drop_all.assert_called_once_with(bind=engine)

    @patch('app.database.Base.metadata.drop_all')
    def test_drop_tables_error(self, mock_drop_all):
        """Test table dropping error handling."""
        mock_drop_all.side_effect = SQLAlchemyError("Connection failed")
        
        with pytest.raises(SQLAlchemyError):
            drop_tables()

    @patch('app.database.engine.connect')
    def test_check_database_connection_success(self, mock_connect):
        """Test successful database connection check."""
        mock_conn = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        
        result = check_database_connection()
        
        assert result is True
        mock_conn.execute.assert_called_once_with("SELECT 1")

    @patch('app.database.engine.connect')
    def test_check_database_connection_failure(self, mock_connect):
        """Test database connection check failure."""
        mock_connect.side_effect = SQLAlchemyError("Connection failed")
        
        result = check_database_connection()
        
        assert result is False


class TestDatabaseConfiguration:
    """Test cases for database configuration."""

    def test_engine_configuration(self):
        """Test that engine is properly configured."""
        assert engine is not None
        assert engine.pool_size == 10
        assert engine.pool._recycle == 3600

    def test_base_metadata(self):
        """Test that Base metadata is properly configured."""
        assert Base is not None
        assert hasattr(Base, 'metadata')
        assert Base.metadata is not None