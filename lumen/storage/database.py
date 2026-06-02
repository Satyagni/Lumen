import sqlite3
import os
from pathlib import Path
from lumen.core.logger import logger

class DatabaseManager:
    """Manages the lightweight SQLite storage for application settings."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.initialize_database()

    def get_connection(self):
        """Creates and returns a sqlite3 connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def initialize_database(self):
        """Creates the app_settings table if it doesn't exist."""
        logger.info("Initializing database at: %s", self.db_path)
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS app_settings (
                    setting_key TEXT PRIMARY KEY,
                    setting_value TEXT
                )
            """)
            conn.commit()
            logger.info("Database schema initialized successfully.")
        except Exception as e:
            logger.error("Failed to initialize database: %s", e, exc_info=True)
        finally:
            if conn:
                conn.close()

    def get_setting(self, key: str, default: str = None) -> str:
        """Retrieves a setting value from the app_settings table."""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT setting_value FROM app_settings WHERE setting_key = ?",
                (key,)
            )
            row = cursor.fetchone()
            if row:
                return row["setting_value"]
        except Exception as e:
            logger.error("Error fetching setting '%s': %s", key, e)
        finally:
            if conn:
                conn.close()
        return default

    def set_setting(self, key: str, value: str):
        """Saves or updates a setting in the app_settings table."""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO app_settings (setting_key, setting_value)
                VALUES (?, ?)
                ON CONFLICT(setting_key) DO UPDATE SET setting_value = excluded.setting_value
                """,
                (key, str(value))
            )
            conn.commit()
            logger.debug("Database setting saved: %s = %s", key, value)
        except Exception as e:
            logger.error("Error setting '%s' to '%s': %s", key, value, e)
        finally:
            if conn:
                conn.close()
            
# Global instance placeholder to be configured at launcher boot
db = None

def init_db(db_path: Path):
    global db
    db = DatabaseManager(db_path)
    return db

