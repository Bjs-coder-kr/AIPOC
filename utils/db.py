# utils/db.py
"""
SQLite Database Manager for DocuMind.
Handles caching, history, and settings persistence.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Any, Optional, List, Dict
import threading

logger = logging.getLogger(__name__)

DB_PATH = Path("documind.db")

class SQLiteManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(SQLiteManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.db_path = DB_PATH
        self._init_db()
        self._initialized = True
        logger.info(f"💾 SQLite DB initialized at {self.db_path.absolute()}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get a thread-local connection."""
        # SQLite connections are not thread-safe by default, but we use them in short bursts.
        # Ideally, use a connection pool or thread-local storage if concurrency is high.
        # For this single-user Streamlit app, creating a new connection per request is acceptable mostly,
        # but let's just make a new connection to be safe and simple.
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize database schema."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            
            # 1. Embeddings Cache Table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                text_hash TEXT PRIMARY KEY,
                text TEXT,
                model TEXT,
                vector BLOB,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # 2. Analysis History Table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                file_hash TEXT,
                report_json TEXT,  -- JSON string
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # 3. Settings Table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # 4. Users Table (Authentication)
            cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # 5. Add user_id column to analysis_history if not exists
            cur.execute("PRAGMA table_info(analysis_history)")
            columns = [row[1] for row in cur.fetchall()]
            if "user_id" not in columns:
                cur.execute("ALTER TABLE analysis_history ADD COLUMN user_id TEXT DEFAULT 'anonymous'")
            
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to initialize DB: {e}")
        finally:
            conn.close()

    # ═══════════════════════════════════════════════════════════════════
    # Embeddings Cache
    # ═══════════════════════════════════════════════════════════════════

    def get_cached_embedding(self, text_hash: str, model: str) -> Optional[List[float]]:
        """Retrieve cached embedding vector."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT vector FROM embeddings WHERE text_hash = ? AND model = ?", 
                (text_hash, model)
            )
            row = cur.fetchone()
            if row:
                return json.loads(row["vector"])
            return None
        except Exception as e:
            logger.error(f"DB Error (get_cached_embedding): {e}")
            return None
        finally:
            conn.close()

    def save_embedding(self, text_hash: str, text: str, vector: List[float], model: str):
        """Save embedding vector to cache."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            vector_json = json.dumps(vector)
            cur.execute(
                """
                INSERT OR IGNORE INTO embeddings (text_hash, text, model, vector)
                VALUES (?, ?, ?, ?)
                """,
                (text_hash, text, model, vector_json)
            )
            conn.commit()
        except Exception as e:
            logger.error(f"DB Error (save_embedding): {e}")
        finally:
            conn.close()

    # ═══════════════════════════════════════════════════════════════════
    # Settings
    # ═══════════════════════════════════════════════════════════════════

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a setting value."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT value FROM settings WHERE key = ?", (key,))
            row = cur.fetchone()
            if row:
                return row["value"]
            return default
        except Exception as e:
            logger.error(f"DB Error (get_setting): {e}")
            return default
        finally:
            conn.close()

    def save_setting(self, key: str, value: str):
        """Save a setting value."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT OR REPLACE INTO settings (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                """,
                (key, value)
            )
            conn.commit()
        except Exception as e:
            logger.error(f"DB Error (save_setting): {e}")
        finally:
            conn.close()

    # ═══════════════════════════════════════════════════════════════════
    # Analysis History
    # ═══════════════════════════════════════════════════════════════════

    def save_history(self, filename: str, file_hash: str, report: Dict[str, Any]):
        """Save analysis report history."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            report_json = json.dumps(report, ensure_ascii=False)
            cur.execute(
                """
                INSERT INTO analysis_history (filename, file_hash, report_json)
                VALUES (?, ?, ?)
                """,
                (filename, file_hash, report_json)
            )
            conn.commit()
        except Exception as e:
            logger.error(f"DB Error (save_history): {e}")
        finally:
            conn.close()

    def get_recent_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent analysis history headers."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, filename, created_at 
                FROM analysis_history 
                ORDER BY created_at DESC 
                LIMIT ?
                """,
                (limit,)
            )
            rows = cur.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"DB Error (get_recent_history): {e}")
            return []
        finally:
            conn.close()
            
    def get_history_detail(self, history_id: int) -> Optional[Dict[str, Any]]:
        """Get full report for a history item."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT report_json FROM analysis_history WHERE id = ?", (history_id,))
            row = cur.fetchone()
            if row:
                return json.loads(row["report_json"])
            return None
        except Exception as e:
            logger.error(f"DB Error (get_history_detail): {e}")
            return None
        finally:
            conn.close()

    # ═══════════════════════════════════════════════════════════════════
    # Authentication
    # ═══════════════════════════════════════════════════════════════════

    def register_user(self, username: str, password: str, role: str = "user") -> bool:
        """Register a new user with hashed password."""
        import hashlib
        import secrets
        
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            # Check if user exists
            cur.execute("SELECT username FROM users WHERE username = ?", (username,))
            if cur.fetchone():
                return False  # User already exists
            
            # Generate salt and hash password
            salt = secrets.token_hex(16)
            password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
            
            cur.execute(
                "INSERT INTO users (username, password_hash, salt, role) VALUES (?, ?, ?, ?)",
                (username, password_hash, salt, role)
            )
            conn.commit()
            logger.info(f"✅ User registered: {username} (role: {role})")
            return True
        except Exception as e:
            logger.error(f"DB Error (register_user): {e}")
            return False
        finally:
            conn.close()

    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user and return user info if successful."""
        import hashlib
        
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT username, password_hash, salt, role FROM users WHERE username = ?", (username,))
            row = cur.fetchone()
            if not row:
                return None  # User not found
            
            # Verify password
            expected_hash = hashlib.sha256((password + row["salt"]).encode()).hexdigest()
            if expected_hash == row["password_hash"]:
                return {"username": row["username"], "role": row["role"]}
            return None  # Wrong password
        except Exception as e:
            logger.error(f"DB Error (authenticate_user): {e}")
            return None
        finally:
            conn.close()

    def get_user_history(self, username: str, is_admin: bool = False, limit: int = 10) -> List[Dict[str, Any]]:
        """Get analysis history filtered by user (admin sees all)."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            if is_admin:
                cur.execute(
                    "SELECT id, filename, user_id, created_at FROM analysis_history ORDER BY created_at DESC LIMIT ?",
                    (limit,)
                )
            else:
                cur.execute(
                    "SELECT id, filename, user_id, created_at FROM analysis_history WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                    (username, limit)
                )
            rows = cur.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"DB Error (get_user_history): {e}")
            return []
        finally:
            conn.close()

    def save_history_with_user(self, filename: str, file_hash: str, report: Dict[str, Any], user_id: str):
        """Save analysis report history with user ownership."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            report_json = json.dumps(report, ensure_ascii=False)
            cur.execute(
                """
                INSERT INTO analysis_history (filename, file_hash, report_json, user_id)
                VALUES (?, ?, ?, ?)
                """,
                (filename, file_hash, report_json, user_id)
            )
            conn.commit()
        except Exception as e:
            logger.error(f"DB Error (save_history_with_user): {e}")
        finally:
            conn.close()

    def get_all_users(self) -> List[Dict[str, Any]]:
        """Get all registered users (admin use only)."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT username, role, created_at FROM users ORDER BY created_at DESC")
            rows = cur.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"DB Error (get_all_users): {e}")
            return []
        finally:
            conn.close()

# Global Instance
db_manager = SQLiteManager()
