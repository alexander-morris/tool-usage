"""
Conversation history management with persistent storage.
"""

from typing import Dict, List, Optional, Union
from pathlib import Path
import json
import zlib
import shutil
from datetime import datetime
import sqlite3
import tiktoken
from threading import Lock


def adapt_datetime(dt):
    """SQLite adapter for datetime objects."""
    return dt.isoformat().encode("utf-8")


def convert_datetime(val):
    """SQLite converter for datetime strings."""
    if isinstance(val, bytes):
        val = val.decode("utf-8")
    return datetime.fromisoformat(val)


class ConversationHistoryManager:
    """Manages conversation history with persistent storage."""

    def __init__(self, storage_dir: Union[str, Path] = "conversations"):
        """Initialize the history manager.

        Args:
            storage_dir: Directory for storing conversation data
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.db_path = self.storage_dir / "conversations.db"
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.lock = Lock()

        # Initialize database
        self._init_db()

    def _get_connection(self):
        """Get a SQLite connection with datetime handling."""
        return sqlite3.connect(
            self.db_path,
            detect_types=sqlite3.PARSE_DECLTYPES,
            isolation_level=None,  # Enable autocommit mode
        )

    def _init_db(self):
        """Initialize the database with required tables."""
        sqlite3.register_adapter(datetime, adapt_datetime)
        sqlite3.register_converter("datetime", convert_datetime)

        # Create tables if they don't exist
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at datetime NOT NULL,
                    last_updated datetime NOT NULL,
                    metadata TEXT
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content BLOB,
                    timestamp datetime NOT NULL,
                    token_count INTEGER,
                    metadata TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                )
            """
            )

    def create_conversation(self, title: str = "", metadata: Dict = None) -> str:
        """Create a new conversation.

        Args:
            title: Optional conversation title
            metadata: Optional metadata dictionary

        Returns:
            Conversation ID
        """
        conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        now = datetime.now()

        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO conversations (id, title, created_at, last_updated, metadata) VALUES (?, ?, ?, ?, ?)",
                (conversation_id, title, now, now, json.dumps(metadata or {})),
            )

        return conversation_id

    def add_message(
        self, conversation_id: str, role: str, content: str, metadata: Dict = None
    ) -> int:
        """Add a message to a conversation.

        Args:
            conversation_id: ID of conversation to add to
            role: Message role (e.g. 'user', 'assistant')
            content: Message content
            metadata: Optional message metadata

        Returns:
            Message ID
        """
        # Count tokens
        token_count = len(self.encoder.encode(content))

        # Compress content
        compressed = sqlite3.Binary(zlib.compress(content.encode("utf-8")))
        now = datetime.now()

        with self._get_connection() as conn:
            # Update conversation last_updated
            conn.execute(
                "UPDATE conversations SET last_updated = ? WHERE id = ?",
                (now, conversation_id),
            )

            # Insert message
            cursor = conn.execute(
                """INSERT INTO messages
                   (conversation_id, timestamp, role, content, token_count, metadata)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    conversation_id,
                    now,
                    role,
                    compressed,
                    token_count,
                    json.dumps(metadata or {}),
                ),
            )

            return cursor.lastrowid

    def get_messages(
        self,
        conversation_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        max_tokens: Optional[int] = None,
    ) -> List[Dict]:
        """Get messages from a conversation.

        Args:
            conversation_id: Conversation to get messages from
            start_time: Optional start time filter
            end_time: Optional end time filter
            max_tokens: Optional maximum total tokens to return

        Returns:
            List of message dictionaries
        """
        query = "SELECT * FROM messages WHERE conversation_id = ?"
        params = [conversation_id]

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)

        query += " ORDER BY timestamp ASC"

        messages = []
        total_tokens = 0

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)

            for row in cursor:
                # Decompress content
                content = zlib.decompress(bytes(row[3])).decode("utf-8")

                message = {
                    "id": row[0],
                    "conversation_id": row[1],
                    "role": row[2],
                    "content": content,
                    "timestamp": row[4],
                    "token_count": row[5],
                    "metadata": json.loads(row[6]),
                }

                if max_tokens:
                    if total_tokens + message["token_count"] > max_tokens:
                        break
                    total_tokens += message["token_count"]

                messages.append(message)

        return messages

    def get_conversation_metadata(self, conversation_id: str) -> Dict:
        """Get conversation metadata.

        Args:
            conversation_id: Conversation ID

        Returns:
            Dictionary with conversation metadata
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT created_at, last_updated, title, metadata FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()

            if not row:
                raise ValueError(f"Conversation {conversation_id} not found")

            return {
                "created_at": row[0],
                "last_updated": row[1],
                "title": row[2],
                "metadata": json.loads(row[3]),
            }

    def list_conversations(
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> List[Dict]:
        """List all conversations in date range.

        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            List of conversation metadata dictionaries
        """
        query = "SELECT * FROM conversations"
        params = []

        if start_date:
            query += " WHERE created_at >= ?"
            params.append(start_date)
            if end_date:
                query += " AND created_at <= ?"
                params.append(end_date)
        elif end_date:
            query += " WHERE created_at <= ?"
            params.append(end_date)

        query += " ORDER BY created_at DESC"

        conversations = []
        with self._get_connection() as conn:
            for row in conn.execute(query, params):
                conversations.append(
                    {
                        "id": row[0],
                        "title": row[1],
                        "created_at": row[2],
                        "last_updated": row[3],
                        "metadata": json.loads(row[4]),
                    }
                )

        return conversations

    def delete_conversation(self, conversation_id: str):
        """Delete a conversation and all its messages.

        Args:
            conversation_id: ID of conversation to delete
        """
        with self._get_connection() as conn:
            conn.execute(
                "DELETE FROM messages WHERE conversation_id = ?", (conversation_id,)
            )
            conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))

    def prune_old_conversations(self, before_date: datetime):
        """Delete conversations older than specified date.

        Args:
            before_date: Delete conversations older than this date
        """
        with self._get_connection() as conn:
            # Get IDs of old conversations
            old_ids = [
                row[0]
                for row in conn.execute(
                    "SELECT id FROM conversations WHERE created_at < ?", (before_date,)
                )
            ]

            # Delete messages and conversations
            for conv_id in old_ids:
                self.delete_conversation(conv_id)

    def export_conversation(self, conversation_id: str, output_file: Union[str, Path]):
        """Export a conversation to JSON file.

        Args:
            conversation_id: Conversation to export
            output_file: Path to save JSON file
        """
        data = {
            "metadata": self.get_conversation_metadata(conversation_id),
            "messages": self.get_messages(conversation_id),
        }

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def import_conversation(self, input_file: Union[str, Path]) -> str:
        """Import a conversation from JSON file.

        Args:
            input_file: Path to JSON file

        Returns:
            ID of imported conversation
        """
        with open(input_file) as f:
            data = json.load(f)

        # Create new conversation
        conv_id = self.create_conversation(
            title=data["metadata"].get("title", ""),
            metadata=data["metadata"].get("metadata", {}),
        )

        # Add messages
        for msg in data["messages"]:
            self.add_message(
                conv_id, msg["role"], msg["content"], msg.get("metadata", {})
            )

        return conv_id

    def backup(self, backup_dir: Union[str, Path]):
        """Create backup of conversation database.

        Args:
            backup_dir: Directory to store backup
        """
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"conversations_{timestamp}.db"

        with self.lock:
            shutil.copy2(self.db_path, backup_path)

    def restore(self, backup_file: Union[str, Path]):
        """Restore conversation database from backup.

        Args:
            backup_file: Backup file to restore from
        """
        with self.lock:
            shutil.copy2(backup_file, self.db_path)
