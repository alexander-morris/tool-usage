"""Tests for conversation history management."""

import pytest
from datetime import datetime
from pathlib import Path
import tempfile
import shutil
from src.history_manager import ConversationHistoryManager


@pytest.fixture
def temp_storage():
    """Create temporary storage directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def history_manager(temp_storage):
    """Create history manager with temporary storage."""
    return ConversationHistoryManager(temp_storage)


def test_create_conversation(history_manager):
    """Test creating a new conversation."""
    title = "Test Conversation"
    metadata = {"test_key": "test_value"}

    conv_id = history_manager.create_conversation(title, metadata)

    assert conv_id is not None
    meta = history_manager.get_conversation_metadata(conv_id)
    assert meta["title"] == title
    assert meta["metadata"]["test_key"] == "test_value"


def test_add_message(history_manager):
    """Test adding messages to a conversation."""
    conv_id = history_manager.create_conversation()

    # Add messages
    msg1_id = history_manager.add_message(conv_id, "user", "Hello")
    msg2_id = history_manager.add_message(conv_id, "assistant", "Hi there")

    assert msg1_id is not None
    assert msg2_id is not None

    # Verify messages
    messages = history_manager.get_messages(conv_id)
    assert len(messages) == 2
    assert messages[0]["content"] == "Hello"
    assert messages[1]["content"] == "Hi there"


def test_get_messages_with_filters(history_manager):
    """Test getting messages with time and token filters."""
    conv_id = history_manager.create_conversation()

    # Add messages at different times
    history_manager.add_message(conv_id, "user", "Message 1")
    mid_time = datetime.now()
    history_manager.add_message(conv_id, "assistant", "Message 2")
    end_time = datetime.now()

    # Test time filters
    messages = history_manager.get_messages(
        conv_id, start_time=mid_time, end_time=end_time
    )
    assert len(messages) == 1
    assert messages[0]["content"] == "Message 2"

    # Test token limit
    messages = history_manager.get_messages(conv_id, max_tokens=10)
    assert len(messages) <= 2  # May be less depending on token count


def test_list_conversations(history_manager):
    """Test listing conversations with filters."""
    # Create conversations at different times
    history_manager.create_conversation("Conv 1")
    start_date = datetime.now()
    history_manager.create_conversation("Conv 2")
    end_date = datetime.now()
    history_manager.create_conversation("Conv 3")

    # Test date filters
    convs = history_manager.list_conversations(start_date=start_date, end_date=end_date)
    assert len(convs) == 1
    assert convs[0]["title"] == "Conv 2"


def test_delete_conversation(history_manager):
    """Test deleting a conversation."""
    conv_id = history_manager.create_conversation()
    history_manager.add_message(conv_id, "user", "Test message")

    # Delete conversation
    history_manager.delete_conversation(conv_id)

    # Verify deletion
    with pytest.raises(ValueError):
        history_manager.get_conversation_metadata(conv_id)
    assert history_manager.get_messages(conv_id) == []


def test_prune_old_conversations(history_manager):
    """Test pruning old conversations."""
    # Create old and new conversations
    history_manager.create_conversation("Old")
    cutoff_date = datetime.now()
    new_id = history_manager.create_conversation("New")

    # Prune old conversations
    history_manager.prune_old_conversations(cutoff_date)

    # Verify pruning
    convs = history_manager.list_conversations()
    assert len(convs) == 1
    assert convs[0]["id"] == new_id


def test_export_import_conversation(history_manager, temp_storage):
    """Test exporting and importing conversations."""
    # Create conversation with messages
    conv_id = history_manager.create_conversation("Export Test")
    history_manager.add_message(conv_id, "user", "Test message")

    # Export conversation
    export_path = Path(temp_storage) / "export.json"
    history_manager.export_conversation(conv_id, export_path)

    # Import conversation
    new_id = history_manager.import_conversation(export_path)

    # Verify imported conversation
    messages = history_manager.get_messages(new_id)
    assert len(messages) == 1
    assert messages[0]["content"] == "Test message"


def test_backup_restore(history_manager, temp_storage):
    """Test backup and restore functionality."""
    # Create conversation with messages
    conv_id = history_manager.create_conversation("Backup Test")
    history_manager.add_message(conv_id, "user", "Test message")

    # Create backup
    backup_dir = Path(temp_storage) / "backups"
    history_manager.backup(backup_dir)

    # Create new manager and restore backup
    backup_file = next(backup_dir.glob("conversations_*.db"))
    new_manager = ConversationHistoryManager(temp_storage + "_restored")
    new_manager.restore(backup_file)

    # Verify restored data
    messages = new_manager.get_messages(conv_id)
    assert len(messages) == 1
    assert messages[0]["content"] == "Test message"
