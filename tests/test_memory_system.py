"""Tests for the memory system (memory_models + memory_manager)."""

import pytest
import os
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch


class TestMemoryConfig:
    """Test MemoryConfig dataclass."""

    def test_default_config(self):
        from memory_models import MemoryConfig
        config = MemoryConfig()
        assert config.llm_model == "gpt-5.2"
        assert config.base_dir == Path.home() / '.openclaw'
        assert config.embedding.dimension == 1536
        assert config.chunking.chunk_tokens == 400

    def test_from_yaml(self):
        """Test loading from YAML file."""
        from memory_models import MemoryConfig
        config = MemoryConfig.from_yaml()
        # Should load from memory_config.yaml or fall back to defaults
        assert config.llm_model == "gpt-5.2"
        assert config.embedding.provider in ("auto", "openai", "local", "gemini")
        assert config.search.min_score > 0

    def test_from_yaml_missing_file(self):
        """Should return defaults for missing YAML."""
        from memory_models import MemoryConfig
        config = MemoryConfig.from_yaml("/nonexistent/path.yaml")
        assert config.llm_model == "gpt-5.2"

    def test_memory_dir_property(self):
        from memory_models import MemoryConfig
        config = MemoryConfig(base_dir=Path("/tmp/test_openclaw"))
        assert config.memory_dir == Path("/tmp/test_openclaw/memory")
        assert config.daily_dir == Path("/tmp/test_openclaw/memory/daily")
        assert config.sessions_dir == Path("/tmp/test_openclaw/sessions")

    def test_to_dict(self):
        from memory_models import MemoryConfig
        config = MemoryConfig()
        d = config.to_dict()
        assert 'base_dir' in d
        assert 'llm_model' in d
        assert 'embedding' in d
        assert 'search' in d


class TestEmbeddingConfig:
    """Test EmbeddingConfig."""

    def test_default_model_name(self):
        from memory_models import EmbeddingConfig
        config = EmbeddingConfig(provider='openai')
        assert config.get_model_name() == 'text-embedding-3-small'

    def test_custom_model_name(self):
        from memory_models import EmbeddingConfig
        config = EmbeddingConfig(model_name='custom-model')
        assert config.get_model_name() == 'custom-model'


class TestMemoryChunk:
    """Test MemoryChunk."""

    def test_content_hash(self):
        from memory_models import MemoryChunk
        chunk = MemoryChunk(
            content="test content",
            line_start=0, line_end=10,
            chunk_index=0, total_chunks=1,
        )
        assert chunk.content_hash != ""
        assert len(chunk.content_hash) == 64  # SHA256


class TestMemoryTypes:
    """Test memory type enums."""

    def test_memory_type_values(self):
        from memory_models import MemoryType
        assert MemoryType.EPISODIC.value == "episodic"
        assert MemoryType.SEMANTIC.value == "semantic"
        assert MemoryType.PROCEDURAL.value == "procedural"
        assert MemoryType.PREFERENCE.value == "preference"

    def test_memory_priority_ordering(self):
        from memory_models import MemoryPriority
        assert MemoryPriority.CRITICAL.value > MemoryPriority.HIGH.value
        assert MemoryPriority.HIGH.value > MemoryPriority.NORMAL.value
