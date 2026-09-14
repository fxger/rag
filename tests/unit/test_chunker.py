"""Tests for the SmartChunker."""
import pytest


class TestSmartChunker:

    def test_basic_sliding_window(self):
        from src.core.document.chunker import SmartChunker
        chunker = SmartChunker(chunk_size=50, chunk_overlap=10)
        text = "word " * 100  # 500 chars
        chunks = chunker.chunk(text, file_type="txt")
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.content
            assert chunk.index >= 0

    def test_empty_content(self):
        from src.core.document.chunker import SmartChunker
        chunker = SmartChunker()
        chunks = chunker.chunk("", file_type="txt")
        assert chunks == []

    def test_short_content_single_chunk(self):
        from src.core.document.chunker import SmartChunker
        chunker = SmartChunker(chunk_size=1000)
        chunks = chunker.chunk("Short text.", file_type="txt")
        assert len(chunks) == 1
        assert chunks[0].content == "Short text."

    def test_token_estimation(self):
        from src.core.document.chunker import SmartChunker
        chunker = SmartChunker()
        # English: ~4 chars per token
        assert chunker._estimate_tokens("abcd") >= 1
        # CJK: ~2 chars per token
        assert chunker._estimate_tokens("你好世界") >= 1

    def test_chunk_metadata_preserved(self):
        from src.core.document.chunker import SmartChunker
        chunker = SmartChunker(chunk_size=50)
        text = "A" * 200
        chunks = chunker.chunk(text, file_type="txt", metadata={"source": "test"})
        for chunk in chunks:
            assert chunk.metadata is not None
            assert chunk.metadata["source"] == "test"