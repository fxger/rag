"""Tests for document parsers."""
import pytest
import tempfile
import os
from unittest.mock import MagicMock, patch


class TestPDFParser:
    """Test PDF document parser."""

    @pytest.mark.asyncio
    async def test_parse_returns_parsed_document(self):
        from src.core.document.parsers.pdf_parser import PDFParser
        parser = PDFParser()
        # Mock pdfplumber
        with patch("src.core.document.parsers.pdf_parser.pdfplumber") as mock_pdf:
            mock_page = MagicMock()
            mock_page.extract_text.return_value = "Sample PDF text content"
            mock_doc = MagicMock()
            mock_doc.pages = [mock_page]
            mock_doc.metadata = {"Title": "Test PDF", "Author": "Test"}
            mock_doc.__enter__ = MagicMock(return_value=mock_doc)
            mock_doc.__exit__ = MagicMock(return_value=False)
            mock_pdf.open.return_value = mock_doc

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(b"%PDF-1.4 fake content")
                tmp_path = f.name

            try:
                result = await parser.parse(tmp_path)
                assert result.content is not None
                assert len(result.content) > 0
                assert result.page_count == 1
            finally:
                os.unlink(tmp_path)


class TestTextParser:
    """Test plain text parser."""

    @pytest.mark.asyncio
    async def test_parse_plain_text(self):
        from src.core.document.parsers.text_parser import TextParser
        parser = TextParser()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("Hello World\nThis is a test document.\n")
            tmp_path = f.name

        try:
            result = await parser.parse(tmp_path)
            assert "Hello World" in result.content
            assert "test document" in result.content
        finally:
            os.unlink(tmp_path)


class TestMarkdownParser:
    """Test Markdown parser."""

    @pytest.mark.asyncio
    async def test_parse_markdown(self):
        from src.core.document.parsers.markdown_parser import MarkdownParser
        parser = MarkdownParser()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write("# Title\n\nSome **bold** text.\n\n## Section\n\nMore content.\n")
            tmp_path = f.name

        try:
            result = await parser.parse(tmp_path)
            assert "Title" in result.content
            assert len(result.content) > 0
        finally:
            os.unlink(tmp_path)