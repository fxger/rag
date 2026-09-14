"""Excel document parser using openpyxl."""
import logging
from typing import Any
from pathlib import Path

from openpyxl import load_workbook

from src.core.document.parser import BaseDocumentParser, ParsedDocument

logger = logging.getLogger(__name__)


class ExcelParser(BaseDocumentParser):
    """Extract text and metadata from .xlsx files.
    
    Uses ``openpyxl`` to iterate every sheet and produce a textual
    representation suitable for RAG ingestion.
    """

    async def parse(self, file_path: str) -> ParsedDocument:
        """Parse an Excel workbook and return its content and metadata.
        
        Each sheet is rendered as a section with headers, followed by
        row data.  Empty sheets are skipped.
        
        Args:
            file_path: Path to the .xlsx file.
            
        Returns:
            ``ParsedDocument`` with extracted text, metadata and page count.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Excel file not found: {file_path}")

        try:
            wb = load_workbook(str(path), read_only=True, data_only=True)
        except Exception as exc:
            logger.error("Failed to open Excel file '%s': %s", file_path, exc)
            raise RuntimeError(f"Excel parsing failed: {exc}") from exc

        sheet_sections: list[str] = []
        sheet_names: list[str] = []
        total_rows = 0

        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            rows = list(ws.iter_rows(values_only=True))
            if not rows:
                continue
            sheet_names.append(sheet_name)

            section_lines = [f"=== Sheet: {sheet_name} ==="]
            for row_idx, row in enumerate(rows):
                # Convert None values to empty strings for readability
                cells = [str(cell) if cell is not None else "" for cell in row]
                line = " | ".join(cells)
                section_lines.append(line)
            total_rows += len(rows)
            sheet_sections.append("\n".join(section_lines))

        wb.close()

        raw_text = "\n\n".join(sheet_sections)
        metadata: dict[str, Any] = {
            "sheet_count": len(sheet_names),
            "sheet_names": sheet_names,
            "total_rows": total_rows,
        }

        return ParsedDocument(
            content=raw_text,
            metadata=metadata,
            page_count=len(sheet_names),
            raw_text=raw_text,
        )

    async def extract_metadata(self, file_path: str) -> dict[str, Any]:
        """Extract metadata only, without parsing cell data.
        
        Args:
            file_path: Path to the .xlsx file.
            
        Returns:
            Dictionary of workbook metadata fields.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Excel file not found: {file_path}")

        try:
            wb = load_workbook(str(path), read_only=True, data_only=True)
            names = list(wb.sheetnames)
            wb.close()
            return {
                "sheet_count": len(names),
                "sheet_names": names,
            }
        except Exception as exc:
            logger.error("Failed to extract Excel metadata '%s': %s", file_path, exc)
            raise RuntimeError(f"Excel metadata extraction failed: {exc}") from exc
