"""Document parser registry — maps file types to parser classes."""
from src.core.document.parsers.pdf_parser import PDFParser
from src.core.document.parsers.word_parser import WordParser
from src.core.document.parsers.excel_parser import ExcelParser
from src.core.document.parsers.markdown_parser import MarkdownParser
from src.core.document.parsers.text_parser import TextParser

PARSER_REGISTRY: dict[str, type] = {
    "pdf": PDFParser,
    "docx": WordParser,
    "xlsx": ExcelParser,
    "md": MarkdownParser,
    "txt": TextParser,
}

def get_parser(file_type: str):
    """Get parser instance for a given file type.
    
    Args:
        file_type: File extension without dot (e.g. "pdf", "docx").
        
    Returns:
        Instantiated parser for the requested type.
        
    Raises:
        ValueError: If the file type is not supported.
    """
    parser_cls = PARSER_REGISTRY.get(file_type.lower())
    if parser_cls is None:
        supported = ", ".join(sorted(PARSER_REGISTRY))
        raise ValueError(
            f"Unsupported file type: '{file_type}'. "
            f"Supported types: {supported}"
        )
    return parser_cls()
