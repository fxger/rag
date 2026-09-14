"""Content cleaning and normalization utilities."""
import re
import html
import unicodedata
from bs4 import BeautifulSoup


class ContentCleaner:
    """Clean and normalize extracted text content.
    
    The ``clean`` method runs the full pipeline:
    
    1. ``strip_html`` — remove HTML/XML tags.
    2. ``normalize_encoding`` — NFC-normalize Unicode and collapse
       control characters.
    3. ``normalize_whitespace`` — collapse runs of whitespace and
       blank lines.
    """

    # Control characters (except tab, newline, carriage return)
    _CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

    # Multiple blank lines → single blank line
    _MULTI_BLANK_RE = re.compile(r"\n{3,}")

    # Multiple spaces → single space
    _MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean(self, text: str) -> str:
        """Full cleaning pipeline.
        
        Args:
            text: Raw text potentially containing HTML, odd encoding,
                or excessive whitespace.
                
        Returns:
            Cleaned, normalized text.
        """
        text = self.strip_html(text)
        text = self.normalize_encoding(text)
        text = self.normalize_whitespace(text)
        return text.strip()

    def strip_html(self, text: str) -> str:
        """Remove HTML/XML tags and decode HTML entities.
        
        Args:
            text: Text that may contain HTML markup.
            
        Returns:
            Plain text with all tags removed.
        """
        if not text:
            return ""

        # First unescape HTML entities (&amp; → &, etc.)
        text = html.unescape(text)

        # Parse and extract text
        soup = BeautifulSoup(text, "html.parser")

        # Remove script and style elements entirely
        for tag in soup(["script", "style"]):
            tag.decompose()

        return soup.get_text(separator=" ")

    def normalize_encoding(self, text: str) -> str:
        """Normalize Unicode and strip control characters.
        
        Applies NFC normalization so that canonically equivalent
        sequences are represented identically.  Control characters
        (except ``\\t``, ``\\n``, ``\\r``) are removed.
        
        Args:
            text: Text with potentially mixed Unicode forms.
            
        Returns:
            NFC-normalized text with control characters removed.
        """
        if not text:
            return ""

        # NFC normalization
        text = unicodedata.normalize("NFC", text)

        # Remove control characters (keep tab, newline, CR)
        text = self._CTRL_RE.sub("", text)

        return text

    def normalize_whitespace(self, text: str) -> str:
        """Collapse excessive whitespace.
        
        * Multiple spaces/tabs → single space (except at line starts).
        * Multiple blank lines → single blank line.
        * Trailing whitespace on each line is stripped.
        
        Args:
            text: Text with potentially messy whitespace.
            
        Returns:
            Text with normalized whitespace.
        """
        if not text:
            return ""

        # Collapse multiple spaces/tabs into one space
        text = self._MULTI_SPACE_RE.sub(" ", text)

        # Strip trailing whitespace per line
        lines = [line.rstrip() for line in text.splitlines()]
        text = "\n".join(lines)

        # Collapse multiple blank lines
        text = self._MULTI_BLANK_RE.sub("\n\n", text)

        return text
