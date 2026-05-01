from pathlib import Path

from rag.parser.pdf import parse_pdf
from rag.parser.docx import parse_docx
from rag.parser.excel import parse_excel
from rag.parser.txt import parse_txt
from rag.parser.utils import ParsedChunk

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".xlsx", ".csv"}


def parse(file_path: str | Path) -> list[ParsedChunk]:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return parse_pdf(file_path)
    elif ext == ".docx":
        return parse_docx(file_path)
    elif ext in {".xlsx", ".csv"}:
        return parse_excel(file_path)
    elif ext == ".txt":
        return parse_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {SUPPORTED_EXTENSIONS}")
