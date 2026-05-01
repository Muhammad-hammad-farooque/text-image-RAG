from pathlib import Path

from core.config import settings
from core.logger import logger
from rag.parser.utils import ParsedChunk, split_text


def parse_txt(file_path: str | Path) -> list[ParsedChunk]:
    source = Path(file_path).name
    chunks: list[ParsedChunk] = []

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    if text.strip():
        for i, chunk_text in enumerate(split_text(text, settings.chunk_size, settings.chunk_overlap)):
            chunks.append(ParsedChunk(
                text=chunk_text,
                type="text",
                page=0,
                source=source,
                metadata={"chunk_index": i},
            ))

    logger.info(f"Parsed TXT {source}: {len(chunks)} chunks")
    return chunks
