import io
import base64
from pathlib import Path

import fitz
import pdfplumber
from PIL import Image

from core.config import settings
from core.logger import logger
from rag.parser.utils import ParsedChunk, table_to_markdown, split_text


def parse_pdf(file_path: str | Path) -> list[ParsedChunk]:
    source = Path(file_path).name
    chunks: list[ParsedChunk] = []

    # Tables — pdfplumber (geometric boundary detection)
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            for table_idx, table in enumerate(tables or []):
                markdown, headers = table_to_markdown(table)
                if markdown.strip():
                    chunks.append(ParsedChunk(
                        text=markdown,
                        type="table",
                        page=page_num,
                        source=source,
                        metadata={"table_index": table_idx, "headers": headers},
                    ))

    # Text + Images — PyMuPDF
    doc = fitz.open(str(file_path))
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            for i, chunk_text in enumerate(split_text(text, settings.chunk_size, settings.chunk_overlap)):
                chunks.append(ParsedChunk(
                    text=chunk_text,
                    type="text",
                    page=page_num,
                    source=source,
                    metadata={"chunk_index": i},
                ))

        for img_idx, img in enumerate(page.get_images(full=True)):
            try:
                base_image = doc.extract_image(img[0])
                pil_image = Image.open(io.BytesIO(base_image["image"])).convert("RGB")
                buf = io.BytesIO()
                pil_image.save(buf, format="PNG")
                image_b64 = base64.b64encode(buf.getvalue()).decode()
                chunks.append(ParsedChunk(
                    text=f"[Image on page {page_num}]",
                    type="image",
                    page=page_num,
                    source=source,
                    metadata={"image_index": img_idx, "image_base64": image_b64},
                ))
            except Exception as e:
                logger.warning(f"Skipped image on page {page_num}: {e}")

    doc.close()
    logger.info(f"Parsed PDF {source}: {len(chunks)} chunks")
    return chunks
