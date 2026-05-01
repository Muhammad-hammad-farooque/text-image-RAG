from pathlib import Path

import pandas as pd

from core.logger import logger
from rag.parser.utils import ParsedChunk, table_to_markdown


def parse_excel(file_path: str | Path) -> list[ParsedChunk]:
    source = Path(file_path).name
    ext = Path(file_path).suffix.lower()
    chunks: list[ParsedChunk] = []

    if ext == ".csv":
        sheets = {"Sheet1": pd.read_csv(file_path)}
    else:
        xl = pd.ExcelFile(file_path)
        sheets = {name: xl.parse(name) for name in xl.sheet_names}

    for sheet_name, df in sheets.items():
        df = df.dropna(how="all").fillna("")
        if df.empty:
            continue

        headers = [str(col) for col in df.columns]
        rows = [headers] + [[str(v) for v in row] for row in df.values.tolist()]
        markdown, headers = table_to_markdown(rows)

        if markdown.strip():
            chunks.append(ParsedChunk(
                text=markdown,
                type="table",
                page=0,
                source=source,
                metadata={"table_index": 0, "sheet_name": sheet_name, "headers": headers},
            ))

    logger.info(f"Parsed Excel/CSV {source}: {len(chunks)} chunks")
    return chunks
