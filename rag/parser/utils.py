from typing import TypedDict


class ParsedChunk(TypedDict):
    text: str
    type: str        # "text" | "table" | "image"
    page: int
    source: str
    metadata: dict


def table_to_markdown(rows: list[list[str]]) -> tuple[str, list[str]]:
    """Convert a list of rows (first row = headers) to a markdown table string."""
    if not rows or not rows[0]:
        return "", []

    rows = [[cell if cell is not None else "" for cell in row] for row in rows]
    headers = [str(h) for h in rows[0]]

    header_row = "| " + " | ".join(headers) + " |"
    separator  = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_rows  = ["| " + " | ".join(str(c) for c in row) + " |" for row in rows[1:]]

    return "\n".join([header_row, separator] + data_rows), headers


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Recursive character text splitter — mirrors LangChain's RecursiveCharacterTextSplitter."""
    separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def _split(text: str, separators: list[str]) -> list[str]:
        sep = separators[0]
        next_seps = separators[1:]

        if sep == "":
            parts = list(text)
        else:
            parts = text.split(sep)

        chunks: list[str] = []
        current = ""

        for part in parts:
            candidate = (current + sep + part).lstrip(sep) if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) > chunk_size and next_seps:
                    chunks.extend(_split(part, next_seps))
                else:
                    current = part

        if current:
            chunks.append(current)

        return chunks

    raw_chunks = _split(text.strip(), separators)

    # Apply overlap by re-merging with look-back
    if chunk_overlap == 0 or len(raw_chunks) <= 1:
        return [c for c in raw_chunks if c.strip()]

    result: list[str] = []
    for i, chunk in enumerate(raw_chunks):
        if i == 0:
            result.append(chunk)
            continue
        overlap_start = max(0, len(result[-1]) - chunk_overlap)
        merged = result[-1][overlap_start:] + " " + chunk
        if len(merged) <= chunk_size:
            result[-1] = merged
        else:
            result.append(chunk)

    return [c for c in result if c.strip()]
