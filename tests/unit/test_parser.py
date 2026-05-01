import tempfile
from pathlib import Path

import pytest

from rag.parser.utils import ParsedChunk, split_text, table_to_markdown
from rag.parser.txt import parse_txt


def test_split_text_produces_multiple_chunks():
    text = "Hello world. " * 50
    chunks = split_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1


def test_split_text_respects_chunk_size():
    text = "Hello world. " * 50
    chunks = split_text(text, chunk_size=100, chunk_overlap=0)
    for chunk in chunks:
        assert len(chunk) <= 120  # allow small tolerance for separator length


def test_split_text_preserves_all_content():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = split_text(text, chunk_size=50, chunk_overlap=0)
    joined = " ".join(chunks)
    assert "First" in joined
    assert "Second" in joined
    assert "Third" in joined


def test_split_text_single_chunk_if_small():
    text = "Short text."
    chunks = split_text(text, chunk_size=500, chunk_overlap=100)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_table_to_markdown_basic():
    rows = [["Name", "Age", "City"], ["Alice", "30", "Berlin"], ["Bob", "25", "Paris"]]
    md, headers = table_to_markdown(rows)
    assert "| Name | Age | City |" in md
    assert "| Alice | 30 | Berlin |" in md
    assert headers == ["Name", "Age", "City"]


def test_table_to_markdown_separator_row():
    rows = [["A", "B"], ["1", "2"]]
    md, _ = table_to_markdown(rows)
    assert "|---|---|" in md


def test_table_to_markdown_empty():
    md, headers = table_to_markdown([])
    assert md == ""
    assert headers == []


def test_table_to_markdown_none_cells():
    rows = [["Col1", None], [None, "val"]]
    md, headers = table_to_markdown(rows)
    assert "Col1" in md
    assert "|  |" in md or "| Col1 |" in md


def test_parse_txt_returns_text_chunks():
    content = "This is a test document. " * 30
    with tempfile.NamedTemporaryFile(
        suffix=".txt", mode="w", delete=False, encoding="utf-8"
    ) as f:
        f.write(content)
        tmp = f.name

    chunks = parse_txt(tmp)
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk["type"] == "text"
        assert chunk["text"].strip()

    Path(tmp).unlink()


def test_parse_txt_empty_file():
    with tempfile.NamedTemporaryFile(
        suffix=".txt", mode="w", delete=False, encoding="utf-8"
    ) as f:
        f.write("   ")
        tmp = f.name

    chunks = parse_txt(tmp)
    assert chunks == []
    Path(tmp).unlink()
