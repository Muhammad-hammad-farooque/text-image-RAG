import numpy as np
import pytest

from rag.retriever import _rrf_merge, _tokenize


def test_tokenize_basic():
    assert _tokenize("Hello World") == ["hello", "world"]


def test_tokenize_empty():
    assert _tokenize("") == []


def test_tokenize_lowercase():
    assert _tokenize("FOO BAR") == ["foo", "bar"]


def test_rrf_merge_prefers_double_ranked():
    vector_results = [
        {"score": 0.9, "payload": {"text": "chunk A"}},
        {"score": 0.8, "payload": {"text": "chunk B"}},
    ]
    bm25_results = [
        {"score": 5.0, "payload": {"text": "chunk B"}},
        {"score": 3.0, "payload": {"text": "chunk C"}},
    ]
    merged = _rrf_merge(vector_results, bm25_results)
    texts = [r["payload"]["text"] for r in merged]
    # chunk B appears in both lists → highest RRF score
    assert texts[0] == "chunk B"


def test_rrf_merge_returns_all_unique():
    vector_results = [{"score": 0.9, "payload": {"text": "A"}}]
    bm25_results = [{"score": 5.0, "payload": {"text": "B"}}]
    merged = _rrf_merge(vector_results, bm25_results)
    assert len(merged) == 2


def test_rrf_merge_empty_inputs():
    assert _rrf_merge([], []) == []


def test_rrf_merge_one_empty():
    results = [
        {"score": 0.9, "payload": {"text": "only result"}},
        {"score": 0.7, "payload": {"text": "second result"}},
    ]
    merged = _rrf_merge(results, [])
    assert len(merged) == 2
    assert merged[0]["payload"]["text"] == "only result"


def test_rrf_merge_deduplicates():
    same = [{"score": 0.9, "payload": {"text": "dup"}}]
    merged = _rrf_merge(same, same)
    assert len(merged) == 1
