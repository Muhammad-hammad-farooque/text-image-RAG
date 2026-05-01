import numpy as np
import pytest

from rag.embedder import embed_text, embed_texts


def test_embed_text_shape():
    vec = embed_text("Hello world")
    assert vec.shape == (384,)


def test_embed_text_is_normalized():
    vec = embed_text("Hello world")
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-5


def test_embed_texts_batch_shape():
    texts = ["Hello", "World", "Test sentence"]
    vecs = embed_texts(texts)
    assert vecs.shape == (3, 384)


def test_embed_texts_all_normalized():
    texts = ["Hello", "World"]
    vecs = embed_texts(texts)
    for vec in vecs:
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5


def test_embed_text_deterministic():
    text = "Deterministic test"
    vec1 = embed_text(text)
    vec2 = embed_text(text)
    np.testing.assert_array_almost_equal(vec1, vec2)


def test_multilingual_similarity():
    en = embed_text("The cat sits on the mat")
    de = embed_text("Die Katze sitzt auf der Matte")
    similarity = float(np.dot(en, de))
    assert similarity > 0.5, f"Expected cross-lingual similarity > 0.5, got {similarity:.3f}"


def test_embed_single_vs_batch_consistency():
    texts = ["First sentence", "Second sentence"]
    batch = embed_texts(texts)
    single_0 = embed_text(texts[0])
    single_1 = embed_text(texts[1])
    np.testing.assert_array_almost_equal(batch[0], single_0, decimal=5)
    np.testing.assert_array_almost_equal(batch[1], single_1, decimal=5)
