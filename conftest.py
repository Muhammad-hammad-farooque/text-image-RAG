import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests that require running infrastructure (Qdrant, Redis, GROQ_API_KEY)",
    )
