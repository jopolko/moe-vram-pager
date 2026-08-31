"""Shared fixtures / path setup for the pentest subsystem tests.

Runs with or without pytest installed - test_pentest_ontology.py has a
__main__ harness that executes every test_* function directly. When pytest is
available, `python -m pytest tools/tests` works too.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tools/

try:
    import pytest

    @pytest.fixture
    def fake_resolver():
        from _ontology_fakes import FakeResolver
        return FakeResolver()
except ImportError:  # pytest not installed - the __main__ harness handles it
    pass
