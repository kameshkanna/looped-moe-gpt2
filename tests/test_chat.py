"""Unit tests for scripts/chat.py's prompt construction (template + system prompt)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from chat import build_prompt


def test_build_prompt_no_template_no_system_prompt() -> None:
    assert build_prompt("hello", template="none", system_prompt=None) == "hello"


def test_build_prompt_math_template() -> None:
    result = build_prompt("John has 5 apples.", template="math", system_prompt=None)
    assert result == "Problem: John has 5 apples.\nSolution:"


def test_build_prompt_system_prompt_only() -> None:
    result = build_prompt("hello", template="none", system_prompt="You are a helpful assistant.")
    assert result == "You are a helpful assistant.\n\nhello"


def test_build_prompt_system_prompt_and_template_combined() -> None:
    result = build_prompt("John has 5 apples.", template="math", system_prompt="Solve carefully.")
    assert result == "Solve carefully.\n\nProblem: John has 5 apples.\nSolution:"


def test_build_prompt_unknown_template_raises() -> None:
    with pytest.raises(ValueError, match="Unknown template"):
        build_prompt("hello", template="not_a_real_template", system_prompt=None)
