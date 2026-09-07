"""Unit tests for GSM8K answer extraction logic used by scripts/eval_reasoning_accuracy.py.

These guard the correctness of the accuracy metric itself -- a bug in number extraction would
silently produce a meaningless accuracy score without any visible error, which is worse than no
eval at all (a wrong-but-confident number is more dangerous than an absent one).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from eval_reasoning_accuracy import extract_gsm8k_ground_truth, extract_model_answer


def test_extract_ground_truth_standard_format() -> None:
    answer_field = (
        "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
        "Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n"
        "#### 72"
    )
    assert extract_gsm8k_ground_truth(answer_field) == 72.0


def test_extract_ground_truth_negative_decimal() -> None:
    assert extract_gsm8k_ground_truth("some reasoning text #### -5.5") == -5.5


def test_extract_ground_truth_comma_formatted() -> None:
    assert extract_gsm8k_ground_truth("some reasoning text #### 1,234") == 1234.0


def test_extract_ground_truth_missing_marker_returns_none() -> None:
    assert extract_gsm8k_ground_truth("no marker here, just text") is None


def test_extract_model_answer_with_marker() -> None:
    generated = "Let me solve this. First 48+24=72. #### 72"
    assert extract_model_answer(generated) == 72.0


def test_extract_model_answer_fallback_to_last_number() -> None:
    """Without a '####' marker, the last number mentioned anywhere in free-form text is used."""
    generated = "The store had 48 clips, then sold 24 more, for a total of 72 clips sold."
    assert extract_model_answer(generated) == 72.0


def test_extract_model_answer_no_number_returns_none() -> None:
    assert extract_model_answer("I cannot solve this problem.") is None


def test_extract_model_answer_bare_commas_do_not_crash() -> None:
    """Regression test for a real crash hit during a live GSM8K eval run: the original regex
    `[\\d,]+` matches one-or-more of "digit OR comma", so a bare comma with zero actual digits
    (e.g. free-form model text like "a, b, c") matched as "," -- then `float(",")` (after
    stripping commas, leaving an empty string) raised ValueError and crashed the whole eval run
    partway through. The fixed pattern requires the match to START with an actual digit."""
    assert extract_model_answer("a, b, c") is None
    assert extract_model_answer(", , ,") is None
    assert extract_model_answer("well, I think, the answer, is unclear") is None


def test_extract_model_answer_comma_with_digits_still_works() -> None:
    """The fix for bare commas must not break legitimate comma-formatted numbers."""
    assert extract_model_answer("the total came to 1,234 dollars") == 1234.0


def test_extract_model_answer_prefers_marker_over_fallback() -> None:
    """If both a '####' marker AND other numbers are present, the marker takes precedence --
    the marker is the model's explicit final-answer signal, not just any number mentioned."""
    generated = "I considered 48 and 24 as intermediate values, but made an error. #### 100"
    assert extract_model_answer(generated) == 100.0


def test_extract_model_answer_negative_fallback() -> None:
    generated = "The temperature dropped by 10 degrees to reach -5."
    assert extract_model_answer(generated) == -5.0
