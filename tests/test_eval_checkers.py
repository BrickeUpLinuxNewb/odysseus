"""Checker-level tests for the eval harness: every kind, pass and fail."""
import pytest

from evals.checkers import (
    KNOWN_KINDS,
    check,
    derive_passing_output,
    extract_json,
    normalize,
)


def test_normalize_strips_case_punct_ws():
    assert normalize("  Canberra.  ") == "canberra"
    assert normalize('"2.7.1"') == "2.7.1"
    assert normalize("A  B\nC") == "a b c"


# -- choice -------------------------------------------------------------------

CHOICE = {"choices": ["email", "code", "calendar"], "answer": "email"}

def test_choice_exact_label():
    assert check("choice", "email", CHOICE)[0] == 1.0

def test_choice_label_in_sentence():
    assert check("choice", "I'd route this to Email.", CHOICE)[0] == 1.0

def test_choice_wrong_label():
    score, detail = check("choice", "code", CHOICE)
    assert score == 0.0 and "expected" in detail

def test_choice_ambiguous_multi_label_fails():
    # Naming several labels is a non-answer even if the right one appears.
    score, detail = check("choice", "either email or calendar", CHOICE)
    assert score == 0.0 and "ambiguous" in detail

def test_choice_no_label():
    assert check("choice", "hmm, not sure", CHOICE)[0] == 0.0


# -- exact / contains ---------------------------------------------------------

def test_exact_normalized():
    assert check("exact", "  DENVER. ", {"value": "Denver"})[0] == 1.0
    assert check("exact", "Denver and Portland", {"value": "Denver"})[0] == 0.0

def test_contains_casefold():
    assert check("contains", "It lands in Canberra, obviously.", {"value": "canberra"})[0] == 1.0
    assert check("contains", "Sydney", {"value": "canberra"})[0] == 0.0

def test_contains_all_partial_credit():
    expect = {"values": ["Marcus", "Elena", "Priya", "Dan"]}
    score, detail = check("contains_all", "Marcus, Elena and Priya were there", expect)
    assert score == pytest.approx(0.75)
    assert "Dan" in detail
    assert check("contains_all", "marcus elena priya dan", expect)[0] == 1.0


# -- regex / numeric ----------------------------------------------------------

def test_regex_word_boundary():
    expect = {"pattern": r"\bSn\b"}
    assert check("regex", "The symbol is Sn.", expect)[0] == 1.0
    assert check("regex", "snake", expect)[0] == 0.0

def test_numeric_tolerances_and_restated_operands():
    # Lenient default: the answer among restated operands still passes.
    assert check("numeric", "17 x 23 = 391", {"value": 391, "tol": 0})[0] == 1.0
    assert check("numeric", "about 8.05 km", {"value": 8.05, "rel_tol": 0.02})[0] == 1.0
    assert check("numeric", "roughly 9 km", {"value": 8.05, "rel_tol": 0.02})[0] == 0.0
    assert check("numeric", "no digits here", {"value": 5})[0] == 0.0

def test_numeric_strict_requires_single_number():
    assert check("numeric", "17 x 23 = 391", {"value": 391, "strict": True})[0] == 0.0
    assert check("numeric", "391", {"value": 391, "strict": True})[0] == 1.0

def test_numeric_comma_grouping():
    assert check("numeric", "Total: $1,849.50", {"value": 1849.5, "tol": 0.01})[0] == 1.0


# -- json -----------------------------------------------------------------------

JSON_EXPECT = {
    "required": ["route", "confidence"],
    "values": {"route": "email"},
    "types": {"confidence": "number"},
    "enum": {"route": ["email", "code"]},
}

def test_json_happy_path_with_fences():
    out = '```json\n{"route": "email", "confidence": 0.9}\n```'
    assert check("json", out, JSON_EXPECT)[0] == 1.0

def test_json_prose_wrapped_object():
    out = 'Sure! Here you go: {"route": "email", "confidence": 0.4} hope that helps'
    assert check("json", out, JSON_EXPECT)[0] == 1.0

def test_json_missing_key():
    assert check("json", '{"route": "email"}', JSON_EXPECT)[0] == 0.0

def test_json_wrong_type():
    out = '{"route": "email", "confidence": "high"}'
    score, detail = check("json", out, JSON_EXPECT)
    assert score == 0.0 and "confidence" in detail

def test_json_enum_violation():
    out = '{"route": "fax", "confidence": 0.5}'
    assert check("json", out, JSON_EXPECT)[0] == 0.0

def test_json_boolean_is_not_a_number():
    out = '{"route": "email", "confidence": true}'
    assert check("json", out, JSON_EXPECT)[0] == 0.0

def test_json_numeric_value_match():
    expect = {"required": ["age"], "values": {"age": 34}}
    assert check("json", '{"age": 34}', expect)[0] == 1.0
    assert check("json", '{"age": 33}', expect)[0] == 0.0

def test_json_not_json():
    assert check("json", "definitely not json", JSON_EXPECT)[0] == 0.0

def test_extract_json_returns_none_on_garbage():
    assert extract_json("{{{") is None


# -- meta -----------------------------------------------------------------------

def test_unknown_kind_scores_zero_not_raise():
    score, detail = check("vibes", "anything", {})
    assert score == 0.0 and "unknown kind" in detail

def test_derive_passing_output_covers_all_kinds():
    cases = {
        "choice": {"choices": ["a", "b"], "answer": "b"},
        "exact": {"value": "Denver"},
        "contains": {"value": "canberra"},
        "contains_all": {"values": ["x", "y"]},
        "numeric": {"value": 391, "tol": 0},
        "json": JSON_EXPECT,
        "regex": {"pattern": r"\bSn\b", "sample_pass": "Sn"},
    }
    assert set(cases) == set(KNOWN_KINDS)
    for kind, expect in cases.items():
        derived = derive_passing_output(kind, expect)
        assert derived is not None, kind
        assert check(kind, derived, expect)[0] == 1.0, kind

def test_derive_regex_without_sample_is_none():
    assert derive_passing_output("regex", {"pattern": r"\d+"}) is None
