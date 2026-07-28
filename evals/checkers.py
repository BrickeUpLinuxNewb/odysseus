"""Programmatic scoring for eval tasks -- no LLM judge, ever.

Each checker takes the model's raw output plus the task's ``expect`` dict and
returns ``(score, detail)`` where score is 0.0..1.0 and detail says why. All
kinds are deterministic and machine-checkable; that is the whole point --
using a small local model to judge small local models would make the ruler
out of the same rubber it is measuring.

Task kinds:
    choice        output must resolve to one label from ``choices``; scored
                  against ``answer``. For routing/classification tasks.
    exact         normalized string equality with ``value``.
    contains      ``value`` substring present (casefolded).
    contains_all  every string in ``values`` present.
    regex         ``pattern`` found (or fullmatch with ``full: true``).
    numeric       first number in the output within ``tol`` (absolute) or
                  ``rel_tol`` (relative) of ``value``.
    json          output parses as JSON (code fences tolerated) and satisfies
                  ``required`` keys / ``types`` / ``enum`` / ``values``.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

_WS_RE = re.compile(r"\s+")
_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*|\s*```$")
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9_.\-]*")

# JSON-ish type names accepted in a task's ``types`` map.
_TYPE_MAP = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def normalize(text: str) -> str:
    """Casefold, collapse whitespace, strip surrounding quotes/punctuation."""
    t = _WS_RE.sub(" ", (text or "").strip()).casefold()
    return t.strip(" \t\"'.,:;!`*")


def strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = _FENCE_RE.sub("", t).strip()
    return t


def extract_json(text: str) -> Optional[Any]:
    """Parse JSON from raw output: direct, fence-stripped, or first {...}."""
    t = strip_code_fences(text)
    for candidate in (t,):
        try:
            return json.loads(candidate)
        except Exception:
            pass
    match = re.search(r"\{.*\}", t, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def all_numbers(text: str) -> List[float]:
    out: List[float] = []
    for match in _NUM_RE.finditer(text or ""):
        try:
            out.append(float(match.group(0).replace(",", "")))
        except ValueError:
            continue
    return out


# -- individual checkers -----------------------------------------------------

def _check_choice(output: str, expect: Dict[str, Any]) -> Tuple[float, str]:
    choices = [normalize(c) for c in expect.get("choices", [])]
    answer = normalize(str(expect.get("answer", "")))
    if not choices or not answer:
        return 0.0, "task error: choice needs 'choices' and 'answer'"
    norm = normalize(output)
    if norm == answer:
        return 1.0, "exact label"
    # Otherwise the output must mention exactly ONE known label -- an output
    # naming several labels is a non-answer, not a lucky hit.
    words = set(_WORD_RE.findall(norm))
    mentioned = [c for c in choices if c in words]
    if len(mentioned) == 1:
        return (1.0, "single label mentioned") if mentioned[0] == answer \
            else (0.0, f"answered {mentioned[0]!r}, expected {answer!r}")
    if not mentioned:
        return 0.0, "no known label in output"
    return 0.0, f"ambiguous: mentioned {mentioned}"


def _check_exact(output: str, expect: Dict[str, Any]) -> Tuple[float, str]:
    value = normalize(str(expect.get("value", "")))
    norm = normalize(output)
    if norm == value:
        return 1.0, "exact match"
    return 0.0, f"got {norm[:80]!r}, expected {value!r}"


def _check_contains(output: str, expect: Dict[str, Any]) -> Tuple[float, str]:
    value = str(expect.get("value", ""))
    if value.casefold() in (output or "").casefold():
        return 1.0, "substring present"
    return 0.0, f"missing {value!r}"


def _check_contains_all(output: str, expect: Dict[str, Any]) -> Tuple[float, str]:
    values: List[str] = [str(v) for v in expect.get("values", [])]
    if not values:
        return 0.0, "task error: contains_all needs 'values'"
    low = (output or "").casefold()
    missing = [v for v in values if v.casefold() not in low]
    if not missing:
        return 1.0, f"all {len(values)} present"
    # Partial credit proportional to hits: extraction of 3-of-4 names is
    # meaningfully better than 0-of-4 and the delta should be visible.
    score = (len(values) - len(missing)) / len(values)
    return score, f"missing {missing}"


def _check_regex(output: str, expect: Dict[str, Any]) -> Tuple[float, str]:
    pattern = expect.get("pattern", "")
    if not pattern:
        return 0.0, "task error: regex needs 'pattern'"
    flags = re.IGNORECASE if expect.get("ignore_case", True) else 0
    if expect.get("full"):
        ok = re.fullmatch(pattern, (output or "").strip(), flags) is not None
    else:
        ok = re.search(pattern, output or "", flags) is not None
    return (1.0, "pattern matched") if ok else (0.0, f"pattern {pattern!r} not found")


def _check_numeric(output: str, expect: Dict[str, Any]) -> Tuple[float, str]:
    try:
        value = float(expect["value"])
    except (KeyError, TypeError, ValueError):
        return 0.0, "task error: numeric needs numeric 'value'"
    tol = float(expect.get("tol", 0.0))
    rel_tol = float(expect.get("rel_tol", 0.0))
    limit = max(tol, abs(value) * rel_tol)

    # Answer-marker mode: score ONLY the model's declared final answer, not any
    # intermediate number in its working. This removes two confounds at once --
    # a verbose chain-of-thought no longer wins by emitting a matching
    # intermediate value, and reasoning is no longer suppressed (so a math task
    # measures math, not instruction-compliance-under-a-terseness-order). The
    # prompt must ask the model to end with e.g. "ANSWER: <number>".
    prefix = expect.get("answer_prefix")
    if prefix:
        idx = output.lower().rfind(prefix.lower())
        candidates = all_numbers(output[idx + len(prefix):]) if idx != -1 else []
        if not candidates:
            candidates = all_numbers(output)[-1:]  # fallback: last number stated
        if not candidates:
            return 0.0, f"no number after {prefix!r}"
        got = candidates[0]
        if abs(got - value) <= limit:
            return 1.0, f"got {got}"
        return 0.0, f"final answer {got}, expected {value} (±{limit})"

    numbers = all_numbers(output)
    if not numbers:
        return 0.0, "no number in output"
    # Legacy lenient path (no marker): models restate operands ("17 x 23 = 391"),
    # so any in-tolerance number passes. ``strict: true`` demands exactly one.
    if expect.get("strict") and len(numbers) != 1:
        return 0.0, f"strict: expected exactly one number, got {len(numbers)}"
    hits = [n for n in numbers if abs(n - value) <= limit]
    if hits:
        return 1.0, f"got {hits[0]}"
    nearest = min(numbers, key=lambda n: abs(n - value))
    return 0.0, f"nearest {nearest}, expected {value} (±{limit})"


def _check_json(output: str, expect: Dict[str, Any]) -> Tuple[float, str]:
    data = extract_json(output)
    if data is None:
        return 0.0, "output is not parseable JSON"
    if not isinstance(data, dict):
        return 0.0, f"JSON is {type(data).__name__}, expected object"
    problems: List[str] = []
    for key in expect.get("required", []):
        if key not in data:
            problems.append(f"missing key {key!r}")
    for key, type_name in (expect.get("types") or {}).items():
        expected_type = _TYPE_MAP.get(type_name)
        if expected_type is None:
            problems.append(f"task error: unknown type {type_name!r}")
        elif key in data and not isinstance(data[key], expected_type):
            # bool is an int subclass; a true/false is not an acceptable number.
            problems.append(f"{key!r} is {type(data[key]).__name__}, expected {type_name}")
        elif key in data and type_name in ("number", "integer") and isinstance(data[key], bool):
            problems.append(f"{key!r} is boolean, expected {type_name}")
    for key, allowed in (expect.get("enum") or {}).items():
        if key in data and normalize(str(data[key])) not in [normalize(str(a)) for a in allowed]:
            problems.append(f"{key!r}={data[key]!r} not in {allowed}")
    for key, wanted in (expect.get("values") or {}).items():
        if key not in data:
            problems.append(f"missing key {key!r}")
        elif isinstance(wanted, (int, float)) and not isinstance(wanted, bool):
            got_num = data[key]
            if isinstance(got_num, bool) or not isinstance(got_num, (int, float)) \
                    or abs(float(got_num) - float(wanted)) > 1e-9:
                problems.append(f"{key!r}={data[key]!r}, expected {wanted!r}")
        elif normalize(str(data.get(key))) != normalize(str(wanted)):
            problems.append(f"{key!r}={data[key]!r}, expected {wanted!r}")
    if problems:
        return 0.0, "; ".join(problems[:4])
    return 1.0, "JSON shape ok"


_CHECKERS = {
    "choice": _check_choice,
    "exact": _check_exact,
    "contains": _check_contains,
    "contains_all": _check_contains_all,
    "regex": _check_regex,
    "numeric": _check_numeric,
    "json": _check_json,
}

KNOWN_KINDS = tuple(_CHECKERS)


def check(kind: str, output: str, expect: Dict[str, Any]) -> Tuple[float, str]:
    """Score one output against one task rubric. Never raises."""
    checker = _CHECKERS.get(kind)
    if checker is None:
        return 0.0, f"task error: unknown kind {kind!r}"
    try:
        return checker(output or "", expect or {})
    except Exception as exc:  # a bad rubric must not kill the run
        return 0.0, f"checker crashed: {exc}"


def derive_passing_output(kind: str, expect: Dict[str, Any]) -> Optional[str]:
    """Construct an output that SHOULD score 1.0 -- used by the selftest
    target to lint task files: if the harness can't pass its own rubric with
    a known-good answer, the task is broken, not the model."""
    if "sample_pass" in expect:
        return str(expect["sample_pass"])
    if kind == "choice":
        return str(expect.get("answer", ""))
    if kind == "exact":
        return str(expect.get("value", ""))
    if kind == "contains":
        return str(expect.get("value", ""))
    if kind == "contains_all":
        return ", ".join(str(v) for v in expect.get("values", []))
    if kind == "numeric":
        prefix = expect.get("answer_prefix")
        return f"{prefix} {expect.get('value', '')}" if prefix else str(expect.get("value", ""))
    if kind == "json":
        obj: Dict[str, Any] = {}
        for key in expect.get("required", []):
            obj[key] = ""
        for key, wanted in (expect.get("values") or {}).items():
            obj[key] = wanted
        for key, allowed in (expect.get("enum") or {}).items():
            if key not in (expect.get("values") or {}):
                obj[key] = allowed[0] if allowed else ""
        for key, type_name in (expect.get("types") or {}).items():
            if key not in obj or obj[key] == "":
                obj[key] = {"string": "x", "number": 0.5, "integer": 1,
                            "boolean": True, "array": [], "object": {}}.get(type_name, "x")
        return json.dumps(obj)
    return None  # regex without sample_pass: not derivable
