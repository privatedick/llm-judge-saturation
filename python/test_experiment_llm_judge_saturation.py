"""Regression tests for `_parse_verdict` in experiment_llm_judge_saturation.py.

Round-3 review (2026-08-12) claimed the parser bug fixed in round 1
(uppercasing content before matching, so the English article "a" could match
as a verdict of A) was "demonstrated, in both directions" against four example
strings. Checked empirically before applying anything -- only ONE of the four
reproduced, and via a DIFFERENT mechanism than claimed (not the article-hijack
bug, which round 1 already fixed and stays fixed; a genuine "explain-then-
conclude" ambiguity where the first \\b[AB]\\b match isn't the actual verdict).
This file locks in both: the real fix, and the false claims staying false, so
neither regresses silently.
"""

from __future__ import annotations

from experiment_llm_judge_saturation import _parse_verdict


def test_terse_single_letter_still_parses():
    assert _parse_verdict("A") == "A"
    assert _parse_verdict("B") == "B"
    assert _parse_verdict("B.") == "B"
    assert _parse_verdict("Answer A is correct") == "A"


def test_lowercase_article_does_not_hijack_the_verdict():
    """Round-1 fix (case-sensitive matching) must stay fixed. Round-3's claimed
    demonstrations of a residual bug did NOT reproduce -- verified here."""
    assert _parse_verdict("It's a tie, but B") == "B"
    assert _parse_verdict("As a judge, I pick B") == "B"
    assert _parse_verdict("B is more precise. Overall it is a close call though.") == "B"


def test_explain_then_conclude_takes_the_concluding_letter():
    """The one round-3 finding that DID reproduce: content that mentions the
    non-chosen option by letter before the actual verdict defeated the old
    first-match heuristic. Fixed by taking the LAST \\b[AB]\\b match, matching
    the reasoning-fallback's own "verdict concludes the trace" assumption."""
    assert _parse_verdict("I think A is worse, B is better") == "B"


def test_no_letters_returns_none():
    assert _parse_verdict("") is None
    assert _parse_verdict("I cannot decide.") is None


# --------------------------------------------------------------------------- #
# Broader robustness battery (UltraReview follow-up, 2026-08-12): "verdict
# extraction remains a practical weak point for reasoning models... broader
# robustness testing would be useful." Realistic completion shapes a judge
# might produce despite JUDGE_SYSTEM's "reply with exactly one character"
# instruction, since reasoning/verbose models routinely ignore terse-output
# instructions.
# --------------------------------------------------------------------------- #

MARKDOWN_AND_WRAPPING_CASES = [
    ("**A**", "A"),
    ("**B**.", "B"),
    ("(A)", "A"),
    ("Answer: B", "B"),
    ("```\nB\n```", "B"),
    ('{"verdict": "A"}', "A"),
    ("Verdict: A.", "A"),
    ("My answer is B", "B"),
]


def test_markdown_and_wrapping_do_not_defeat_the_parser():
    for content, expected in MARKDOWN_AND_WRAPPING_CASES:
        assert _parse_verdict(content) == expected, content


def test_refusal_and_hedge_return_none_not_a_guess():
    """A judge that refuses to pick, or hedges without ever naming a letter,
    must return None -- guessing here would silently manufacture a verdict
    (and a position/label statistic) out of missing data, the same failure
    class as round 2's _dist_stats([]) bug at the aggregation layer."""
    assert _parse_verdict("I cannot determine which is better without more context.") is None
    assert _parse_verdict("Both answers are equally valid.") is None
    assert _parse_verdict("This depends on the intended audience.") is None


def test_lowercase_only_never_matches():
    """JUDGE_SYSTEM demands uppercase A/B; lowercase a/b alone (no uppercase
    anywhere) must never be treated as a verdict -- that would reopen the
    exact class of bug round 1 fixed (the case-INsensitive match), just via a
    different code path (matching lowercase directly instead of uppercasing
    first)."""
    assert _parse_verdict("i think a is the better one, actually b too") is None


def test_reasoning_style_multi_sentence_content_takes_final_verdict():
    """Same function parses both `content` and the `reasoning` fallback text
    in _call() -- a reasoning model that puts its whole chain of thought in
    `content` despite the terse-output instruction must still resolve to the
    concluding letter, not an early one mentioned in passing."""
    text = (
        "Let me think through this. Option A addresses the question directly "
        "but option B has a clearer structure. On balance, B is more precise. "
        "Overall it is a close call though."
    )
    assert _parse_verdict(text) == "B"
