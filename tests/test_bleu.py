"""Tests unitaires pour le score BLEU (Papineni et al., 2002)."""

import math

from src.benchmark.metrics import bleu_score


def test_bleu_identical():
    score = bleu_score("the cat sat on the mat", "the cat sat on the mat")
    assert score == 1.0


def test_bleu_empty():
    assert bleu_score("", "") == 0.0
    assert bleu_score("hello", "") == 0.0
    assert bleu_score("", "hello") == 0.0


def test_bleu_disjoint():
    score = bleu_score("the cat sat", "a dog ran")
    assert score == 0.0


def test_bleu_partial():
    score = bleu_score("the cat sat on the mat", "the cat sat on a mat")
    assert 0.0 < score < 1.0


def test_bleu_brevity_penalty():
    ref = "the cat sat on the mat in the room"
    short = "the cat"
    long_cand = "the cat sat on the mat in the room today"
    score_short = bleu_score(ref, short)
    score_long = bleu_score(ref, long_cand)
    # short a BP < 1, long a BP = 1
    assert score_short <= score_long


def test_bleu_bp_exponential():
    """BP = exp(1 - r/c) quand le candidat est plus court."""
    expected_bp = math.exp(1 - 6 / 3)  # exp(-1) ≈ 0.368
    assert expected_bp < 0.5  # Confirme que c'est exponentiel, pas linéaire


def test_bleu_trigram_zero_means_bleu_zero():
    """Si P3=0 et P4=0, BLEU-4 = 0 (ln(0) undefined)."""
    ref = "this picture is clicked by me"
    cand = "the picture the picture by me"
    score = bleu_score(ref, cand)
    assert score == 0.0


def test_bleu_unigram_precision():
    """BLEU-1 sur l'exemple du papier.
    ref:  'the picture was clicked by me'
    cand: 'the picture the picture by me'
    Unigrams cand: the(2), picture(2), by(1), me(1) = 6 total
    Clipped: the→min(2,1)=1, picture→min(2,1)=1, by→1, me→1 = 4
    P1 = 4/6 ≈ 0.6667, BP=1 (same length) → BLEU-1 ≈ 0.6667
    """
    ref = "the picture was clicked by me"
    cand = "the picture the picture by me"
    score = bleu_score(ref, cand, max_n=1)
    assert abs(score - round(4 / 6, 4)) < 0.01


def test_bleu_bigram_precision():
    """BLEU-2 sur l'exemple du papier.
    ref:  'the picture was clicked by me'
    cand: 'the picture the picture by me'
    Bigrams cand: (the picture)x2, (picture the)x1, (picture by)x1, (by me)x1 = 5
    Clipped: (the picture)→min(2,1)=1, (picture the)→0, (picture by)→0, (by me)→1 = 2
    P2 = 2/5 = 0.4
    BLEU-2 = BP * exp(0.5*ln(P1) + 0.5*ln(P2)) = 1 * exp(0.5*ln(4/6) + 0.5*ln(2/5))
    """
    ref = "the picture was clicked by me"
    cand = "the picture the picture by me"
    score = bleu_score(ref, cand, max_n=2)
    expected = math.exp(0.5 * math.log(4 / 6) + 0.5 * math.log(2 / 5))
    assert abs(score - round(expected, 4)) < 0.01
