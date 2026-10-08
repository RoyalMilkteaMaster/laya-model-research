import pytest

from multihop_benchmark.evaluation.answer_scorer import exact_match, f1


def test_normalization_ignores_case_articles_and_punctuation():
    assert exact_match("The United States", "the united states.") == 1


def test_alias_match_counts_as_exact():
    assert exact_match("USA", "United States", aliases=["USA"]) == 1
    assert exact_match("USA", "United States") == 0


def test_partial_overlap_f1():
    assert f1("Barack Obama", "Obama") == pytest.approx(0.667, abs=0.001)


def test_empty_prediction_scores_zero():
    assert exact_match("", "Obama") == 0
    assert f1("", "Obama") == 0.0
    assert exact_match("  the . ", "Obama", aliases=["Barack Obama"]) == 0
    assert f1("  the . ", "Obama", aliases=["Barack Obama"]) == 0.0


def test_f1_takes_best_over_gold_and_aliases():
    assert f1("Barack Obama", "Obama", aliases=["Barack Hussein Obama", "Barack Obama"]) == 1.0
    assert f1("Paris", "London", aliases=[]) == 0.0


def test_whitespace_is_collapsed():
    assert exact_match("  New\tYork   City ", "new york city") == 1
