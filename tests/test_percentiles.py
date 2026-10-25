"""Batched quantiles keep the frozen interpolation and rounding convention."""
from random import Random
from unittest.mock import patch

from vera_mot.reid_data import percentile, percentiles


def test_batched_quantiles_match_individual_queries_without_mutation():
    rng = Random(31)
    for size in (0, 1, 2, 3, 1001):
        values = [rng.randrange(-1000, 1000) for _ in range(size)]
        original = values.copy()
        expected = {str(p): percentile(values, p) for p in (0, 25, 50, 75, 100)}
        assert percentiles(values) == expected
        assert values == original


def test_five_quantiles_only_sort_the_input_once():
    with patch("builtins.sorted", wraps=sorted) as sort:
        assert percentiles([9, 1, 5]) == {"0": 1, "25": 3, "50": 5, "75": 7, "100": 9}
    assert sort.call_count == 1
