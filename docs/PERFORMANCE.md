# Percentile summary performance

Batch dimension summaries sort each input vector once and reuse that ordering
for the 0th, 25th, 50th, 75th and 100th percentiles. Previously each percentile
sorted the same vector independently. Both approaches remain O(n log n) for
the fixed five percentiles: this is a constant-factor improvement, not a new
asymptotic bound. The interpolation, rounding, empty-input handling and input
order are unchanged. No dataset, split, model or published measurement changes.

Run: PYTHONPATH=src python -m pytest -q tests/test_percentiles.py
