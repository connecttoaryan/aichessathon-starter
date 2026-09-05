# Agent versions

| Version | Description | Key features | Benchmark command | Benchmark result |
| --- | --- | --- | --- | --- |
| v1 | First alpha-beta engine | Material and piece-square evaluation; alpha-beta negamax; tactical move ordering; iterative deepening; capture quiescence; safe clock management | `uv run python -m harness.arena --agent versions/v1_first_engine --opponent baselines/minimax --games 20` | record after final minimax run |
