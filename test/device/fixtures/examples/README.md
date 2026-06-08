# API schema examples

Sanitized JSON dumps captured from the live qlam-core API, used as the
reference wire shapes for the builders in `../remote.py`.

**Verified against qlam-core v0.2.0** (the `qlam-core>=0.2.0` pin in
`pyproject.toml`). If that pin is bumped, re-capture these dumps and re-check
the `remote.py` builders against the new shapes.

Notable shape these pin down: a results-envelope subtask carries **no**
`subtask_index`/`subtask_id`; the index appears only on each `shot_results`
entry (see `results_envelope_completed.json`).
