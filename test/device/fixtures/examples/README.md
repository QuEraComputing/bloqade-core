# API schema examples

Sanitized JSON dumps captured from the live qlam-core API, used as the
reference wire shapes for the builders in `../remote.py`.

`test_wire_schema.py` validates these directly against the installed
qlam-core models (pinned to `qlam-core~=0.7.0` in `pyproject.toml`;
a test asserts the installed version matches the pin).

| capture | status |
|---|---|
| `task_*.json` | current — captured 2026-08-19 against 0.7.0 (carry `group`) |
| `task_definition_response_*.json` | **stale** — 0.6.x capture, missing the now-required `group`; marked `xfail(strict=True)` |
| `public_compilation_*.json` | **stale** — same |
| `results_envelope_*.json` | not schema-validated (qlam-core does not model the results envelope); still the reference for `remote.make_result_*` |
| `task_list_page.json` | not schema-validated |

To refresh a stale capture, re-fetch it from the live API and drop the
`xfail` marker for that file — the strict marker will fail loudly if the
capture is refreshed without removing it.

Notable shape these pin down: a results-envelope subtask carries **no**
`subtask_index`/`subtask_id`; the index appears only on each `shot_results`
entry (see `results_envelope_completed.json`).
