# API schema examples

Sanitized JSON dumps captured from the live qlam-core API, used as the
reference wire shapes for the builders in `../remote.py`.

These captures reflect the `qlam-core~=0.7.0` wire shapes used by the remote
fixture builders.

| capture | status |
|---|---|
| `task_*.json` | current — re-checked 2026-09-17 against 0.7.0 (`group`, nullable `profile_id`) |
| `task_definition_response_*.json` | current — re-captured 2026-09-17 (`group` is required) |
| `public_compilation_*.json` | current — re-captured 2026-09-17 (`group` is required) |
| `results_envelope_*.json` | current — re-checked 2026-09-17; qlam-core does not model this envelope, so its shape is asserted directly |
| `task_list_page.json` | current — re-checked 2026-09-17; each item is validated as a `Task` |

Identifiers, timestamps, program contents, and measurements are sanitized;
field presence and nesting reflect the live response.

Notable shapes these pin down: each results-envelope element carries `group`,
while a results-envelope subtask carries **no** `subtask_index`/`subtask_id`;
the index appears only on each `shot_results` entry (see
`results_envelope_completed.json`).
