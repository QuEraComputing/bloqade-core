from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TypeVar

import numpy as np
from typing_extensions import Self

from .local_storage import ShotFilter, ShotResult, StorageBackend, StorageFilter

ResultType = TypeVar("ResultType", bound="Result")


def _default_shot_filter() -> ShotFilter:
    return ShotFilter(frame_type="DETECTED")


@dataclass(kw_only=True)
class Result:
    """Result view over stored shots.

    Merge-oriented methods assume each selected task ID has the same subtask
    structure.

    Attributes:
        storage (StorageBackend): Storage backend that holds shots and task
            metadata.
        shot_filter (ShotFilter): Filter used when reading shots and deriving
            subtask scope. Defaults to the DETECTED frame type.
    """

    storage: StorageBackend
    shot_filter: ShotFilter = field(default_factory=_default_shot_filter)

    _is_valid: bool = field(init=False, default=False)

    @property
    def storage_filter(self) -> StorageFilter:
        """Return the subtask-level portion of `shot_filter`.

        Returns:
            StorageFilter: Filter containing task IDs, subtask indices, and
                task-subtask pairs from `shot_filter`.
        """
        return StorageFilter(
            task_ids=self.shot_filter.task_ids,
            subtask_indices=self.shot_filter.subtask_indices,
            task_subtask_pairs=self.shot_filter.task_subtask_pairs,
        )

    def validate(self) -> None:
        """Validate that selected task IDs can be merged by subtask index.

        Compatible task IDs have the same `program_index` and equal arguments
        for each shared `subtask_index`. Different shot counts are allowed, and
        None and empty dictionaries are treated as equivalent arguments.

        Raises:
            ValueError: If selected task IDs disagree on `program_index` or
                arguments for the same `subtask_index`.
        """

        if self._is_valid:
            return

        if self.shot_filter.task_ids is None:
            task_ids = self.storage.task_ids()
        else:
            task_ids = self.shot_filter.task_ids

        if len(task_ids) == 1:
            # Just a single task, we're safe
            self._is_valid = True
            return

        full_subtasks = self.full_subtasks()

        subtask_index_groups: dict[int, list[dict]] = {}
        for subtask in full_subtasks:
            subtask_index_groups.setdefault(subtask["subtask_index"], []).append(
                subtask
            )

        _VERIFY_HINT = "Pass verify=False to skip this check."

        for idx, group in subtask_index_groups.items():
            if len(group) <= 1:
                continue
            ref_subtask = group[0]
            for other_subtask in group[1:]:
                if other_subtask["program_index"] != ref_subtask["program_index"]:
                    raise ValueError(
                        f"task_ids disagree on program_index for subtask_index={idx}: "
                        f"{ref_subtask['task_id']!r} -> {ref_subtask['program_index']}, "
                        f"{other_subtask['task_id']!r} -> {other_subtask['program_index']}. "
                        + _VERIFY_HINT
                    )
                ref_arguments = ref_subtask["arguments"]
                other_arguments = other_subtask["arguments"]
                if not ref_arguments and not other_arguments:
                    # NOTE: treat empty dict and None as equal
                    continue
                if other_subtask["arguments"] != ref_subtask["arguments"]:
                    raise ValueError(
                        f"task_ids disagree on arguments for subtask_index={idx}: "
                        f"{ref_subtask['task_id']!r} -> {ref_subtask['arguments']!r}, "
                        f"{other_subtask['task_id']!r} -> {other_subtask['arguments']!r}. "
                        + _VERIFY_HINT
                    )

        self._is_valid = True

    def _shot_results_for_subtasks(self, subtasks: list[dict]) -> list[np.ndarray]:
        shot_results = []
        for subtask in subtasks:
            shot_filter = replace(
                self.shot_filter, subtask_indices=(subtask["subtask_index"],)
            )
            shot_results.append(self.storage.get_shots(shot_filter=shot_filter))

        shots_per_subtask = [
            np.array([shot_result.bitstring for shot_result in shot_results[i]])
            for i in range(len(shot_results))
        ]

        return shots_per_subtask

    def shot_results(self, verify: bool = True) -> list[np.ndarray]:
        """Return physical shot bitstrings grouped by merged subtask.

        Args:
            verify (bool): Whether to validate that selected task IDs can be
                merged before reading shots. Defaults to True.

        Returns:
            list[np.ndarray]: One two-dimensional boolean array per merged
                subtask, ordered by subtask index.

        Raises:
            ValueError: If `verify` is True and selected task IDs cannot be
                merged.
        """
        subtasks = self.subtasks(verify=verify)
        return self._shot_results_for_subtasks(subtasks)

    def arguments(self, verify: bool = True) -> list[dict | None]:
        """Return subtask arguments after merging selected task IDs.

        Args:
            verify (bool): Whether to validate that selected task IDs can be
                merged before reading arguments. Defaults to True.

        Returns:
            list[dict | None]: One arguments entry per merged subtask, ordered
                by subtask index.

        Raises:
            ValueError: If `verify` is True and selected task IDs cannot be
                merged.
        """
        subtasks = self.subtasks(verify=verify)
        return [subtask["arguments"] for subtask in subtasks]

    def full_arguments(self) -> list[dict | None]:
        """Return all stored subtask arguments without merging task IDs.

        Returns:
            list[dict | None]: Arguments for every selected stored subtask row,
                including `None` for rows without arguments.
        """
        return self.storage.get_arguments(storage_filter=self.storage_filter)

    def subtasks(self, verify: bool = True) -> list[dict]:
        """Return subtasks merged across selected task IDs.

        The merged view is ordered by `subtask_index`, removes ambiguous
        per-task fields (`task_id` and `metadata`), and aggregates `num_shots`.
        Use `full_subtasks` when task IDs or metadata must be preserved.

        Args:
            verify (bool): Whether to validate that selected task IDs can be
                merged before returning subtasks. Defaults to True.

        Returns:
            list[dict]: Merged subtask dictionaries ordered by subtask index.

        Raises:
            ValueError: If `verify` is True and selected task IDs cannot be
                merged.
        """
        if verify:
            self.validate()

        subtasks = self.storage.get_subtasks(storage_filter=self.storage_filter)
        subtasks_by_index: dict[int, dict] = {}
        for subtask in subtasks:
            # NOTE: remove task_id and metadata since we're merging multiple ones
            subtask.pop("task_id")
            subtask.pop("metadata")
            idx = subtask["subtask_index"]
            if idx in subtasks_by_index:
                subtasks_by_index[idx]["num_shots"] += subtask["num_shots"]
            else:
                subtasks_by_index[idx] = subtask
        return [subtasks_by_index[i] for i in sorted(subtasks_by_index)]

    def full_subtasks(self) -> list[dict]:
        """Return selected stored subtasks without merging task IDs.

        If selected task IDs share the same subtask structure, this view
        contains one row per task and subtask.

        Returns:
            list[dict]: Full subtask dictionaries selected by `storage_filter`.
        """
        return self.storage.get_subtasks(storage_filter=self.storage_filter)

    def task_ids(self) -> set[str]:
        """Return task IDs selected by this result view.

        Returns:
            set[str]: Task IDs from `shot_filter.task_ids` when set; otherwise
                all task IDs known to storage.
        """
        if self.shot_filter.task_ids is not None:
            return set(self.shot_filter.task_ids)

        return self.storage.task_ids()

    def _from_where_filters(
        self,
        where_filter: StorageFilter,
    ) -> Self:
        """Build a narrowed result by overlaying `where_filter`'s `task_subtask_pairs`
        onto `self.shot_filter`. All other scoping on `self` is preserved.
        """
        shot_filter = replace(
            self.shot_filter,
            task_subtask_pairs=where_filter.task_subtask_pairs,
        )
        return type(self)(
            storage=self.storage,
            shot_filter=shot_filter,
        )

    def where_subtasks(
        self,
        predicate: Callable[[dict], bool],
    ) -> Self:
        """Return a result narrowed to subtasks matching a predicate.

        The predicate sees only subtasks in the scope of `self.shot_filter`; the
        returned result inherits that scope intersected with the matches.

        Args:
            predicate (Callable[[dict], bool]): Predicate applied to each
                selected subtask dictionary.

        Returns:
            Self: A narrowed result view.
        """
        where_filter = self.storage.filter_by_subtasks(
            predicate, storage_filter=self.storage_filter
        )
        return self._from_where_filters(
            where_filter=where_filter,
        )

    def where_arguments(
        self,
        predicate: Callable[[dict | None], bool],
    ) -> Self:
        """Return a result narrowed by subtask arguments.

        The predicate sees only subtasks in the scope of `self.shot_filter`; the
        returned result inherits that scope intersected with the matches.

        NOTE: the Subtask model coerces bool values in `arguments` to float
        (True -> 1.0, False -> 0.0). Predicates that rely on identity
        (`is True`) or strict types will silently match nothing for bool-valued
        arguments. Use `== 1` or `> 0` instead, or store non-bool
        discriminators.

        Args:
            predicate (Callable[[dict | None], bool]): Predicate applied to each
                selected subtask's arguments.

        Returns:
            Self: A narrowed result view.
        """
        where_filter = self.storage.filter_by_arguments(
            predicate=predicate, storage_filter=self.storage_filter
        )
        return self._from_where_filters(
            where_filter=where_filter,
        )

    def where_metadata(
        self,
        predicate: Callable[[dict | None], bool],
    ) -> Self:
        """Return a result narrowed by JSON-decoded user metadata.

        Expects user_metadata to be a JSON-serialized dict; non-JSON values raise. To
        filter on raw strings instead, use `where_subtasks` and parse manually. The
        predicate sees only subtasks in the scope of `self.shot_filter`; the returned
        result inherits that scope intersected with the matches.

        Args:
            predicate (Callable[[dict | None], bool]): Predicate applied to each
                selected subtask's decoded `user_metadata`.

        Returns:
            Self: A narrowed result view.
        """
        where_filter = self.storage.filter_by_metadata(
            predicate=predicate, storage_filter=self.storage_filter
        )
        return self._from_where_filters(
            where_filter=where_filter,
        )

    def where_shots(
        self,
        predicate: Callable[[ShotResult], bool],
        predicate_filter: ShotFilter | None = None,
    ) -> Self:
        """Return a result narrowed by shot-level predicate.

        `predicate_filter` selects the shots the predicate evaluates against;
        defaults to `self.shot_filter` (predicate sees the same shots the
        current result fetches). The returned result inherits
        `self.shot_filter`'s scope (notably `frame_type`) intersected with the
        matching shot pairs.

        To precondition on one frame and return another, such as
        SORTED-was-all-1 -> DETECTED, pass
        `predicate_filter=replace(self.shot_filter, frame_type='SORTED')`.

        Args:
            predicate (Callable[[ShotResult], bool]): Predicate applied to each
                shot selected by `predicate_filter`.
            predicate_filter (ShotFilter | None): Filter used only for predicate
                evaluation. When None, `self.shot_filter` is used. Defaults to
                None.

        Returns:
            Self: A narrowed result view.
        """
        if predicate_filter is None:
            predicate_filter = self.shot_filter

        where_filter = self.storage.filter_by_shots(
            predicate, shot_filter=predicate_filter
        )
        shot_filter = replace(
            self.shot_filter,
            task_shot_pairs=where_filter.task_shot_pairs,
        )
        return type(self)(
            storage=self.storage,
            shot_filter=shot_filter,
        )
