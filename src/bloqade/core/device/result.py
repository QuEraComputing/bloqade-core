import json
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal, TypeVar

import numpy as np
from typing_extensions import Self

from .local_storage import (
    ShotFilter,
    ShotResult,
    StorageBackend,
    StorageFilter,
)

ResultType = TypeVar("ResultType", bound="Result")
ShotValue = TypeVar("ShotValue")


@dataclass(frozen=True)
class ResultScope(StorageFilter):
    task_shot_pairs: tuple[tuple[str, int], ...] | None = None


@dataclass(kw_only=True)
class Result:
    """Result view over stored shots.

    Merge-oriented methods assume each selected task ID has the same subtask
    structure.

    Attributes:
        storage (StorageBackend): Storage backend that holds shots and task
            metadata.
        scope (ResultScope): Filter used when reading shots and deriving
            subtask scope.
    """

    storage: StorageBackend
    scope: ResultScope = field(default_factory=ResultScope)

    _is_valid: bool = field(init=False, default=False)

    @property
    def storage_filter(self) -> StorageFilter:
        """Return the subtask-level portion of `scope`.

        Returns:
            StorageFilter: Filter containing task IDs, subtask indices, and
                task-subtask pairs from `scope`.
        """
        return StorageFilter(
            task_ids=self.scope.task_ids,
            subtask_indices=self.scope.subtask_indices,
            task_subtask_pairs=self.scope.task_subtask_pairs,
        )

    @staticmethod
    def _shot_filter(
        scope: ResultScope, *, frame_type: Literal["SORTED", "DETECTED"]
    ) -> ShotFilter:
        return ShotFilter(
            task_ids=scope.task_ids,
            subtask_indices=scope.subtask_indices,
            task_subtask_pairs=scope.task_subtask_pairs,
            task_shot_pairs=scope.task_shot_pairs,
            frame_type=frame_type,
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

        if self.scope.task_ids is None:
            task_ids = self.storage.task_ids()
        else:
            task_ids = self.scope.task_ids

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

    def _detected_shot_results_for_subtasks(
        self, subtasks: list[dict]
    ) -> list[np.ndarray]:
        shot_results = []
        for subtask in subtasks:
            shot_filter = Result._shot_filter(
                replace(
                    self.scope,
                    subtask_indices=(subtask["subtask_index"],),
                ),
                frame_type="DETECTED",
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
        return self._detected_shot_results_for_subtasks(subtasks)

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

    def group_shots_by_metadata(
        self,
        shots: Sequence[Sequence[ShotValue]],
        metadata_keys: Sequence[str],
    ) -> dict[tuple[Hashable, ...], list[ShotValue]]:
        """Aggregate per-subtask shots by selected user-metadata values.

        ``shots`` must contain one sequence per entry of :meth:`subtasks`, in
        that same order. For a result merged from multiple task IDs, every full
        subtask contributing to one merged subtask must agree on the requested
        metadata values.

        Args:
            shots: Per-subtask shot sequences to aggregate.
            metadata_keys: User-metadata keys whose values form each group key.

        Returns:
            A dictionary mapping metadata-value tuples to flattened shot lists.
            The order within each list follows ``subtasks()`` order, then the
            order supplied within each corresponding shot sequence.

        Raises:
            ValueError: If ``shots`` does not align with the selected merged
                subtasks, a full subtask has invalid/missing user metadata, a
                requested key is missing, metadata disagrees across merged
                task IDs, or a key value is unhashable.
        """
        subtasks = self.subtasks()
        if len(shots) != len(subtasks):
            raise ValueError(
                "shots must contain one sequence per selected merged subtask: "
                f"got {len(shots)} sequences for {len(subtasks)} subtasks."
            )

        metadata_keys_by_subtask_index: dict[int, tuple[Hashable, ...]] = {}
        for subtask in self.full_subtasks():
            subtask_ref = f"{subtask['task_id']!r}/{subtask['subtask_index']}"
            metadata = subtask["metadata"]
            user_metadata_text = (
                None if metadata is None else metadata.get("user_metadata")
            )
            if user_metadata_text is None:
                raise ValueError(f"Subtask {subtask_ref} has no user metadata.")

            try:
                user_metadata = json.loads(user_metadata_text)
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Subtask {subtask_ref} has invalid JSON user metadata."
                ) from exc

            if not isinstance(user_metadata, dict):
                raise TypeError(
                    f"Subtask {subtask_ref} user metadata must be a JSON object."
                )

            missing_keys = [key for key in metadata_keys if key not in user_metadata]
            if missing_keys:
                raise ValueError(
                    f"Subtask {subtask_ref} is missing metadata keys: {missing_keys!r}."
                )

            key = tuple(user_metadata[name] for name in metadata_keys)
            try:
                hash(key)
            except TypeError as exc:
                raise ValueError(
                    f"Subtask {subtask_ref} metadata values for {metadata_keys!r} "
                    "must be hashable."
                ) from exc

            subtask_index = subtask["subtask_index"]
            existing_key = metadata_keys_by_subtask_index.setdefault(subtask_index, key)
            if existing_key != key:
                raise ValueError(
                    "Selected task IDs disagree on metadata values for "
                    f"subtask_index={subtask_index}: {existing_key!r} != {key!r}."
                )

        grouped: dict[tuple[Hashable, ...], list[ShotValue]] = {}
        for subtask, subtask_shots in zip(subtasks, shots, strict=True):
            key = metadata_keys_by_subtask_index[subtask["subtask_index"]]
            grouped.setdefault(key, []).extend(subtask_shots)

        return grouped

    def task_ids(self) -> set[str]:
        """Return task IDs selected by this result view.

        Returns:
            set[str]: Task IDs from `scope.task_ids` when set; otherwise
                all task IDs known to storage.
        """
        if self.scope.task_ids is not None:
            return set(self.scope.task_ids)

        return self.storage.task_ids()

    def _from_where_filters(
        self,
        where_filter: StorageFilter,
    ) -> Self:
        """Build a narrowed result by overlaying `where_filter`'s `task_subtask_pairs`
        onto `self.scope`. All other scoping on `self` is preserved.
        """
        scope = replace(
            self.scope,
            task_subtask_pairs=where_filter.task_subtask_pairs,
        )
        return type(self)(
            storage=self.storage,
            scope=scope,
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

        The predicate sees only subtasks in the scope of `self.scope`; the
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
        predicate sees only subtasks in the scope of `self.scope`; the returned
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
    ) -> Self:
        """Return a result narrowed by shot-level predicate.

        `predicate_filter` selects the shots the predicate evaluates against;
        defaults to `self.scope` (predicate sees the same shots the
        current result fetches). The returned result inherits
        `self.scope`'s scope (notably `frame_type`) intersected with the
        matching shot pairs.

        To precondition on one frame and return another, such as
        SORTED-was-all-1 -> DETECTED, pass
        `predicate_filter=replace(self.shot_filter, frame_type='SORTED')`.

        Args:
            predicate (Callable[[ShotResult], bool]): Predicate applied to each
                shot selected by `predicate_filter`.
            predicate_filter (ShotFilter | None): Filter used only for predicate
                evaluation. When None, `self.scope` is used. Defaults to
                None.

        Returns:
            Self: A narrowed result view.
        """
        return self._where_frame_shots(predicate, "DETECTED")

    def where_sorted_shots(
        self,
        predicate: Callable[[ShotResult], bool],
    ) -> Self:
        """
        return DETECTED shots, filtered based on SORTED shots
        """
        return self._where_frame_shots(predicate, "SORTED")

    def _where_frame_shots(
        self,
        predicate: Callable[[ShotResult], bool],
        frame_type: Literal["DETECTED", "SORTED"],
    ) -> Self:
        shot_filter = self._shot_filter(
            self.scope,
            frame_type=frame_type,
        )
        where_filter = self.storage.filter_by_shots(
            predicate,
            shot_filter=shot_filter,
        )
        scope = replace(
            self.scope,
            task_shot_pairs=where_filter.task_shot_pairs,
        )
        return type(self)(
            storage=self.storage,
            scope=scope,
        )
