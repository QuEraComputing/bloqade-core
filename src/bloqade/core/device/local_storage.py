import datetime
import json
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
from qlam_core.plugins.tasks.api.tasks_models import (
    Program,
    Subtask,
    TaskDefinition,
    TaskMetadata,
)


@dataclass(frozen=True)
class ShotResult:
    """Stored result for one shot and frame type.

    Attributes:
        task_id (str): Backend task ID that produced the shot.
        shot_index (int): Shot index within the task.
        subtask_index (int): Subtask index within the task definition.
        subtask_shot_index (int): Shot index local to the subtask.
        frame_type (str): Result frame type, such as "DETECTED".
        bitstring (np.ndarray): Boolean measurement bitstring.
    """

    task_id: str
    shot_index: int
    subtask_index: int
    subtask_shot_index: int
    frame_type: str
    bitstring: np.ndarray


@dataclass(frozen=True)
class StorageFilter:
    """Filter for task and subtask metadata rows.

    All separate filters are AND-ed. Use `task_subtask_pairs` to specify exact
    pairs rather than combining `task_ids` and `subtask_indices` independently.

    Attributes:
        task_ids (tuple[str, ...] | None): Task IDs to include. Defaults to
            None.
        subtask_indices (tuple[int, ...] | None): Subtask indices to include.
            Defaults to None.
        task_subtask_pairs (tuple[tuple[str, int], ...] | None): Exact
            `(task_id, subtask_index)` pairs to include. Defaults to None.
    """

    task_ids: tuple[str, ...] | None = None
    subtask_indices: tuple[int, ...] | None = None

    task_subtask_pairs: tuple[tuple[str, int], ...] | None = None
    """Exact pairs to use instead of separate task and subtask filters."""


# NOTE: default arguments rely on this being frozen
@dataclass(frozen=True)
class ShotFilter(StorageFilter):
    """Filter for shot result rows.

    Extends `StorageFilter` with frame-type and shot-pair criteria. `frame_type`
    is normalized to uppercase during initialization.

    Attributes:
        frame_type (str | None): Frame type to include. Defaults to None.
        task_shot_pairs (tuple[tuple[str, int], ...] | None): Exact
            `(task_id, shot_index)` pairs to include. Defaults to None.
    """

    frame_type: str | None = None  # TODO: should be an enum
    task_shot_pairs: tuple[tuple[str, int], ...] | None = None

    def __post_init__(self):
        if self.frame_type is not None:
            object.__setattr__(self, "frame_type", self.frame_type.upper())


class _BloqadeSchemaVersion:
    version: str = "0.1.0"


class StorageBackend(ABC):
    """Abstract storage backend for shot results.

    Implementations store shot rows together with enough task metadata to
    reconstruct `TaskDefinition` objects and create `Result` views.

    NOTE: methods that return mutable dictionaries should return independent
    copies. Result helpers may mutate returned metadata records while building
    merged views.
    """

    @abstractmethod
    def add_shots(self, shots: Iterable[ShotResult]) -> None:
        """Store shot result rows.

        Args:
            shots (Iterable[ShotResult]): Shot rows to store.
        """
        ...

    @abstractmethod
    def get_shots(
        self,
        *,
        shot_filter: ShotFilter | None = None,
    ) -> Iterable[ShotResult]:
        """Return stored shot rows matching a filter.

        Keyword Args:
            shot_filter (ShotFilter | None): Optional shot filter. When None,
                all shots are returned. Defaults to None.

        Returns:
            Iterable[ShotResult]: Matching shot rows.
        """
        ...

    @abstractmethod
    def task_ids(self) -> set[str]:
        """Return task IDs with stored task definitions.

        Returns:
            set[str]: Stored task IDs.
        """
        ...

    @abstractmethod
    def add_task_definition(
        self,
        task_id: str,
        task_definition: TaskDefinition,
        creation_time: datetime.datetime,
    ):
        """Store a task definition and its creation time.

        Args:
            task_id (str): Backend task ID.
            task_definition (TaskDefinition): Task definition to store.
            creation_time (datetime.datetime): Backend task creation time.
        """
        ...

    @abstractmethod
    def get_program_language(self, task_id: str) -> str:
        """Return the program language for a stored task definition.

        Args:
            task_id (str): Backend task ID.

        Returns:
            str: Program language stored for the task.

        Raises:
            KeyError: If `task_id` is not present.
        """
        ...

    @abstractmethod
    def get_task_creation_time(self, task_id: str) -> datetime.datetime:
        """Return the creation time for a stored task.

        Args:
            task_id (str): Backend task ID.

        Returns:
            datetime.datetime: Stored task creation time.

        Raises:
            KeyError: If `task_id` is not present.
        """
        ...

    @abstractmethod
    def get_programs(self, task_ids: tuple[str, ...] | None = None) -> list[dict]:
        """Return stored program records.

        Args:
            task_ids (tuple[str, ...] | None): Optional task IDs to include.
                When None, programs for all stored task IDs are returned.
                Defaults to None.

        Returns:
            list[dict]: Program dictionaries with `task_id`, `program_index`,
                and `content`.
        """
        ...

    @abstractmethod
    def get_subtasks(self, storage_filter: StorageFilter | None = None) -> list[dict]:
        """Return stored subtask records.

        Args:
            storage_filter (StorageFilter | None): Optional subtask metadata
                filter. When None, all subtasks are returned. Defaults to None.

        Returns:
            list[dict]: Subtask dictionaries, including task ID, subtask index,
                program index, shot count, arguments, metadata, and completion
                date.
        """
        ...

    @abstractmethod
    def update_subtasks_completed_date(self, task_id, subtasks: list[dict]) -> None:
        """Update completion times for stored subtasks.

        Args:
            task_id (str): Backend task ID.
            subtasks (list[dict]): Subtask dictionaries containing
                `subtask_index` and `completed_date`.
        """
        ...

    def get_task_definition(self, task_id: str) -> TaskDefinition:
        """Reconstruct a task definition from stored metadata.

        Args:
            task_id (str): Backend task ID.

        Returns:
            TaskDefinition: Reconstructed task definition.
        """
        program_language = self.get_program_language(task_id=task_id)

        program_dicts = self.get_programs(task_ids=(task_id,))
        program_dicts.sort(key=lambda prog: prog["program_index"])

        subtask_dicts = self.get_subtasks(
            storage_filter=StorageFilter(task_ids=(task_id,))
        )
        subtask_dicts.sort(key=lambda subtask: subtask["subtask_index"])

        programs = [Program(content=prog["content"]) for prog in program_dicts]
        subtasks = []
        for subtask_dict in subtask_dicts:
            metadata = subtask_dict["metadata"]
            if metadata is None:
                subtask_metadata = None
            else:
                subtask_metadata = TaskMetadata(**metadata)
            subtasks.append(
                Subtask(
                    program_index=subtask_dict["program_index"],
                    num_shots=subtask_dict["num_shots"],
                    arguments=subtask_dict["arguments"],
                    subtask_metadata=subtask_metadata,
                )
            )

        return TaskDefinition(
            program_language=program_language,
            programs=programs,
            subtasks=subtasks,
        )

    def get_arguments(self, storage_filter: StorageFilter | None = None) -> list[dict]:
        """Return arguments from stored subtasks.

        Args:
            storage_filter (StorageFilter | None): Optional subtask metadata
                filter. Defaults to None.

        Returns:
            list[dict]: Arguments from matching subtasks.
        """
        subtasks = self.get_subtasks(storage_filter=storage_filter)
        return [subtask["arguments"] for subtask in subtasks]

    # Building filters in a convenient way
    def filter_by_subtasks(
        self,
        predicate: Callable[[dict], bool],
        storage_filter: StorageFilter | None = None,
    ) -> StorageFilter:
        """Build a filter from subtasks matching a predicate.

        Args:
            predicate (Callable[[dict], bool]): Predicate applied to each
                selected subtask dictionary.
            storage_filter (StorageFilter | None): Optional filter used before
                evaluating the predicate. Defaults to None.

        Returns:
            StorageFilter: Filter containing matching `(task_id, subtask_index)`
                pairs.
        """

        subtasks = self.get_subtasks(storage_filter=storage_filter)
        task_subtask_pairs = []
        for subtask in subtasks:
            if not predicate(subtask):
                continue

            task_subtask_pairs.append((subtask["task_id"], subtask["subtask_index"]))

        return StorageFilter(
            task_subtask_pairs=tuple(task_subtask_pairs),
        )

    def filter_by_metadata(
        self,
        predicate: Callable[[dict | None], bool],
        storage_filter: StorageFilter | None = None,
    ) -> StorageFilter:
        """Build a filter from JSON-decoded user metadata.

        This expects metadata to have been set as a dictionary and serialized as
        JSON. If metadata cannot be deserialized by JSON, fetch and filter the
        subtasks manually.

        Args:
            predicate (Callable[[dict | None], bool]): Predicate applied to each
                selected subtask's decoded `user_metadata`.
            storage_filter (StorageFilter | None): Optional filter used before
                evaluating the predicate. Defaults to None.

        Returns:
            StorageFilter: Filter containing matching `(task_id, subtask_index)`
                pairs.
        """
        subtasks = self.get_subtasks(storage_filter=storage_filter)

        task_subtask_pairs = []
        for subtask in subtasks:
            metadata = subtask["metadata"]
            if metadata is None:
                user_metadata = None
            else:
                user_metadata_str = metadata["user_metadata"]
                user_metadata = (
                    json.loads(user_metadata_str)
                    if user_metadata_str is not None
                    else None
                )

            if not predicate(user_metadata):
                continue

            task_subtask_pairs.append((subtask["task_id"], subtask["subtask_index"]))

        return StorageFilter(task_subtask_pairs=tuple(task_subtask_pairs))

    def filter_by_arguments(
        self,
        predicate: Callable[[dict | None], bool],
        storage_filter: StorageFilter | None = None,
    ) -> StorageFilter:
        """Build a filter from subtask arguments matching a predicate.

        Args:
            predicate (Callable[[dict | None], bool]): Predicate applied to each
                selected subtask's arguments.
            storage_filter (StorageFilter | None): Optional filter used before
                evaluating the predicate. Defaults to None.

        Returns:
            StorageFilter: Filter containing matching `(task_id, subtask_index)`
                pairs.
        """
        subtasks = self.get_subtasks(storage_filter=storage_filter)

        task_subtask_pairs = []
        for subtask in subtasks:
            arguments = subtask.get("arguments")
            if not predicate(arguments):
                continue
            task_subtask_pairs.append((subtask["task_id"], subtask["subtask_index"]))

        return StorageFilter(task_subtask_pairs=tuple(task_subtask_pairs))

    def filter_by_shots(
        self,
        predicate: Callable[[ShotResult], bool],
        shot_filter: ShotFilter | None = None,
    ) -> ShotFilter:
        """Build a filter from shots matching a predicate.

        Args:
            predicate (Callable[[ShotResult], bool]): Predicate applied to each
                selected shot.
            shot_filter (ShotFilter | None): Optional filter used before
                evaluating the predicate. Defaults to None.

        Returns:
            ShotFilter: Filter containing matching `(task_id, shot_index)`
                pairs.
        """
        shots = self.get_shots(shot_filter=shot_filter)

        task_shot_pairs = []
        for shot in shots:
            if not predicate(shot):
                continue

            task_shot_pairs.append((shot.task_id, shot.shot_index))

        return ShotFilter(
            task_shot_pairs=tuple(task_shot_pairs),
        )


@dataclass
class DictStorage(StorageBackend):
    """In-memory storage backend for shot results.

    This backend is useful for tests, examples, and short-lived sessions. Data
    is not persisted across Python processes.
    """

    _data: dict[tuple, ShotResult] = field(init=False, default_factory=dict)
    _metadata: dict = field(init=False, default_factory=dict)

    def __repr__(self) -> str:
        return repr(self._data)

    def add_shots(self, shots: Iterable[ShotResult]) -> None:
        """Store shot result rows in memory.

        Args:
            shots (Iterable[ShotResult]): Shot rows to store.
        """
        for shot in shots:
            key = (
                shot.task_id,
                shot.shot_index,
                shot.frame_type,
            )
            self._data[key] = shot

    def get_shots(
        self,
        *,
        shot_filter: ShotFilter | None = None,
    ) -> Iterable[ShotResult]:
        """Return in-memory shot rows matching a filter.

        Keyword Args:
            shot_filter (ShotFilter | None): Optional shot filter. When None,
                all shots are returned. Defaults to None.

        Returns:
            Iterable[ShotResult]: Matching shot rows.
        """
        if shot_filter is None:
            yield from self._data.values()
            return

        for val in self._data.values():
            if (
                shot_filter.task_ids is not None
                and val.task_id not in shot_filter.task_ids
            ):
                continue
            if (
                shot_filter.frame_type is not None
                and val.frame_type.upper() != shot_filter.frame_type
            ):
                continue
            if (
                shot_filter.subtask_indices is not None
                and val.subtask_index not in shot_filter.subtask_indices
            ):
                continue
            if (
                shot_filter.task_subtask_pairs is not None
                and (val.task_id, val.subtask_index)
                not in shot_filter.task_subtask_pairs
            ):
                continue

            if (
                shot_filter.task_shot_pairs is not None
                and (val.task_id, val.shot_index) not in shot_filter.task_shot_pairs
            ):
                continue

            yield val

    def task_ids(self) -> set[str]:
        """Return task IDs with stored task definitions.

        Returns:
            set[str]: Stored task IDs.
        """
        task_defs = self._metadata.get("task_definitions")
        if task_defs is None:
            return set()

        return set(task_defs.keys())

    def add_task_definition(
        self,
        task_id: str,
        task_definition: TaskDefinition,
        creation_time: datetime.datetime,
    ):
        """Store a task definition and creation time in memory.

        Args:
            task_id (str): Backend task ID.
            task_definition (TaskDefinition): Task definition to store.
            creation_time (datetime.datetime): Backend task creation time.
        """
        current_defs = self._metadata.get("task_definitions", {})
        current_defs[task_id] = {
            "task_id": task_id,
            "program_language": task_definition.program_language,
            "creation_time": creation_time,
        }
        self._metadata["task_definitions"] = current_defs

        current_programs = self._metadata.get("programs", {})
        for i, program in enumerate(task_definition.programs):
            current_programs[(task_id, i)] = {
                "task_id": task_id,
                "program_index": i,
                "content": program.content,
            }
        self._metadata["programs"] = current_programs

        current_subtasks = self._metadata.get("subtasks", {})
        for i, subtask in enumerate(task_definition.subtasks):
            if subtask.subtask_metadata is None:
                metadata = None
            else:
                metadata = subtask.subtask_metadata.model_dump()

            current_subtasks[(task_id, i)] = {
                "task_id": task_id,
                "subtask_index": i,
                "program_index": subtask.program_index,
                "num_shots": subtask.num_shots,
                "arguments": subtask.arguments,
                "metadata": metadata,
                "completed_date": None,
            }
        self._metadata["subtasks"] = current_subtasks

    def get_program_language(self, task_id: str) -> str:
        """Return the program language for a stored task definition.

        Args:
            task_id (str): Backend task ID.

        Returns:
            str: Stored program language.

        Raises:
            KeyError: If `task_id` is not present.
        """
        return self._metadata["task_definitions"][task_id]["program_language"]

    def get_task_creation_time(self, task_id: str) -> datetime.datetime:
        """Return the creation time for a stored task.

        Args:
            task_id (str): Backend task ID.

        Returns:
            datetime.datetime: Stored task creation time.

        Raises:
            KeyError: If `task_id` is not present.
        """
        return self._metadata["task_definitions"][task_id]["creation_time"]

    def get_programs(self, task_ids: tuple[str, ...] | None = None) -> list[dict]:
        """Return stored program records.

        Args:
            task_ids (tuple[str, ...] | None): Optional task IDs to include.
                When None, programs for all stored tasks are returned. Defaults
                to None.

        Returns:
            list[dict]: Independent program dictionaries.
        """
        programs = self._metadata.get("programs")
        if programs is None:
            return []

        # NOTE: extra dict call to create a copy so we can safely mutate the return programs
        if task_ids is None:
            return list(map(dict, programs.values()))
        else:
            return [
                dict(prog) for prog in programs.values() if prog["task_id"] in task_ids
            ]

    def get_subtasks(self, storage_filter: StorageFilter | None = None) -> list[dict]:
        """Return stored subtask records.

        Args:
            storage_filter (StorageFilter | None): Optional metadata filter.
                When None, all subtasks are returned. Defaults to None.

        Returns:
            list[dict]: Independent subtask dictionaries.
        """
        subtasks = self._metadata.get("subtasks")
        if subtasks is None:
            return []

        if storage_filter is None:
            # NOTE: extra dict call so we return a copy and can safely mutate the result
            return list(map(dict, subtasks.values()))

        filtered_subtasks = []
        for subtask in subtasks.values():
            if (
                storage_filter.task_ids is not None
                and subtask["task_id"] not in storage_filter.task_ids
            ):
                continue
            if (
                storage_filter.subtask_indices is not None
                and subtask["subtask_index"] not in storage_filter.subtask_indices
            ):
                continue

            if (
                storage_filter.task_subtask_pairs is not None
                and (subtask["task_id"], subtask["subtask_index"])
                not in storage_filter.task_subtask_pairs
            ):
                continue

            # NOTE: extra dict call so we return a copy and can safely mutate the result
            filtered_subtasks.append(dict(subtask))
        return filtered_subtasks

    def update_subtasks_completed_date(self, task_id, subtasks: list[dict]) -> None:
        """Update completion times for stored subtasks.

        Args:
            task_id (str): Backend task ID.
            subtasks (list[dict]): Subtask dictionaries containing
                `subtask_index` and `completed_date`.
        """
        current_subtasks = self._metadata.get("subtasks")
        if current_subtasks is None:
            return

        for subtask in subtasks:
            completed_date = subtask["completed_date"]
            if completed_date is None:
                continue

            idx = subtask["subtask_index"]
            if (task_id, idx) not in current_subtasks:
                continue

            if isinstance(completed_date, str):
                completed_date = datetime.datetime.fromisoformat(completed_date)

            current_subtasks[(task_id, idx)]["completed_date"] = completed_date


class SQLiteStorage(StorageBackend):
    """SQLite-backed storage for shot results.

    Use this backend to persist shots and task metadata across Python sessions.
    When used with a future, close the connection after fetching is complete,
    for example:

    ```python
    with SQLiteStorage("my_database.sql") as store:
        future = task.run_async(..., storage=store)
    ```

    Otherwise, garbage collection closes the connection later, which may keep
    the file lock open longer than expected.
    """

    def __init__(self, db_file: str):
        """Initialize a SQLite storage backend.

        Args:
            db_file (str): Path to the SQLite database file.

        Raises:
            ValueError: If the stored schema version does not match this
                package's expected schema version.
        """
        self.conn = sqlite3.connect(db_file)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS results (
                row_number INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                shot_index INTEGER NOT NULL,
                subtask_index INTEGER NOT NULL,
                subtask_shot_index INTEGER NOT NULL,
                frame_type TEXT NOT NULL,
                bitstring TEXT NOT NULL,
                UNIQUE(task_id, shot_index, frame_type)
            )
            """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS bloqade_schema (
                version_number TEXT PRIMARY KEY
            )
            """)
        self.conn.execute(
            """
            INSERT OR IGNORE INTO bloqade_schema (version_number) VALUES (?)
            """,
            (_BloqadeSchemaVersion.version,),
        )

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS programs (
                task_id TEXT,
                program_index INT,
                content TEXT NOT NULL,
                PRIMARY KEY (task_id, program_index)
            )
            """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS subtasks (
                task_id TEXT NOT NULL,
                subtask_index INT NOT NULL,
                program_index INT NOT NULL,
                num_shots INT NOT NULL,
                arguments TEXT,
                metadata TEXT,
                completed_date TEXT,
                PRIMARY KEY (task_id, subtask_index)
            )
            """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS task_definitions (
                task_id TEXT PRIMARY KEY,
                program_language TEXT NOT NULL,
                creation_time TEXT NOT NULL
            )
            """)

        cur = self.conn.execute("SELECT version_number FROM bloqade_schema")
        (stored_version,) = cur.fetchone()
        if stored_version != _BloqadeSchemaVersion.version:
            raise ValueError(
                f"Schema version mismatch: expected {_BloqadeSchemaVersion.version}, found {stored_version}"
            )

        self.conn.commit()

    def add_shots(self, shots: Iterable[ShotResult]) -> None:
        """Store shot result rows in SQLite.

        Args:
            shots (Iterable[ShotResult]): Shot rows to store.
        """
        # TODO: optionally compress bit string data
        self.conn.executemany(
            """
            INSERT OR IGNORE INTO results (task_id, shot_index, subtask_index, subtask_shot_index, frame_type, bitstring)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    shot.task_id,
                    shot.shot_index,
                    shot.subtask_index,
                    shot.subtask_shot_index,
                    shot.frame_type,
                    "".join(str(int(b)) for b in shot.bitstring),
                )
                for shot in shots
            ),
        )
        self.conn.commit()

    @staticmethod
    def _storage_filter_to_where_clause(
        storage_filter: StorageFilter | None,
    ) -> tuple[list, list]:
        if storage_filter is None:
            return [], []

        where_clauses = []
        filter_values = []

        if storage_filter.task_ids is not None:
            if len(storage_filter.task_ids) == 0:
                where_clauses.append("0")  # NOTE: match nothing
            else:
                task_ids = storage_filter.task_ids
                placeholders = ", ".join("?" for _ in task_ids)
                where_clauses.append(f"task_id IN ({placeholders})")
                filter_values.extend(task_ids)

        if storage_filter.subtask_indices is not None:
            if len(storage_filter.subtask_indices) == 0:
                where_clauses.append("0")  # NOTE: match nothing
            else:
                subtask_indices = storage_filter.subtask_indices
                placeholders = ", ".join("?" for _ in subtask_indices)
                where_clauses.append(f"subtask_index IN ({placeholders})")
                filter_values.extend(subtask_indices)

        if storage_filter.task_subtask_pairs is not None:
            if len(storage_filter.task_subtask_pairs) == 0:
                where_clauses.append("0")  # NOTE: match nothing
            else:
                task_subtask_pairs = storage_filter.task_subtask_pairs
                placeholders = ", ".join("(?, ?)" for _ in task_subtask_pairs)
                where_clauses.append(
                    f"(task_id, subtask_index) IN (VALUES {placeholders})"
                )
                for task_id, subtask_index in task_subtask_pairs:
                    filter_values.extend([task_id, subtask_index])

        return where_clauses, filter_values

    @staticmethod
    def _shot_filter_to_where_clause(
        shot_filter: ShotFilter | None,
    ) -> tuple[list, list]:
        if shot_filter is None:
            return [], []

        where_clauses, filter_values = SQLiteStorage._storage_filter_to_where_clause(
            shot_filter
        )

        if shot_filter.frame_type is not None:
            where_clauses.append("frame_type = (?)")
            filter_values.append(shot_filter.frame_type)

        if shot_filter.task_shot_pairs is not None:
            if len(shot_filter.task_shot_pairs) == 0:
                where_clauses.append("0")
            else:
                task_shot_pairs = shot_filter.task_shot_pairs
                # NOTE: since we can have many shots, we need to work around
                # SQLite's limit of how many filter values you can apply of ~32k
                where_clauses.append(
                    "(task_id, shot_index) IN ("
                    "SELECT json_extract(value, '$[0]'), json_extract(value, '$[1]') "
                    "FROM json_each(?))"
                )
                filter_values.append(json.dumps([[t, s] for t, s in task_shot_pairs]))

        return where_clauses, filter_values

    def get_shots(
        self,
        *,
        shot_filter: ShotFilter | None = None,
    ) -> Iterable[ShotResult]:
        """Return SQLite shot rows matching a filter.

        Keyword Args:
            shot_filter (ShotFilter | None): Optional shot filter. When None,
                all shots are returned. Defaults to None.

        Returns:
            Iterable[ShotResult]: Matching shot rows.
        """
        query = "SELECT * FROM results"

        where_clauses, filter_values = self._shot_filter_to_where_clause(shot_filter)
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)

        cur = self.conn.execute(query, filter_values)

        for row in cur:
            shot = ShotResult(
                task_id=row["task_id"],
                shot_index=row["shot_index"],
                subtask_index=row["subtask_index"],
                subtask_shot_index=row["subtask_shot_index"],
                frame_type=row["frame_type"],
                bitstring=np.array(list(row["bitstring"]), dtype=np.uint8).view(bool),
            )
            yield shot

    def close(self):
        """Close the SQLite connection."""
        self.conn.close()

    def task_ids(self) -> set[str]:
        """Return task IDs with stored task definitions.

        Returns:
            set[str]: Stored task IDs.
        """
        cur = self.conn.execute("SELECT task_id FROM task_definitions")
        return {row["task_id"] for row in cur}

    @staticmethod
    def _datetime_to_sql_txt(time: datetime.datetime) -> str:
        return time.isoformat()

    @staticmethod
    def _sql_txt_to_datetime(time_str: str) -> datetime.datetime:
        return datetime.datetime.fromisoformat(time_str)

    def add_task_definition(
        self,
        task_id: str,
        task_definition: TaskDefinition,
        creation_time: datetime.datetime,
    ):
        """Store a task definition and creation time in SQLite.

        Args:
            task_id (str): Backend task ID.
            task_definition (TaskDefinition): Task definition to store.
            creation_time (datetime.datetime): Backend task creation time.
        """
        programs = task_definition.programs
        subtasks = task_definition.subtasks

        # task_id, program_index, content
        program_rows = [(task_id, i, programs[i].content) for i in range(len(programs))]

        subtask_rows = []
        for i in range(len(subtasks)):
            # task_id, subtask_index, program_index, num_shots, arguments, metadata
            subtask_row = [task_id, i, subtasks[i].program_index, subtasks[i].num_shots]

            if subtasks[i].arguments is not None:
                subtask_row.append(json.dumps(subtasks[i].arguments))
            else:
                subtask_row.append(None)

            subtask_metadata = subtasks[i].subtask_metadata
            if subtask_metadata is not None:
                subtask_row.append(subtask_metadata.model_dump_json())
            else:
                subtask_row.append(None)

            subtask_rows.append(tuple(subtask_row))

        self.conn.executemany(
            "INSERT OR IGNORE INTO programs (task_id, program_index, content) VALUES (?, ?, ?)",
            program_rows,
        )

        self.conn.executemany(
            "INSERT OR IGNORE INTO subtasks (task_id, subtask_index, program_index, num_shots, arguments, metadata) VALUES (?, ?, ?, ?, ?, ?)",
            subtask_rows,
        )

        creation_time_str = self._datetime_to_sql_txt(creation_time)
        self.conn.execute(
            "INSERT OR IGNORE INTO task_definitions (task_id, program_language, creation_time) VALUES (?, ?, ?)",
            (task_id, task_definition.program_language, creation_time_str),
        )

        self.conn.commit()

    def get_programs(self, task_ids: tuple[str, ...] | None = None) -> list[dict]:
        """Return stored program records.

        Args:
            task_ids (tuple[str, ...] | None): Optional task IDs to include.
                When None, programs for all stored tasks are returned. Defaults
                to None.

        Returns:
            list[dict]: Program dictionaries ordered by task ID and program
                index.
        """
        query = "SELECT * FROM programs"
        if task_ids is not None:
            placeholders = ", ".join("?" for _ in task_ids)
            query += f" WHERE task_id IN ({placeholders})"
        else:
            task_ids = ()
        query += " ORDER BY task_id, program_index"
        cursor = self.conn.execute(
            query,
            task_ids,
        )
        return list(map(dict, cursor))

    def get_subtasks(self, storage_filter: StorageFilter | None = None) -> list[dict]:
        """Return stored subtask records.

        Args:
            storage_filter (StorageFilter | None): Optional metadata filter.
                When None, all subtasks are returned. Defaults to None.

        Returns:
            list[dict]: Subtask dictionaries ordered by task ID and subtask
                index.
        """
        query = "SELECT * FROM subtasks"
        where_clauses, filter_values = self._storage_filter_to_where_clause(
            storage_filter
        )
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)

        query += " ORDER BY task_id, subtask_index"

        cursor = self.conn.execute(query, filter_values)
        subtasks = []
        for row in cursor:
            subtask = dict(row)
            arguments = row["arguments"]
            if arguments is not None:
                subtask["arguments"] = json.loads(arguments)
            metadata = row["metadata"]
            if metadata is not None:
                subtask["metadata"] = json.loads(metadata)
            completed_date = subtask["completed_date"]
            if completed_date is not None:
                subtask["completed_date"] = self._sql_txt_to_datetime(completed_date)
            subtasks.append(subtask)
        return subtasks

    def update_subtasks_completed_date(self, task_id, subtasks: list[dict]) -> None:
        """Update completion times for stored subtasks.

        Args:
            task_id (str): Backend task ID.
            subtasks (list[dict]): Subtask dictionaries containing
                `subtask_index` and `completed_date`.
        """
        values = []
        for subtask in subtasks:
            completed_date = subtask.get("completed_date")
            if isinstance(completed_date, datetime.datetime):
                completed_date_str = completed_date.isoformat()
            else:
                completed_date_str = completed_date
            values.append((completed_date_str, task_id, subtask["subtask_index"]))
        self.conn.executemany(
            "UPDATE subtasks SET completed_date = (?) WHERE task_id = (?) AND subtask_index = (?)",
            values,
        )
        self.conn.commit()

    def get_program_language(self, task_id: str) -> str:
        """Return the program language for a stored task definition.

        Args:
            task_id (str): Backend task ID.

        Returns:
            str: Stored program language.

        Raises:
            KeyError: If `task_id` is not present.
        """
        cursor = self.conn.execute(
            "SELECT program_language FROM task_definitions WHERE task_id = (?)",
            (task_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise KeyError(task_id)
        else:
            return row[0]

    def get_task_creation_time(self, task_id: str) -> datetime.datetime:
        """Return the creation time for a stored task.

        Args:
            task_id (str): Backend task ID.

        Returns:
            datetime.datetime: Stored task creation time.

        Raises:
            KeyError: If `task_id` is not present.
        """
        cursor = self.conn.execute(
            "SELECT creation_time FROM task_definitions WHERE task_id = (?)",
            (task_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise KeyError(task_id)
        else:
            return self._sql_txt_to_datetime(row[0])

    def __enter__(self):
        """Return this storage backend for use as a context manager.

        Returns:
            SQLiteStorage: This storage instance.
        """
        return self

    def __exit__(self, type, value, traceback):
        """Close the SQLite connection when leaving a context manager."""
        self.close()
