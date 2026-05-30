"""Load a scenario directory into typed Python objects."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Tune this to change min TAs derived from class size.
# Default 1000 -> every course needs exactly 1 TA. Set to e.g. 30 for "1 TA per 30 students".
STUDENTS_PER_TA = 1000

REQUIRED_MATH_COURSE_NUMBERS = {
    "111",
    "120",
    "124",
    "125",
    "126",
    "207",
    "208",
    "300",
    "327",
    "381",
    "394",
    "402",
    "407",
}


@dataclass(frozen=True)
class TA:
    ta_id: str
    name: str
    max_load: int


@dataclass(frozen=True)
class Course:
    course_id: str
    course_code: str
    course_title: str
    enrollment: int
    is_required: bool
    priority: int       # 10 = must teach; lower = lower priority optional
    timeslot_id: str | None = None
    room_id: str | None = None


@dataclass(frozen=True)
class Timeslot:
    timeslot_id: str
    day: str
    start_time: str
    end_time: str


@dataclass(frozen=True)
class Preference:
    ta_id: str
    course_id: str
    qualified: bool
    rank: int | None        # 1-4 if ranked, None if no preference


@dataclass(frozen=True)
class TimePreference:
    ta_id: str
    timeslot_id: str
    available: bool
    score: float            # 0-5; 0 means unavailable


@dataclass(frozen=True)
class Room:
    room_id: str
    building: str
    room_number: str
    capacity: int
    board_type: str         # "blackboard", "whiteboard", "both"
    has_projector: bool
    is_lab: bool


@dataclass(frozen=True)
class RoomSuitability:
    course_id: str
    room_id: str
    suitable: bool
    room_score: int         # 1-5; 5 = most suitable, 0 = not suitable


@dataclass
class ScenarioData:
    name: str
    metadata: dict
    tas: list[TA]
    courses: list[Course]
    timeslots: list[Timeslot]
    rooms: list[Room]
    preferences: list[Preference]
    time_preferences: list[TimePreference]
    room_suitabilities: list[RoomSuitability]

    # --- Step 1 indexes ---
    min_tas: dict[str, int] = field(default_factory=dict)
    qualified_pairs: set[tuple[str, str]] = field(default_factory=set)
    pref_by_pair: dict[tuple[str, str], Preference] = field(default_factory=dict)
    qualified_tas_for: dict[str, list[str]] = field(default_factory=dict)
    qualified_courses_for: dict[str, list[str]] = field(default_factory=dict)

    # --- Step 2 indexes ---
    available_ta_time: set[tuple[str, str]] = field(default_factory=set)
    time_pref_score: dict[tuple[str, str], float] = field(default_factory=dict)
    # (prof_id, course_id, ts_id) where prof qualified for course AND available at timeslot
    qualified_triples: set[tuple[str, str, str]] = field(default_factory=set)
    # best room_score (1-5) available for each course across all suitable rooms
    best_room_score: dict[str, int] = field(default_factory=dict)
    # (course_id, room_id) -> RoomSuitability
    suitability_by_course_room: dict[tuple[str, str], RoomSuitability] = field(default_factory=dict)

    # --- Lookups ---
    ta_by_id: dict[str, TA] = field(default_factory=dict)
    course_by_id: dict[str, Course] = field(default_factory=dict)
    timeslot_by_id: dict[str, Timeslot] = field(default_factory=dict)
    room_by_id: dict[str, Room] = field(default_factory=dict)

    @property
    def required_course_ids(self) -> set[str]:
        return {c.course_id for c in self.courses if c.is_required}

    @property
    def optional_course_ids(self) -> set[str]:
        return {c.course_id for c in self.courses if not c.is_required}


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def required_by_course_code(course_code: str) -> bool:
    """Return True for required UW MATH course numbers, including section suffixes."""
    match = re.match(r"^MATH\s+(\d{3})(?:\b|[-\s])", course_code.strip())
    return bool(match and match.group(1) in REQUIRED_MATH_COURSE_NUMBERS)


def load_scenario(scenario_dir: str | Path) -> ScenarioData:
    p = Path(scenario_dir)
    metadata = json.loads((p / "metadata.json").read_text())

    tas = [
        TA(
            ta_id=r["professor_id"],
            name=r["professor_name"],
            max_load=int(r["max_load"]),
        )
        for r in _read_csv(p / "professors.csv")
    ]

    courses = []
    for r in _read_csv(p / "courses.csv"):
        is_required = required_by_course_code(r["course_code"])
        courses.append(
            Course(
                course_id=r["course_id"],
                course_code=r["course_code"],
                course_title=r["course_title"],
                enrollment=int(r["enrollment"]),
                is_required=is_required,
                priority=10 if is_required else 2,
                timeslot_id=r.get("timeslot_id") or None,
                room_id=r.get("room_id") or None,
            )
        )

    timeslots = [
        Timeslot(
            timeslot_id=r["timeslot_id"],
            day=r["day"],
            start_time=r["start_time"],
            end_time=r["end_time"],
        )
        for r in _read_csv(p / "timeslots.csv")
    ]

    preferences = []
    for r in _read_csv(p / "professor_course_preferences.csv"):
        rank_str = r["course_pref_rank"]
        preferences.append(
            Preference(
                ta_id=r["professor_id"],
                course_id=r["course_id"],
                qualified=bool(int(r["qualified"])),
                rank=int(rank_str) if rank_str else None,
            )
        )

    time_preferences = [
        TimePreference(
            ta_id=r["professor_id"],
            timeslot_id=r["timeslot_id"],
            available=bool(int(r["available"])),
            score=float(r["time_pref_score"]),
        )
        for r in _read_csv(p / "professor_time_preferences.csv")
    ]

    rooms = [
        Room(
            room_id=r["room_id"],
            building=r["building"],
            room_number=r["room_number"],
            capacity=int(r["capacity"]),
            board_type=r["board_type"],
            has_projector=bool(int(r["has_projector"])),
            is_lab=bool(int(r["is_lab"])),
        )
        for r in _read_csv(p / "rooms.csv")
    ]

    room_suitabilities = [
        RoomSuitability(
            course_id=r["course_id"],
            room_id=r["room_id"],
            suitable=bool(int(r["suitable"])),
            room_score=int(r["room_score"]),
        )
        for r in _read_csv(p / "room_suitability.csv")
    ]

    data = ScenarioData(
        name=p.name,
        metadata=metadata,
        tas=tas,
        courses=courses,
        timeslots=timeslots,
        rooms=rooms,
        preferences=preferences,
        time_preferences=time_preferences,
        room_suitabilities=room_suitabilities,
    )

    # Step 1 indexes
    data.min_tas = {
        c.course_id: max(1, math.ceil(c.enrollment / STUDENTS_PER_TA))
        for c in courses
    }
    data.pref_by_pair = {(pr.ta_id, pr.course_id): pr for pr in preferences}
    data.qualified_pairs = {
        (pr.ta_id, pr.course_id) for pr in preferences if pr.qualified
    }

    qfor_course: dict[str, list[str]] = defaultdict(list)
    qfor_ta: dict[str, list[str]] = defaultdict(list)
    for pr in preferences:
        if pr.qualified:
            qfor_course[pr.course_id].append(pr.ta_id)
            qfor_ta[pr.ta_id].append(pr.course_id)
    data.qualified_tas_for = dict(qfor_course)
    data.qualified_courses_for = dict(qfor_ta)

    # Step 2 indexes
    data.available_ta_time = {
        (tp.ta_id, tp.timeslot_id) for tp in time_preferences if tp.available
    }
    data.time_pref_score = {
        (tp.ta_id, tp.timeslot_id): tp.score for tp in time_preferences
    }
    ta_available_slots: dict[str, list[str]] = defaultdict(list)
    for tp in time_preferences:
        if tp.available:
            ta_available_slots[tp.ta_id].append(tp.timeslot_id)

    fixed_times_by_course = {
        course.course_id: course.timeslot_id
        for course in courses
        if course.timeslot_id
    }
    triples: set[tuple[str, str, str]] = set()
    for ta_id, c_id in data.qualified_pairs:
        fixed_ts = fixed_times_by_course.get(c_id)
        if fixed_ts:
            if fixed_ts in ta_available_slots.get(ta_id, []):
                triples.add((ta_id, c_id, fixed_ts))
        else:
            for ts_id in ta_available_slots.get(ta_id, []):
                triples.add((ta_id, c_id, ts_id))
    data.qualified_triples = triples

    # Room indexes
    data.suitability_by_course_room = {
        (rs.course_id, rs.room_id): rs for rs in room_suitabilities
    }
    best: dict[str, int] = defaultdict(int)
    for rs in room_suitabilities:
        if rs.suitable and rs.room_score > best[rs.course_id]:
            best[rs.course_id] = rs.room_score
    data.best_room_score = dict(best)

    # Lookups
    data.ta_by_id = {ta.ta_id: ta for ta in tas}
    data.course_by_id = {c.course_id: c for c in courses}
    data.timeslot_by_id = {ts.timeslot_id: ts for ts in timeslots}
    data.room_by_id = {r.room_id: r for r in rooms}

    return data
