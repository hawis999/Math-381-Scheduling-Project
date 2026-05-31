"""Generate reproducible UW-style Autumn 2026 Math scheduling scenarios.

The course sections are fixed from the scraped Autumn 2026 Math schedule.
Each seed assigns those same sections to sensible randomized time blocks and
rooms, then generates synthetic professor/course and professor/time
preferences.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from data import REQUIRED_MATH_COURSE_NUMBERS, required_by_course_code


DEFAULT_OUTPUT_DIR = Path("data/synthetic_scheduling_data")


@dataclass(frozen=True)
class Section:
    number: str
    section: str
    title: str
    credits: int
    enrollment: int
    tags: tuple[str, ...]


@dataclass(frozen=True)
class ProfessorSeed:
    name: str
    tags: tuple[str, ...]
    rank: str
    max_load: int


TIMESLOTS = [
    ("T_MWF_0830_0920", "MWF", "08:30", "09:20", "mwf", "morning"),
    ("T_MWF_0930_1020", "MWF", "09:30", "10:20", "mwf", "morning"),
    ("T_MWF_1030_1120", "MWF", "10:30", "11:20", "mwf", "midday"),
    ("T_MWF_1130_1220", "MWF", "11:30", "12:20", "mwf", "midday"),
    ("T_MWF_1230_1320", "MWF", "12:30", "13:20", "mwf", "midday"),
    ("T_MWF_1330_1420", "MWF", "13:30", "14:20", "mwf", "afternoon"),
    ("T_MWF_1430_1520", "MWF", "14:30", "15:20", "mwf", "afternoon"),
    ("T_MWF_1530_1620", "MWF", "15:30", "16:20", "mwf", "afternoon"),
    ("T_TTH_0830_0950", "TTh", "08:30", "09:50", "tth", "morning"),
    ("T_TTH_1000_1120", "TTh", "10:00", "11:20", "tth", "midday"),
    ("T_TTH_1130_1250", "TTh", "11:30", "12:50", "tth", "midday"),
    ("T_TTH_1300_1420", "TTh", "13:00", "14:20", "tth", "afternoon"),
    ("T_TTH_1430_1550", "TTh", "14:30", "15:50", "tth", "afternoon"),
    ("T_TTH_1600_1720", "TTh", "16:00", "17:20", "tth", "late"),
    ("T_MW_1000_1120", "MW", "10:00", "11:20", "mw", "midday"),
    ("T_MW_1430_1550", "MW", "14:30", "15:50", "mw", "afternoon"),
    ("T_F_1530_1720", "F", "15:30", "17:20", "seminar", "late"),
    ("T_W_1500_1650", "W", "15:00", "16:50", "seminar", "late"),
]


ROOMS = [
    ("R_KNE_210", "KNE", "210", 240, "blackboard", 1, 0),
    ("R_KNE_220", "KNE", "220", 240, "blackboard", 1, 0),
    ("R_GUG_220", "GUG", "220", 240, "whiteboard", 1, 0),
    ("R_SMI_120", "SMI", "120", 240, "whiteboard", 1, 0),
    ("R_SAV_260", "SAV", "260", 170, "whiteboard", 1, 0),
    ("R_AND_205", "AND", "205", 150, "whiteboard", 1, 0),
    ("R_OTB_014", "OTB", "014", 100, "whiteboard", 1, 0),
    ("R_MUE_153", "MUE", "153", 90, "blackboard", 1, 0),
    ("R_SMI_102", "SMI", "102", 80, "whiteboard", 1, 0),
    ("R_MGH_389", "MGH", "389", 70, "whiteboard", 1, 0),
    ("R_LOW_206", "LOW", "206", 50, "blackboard", 1, 0),
    ("R_MEB_248", "MEB", "248", 45, "blackboard", 1, 0),
    ("R_PDL_C038", "PDL", "C038", 35, "blackboard", 1, 0),
    ("R_PDL_C401", "PDL", "C401", 25, "blackboard", 1, 0),
    ("R_LOW_113", "LOW", "113", 20, "whiteboard", 1, 0),
]


SECTIONS = [
    Section("102", "A", "Algebra", 5, 40, ("service", "algebra", "precalculus")),
    Section("103", "A", "Introduction to Elementary Functions", 5, 40, ("service", "precalculus")),
    Section("111", "A", "Algebra with Applications", 5, 240, ("service", "algebra", "applications")),
    Section("111", "B", "Algebra with Applications", 5, 240, ("service", "algebra", "applications")),
    Section("112", "A", "Business and Economic Calculus", 5, 90, ("service", "calculus", "optimization")),
    Section("120", "A", "Precalculus", 5, 240, ("service", "precalculus", "calculus")),
    Section("120", "B", "Precalculus", 5, 240, ("service", "precalculus", "calculus")),
    Section("124", "A", "Calculus with Analytic Geometry I", 5, 240, ("service", "calculus")),
    Section("124", "B", "Calculus with Analytic Geometry I", 5, 240, ("service", "calculus")),
    Section("124", "C", "Calculus with Analytic Geometry I", 5, 240, ("service", "calculus")),
    Section("124", "D", "Calculus with Analytic Geometry I", 5, 240, ("service", "calculus")),
    Section("124", "E", "Calculus with Analytic Geometry I", 5, 240, ("service", "calculus")),
    Section("124", "F", "Calculus with Analytic Geometry I", 5, 240, ("service", "calculus")),
    Section("125", "A", "Calculus with Analytic Geometry II", 5, 240, ("service", "calculus")),
    Section("125", "B", "Calculus with Analytic Geometry II", 5, 240, ("service", "calculus")),
    Section("125", "C", "Calculus with Analytic Geometry II", 5, 240, ("service", "calculus")),
    Section("126", "A", "Calculus with Analytic Geometry III", 5, 240, ("service", "calculus", "geometry")),
    Section("126", "B", "Calculus with Analytic Geometry III", 5, 240, ("service", "calculus", "geometry")),
    Section("126", "C", "Calculus with Analytic Geometry III", 5, 240, ("service", "calculus", "geometry")),
    Section("126", "D", "Calculus with Analytic Geometry III", 5, 240, ("service", "calculus", "geometry")),
    Section("134", "A", "Accelerated Honors Calculus", 5, 45, ("calculus", "honors", "linear_algebra")),
    Section("200", "A", "Discrete Mathematics I", 5, 300, ("combinatorics", "probability", "number_theory")),
    Section("207", "A", "Introduction to Differential Equations", 4, 230, ("differential_equations", "applied_math")),
    Section("207", "C", "Introduction to Differential Equations", 4, 220, ("differential_equations", "applied_math")),
    Section("207", "D", "Introduction to Differential Equations", 4, 50, ("differential_equations", "applied_math")),
    Section("208", "A", "Matrix Algebra with Applications", 4, 440, ("linear_algebra", "applied_math")),
    Section("208", "B", "Matrix Algebra with Applications", 4, 440, ("linear_algebra", "applied_math")),
    Section("209", "A", "Linear Analysis", 4, 50, ("differential_equations", "pde", "analysis")),
    Section("224", "A", "Advanced Multivariable Calculus", 4, 305, ("calculus", "geometry", "analysis")),
    Section("224", "B", "Advanced Multivariable Calculus", 4, 50, ("calculus", "geometry", "analysis")),
    Section("300", "A", "Introduction to Mathematical Reasoning", 4, 80, ("proofs", "number_theory", "combinatorics")),
    Section("300", "B", "Introduction to Mathematical Reasoning", 4, 70, ("proofs", "number_theory", "combinatorics")),
    Section("327", "A", "Introductory Real Analysis I", 4, 75, ("analysis", "real_analysis")),
    Section("334", "A", "Honors Accelerated Advanced Calculus", 5, 45, ("analysis", "pde", "complex_analysis")),
    Section("342", "A", "Art of Problem Solving", 4, 25, ("combinatorics", "number_theory", "geometry", "probability")),
    Section("381", "A", "Discrete Mathematical Modeling", 4, 80, ("combinatorics", "graph_theory", "optimization")),
    Section("394", "A", "Probability I", 4, 170, ("probability", "statistics")),
    Section("398", "B", "Special Topics: WDRP", 2, 15, ("general_math", "seminar")),
    Section("402", "A", "Introduction to Modern Algebra", 4, 100, ("algebra", "number_theory")),
    Section("407", "A", "Linear Optimization", 4, 150, ("optimization", "linear_algebra")),
    Section("424", "A", "Fundamental Concepts of Analysis", 4, 50, ("analysis", "real_analysis")),
    Section("427", "A", "Complex Analysis", 4, 50, ("complex_analysis", "analysis")),
    Section("441", "A", "Topology", 4, 90, ("topology", "geometry")),
    Section("444", "A", "Geometries I", 4, 50, ("geometry", "education")),
    Section("461", "A", "Combinatorial Theory I", 4, 80, ("combinatorics", "graph_theory")),
    Section("491", "A", "Introduction to Stochastic Processes I", 4, 45, ("probability", "stochastic_processes")),
    Section("504", "A", "Modern Algebra", 5, 35, ("algebra", "algebraic_geometry")),
    Section("507", "A", "Algebraic Structures", 3, 25, ("algebra", "representation_theory")),
    Section("514", "A", "Network Optimization", 3, 20, ("optimization", "combinatorics", "graph_theory")),
    Section("521", "A", "Advanced Probability", 3, 15, ("probability", "analysis")),
    Section("526", "A", "Real Analysis", 5, 20, ("analysis", "real_analysis")),
    Section("534", "A", "Complex Analysis", 5, 25, ("complex_analysis", "analysis")),
    Section("544", "A", "Topology and Geometry of Manifolds", 5, 35, ("topology", "geometry")),
    Section("547", "A", "Geometric Structures", 3, 25, ("geometry", "differential_geometry")),
    Section("550", "D", "Seminar in Geometry", 3, 20, ("geometry", "seminar")),
    Section("580", "B", "Departmental Colloquium", 2, 60, ("general_math", "seminar")),
    Section("581", "C", "Special Topics: Inverse Problems and PDE", 3, 25, ("pde", "inverse_problems")),
    Section("584", "A", "Numerical Linear Algebra", 5, 25, ("numerical_analysis", "linear_algebra")),
    Section("590", "A", "Seminar in Probability", 3, 40, ("probability", "seminar")),
]

INTRO_TEACHING_TAGS = ("general_math", "service", "algebra", "calculus", "precalculus", "education")


PROFESSORS = [
    ProfessorSeed("Jarod Alper", ("algebra", "algebraic_geometry", "geometry"), "professor", 2),
    ProfessorSeed("Aleksandr Aravkin", ("optimization", "applied_math"), "adjunct", 1),
    ProfessorSeed("Jayadev Athreya", ("dynamical_systems", "geometry"), "professor", 2),
    ProfessorSeed("Ebru Bekyel", ("number_theory", "analysis", "calculus"), "teaching", 3),
    ProfessorSeed("Sara Billey", ("combinatorics", "algebraic_geometry", "representation_theory"), "professor", 2),
    ProfessorSeed("Kenneth Bube", ("numerical_analysis", "pde"), "professor", 2),
    ProfessorSeed("Krzysztof Burdzy", ("probability",), "professor", 2),
    ProfessorSeed("Charles Camacho", ("calculus", "combinatorics", "education", "geometry"), "teaching", 3),
    ProfessorSeed("Zhenqing Chen", ("probability",), "professor", 2),
    ProfessorSeed("Tony Chiang", ("applied_math", "combinatorics", "data_science", "probability", "statistics"), "affiliate", 1),
    ProfessorSeed("David Collingwood", ("representation_theory", "genomics"), "professor", 2),
    ProfessorSeed("Matthew Conroy", ("number_theory", "education", "general_math"), "teaching", 3),
    ProfessorSeed("Bernard Deconinck", ("applied_math", "pde"), "adjunct", 1),
    ProfessorSeed("Fanny Dos Reis", ("differential_geometry", "geometric_measure_theory"), "lecturer", 2),
    ProfessorSeed("Alexis Drouot", ("analysis", "pde"), "professor", 2),
    ProfessorSeed("Maryam Fazel", ("optimization",), "adjunct", 1),
    ProfessorSeed("Giovanni Inchiostro", ("algebra", "algebraic_geometry", "combinatorics", "geometry"), "professor", 2),
    ProfessorSeed("Neal Koblitz", ("cryptography", "number_theory"), "professor", 2),
    ProfessorSeed("Sandor Kovacs", ("algebra", "algebraic_geometry"), "professor", 2),
    ProfessorSeed("Gaku Liu", ("combinatorics", "geometry"), "professor", 2),
    ProfessorSeed("Ricky Liu", ("algebraic_combinatorics", "combinatorics"), "professor", 2),
    ProfessorSeed("Andrew Loveless", ("education", "number_theory", "calculus"), "teaching", 3),
    ProfessorSeed("Max Lieblich", ("algebraic_geometry",), "professor", 1),
    ProfessorSeed("Monty McGovern", ("representation_theory",), "professor", 2),
    ProfessorSeed("Dan Mikulincer", ("data_science", "optimization", "probability", "theoretical_computer_science"), "professor", 2),
    ProfessorSeed("Natalie Naehrig", ("algebra", "education", "representation_theory", "calculus"), "teaching", 3),
    ProfessorSeed("Alexandra Nichifor", ("algebra", "number_theory", "calculus"), "teaching", 3),
    ProfessorSeed("Isabella Novik", ("combinatorics", "algebraic_combinatorics", "geometric_combinatorics"), "professor", 2),
    ProfessorSeed("Jonah Ostroff", ("combinatorics", "calculus"), "teaching", 3),
    ProfessorSeed("John Palmieri", ("algebra", "algebraic_topology", "representation_theory"), "professor", 2),
    ProfessorSeed("Gabriel Paternain", ("differential_geometry", "dynamical_systems", "inverse_problems"), "professor", 2),
    ProfessorSeed("Patrick Perkins", ("algebra", "combinatorics", "education"), "teaching", 3),
    ProfessorSeed("Julia Pevtsova", ("algebra", "representation_theory", "noncommutative_algebra"), "professor", 2),
    ProfessorSeed("Elena Pezzoli", ("logic", "computer_science", "calculus"), "teaching", 3),
    ProfessorSeed("Daniel Pollack", ("differential_geometry", "general_relativity", "pde"), "professor", 1),
    ProfessorSeed("Piotr Pstragowski", ("algebraic_topology",), "professor", 2),
    ProfessorSeed("Steffen Rohde", ("complex_analysis", "numerical_analysis", "probability"), "professor", 2),
    ProfessorSeed("François Clément", ("optimization",), "postdoctoral", 1),
    ProfessorSeed("David Aldous", INTRO_TEACHING_TAGS, "affiliate", 1),
    ProfessorSeed("Christian Gorski", INTRO_TEACHING_TAGS, "postdoctoral", 1),
    ProfessorSeed("Elena Hafner", INTRO_TEACHING_TAGS, "postdoctoral", 1),
    ProfessorSeed("Andy Heald", ("algebra", "number_theory"), "lecturer", 2),
    ProfessorSeed("Christopher Hoffman", ("probability",), "professor", 2),
    ProfessorSeed("Alexander Holroyd", ("probability",), "affiliate", 1),
    ProfessorSeed("John Garnett", ("complex_analysis",), "affiliate", 1),
    ProfessorSeed("Ken Goodearl", ("noncommutative_algebra", "algebra"), "affiliate", 1),
    ProfessorSeed("Steven Klee", ("algebraic_combinatorics", "geometric_combinatorics", "combinatorics"), "affiliate", 1),
    ProfessorSeed("Henry Kvinge", ("geometry", "algebra", "topology", "data_science", "representation_theory"), "affiliate", 1),
    ProfessorSeed("Kristin Lauter", ("algebraic_geometry", "cryptography", "number_theory"), "affiliate", 1),
    ProfessorSeed("Yutao Liu", INTRO_TEACHING_TAGS, "postdoctoral", 1),
    ProfessorSeed("Eyal Lubetzky", ("combinatorics", "graph_theory", "probability"), "affiliate", 1),
    ProfessorSeed("Mehran Mesbahi", ("optimization", "applied_math"), "adjunct", 1),
    ProfessorSeed("Kyle Ormsby", ("algebraic_geometry", "algebraic_topology", "combinatorics"), "affiliate", 1),
    ProfessorSeed("Soumik Pal", ("probability",), "professor", 2),
    ProfessorSeed("Michele Pernice", INTRO_TEACHING_TAGS, "acting", 1),
    ProfessorSeed("Larry Pierce", ("medical_imaging", "applied_math"), "lecturer", 1),
    ProfessorSeed("Linda Simonsen", INTRO_TEACHING_TAGS, "adjunct", 1),
    ProfessorSeed("Mariana Smit Vega Garcia", INTRO_TEACHING_TAGS, "affiliate", 1),
    ProfessorSeed("Boris Solomyak", ("fractals", "analysis"), "affiliate", 1),
    ProfessorSeed("Tom Trogdon", ("applied_math", "numerical_analysis"), "adjunct", 1),
    ProfessorSeed("David Bruce Wilson", ("combinatorics", "probability"), "affiliate", 1),
    ProfessorSeed("Libin Zhu", INTRO_TEACHING_TAGS, "postdoctoral", 1),
    ProfessorSeed("Judith M. Arms", INTRO_TEACHING_TAGS, "emeritus", 1),
    ProfessorSeed("James Burke", INTRO_TEACHING_TAGS, "emeritus", 1),
    ProfessorSeed("Ethan Devinatz", ("algebraic_topology",), "emeritus", 1),
    ProfessorSeed("Thomas E. Duchamp", ("differential_geometry",), "emeritus", 1),
    ProfessorSeed("K. Bruce Erickson", ("probability",), "emeritus", 1),
    ProfessorSeed("Gerald B. Folland", ("real_analysis", "analysis"), "emeritus", 1),
    ProfessorSeed("Ramesh A. Gangolli", ("probability", "real_analysis", "representation_theory"), "emeritus", 1),
    ProfessorSeed("C. Robin Graham", ("differential_geometry", "mathematical_physics", "pde"), "emeritus", 1),
    ProfessorSeed("Anne Greenbaum", INTRO_TEACHING_TAGS, "emeritus", 1),
    ProfessorSeed("Ralph Greenberg", ("number_theory",), "emeritus", 1),
    ProfessorSeed("Ron Irving", ("noncommutative_algebra", "representation_theory", "algebra"), "emeritus", 1),
    ProfessorSeed("James R. King", INTRO_TEACHING_TAGS, "emeritus", 1),
    ProfessorSeed("John M. Lee", ("differential_geometry", "general_relativity", "geometry", "pde"), "emeritus", 1),
    ProfessorSeed("Doug Lind", ("dynamical_systems",), "emeritus", 1),
    ProfessorSeed("Don Marshall", ("complex_analysis", "analysis", "numerical_analysis"), "emeritus", 1),
    ProfessorSeed("Steven G. Monk", ("education",), "emeritus", 1),
    ProfessorSeed("Vilnis Ozols", ("representation_theory",), "emeritus", 1),
    ProfessorSeed("R. Tyrrell Rockafellar", ("optimization",), "emeritus", 1),
    ProfessorSeed("Jack Segal", ("algebraic_topology",), "emeritus", 1),
    ProfessorSeed("S. Paul Smith", ("algebra", "algebraic_geometry", "noncommutative_algebra", "representation_theory"), "emeritus", 1),
    ProfessorSeed("E. Lee Stout", ("complex_analysis",), "emeritus", 1),
    ProfessorSeed("John Sylvester", ("pde",), "emeritus", 1),
    ProfessorSeed("Jennifer Taggart", ("number_theory",), "emeritus", 1),
    ProfessorSeed("Selim Tuncel", ("dynamical_systems",), "emeritus", 1),
    ProfessorSeed("Virginia M. Warfield", ("education", "probability"), "emeritus", 1),
    ProfessorSeed("Garth Warner", ("representation_theory",), "emeritus", 1),
    ProfessorSeed("Thomas Rothvoss", ("optimization", "theoretical_computer_science"), "professor", 2),
    ProfessorSeed("Danny Shi", ("algebraic_topology", "topology"), "professor", 2),
    ProfessorSeed("Farbod Shokrieh", ("algebra", "algebraic_geometry", "combinatorics", "number_theory"), "professor", 2),
    ProfessorSeed("Hart Smith", ("pde", "analysis", "real_analysis"), "professor", 2),
    ProfessorSeed("Stefan Steinerberger", ("applied_math", "combinatorics", "data_science", "pde", "probability", "analysis"), "professor", 2),
    ProfessorSeed("Zhixu Su", ("topology", "geometry", "education"), "teaching", 3),
    ProfessorSeed("Jinwoo Sung", ("complex_analysis", "probability"), "professor", 2),
    ProfessorSeed("Rekha Thomas", ("optimization",), "professor", 2),
    ProfessorSeed("Tatiana Toro", ("geometric_measure_theory", "pde", "analysis"), "professor", 2),
    ProfessorSeed("Gunther Uhlmann", ("pde", "inverse_problems"), "professor", 2),
    ProfessorSeed("Cynthia Vinzant", ("optimization", "algebraic_geometry"), "professor", 2),
    ProfessorSeed("Bianca Viray", ("algebraic_geometry", "number_theory"), "professor", 2),
    ProfessorSeed("Bobby Wilson", ("geometric_measure_theory", "pde", "analysis"), "professor", 2),
    ProfessorSeed("Yu Yuan", ("differential_geometry", "pde"), "professor", 2),
    ProfessorSeed("James Zhang", ("algebra",), "professor", 2),
    ProfessorSeed("Jonathan Zhu", ("differential_geometry", "pde"), "professor", 2),
]


PROFILE_PREFERENCES = {
    "morning": {"morning": 5, "midday": 4, "afternoon": 2, "late": 1},
    "midday": {"morning": 3, "midday": 5, "afternoon": 4, "late": 2},
    "afternoon": {"morning": 2, "midday": 4, "afternoon": 5, "late": 3},
    "flexible": {"morning": 4, "midday": 5, "afternoon": 4, "late": 3},
}


def course_code(section: Section) -> str:
    return f"MATH {section.number}-{section.section}"


def course_id(index: int) -> str:
    return f"C{index:03d}"


def professor_id(index: int) -> str:
    return f"P{index:03d}"


def clone_sections(scale: int) -> list[Section]:
    if scale <= 1:
        return list(SECTIONS)

    sections: list[Section] = []
    for copy_num in range(1, scale + 1):
        for section in SECTIONS:
            sections.append(
                Section(
                    number=section.number,
                    section=f"{section.section}{copy_num:02d}",
                    title=section.title,
                    credits=section.credits,
                    enrollment=section.enrollment,
                    tags=section.tags,
                )
            )
    return sections


def stress_professors(scale: int) -> list[ProfessorSeed]:
    professors = list(PROFESSORS)
    broad_tags = tuple(
        sorted(
            {
                "algebra",
                "algebraic_geometry",
                "algebraic_topology",
                "analysis",
                "applications",
                "applied_math",
                "calculus",
                "combinatorics",
                "complex_analysis",
                "differential_equations",
                "differential_geometry",
                "education",
                "general_math",
                "geometry",
                "graph_theory",
                "honors",
                "inverse_problems",
                "linear_algebra",
                "number_theory",
                "numerical_analysis",
                "optimization",
                "pde",
                "precalculus",
                "probability",
                "proofs",
                "real_analysis",
                "seminar",
                "service",
                "stochastic_processes",
                "topology",
            }
        )
    )
    for idx in range(1, max(10, scale * 8) + 1):
        professors.append(
            ProfessorSeed(
                name=f"Stress Lecturer Pool {idx:03d}",
                tags=broad_tags,
                rank="lecturer_pool",
                max_load=12,
            )
        )
    return professors


def pick_timeslot(section: Section, rng: random.Random) -> tuple[str, str, str, str, str, str]:
    candidates = []
    for slot in TIMESLOTS:
        _, _, _, _, pattern, period = slot
        weight = 1
        number = int(section.number)
        if "seminar" in section.tags:
            weight += 8 if pattern == "seminar" or period == "late" else 0
        elif section.credits >= 5 and number < 300:
            weight += 8 if pattern == "mwf" else 1
            weight += 4 if period in {"morning", "midday"} else 0
        elif section.credits == 4:
            weight += 5 if pattern in {"mwf", "tth"} else 0
            weight += 3 if period in {"midday", "afternoon"} else 0
        elif section.credits <= 3 or number >= 500:
            weight += 5 if pattern in {"mw", "tth", "seminar"} else 0
            weight += 3 if period in {"afternoon", "late"} else 0
        if section.enrollment >= 150 and period == "late":
            weight = max(1, weight - 5)
        candidates.append((slot, weight))
    total = sum(weight for _, weight in candidates)
    mark = rng.uniform(0, total)
    running = 0.0
    for slot, weight in candidates:
        running += weight
        if running >= mark:
            return slot
    return candidates[-1][0]


def pick_room(section: Section, rng: random.Random) -> tuple[str, str, str, int, str, int, int]:
    suitable = [room for room in ROOMS if room[3] >= section.enrollment]
    if not suitable:
        suitable = [max(ROOMS, key=lambda room: room[3])]
    tight = sorted(suitable, key=lambda room: (room[3], room[0]))[:4]
    return rng.choice(tight)


def topic_score(prof_tags: set[str], course_tags: set[str], section: Section) -> int:
    overlap = prof_tags & course_tags
    if overlap:
        return min(5, 2 + len(overlap))
    if "general_math" in prof_tags and (int(section.number) < 400 or "service" in course_tags):
        return 3
    if "education" in prof_tags and ("service" in course_tags or "proofs" in course_tags):
        return 3
    if "calculus" in prof_tags and ("calculus" in course_tags or "precalculus" in course_tags):
        return 4
    return 0


def rank_from_score(score: int) -> str:
    if score >= 5:
        return "1"
    if score == 4:
        return "2"
    if score == 3:
        return "3"
    if score == 2:
        return "4"
    return ""


def time_score(profile: str, period: str, rng: random.Random) -> int:
    base = PROFILE_PREFERENCES[profile][period]
    return max(1, min(5, base + rng.choice([-1, 0, 0, 1])))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_scenario(seed: int, output_dir: Path, stress: bool = False, scale: int = 25) -> Path:
    rng = random.Random(seed)
    if stress:
        scenario_name = f"uw_math_autumn_2026_stress_seed_{seed}_scale_{scale}"
    else:
        scenario_name = f"uw_math_autumn_2026_seed_{seed}"
    scenario_dir = output_dir / scenario_name
    if scenario_dir.exists():
        shutil.rmtree(scenario_dir)
    scenario_dir.mkdir(parents=True)

    sections = clone_sections(scale) if stress else list(SECTIONS)
    professors = stress_professors(scale) if stress else list(PROFESSORS)

    assigned_times = {section: pick_timeslot(section, rng) for section in sections}
    assigned_rooms = {section: pick_room(section, rng) for section in sections}

    metadata = {
        "name": scenario_dir.name,
        "seed": seed,
        "source": "UW Math Autumn 2026 scrape, lecture-like sections only",
        "time_policy": "Fixed randomized times generated before solving",
        "stress": stress,
        "scale": scale if stress else 1,
        "base_sections": len(SECTIONS),
        "generated_sections": len(sections),
        "generated_professors": len(professors),
        "required_math_course_numbers": sorted(REQUIRED_MATH_COURSE_NUMBERS),
    }
    (scenario_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    professor_rows = []
    for idx, prof in enumerate(professors, start=1):
        professor_rows.append(
            {
                "professor_id": professor_id(idx),
                "professor_name": prof.name,
                "min_load": 0,
                "max_load": prof.max_load,
                "department": "MATH",
                "seniority": prof.rank,
                "research_tags": ";".join(prof.tags),
                "priority_mode": rng.choice(["time_first", "interest_first"]),
            }
        )

    time_rows = []
    professor_profiles: dict[str, str] = {}
    for idx, _prof in enumerate(professors, start=1):
        pid = professor_id(idx)
        profile = rng.choice(list(PROFILE_PREFERENCES))
        professor_profiles[pid] = profile
        for slot_id, _day, _start, _end, _pattern, period in TIMESLOTS:
            score = time_score(profile, period, rng)
            time_rows.append(
                {
                    "professor_id": pid,
                    "timeslot_id": slot_id,
                    "available": 1,
                    "time_pref_score": score,
                }
            )

    course_rows = []
    for idx, section in enumerate(sections, start=1):
        cid = course_id(idx)
        slot_id = assigned_times[section][0]
        room_id = assigned_rooms[section][0]
        required = required_by_course_code(course_code(section))
        course_rows.append(
            {
                "course_id": cid,
                "course_code": course_code(section),
                "course_title": section.title,
                "is_required": int(required),
                "priority": 10 if required else 2,
                "enrollment": section.enrollment,
                "required_board": "any",
                "needs_projector": 1 if section.enrollment >= 80 else 0,
                "needs_lab": 0,
                "credits": section.credits,
                "timeslot_id": slot_id,
                "room_id": room_id,
                "course_tags": ";".join(section.tags),
            }
        )

    timeslot_rows = [
        {
            "timeslot_id": slot_id,
            "day": day,
            "start_time": start,
            "end_time": end,
        }
        for slot_id, day, start, end, _pattern, _period in TIMESLOTS
    ]

    room_rows = [
        {
            "room_id": room_id,
            "building": building,
            "room_number": room_number,
            "capacity": capacity,
            "board_type": board_type,
            "has_projector": has_projector,
            "is_lab": is_lab,
        }
        for room_id, building, room_number, capacity, board_type, has_projector, is_lab in ROOMS
    ]

    pref_rows = []
    for pidx, prof in enumerate(professors, start=1):
        pid = professor_id(pidx)
        prof_tags = set(prof.tags)
        for cidx, section in enumerate(sections, start=1):
            cid = course_id(cidx)
            score = topic_score(prof_tags, set(section.tags), section)
            qualified = score > 0
            pref_rows.append(
                {
                    "professor_id": pid,
                    "course_id": cid,
                    "qualified": int(qualified),
                    "course_pref_rank": rank_from_score(score) if qualified else "",
                    "course_pref_score": score,
                    "topic_score": score,
                }
            )

    room_suitability_rows = []
    for cidx, section in enumerate(sections, start=1):
        cid = course_id(cidx)
        for room_id, _building, _room_number, capacity, _board_type, has_projector, _is_lab in ROOMS:
            suitable = capacity >= section.enrollment
            if not suitable:
                score = 0
            else:
                extra = capacity - section.enrollment
                score = 5 if extra <= 40 else 4 if extra <= 120 else 3
                if section.enrollment >= 80 and not has_projector:
                    score = max(1, score - 1)
            room_suitability_rows.append(
                {
                    "course_id": cid,
                    "room_id": room_id,
                    "suitable": int(suitable),
                    "room_score": score,
                }
            )

    pref_lookup = {
        (row["professor_id"], row["course_id"]): row
        for row in pref_rows
    }
    time_lookup = {
        (row["professor_id"], row["timeslot_id"]): row
        for row in time_rows
    }
    room_lookup = {
        (row["course_id"], row["room_id"]): row
        for row in room_suitability_rows
    }
    allowed_rows = []
    for cidx, section in enumerate(sections, start=1):
        cid = course_id(cidx)
        slot_id = assigned_times[section][0]
        room_id = assigned_rooms[section][0]
        room_score = int(room_lookup[(cid, room_id)]["room_score"])
        for pidx, _prof in enumerate(professors, start=1):
            pid = professor_id(pidx)
            pref = pref_lookup[(pid, cid)]
            if not int(pref["qualified"]):
                continue
            tscore = int(time_lookup[(pid, slot_id)]["time_pref_score"])
            priority = 10 if required_by_course_code(course_code(section)) else 2
            total = float(pref["course_pref_score"]) + tscore + room_score + priority / 2
            allowed_rows.append(
                {
                    "professor_id": pid,
                    "course_id": cid,
                    "timeslot_id": slot_id,
                    "room_id": room_id,
                    "total_score": f"{total:.2f}",
                }
            )

    write_csv(
        scenario_dir / "professors.csv",
        ["professor_id", "professor_name", "min_load", "max_load", "department", "seniority", "research_tags", "priority_mode"],
        professor_rows,
    )
    write_csv(
        scenario_dir / "courses.csv",
        ["course_id", "course_code", "course_title", "is_required", "priority", "enrollment", "required_board", "needs_projector", "needs_lab", "credits", "timeslot_id", "room_id", "course_tags"],
        course_rows,
    )
    write_csv(scenario_dir / "timeslots.csv", ["timeslot_id", "day", "start_time", "end_time"], timeslot_rows)
    write_csv(
        scenario_dir / "rooms.csv",
        ["room_id", "building", "room_number", "capacity", "board_type", "has_projector", "is_lab"],
        room_rows,
    )
    write_csv(
        scenario_dir / "professor_course_preferences.csv",
        ["professor_id", "course_id", "qualified", "course_pref_rank", "course_pref_score", "topic_score"],
        pref_rows,
    )
    write_csv(
        scenario_dir / "professor_time_preferences.csv",
        ["professor_id", "timeslot_id", "available", "time_pref_score"],
        time_rows,
    )
    write_csv(
        scenario_dir / "room_suitability.csv",
        ["course_id", "room_id", "suitable", "room_score"],
        room_suitability_rows,
    )
    write_csv(
        scenario_dir / "allowed_assignments.csv",
        ["professor_id", "course_id", "timeslot_id", "room_id", "total_score"],
        allowed_rows,
    )

    return scenario_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate UW-style Math Autumn 2026 scenarios.")
    parser.add_argument("--seed", type=int, default=381)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--stress", action="store_true", help="generate a very large cloned stress scenario")
    parser.add_argument("--scale", type=int, default=25, help="number of clones per base section for --stress")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    for offset in range(args.count):
        scenario_dir = generate_scenario(
            seed=args.seed + offset,
            output_dir=output_dir,
            stress=args.stress,
            scale=args.scale,
        )
        print(f"wrote {scenario_dir}")


if __name__ == "__main__":
    main()
