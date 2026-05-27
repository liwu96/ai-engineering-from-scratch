#!/usr/bin/env python3
"""Translate AI Engineering curriculum from English to Chinese.

This script manages the translation of:
- Lesson docs (en.md → zh.md)
- Quiz files (quiz.json)
- Glossary (terms.md)

Usage:
    python scripts/translate_lessons.py --phase 1          # Translate phase 1
    python scripts/translate_lessons.py --phase all         # Translate all phases
    python scripts/translate_lessons.py --quiz              # Translate quiz files
    python scripts/translate_lessons.py --glossary         # Translate glossary
    python scripts/translate_lessons.py --status           # Show translation status
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
PHASES_DIR = ROOT / "phases"
GLOSSARY_PATH = ROOT / "glossary" / "terms.md"

PHASE_DIR_RE = re.compile(r"^([0-9]{2})-([a-z0-9][a-z0-9-]*)$")
LESSON_DIR_RE = re.compile(r"^([0-9]{2})-([a-z0-9][a-z0-9-]*)$")


@dataclass
class TranslationStats:
    total_lessons: int = 0
    translated_docs: int = 0
    translated_quizzes: int = 0
    missing_docs: list[str] = field(default_factory=list)
    missing_quizzes: list[str] = field(default_factory=list)


def iter_lesson_dirs(phase_filter: int | None = None) -> Iterable[Path]:
    """Iterate over all lesson directories."""
    if not PHASES_DIR.is_dir():
        return
    for phase in sorted(PHASES_DIR.iterdir()):
        if not phase.is_dir():
            continue
        m = PHASE_DIR_RE.match(phase.name)
        if not m:
            continue
        phase_num = int(m.group(1))
        if phase_filter is not None and phase_num != phase_filter:
            continue
        for lesson in sorted(phase.iterdir()):
            if lesson.is_dir() and LESSON_DIR_RE.match(lesson.name):
                yield lesson


def get_translation_status() -> TranslationStats:
    """Get current translation status."""
    stats = TranslationStats()
    for lesson in iter_lesson_dirs():
        stats.total_lessons += 1
        docs_dir = lesson / "docs"
        en_md = docs_dir / "en.md"
        zh_md = docs_dir / "zh.md"
        quiz_json = lesson / "quiz.json"
        zh_quiz_json = lesson / "quiz.zh.json"

        if zh_md.exists():
            stats.translated_docs += 1
        elif en_md.exists():
            stats.missing_docs.append(str(lesson.relative_to(PHASES_DIR)))

        if zh_quiz_json.exists():
            stats.translated_quizzes += 1
        elif quiz_json.exists():
            stats.missing_quizzes.append(str(lesson.relative_to(PHASES_DIR)))

    return stats


def print_status():
    """Print translation status report."""
    stats = get_translation_status()
    print("=" * 60)
    print("TRANSLATION STATUS REPORT")
    print("=" * 60)
    print(f"\nTotal lessons: {stats.total_lessons}")
    print(f"\nDocs translated: {stats.translated_docs} / {stats.total_lessons} ({100*stats.translated_docs/stats.total_lessons:.1f}%)")
    print(f"Quizzes translated: {stats.translated_quizzes} / {stats.total_lessons} ({100*stats.translated_quizzes/stats.total_lessons:.1f}%)")
    print(f"\nMissing docs: {len(stats.missing_docs)}")
    print(f"Missing quizzes: {len(stats.missing_quizzes)}")

    if stats.missing_docs:
        print("\n" + "-" * 60)
        print("SAMPLE OF MISSING DOCS (first 10):")
        for path in stats.missing_docs[:10]:
            print(f"  - {path}")


def list_phase_lessons(phase_num: int) -> list[Path]:
    """List all lessons in a phase."""
    return list(iter_lesson_dirs(phase_num))


def create_translation_script(phase_num: int):
    """Create a translation script for a specific phase."""
    lessons = list_phase_lessons(phase_num)
    phase_dir = PHASES_DIR / f"{phase_num:02d}-phase-name"  # Placeholder

    print(f"Phase {phase_num}: {len(lessons)} lessons")
    print("\nLessons to translate:")
    for lesson in lessons:
        en_file = lesson / "docs" / "en.md"
        zh_file = lesson / "docs" / "zh.md"
        status = "✓" if zh_file.exists() else "✗"
        print(f"  [{status}] {lesson.name}")


def main():
    parser = argparse.ArgumentParser(description="Manage curriculum translation")
    parser.add_argument("--status", action="store_true", help="Show translation status")
    parser.add_argument("--phase", type=str, help="Phase number or 'all'")
    parser.add_argument("--quiz", action="store_true", help="Translate quiz files")
    parser.add_argument("--glossary", action="store_true", help="Translate glossary")
    parser.add_argument("--list", action="store_true", help="List lessons in phase")

    args = parser.parse_args()

    if args.status or len(sys.argv) == 1:
        print_status()
        return

    if args.list and args.phase:
        phase_num = int(args.phase)
        create_translation_script(phase_num)
        return

    print("Translation script ready. Use --status to see current progress.")


if __name__ == "__main__":
    main()
