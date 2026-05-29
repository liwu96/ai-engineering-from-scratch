#!/usr/bin/env python3
"""Translate missing quiz.json files to Chinese as quiz.zh.json."""

import json
import os
import sys
import time
from pathlib import Path

import anthropic

REPO_ROOT = Path(__file__).parent.parent
PHASES_DIR = REPO_ROOT / "phases"
CLIENT = anthropic.Anthropic()
MODEL = "claude-haiku-4-5-20251001"

SYSTEM = """You are a technical translator specializing in AI/ML education. Translate the quiz JSON from English to Simplified Chinese (zh-CN).

Rules:
- Translate: question, options (all array items), explanation
- Keep unchanged: stage, correct (the integer index), all JSON keys
- Keep unchanged: code snippets, model names, technical acronyms (LLM, RAG, RLHF, GPU, API, etc.), proper nouns
- Output ONLY the translated JSON, no markdown fences, no commentary
- Preserve exact JSON structure"""


def find_missing() -> list[Path]:
    missing = []
    for quiz in sorted(PHASES_DIR.rglob("quiz.json")):
        zh = quiz.parent / "quiz.zh.json"
        if not zh.exists():
            missing.append(quiz)
    return missing


def translate(quiz_path: Path) -> dict:
    content = quiz_path.read_text(encoding="utf-8")
    for attempt in range(5):
        try:
            resp = CLIENT.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=SYSTEM,
                messages=[{"role": "user", "content": content}],
            )
            text = resp.content[0].text.strip()
            # Strip markdown fences if model added them
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                text = text.rsplit("```", 1)[0].strip()
            return json.loads(text)
        except anthropic.RateLimitError:
            wait = 2 ** attempt
            print(f"    rate-limited, waiting {wait}s…")
            time.sleep(wait)
        except json.JSONDecodeError as e:
            print(f"    JSON parse error ({e}), retrying…")
            time.sleep(2)
    raise RuntimeError(f"Failed to translate {quiz_path} after 5 attempts")


def main():
    missing = find_missing()
    total = len(missing)
    print(f"Found {total} quiz.json files without quiz.zh.json\n")

    for i, quiz_path in enumerate(missing, 1):
        rel = quiz_path.relative_to(REPO_ROOT)
        out_path = quiz_path.parent / "quiz.zh.json"
        print(f"[{i}/{total}] {rel}")
        try:
            translated = translate(quiz_path)
            out_path.write_text(
                json.dumps(translated, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"    ✓ saved {out_path.name}")
        except Exception as e:
            print(f"    ✗ ERROR: {e}", file=sys.stderr)
        # Small pause to respect rate limits
        time.sleep(0.3)

    print(f"\nDone. Translated {total} files.")


if __name__ == "__main__":
    main()
