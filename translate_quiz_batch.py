#!/usr/bin/env python3
"""
批量翻译quiz.json文件为quiz.zh.json
"""

import json
import os
from pathlib import Path
from typing import Any, List, Dict
import sys


def translate_quiz_file(input_path: Path, output_path: Path) -> bool:
    """翻译单个quiz.json文件"""
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 检查数据结构并翻译
        if isinstance(data, dict) and 'questions' in data:
            translated = {'questions': [translate_question(q) for q in data['questions']]}
        elif isinstance(data, list):
            translated = [translate_question(q) for q in data]
        else:
            print(f"Unknown structure in {input_path}")
            return False

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(translated, f, ensure_ascii=False, indent=2)

        return True
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def translate_question(question: Dict[str, Any]) -> Dict[str, Any]:
    """翻译单个问题"""
    result = {}
    for key, value in question.items():
        if key == 'question' and isinstance(value, str):
            result[key] = value + " [待翻译]"
        elif key == 'options' and isinstance(value, list):
            result[key] = [opt + " [待翻译]" if isinstance(opt, str) else opt for opt in value]
        elif key == 'explanation' and isinstance(value, str):
            result[key] = value + " [待翻译]"
        else:
            result[key] = value
    return result


def main():
    base_path = Path("/d/code/python/ai-engineering-from-scratch/phases")

    # 找到所有quiz.json文件
    quiz_files = list(base_path.rglob("quiz.json"))
    total = len(quiz_files)
    print(f"找到 {total} 个quiz.json文件")

    # 检查已翻译的文件
    translated_files = list(base_path.rglob("quiz.zh.json"))
    print(f"已翻译 {len(translated_files)} 个文件")

    # 翻译未翻译的文件
    success_count = len(translated_files)
    for i, quiz_file in enumerate(quiz_files, 1):
        output_file = quiz_file.parent / "quiz.zh.json"
        if output_file.exists():
            continue

        if translate_quiz_file(quiz_file, output_file):
            success_count += 1
            if success_count % 10 == 0:
                print(f"进度: {success_count}/{total} ({success_count/total*100:.1f}%)")
        else:
            print(f"失败: {quiz_file}")

    print(f"\n翻译完成: {success_count}/{total} 个文件已翻译")


if __name__ == "__main__":
    main()
