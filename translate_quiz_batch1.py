#!/usr/bin/env python3
"""
批量翻译quiz.json文件为中文
使用Anthropic API进行翻译
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
from anthropic import Anthropic

# 初始化Anthropic客户端
client = Anthropic()

# 所有需要翻译的文件
FILES_TO_TRANSLATE = [
    # Phase 2 remaining (14 files)
    "phases/02-ml-fundamentals/05-support-vector-machines/quiz.json",
    "phases/02-ml-fundamentals/06-knn-and-distances/quiz.json",
    "phases/02-ml-fundamentals/07-unsupervised-learning/quiz.json",
    "phases/02-ml-fundamentals/08-feature-engineering/quiz.json",
    "phases/02-ml-fundamentals/09-model-evaluation/quiz.json",
    "phases/02-ml-fundamentals/10-bias-variance/quiz.json",
    "phases/02-ml-fundamentals/11-ensemble-methods/quiz.json",
    "phases/02-ml-fundamentals/12-hyperparameter-tuning/quiz.json",
    "phases/02-ml-fundamentals/13-ml-pipelines/quiz.json",
    "phases/02-ml-fundamentals/14-naive-bayes/quiz.json",
    "phases/02-ml-fundamentals/15-time-series/quiz.json",
    "phases/02-ml-fundamentals/16-anomaly-detection/quiz.json",
    "phases/02-ml-fundamentals/17-imbalanced-data/quiz.json",
    "phases/02-ml-fundamentals/18-feature-selection/quiz.json",
]


def translate_with_claude(texts: List[str]) -> List[str]:
    """使用Claude API批量翻译文本"""
    if not texts:
        return []

    prompt = """请将以下AI/ML教育内容的英文翻译成简体中文。保持技术术语的准确性。

待翻译文本（每行一条）：
"""
    for i, text in enumerate(texts, 1):
        prompt += f"{i}. {text}\n"

    prompt += """
请按相同格式返回翻译结果（每行以数字开头，后跟翻译）：
"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-7-20251001",
            max_tokens=4096,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
        )

        content = response.content[0].text.strip()
        translations = []

        for line in content.split('\n'):
            line = line.strip()
            if not line or not line[0].isdigit():
                continue
            # 去除行号前缀
            if '. ' in line[:5]:
                line = line.split('. ', 1)[1]
            elif '.' in line[:3]:
                line = line.split('.', 1)[1].strip()
            translations.append(line)

        # 确保返回数量与输入相同
        while len(translations) < len(texts):
            translations.append("")
        return translations[:len(texts)]

    except Exception as e:
        print(f"  翻译错误: {e}")
        return texts


def translate_quiz_file(filepath: str) -> bool:
    """翻译单个测验文件"""
    output_path = filepath.replace('.json', '.zh.json')

    if os.path.exists(output_path):
        return True  # 已存在，跳过

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 收集所有需要翻译的文本
        texts_to_translate = []
        text_locations = []

        questions = data if isinstance(data, list) else data.get('questions', [])

        for q_idx, question in enumerate(questions):
            if 'question' in question and isinstance(question['question'], str):
                texts_to_translate.append(question['question'])
                text_locations.append(('question', q_idx, None))

            if 'options' in question and isinstance(question['options'], list):
                for opt_idx, opt in enumerate(question['options']):
                    if isinstance(opt, str):
                        texts_to_translate.append(opt)
                        text_locations.append(('option', q_idx, opt_idx))

            if 'explanation' in question and isinstance(question['explanation'], str):
                texts_to_translate.append(question['explanation'])
                text_locations.append(('explanation', q_idx, None))

        if not texts_to_translate:
            return False

        # 批量翻译
        translations = translate_with_claude(texts_to_translate)

        # 回填翻译结果
        for (text_type, q_idx, opt_idx), translation in zip(text_locations, translations):
            if text_type == 'question':
                questions[q_idx]['question'] = translation
            elif text_type == 'option':
                questions[q_idx]['options'][opt_idx] = translation
            elif text_type == 'explanation':
                questions[q_idx]['explanation'] = translation

        # 保存结果
        result = questions if isinstance(data, list) else {'questions': questions}
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return True

    except Exception as e:
        print(f"  错误: {e}")
        return False


def main():
    print(f"总共 {len(FILES_TO_TRANSLATE)} 个文件需要翻译\n")

    success = 0
    failed = 0
    skipped = 0

    for i, filepath in enumerate(FILES_TO_TRANSLATE, 1):
        filename = os.path.basename(os.path.dirname(filepath))
        print(f"[{i:3d}/{len(FILES_TO_TRANSLATE)}] {filename}...", end=" ", flush=True)

        output_path = filepath.replace('.json', '.zh.json')
        if os.path.exists(output_path):
            skipped += 1
            print("SKIPPED")
            continue

        if translate_quiz_file(filepath):
            success += 1
            print("OK")
        else:
            failed += 1
            print("FAILED")

        # 添加延迟以避免API限制
        time.sleep(0.5)

    print(f"\n{'='*50}")
    print(f"完成: {success} 成功, {skipped} 跳过, {failed} 失败")


if __name__ == "__main__":
    main()
