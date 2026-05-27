#!/usr/bin/env python3
"""
批量翻译quiz.json文件为中文
使用Anthropic API进行翻译
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import concurrent.futures
from anthropic import Anthropic

# 初始化Anthropic客户端
client = Anthropic()


def translate_text_batch(texts: List[str], context: str = "") -> List[str]:
    """
    批量翻译文本
    """
    if not texts:
        return []

    # 构建提示
    prompt = f"""请将以下AI/ML教育内容的英文翻译成简体中文。保持技术术语的准确性。

{context}

待翻译文本（每行一条，保留行号）：
"""
    for i, text in enumerate(texts):
        prompt += f"{i+1}. {text}\n"

    prompt += """
请按相同格式返回翻译结果（每行一条，保留行号）：
"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-7-20251001",
            max_tokens=4096,
            temperature=0.1,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        # 解析响应
        content = response.content[0].text.strip()
        translations = []

        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
            # 去除行号前缀（如 "1. " 或 "1."）
            if '. ' in line[:10]:
                line = line.split('. ', 1)[1]
            elif line[0].isdigit() and line[1] == '.':
                line = line[2:].strip()
            translations.append(line)

        # 确保返回数量与输入相同
        while len(translations) < len(texts):
            translations.append("")
        return translations[:len(texts)]

    except Exception as e:
        print(f"翻译错误: {e}")
        return texts  # 失败时返回原文


def translate_quiz_data(data: Any) -> Any:
    """翻译测验数据"""
    if isinstance(data, list):
        return [translate_question(q) for q in data]
    elif isinstance(data, dict) and 'questions' in data:
        return {'questions': [translate_question(q) for q in data['questions']]}
    else:
        return data


def translate_question(question: Dict[str, Any]) -> Dict[str, Any]:
    """翻译单个问题"""
    result = dict(question)

    # 收集需要翻译的文本
    texts_to_translate = []
    text_map = []  # 记录文本类型和位置

    if 'question' in result and isinstance(result['question'], str):
        texts_to_translate.append(result['question'])
        text_map.append(('question', None))

    if 'options' in result and isinstance(result['options'], list):
        for i, opt in enumerate(result['options']):
            if isinstance(opt, str):
                texts_to_translate.append(opt)
                text_map.append(('option', i))

    if 'explanation' in result and isinstance(result['explanation'], str):
        texts_to_translate.append(result['explanation'])
        text_map.append(('explanation', None))

    if not texts_to_translate:
        return result

    # 批量翻译
    context = f"问题ID: {result.get('id', 'unknown')}, 阶段: {result.get('stage', 'unknown')}"
    translations = translate_text_batch(texts_to_translate, context)

    # 回填翻译结果
    for (text_type, index), translation in zip(text_map, translations):
        if text_type == 'question':
            result['question'] = translation
        elif text_type == 'option':
            result['options'][index] = translation
        elif text_type == 'explanation':
            result['explanation'] = translation

    return result


def process_file(input_path: Path) -> bool:
    """处理单个文件"""
    output_path = input_path.parent / "quiz.zh.json"

    if output_path.exists():
        return True  # 已存在，跳过

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 翻译
        translated = translate_quiz_data(data)

        # 保存
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(translated, f, ensure_ascii=False, indent=2)

        return True
    except Exception as e:
        print(f"处理失败 {input_path}: {e}")
        return False


def get_files_to_translate() -> List[Path]:
    """获取需要翻译的文件列表"""
    base_path = Path("phases")

    # 从之前的脚本中提取的文件列表
    file_list = [
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
        # Phase 3 (13 files)
        "phases/03-deep-learning-core/01-the-perceptron/quiz.json",
        "phases/03-deep-learning-core/02-multi-layer-networks/quiz.json",
        "phases/03-deep-learning-core/03-backpropagation/quiz.json",
        "phases/03-deep-learning-core/04-activation-functions/quiz.json",
        "phases/03-deep-learning-core/05-loss-functions/quiz.json",
        "phases/03-deep-learning-core/06-optimizers/quiz.json",
        "phases/03-deep-learning-core/07-regularization/quiz.json",
        "phases/03-deep-learning-core/08-weight-initialization/quiz.json",
        "phases/03-deep-learning-core/09-learning-rate-schedules/quiz.json",
        "phases/03-deep-learning-core/10-mini-framework/quiz.json",
        "phases/03-deep-learning-core/11-intro-to-pytorch/quiz.json",
        "phases/03-deep-learning-core/12-intro-to-jax/quiz.json",
        "phases/03-deep-learning-core/13-debugging-neural-networks/quiz.json",
    ]

    return [Path(f) for f in file_list if Path(f).exists()]


def main():
    files = get_files_to_translate()
    print(f"需要翻译 {len(files)} 个文件")

    success = 0
    failed = 0

    for i, file_path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] 处理 {file_path}...", end=" ")

        if process_file(file_path):
            success += 1
            print("✓")
        else:
            failed += 1
            print("✗")

    print(f"\n完成: {success} 成功, {failed} 失败")


if __name__ == "__main__":
    main()
