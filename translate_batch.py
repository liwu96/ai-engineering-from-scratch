#!/usr/bin/env python3
"""
Batch translation script for AI Engineering course lessons (Phase 5-9).
Translates docs/en.md to docs/zh.md.
"""

import os
import re
from pathlib import Path

# Terminology mapping for translation consistency
TERMINOLOGY = {
    # Core AI/ML terms
    "AI": "AI",
    "ML": "ML",
    "LLM": "大语言模型",
    "Large Language Model": "大语言模型",
    "Neural Network": "神经网络",
    "Deep Learning": "深度学习",
    "Transformer": "Transformer",
    "Embedding": "嵌入",
    "embeddings": "嵌入向量",
    "Token": "Token/词元",
    "token": "token/词元",
    "tokens": "tokens/词元",
    "Attention": "注意力机制",
    "Self-Attention": "自注意力机制",
    "Multi-Head Attention": "多头注意力机制",
    "Fine-tuning": "微调",
    "fine-tuning": "微调",
    "fine-tune": "微调",
    "Prompt": "提示词",
    "prompt": "提示词",
    "Inference": "推理",
    "Training": "训练",
    "Gradient Descent": "梯度下降",
    "Backpropagation": "反向传播",
    "Loss Function": "损失函数",
    "Activation Function": "激活函数",
    "Optimization": "优化",
    "Regularization": "正则化",
    "Overfitting": "过拟合",
    "Underfitting": "欠拟合",

    # RL terms
    "Reinforcement Learning": "强化学习",
    "RL": "强化学习",
    "Policy": "策略",
    "policy": "策略",
    "Reward": "奖励",
    "reward": "奖励",
    "Agent": "智能体",
    "agent": "智能体",

    # Generative AI
    "VAE": "VAE/变分自编码器",
    "Variational Autoencoder": "变分自编码器",
    "GAN": "GAN/生成对抗网络",
    "Generative Adversarial Network": "生成对抗网络",
    "Diffusion": "扩散模型",
    "diffusion": "扩散模型",
    "Diffusion Model": "扩散模型",

    # NLP terms
    "NER": "命名实体识别",
    "Named Entity Recognition": "命名实体识别",
    "POS": "词性标注",
    "Part-of-Speech": "词性标注",
    "RAG": "RAG/检索增强生成",
    "Retrieval-Augmented Generation": "检索增强生成",
    "NLI": "自然语言推理",
    "Natural Language Inference": "自然语言推理",
    "QA": "问答",
    "Question Answering": "问答",
    "IR": "信息检索",
    "Information Retrieval": "信息检索",
    "NLP": "自然语言处理",
    "Natural Language Processing": "自然语言处理",
    "BPE": "BPE/字节对编码",
    "WordPiece": "WordPiece",
    "Unigram": "Unigram",
    "SentencePiece": "SentencePiece",

    # Evaluation
    "Perplexity": "困惑度",
    "Accuracy": "准确率",
    "Precision": "精确率",
    "Recall": "召回率",
    "F1 Score": "F1分数",
    "Epoch": "轮次",
    "Batch": "批次",
    "Hyperparameter": "超参数",
    "Dataset": "数据集",
    "Model": "模型",
    "model": "模型",
    "Layer": "层",
    "layer": "层",

    # Other
    "Vector": "向量",
    "vector": "向量",
    "Matrix": "矩阵",
    "matrix": "矩阵",
    "Artificial Intelligence": "人工智能",
    "Machine Learning": "机器学习",
}

# Section translations
SECTION_MAP = {
    "## Learning Objectives": "## 学习目标",
    "## The Problem": "## 问题背景",
    "## The Concept": "## 概念讲解",
    "## Build It": "## 动手实践",
    "## Use It": "## 实际应用",
    "## Ship It": "## 产出成果",
    "## Exercises": "## 练习题",
    "## Key Terms": "## 关键术语",
    "## Further Reading": "## 延伸阅读",
}

# Metadata translations
META_MAP = {
    "**Type:**": "**类型:**",
    "**Languages:**": "**语言:**",
    "**Prerequisites:**": "**前置要求:**",
    "**Time:**": "**时间:**",
}

# Table headers
TABLE_MAP = {
    "| Term |": "| 术语 |",
    "| What people say |": "| 人们怎么说 |",
    "| What it actually means |": "| 实际含义 |",
}


def translate_section_headers(text):
    """Translate section headers"""
    lines = text.split('\n')
    result = []

    for line in lines:
        stripped = line.strip()

        # Check for exact section matches
        for eng, chn in SECTION_MAP.items():
            if stripped == eng or stripped.startswith(eng + " "):
                line = line.replace(eng, chn)
                break

        # Check for metadata
        for eng, chn in META_MAP.items():
            if stripped.startswith(eng):
                line = line.replace(eng, chn)
                break

        # Check for table headers
        for eng, chn in TABLE_MAP.items():
            if stripped.startswith(eng):
                line = line.replace(eng, chn)
                break

        # Translate "Key Terms" table separator
        if "|------|-----------------|-----------------------|" in line:
            line = "|------|------------|----------|"

        result.append(line)

    return '\n'.join(result)


def translate_terminology(text):
    """Translate terminology while preserving code blocks"""
    # Split into code and non-code sections
    parts = []
    in_code = False
    current_part = []

    for line in text.split('\n'):
        if line.strip().startswith('```'):
            if current_part:
                parts.append(('text', '\n'.join(current_part)))
                current_part = []
            in_code = not in_code
            parts.append(('code', line))
        else:
            if in_code:
                parts.append(('code', line))
            else:
                current_part.append(line)

    if current_part:
        parts.append(('text', '\n'.join(current_part)))

    # Translate non-code parts
    result = []
    for part_type, content in parts:
        if part_type == 'code':
            result.append(content)
        else:
            # Translate terminology
            translated = content
            for eng, chn in TERMINOLOGY.items():
                # Word boundary matching
                pattern = r'\b' + re.escape(eng) + r'\b'
                translated = re.sub(pattern, chn, translated)
            result.append(translated)

    return '\n'.join(result)


def translate_lesson_file(en_path, zh_path):
    """Translate a single lesson file"""
    try:
        with open(en_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply translations
        translated = translate_section_headers(content)
        translated = translate_terminology(translated)

        with open(zh_path, 'w', encoding='utf-8') as f:
            f.write(translated)

        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


def get_lessons_in_phase(phase_num):
    """Get all lessons in a phase"""
    phase_map = {
        5: "05-nlp-foundations-to-advanced",
        6: "06-speech-and-audio",
        7: "07-transformers-deep-dive",
        8: "08-generative-ai",
        9: "09-reinforcement-learning",
    }

    phase_name = phase_map.get(phase_num)
    if not phase_name:
        return []

    phase_path = Path("phases") / phase_name
    if not phase_path.exists():
        return []

    lessons = []
    for item in phase_path.iterdir():
        if item.is_dir() and item.name[0].isdigit():
            lessons.append(item)

    return sorted(lessons)


def main():
    """Main function"""
    total_translated = 0
    total_skipped = 0

    # Phase 5: lessons 13-29 only
    print("\n" + "=" * 60)
    print("Phase 5: NLP Foundations (lessons 13-29)")
    print("=" * 60)
    phase5_lessons = get_lessons_in_phase(5)
    phase5_count = 0
    for lesson in phase5_lessons:
        lesson_num = int(lesson.name.split('-')[0])
        if lesson_num >= 13:
            en_file = lesson / "docs" / "en.md"
            zh_file = lesson / "docs" / "zh.md"
            if en_file.exists():
                if zh_file.exists():
                    print(f"  SKIP: {lesson.name} (already translated)")
                    total_skipped += 1
                else:
                    print(f"  Translating: {lesson.name}")
                    if translate_lesson_file(en_file, zh_file):
                        phase5_count += 1
                        total_translated += 1

    print(f"\nPhase 5 completed: {phase5_count} lessons translated")

    # Phase 6-9: all lessons
    for phase_num in [6, 7, 8, 9]:
        phase_names = {
            6: "Speech & Audio",
            7: "Transformers Deep Dive",
            8: "Generative AI",
            9: "Reinforcement Learning",
        }

        print("\n" + "=" * 60)
        print(f"Phase {phase_num}: {phase_names[phase_num]}")
        print("=" * 60)

        lessons = get_lessons_in_phase(phase_num)
        phase_count = 0

        for lesson in lessons:
            en_file = lesson / "docs" / "en.md"
            zh_file = lesson / "docs" / "zh.md"
            if en_file.exists():
                if zh_file.exists():
                    print(f"  SKIP: {lesson.name} (already translated)")
                    total_skipped += 1
                else:
                    print(f"  Translating: {lesson.name}")
                    if translate_lesson_file(en_file, zh_file):
                        phase_count += 1
                        total_translated += 1

        print(f"\nPhase {phase_num} completed: {phase_count} lessons translated")

    # Summary
    print("\n" + "=" * 60)
    print("TRANSLATION SUMMARY")
    print("=" * 60)
    print(f"Total translated: {total_translated}")
    print(f"Total skipped (already exists): {total_skipped}")


if __name__ == "__main__":
    main()
