#!/usr/bin/env python3
"""
Comprehensive batch translation script for AI Engineering course lessons.
Uses Anthropic API to translate docs/en.md to docs/zh.md.
"""

import os
import sys
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
PHASES = {
    16: {
        "name": "16-multi-agent-and-swarms",
        "lessons": [
            "02-fipa-acl-heritage", "03-communication-protocols", "04-primitive-model",
            "05-supervisor-orchestrator-pattern", "06-hierarchical-architecture",
            "07-society-of-mind-debate", "08-role-specialization", "09-parallel-swarm-networks",
            "10-group-chat-speaker-selection", "11-handoffs-and-routines", "12-a2a-protocol",
            "13-shared-memory-blackboard", "14-consensus-and-bft", "15-voting-debate-topology",
            "16-negotiation-bargaining", "17-generative-agents-simulation",
            "18-theory-of-mind-coordination", "19-swarm-optimization-pso-aco",
            "20-marl-maddpg-qmix-mappo", "21-agent-economies",
            "22-production-scaling-queues-checkpoints", "23-failure-modes-mast-groupthink",
            "24-evaluation-coordination-benchmarks", "25-case-studies-2026-sota"
        ]
    },
    17: {
        "name": "17-infrastructure-and-production",
        "lessons": [
            "01-managed-llm-platforms", "02-inference-platform-economics",
            "03-gpu-autoscaling-kubernetes", "04-vllm-serving-internals",
            "05-eagle3-speculative-decoding", "06-sglang-radixattention",
            "07-tensorrt-llm-blackwell", "08-inference-metrics-goodput",
            "09-production-quantization", "10-cold-start-mitigation",
            "11-multi-region-kv-locality", "12-edge-inference",
            "13-llm-observability", "14-prompt-semantic-caching",
            "15-batch-apis", "16-model-routing",
            "17-disaggregated-prefill-decode", "18-vllm-production-stack-lmcache",
            "19-ai-gateways", "20-shadow-canary-progressive",
            "21-ab-testing-llm-features", "22-load-testing-llm-apis",
            "23-sre-for-ai", "24-chaos-engineering-llm",
            "25-security-secrets-audit", "26-compliance-frameworks",
            "27-finops-llms", "28-self-hosted-serving-selection"
        ]
    },
    18: {
        "name": "18-ethics-safety-alignment",
        "lessons": [
            "01-instruction-following-alignment-signal", "02-reward-hacking-goodhart",
            "03-direct-preference-optimization-family", "04-sycophancy-rlhf-amplification",
            "05-constitutional-ai-rlaif", "06-mesa-optimization-deceptive-alignment",
            "07-sleeper-agents-persistent-deception", "08-in-context-scheming-frontier-models",
            "09-alignment-faking", "10-ai-control-subversion",
            "11-scalable-oversight-weak-to-strong", "12-red-teaming-pair-automated-attacks",
            "13-many-shot-jailbreaking", "14-ascii-art-visual-jailbreaks",
            "15-indirect-prompt-injection", "16-red-team-tooling-garak-llamaguard-pyrit",
            "17-wmdp-dual-use-evaluation", "18-frontier-safety-frameworks-rsp-pf-fsf",
            "19-model-welfare-research", "20-bias-representational-harm",
            "21-fairness-criteria-group-individual-counterfactual",
            "22-differential-privacy-for-llms",
            "23-watermarking-synthid-stable-signature-c2pa",
            "24-regulatory-frameworks-eu-us-uk-korea", "25-echoleak-cves-for-ai",
            "26-model-system-dataset-cards", "27-data-provenance-training-governance",
            "28-alignment-research-ecosystem",
            "29-moderation-systems-openai-perspective-llamaguard",
            "30-dual-use-risk-cyber-bio-chem-nuclear"
        ]
    },
    19: {
        "name": "19-capstone-projects",
        "lessons": [
            "01-terminal-native-coding-agent", "02-rag-over-codebase",
            "03-realtime-voice-assistant", "04-multimodal-document-qa",
            "05-autonomous-research-agent", "06-devops-troubleshooting-agent",
            "07-end-to-end-fine-tuning-pipeline", "08-production-rag-chatbot",
            "09-code-migration-agent", "10-multi-agent-software-team",
            "11-llm-observability-dashboard", "12-video-understanding-pipeline",
            "13-mcp-server-with-registry", "14-speculative-decoding-server",
            "15-constitutional-safety-harness", "16-github-issue-to-pr-agent",
            "17-personal-ai-tutor"
        ]
    }
}


def get_lessons_to_translate():
    """Get all lessons that need translation."""
    lessons = []
    base_path = Path("phases")

    for phase_num, phase_info in PHASES.items():
        phase_path = base_path / phase_info["name"]
        if not phase_path.exists():
            print(f"Phase directory not found: {phase_path}")
            continue

        for lesson_name in phase_info["lessons"]:
            lesson_path = phase_path / lesson_name
            docs_path = lesson_path / "docs"
            en_file = docs_path / "en.md"
            zh_file = docs_path / "zh.md"

            if en_file.exists() and not zh_file.exists():
                lessons.append({
                    "phase": phase_num,
                    "phase_name": phase_info["name"],
                    "lesson": lesson_name,
                    "en_path": en_file,
                    "zh_path": zh_file,
                })

    return lessons


def translate_lesson_with_claude(en_path, zh_path, lesson_info):
    """Translate a single lesson using Claude API."""
    try:
        import anthropic

        # Read English content
        with open(en_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Skip if too short (might be a stub)
        if len(content) < 100:
            return {"status": "skipped", "reason": "too_short", "lesson": lesson_info}

        # Create translation prompt
        prompt = f"""Translate the following AI Engineering course lesson from English to Chinese (Simplified).

CRITICAL INSTRUCTIONS:
1. Keep ALL code blocks unchanged - do not translate code, variable names, function names
2. Keep ALL file paths, URLs, and technical identifiers unchanged
3. Translate all prose content: headings, paragraphs, lists, table content, mermaid diagram text
4. Keep markdown formatting exactly as is: # headers, ## headers, **bold**, `code`, etc.
5. Use consistent terminology:
   - Multi-Agent -> 多智能体
   - Agent -> 智能体
   - Swarm -> 集群
   - Orchestrator -> 编排器
   - Alignment -> 对齐
   - Safety -> 安全
   - Ethics -> 伦理
   - Infrastructure -> 基础设施
   - Production -> 生产环境
   - Deployment -> 部署
   - Capstone -> 毕业项目
   - MLOps -> MLOps
   - Monitoring -> 监控
   - CI/CD -> CI/CD
   - Red Teaming -> 红队测试
   - Governance -> 治理
   - LLM -> 大语言模型
   - Fine-tuning -> 微调
   - Inference -> 推理
   - Training -> 训练
   - MCP -> MCP
   - A2A -> A2A
   - ANP -> ANP
   - ACP -> ACP

ORIGINAL CONTENT:
{content}

TRANSLATED CONTENT (Chinese):"""

        client = anthropic.Anthropic()

        response = client.messages.create(
            model="claude-sonnet-4-6-20250514",
            max_tokens=8000,
            messages=[{"role": "user", "content": prompt}]
        )

        translated = response.content[0].text

        # Write translated content
        with open(zh_path, 'w', encoding='utf-8') as f:
            f.write(translated)

        return {"status": "success", "lesson": lesson_info}

    except Exception as e:
        return {"status": "error", "error": str(e), "lesson": lesson_info}


def main():
    """Main entry point."""
    lessons = get_lessons_to_translate()

    print(f"\nFound {len(lessons)} lessons needing translation:\n")

    # Group by phase
    by_phase = {}
    for lesson in lessons:
        phase = lesson["phase"]
        if phase not in by_phase:
            by_phase[phase] = []
        by_phase[phase].append(lesson)

    for phase, phase_lessons in sorted(by_phase.items()):
        print(f"  Phase {phase}: {len(phase_lessons)} lessons")

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nWARNING: ANTHROPIC_API_KEY not set. Translation requires API access.")
        print("Set the environment variable and run again.")
        sys.exit(1)

    # Translate in batches
    print(f"\nStarting translation of {len(lessons)} lessons...")

    results = {
        "success": [],
        "error": [],
        "skipped": []
    }

    for i, lesson in enumerate(lessons, 1):
        print(f"\n[{i}/{len(lessons)}] Translating: {lesson['phase_name']}/{lesson['lesson']}")

        result = translate_lesson_with_claude(
            lesson["en_path"],
            lesson["zh_path"],
            lesson
        )

        if result["status"] == "success":
            results["success"].append(result)
            print(f"  ✓ Success")
        elif result["status"] == "skipped":
            results["skipped"].append(result)
            print(f"  ⊘ Skipped: {result.get('reason', 'unknown')}")
        else:
            results["error"].append(result)
            print(f"  ✗ Error: {result.get('error', 'unknown')}")

        # Rate limiting
        time.sleep(1)

    # Summary
    print("\n" + "=" * 60)
    print("TRANSLATION SUMMARY")
    print("=" * 60)
    print(f"Total lessons: {len(lessons)}")
    print(f"Successful: {len(results['success'])}")
    print(f"Skipped: {len(results['skipped'])}")
    print(f"Errors: {len(results['error'])}")

    # Save results
    with open("translation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to translation_results.json")


if __name__ == "__main__":
    main()
