#!/usr/bin/env python3
"""
Batch translate quiz.json files to Chinese.
Usage: python translate_quiz.py
"""

import json
import os
import sys
from pathlib import Path

# List of files to translate (from bash output)
FILES_TO_TRANSLATE = [
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
    "phases/04-computer-vision/01-image-fundamentals/quiz.json",
    "phases/04-computer-vision/02-convolutions-from-scratch/quiz.json",
    "phases/04-computer-vision/03-cnns-lenet-to-resnet/quiz.json",
    "phases/04-computer-vision/04-image-classification/quiz.json",
    "phases/04-computer-vision/05-transfer-learning/quiz.json",
    "phases/04-computer-vision/06-object-detection-yolo/quiz.json",
    "phases/04-computer-vision/07-semantic-segmentation-unet/quiz.json",
    "phases/04-computer-vision/08-instance-segmentation-mask-rcnn/quiz.json",
    "phases/04-computer-vision/09-image-generation-gans/quiz.json",
    "phases/04-computer-vision/10-image-generation-diffusion/quiz.json",
    "phases/04-computer-vision/11-stable-diffusion/quiz.json",
    "phases/04-computer-vision/12-video-understanding/quiz.json",
    "phases/04-computer-vision/13-3d-vision-nerf/quiz.json",
    "phases/04-computer-vision/14-vision-transformers/quiz.json",
    "phases/04-computer-vision/15-real-time-edge/quiz.json",
    "phases/04-computer-vision/16-vision-pipeline-capstone/quiz.json",
    "phases/04-computer-vision/17-self-supervised-vision/quiz.json",
    "phases/04-computer-vision/18-open-vocab-clip/quiz.json",
    "phases/04-computer-vision/19-ocr-document-understanding/quiz.json",
    "phases/04-computer-vision/20-image-retrieval-metric/quiz.json",
    "phases/04-computer-vision/21-keypoint-pose/quiz.json",
    "phases/04-computer-vision/22-3d-gaussian-splatting/quiz.json",
    "phases/04-computer-vision/23-diffusion-transformers-rectified-flow/quiz.json",
    "phases/04-computer-vision/24-sam3-open-vocab-segmentation/quiz.json",
    "phases/04-computer-vision/25-vision-language-models/quiz.json",
    "phases/04-computer-vision/26-monocular-depth/quiz.json",
    "phases/04-computer-vision/27-multi-object-tracking/quiz.json",
    "phases/04-computer-vision/28-world-models-video-diffusion/quiz.json",
    "phases/05-nlp-foundations-to-advanced/01-text-processing/quiz.json",
    "phases/05-nlp-foundations-to-advanced/02-bag-of-words-tfidf/quiz.json",
    "phases/05-nlp-foundations-to-advanced/03-word-embeddings-word2vec/quiz.json",
    "phases/05-nlp-foundations-to-advanced/04-glove-fasttext-subword/quiz.json",
    "phases/05-nlp-foundations-to-advanced/05-sentiment-analysis/quiz.json",
    "phases/05-nlp-foundations-to-advanced/06-named-entity-recognition/quiz.json",
    "phases/05-nlp-foundations-to-advanced/07-pos-tagging-parsing/quiz.json",
    "phases/05-nlp-foundations-to-advanced/08-cnns-rnns-for-text/quiz.json",
    "phases/05-nlp-foundations-to-advanced/09-sequence-to-sequence/quiz.json",
    "phases/05-nlp-foundations-to-advanced/10-attention-mechanism/quiz.json",
    "phases/05-nlp-foundations-to-advanced/11-machine-translation/quiz.json",
    "phases/05-nlp-foundations-to-advanced/12-text-summarization/quiz.json",
    "phases/05-nlp-foundations-to-advanced/13-question-answering/quiz.json",
    "phases/05-nlp-foundations-to-advanced/14-information-retrieval-search/quiz.json",
    "phases/05-nlp-foundations-to-advanced/15-topic-modeling/quiz.json",
    "phases/05-nlp-foundations-to-advanced/16-text-generation-pre-transformer/quiz.json",
    "phases/05-nlp-foundations-to-advanced/17-chatbots-rule-to-neural/quiz.json",
    "phases/05-nlp-foundations-to-advanced/18-multilingual-nlp/quiz.json",
    "phases/05-nlp-foundations-to-advanced/19-subword-tokenization/quiz.json",
    "phases/05-nlp-foundations-to-advanced/20-structured-outputs-constrained-decoding/quiz.json",
    "phases/05-nlp-foundations-to-advanced/21-nli-textual-entailment/quiz.json",
    "phases/05-nlp-foundations-to-advanced/22-embedding-models-deep-dive/quiz.json",
    "phases/05-nlp-foundations-to-advanced/23-chunking-strategies-rag/quiz.json",
    "phases/05-nlp-foundations-to-advanced/24-coreference-resolution/quiz.json",
    "phases/05-nlp-foundations-to-advanced/25-entity-linking/quiz.json",
    "phases/05-nlp-foundations-to-advanced/26-relation-extraction-kg/quiz.json",
    "phases/05-nlp-foundations-to-advanced/27-llm-evaluation-frameworks/quiz.json",
    "phases/05-nlp-foundations-to-advanced/28-long-context-evaluation/quiz.json",
    "phases/05-nlp-foundations-to-advanced/29-dialogue-state-tracking/quiz.json",
    "phases/07-transformers-deep-dive/02-self-attention-from-scratch/quiz.json",
    "phases/10-llms-from-scratch/01-tokenizers/quiz.json",
    "phases/10-llms-from-scratch/02-building-a-tokenizer/quiz.json",
    "phases/10-llms-from-scratch/03-data-pipelines/quiz.json",
    "phases/10-llms-from-scratch/04-pre-training-mini-gpt/quiz.json",
    "phases/10-llms-from-scratch/05-scaling-distributed/quiz.json",
    "phases/10-llms-from-scratch/06-instruction-tuning-sft/quiz.json",
    "phases/10-llms-from-scratch/07-rlhf/quiz.json",
    "phases/10-llms-from-scratch/08-dpo/quiz.json",
    "phases/10-llms-from-scratch/10-evaluation/quiz.json",
    "phases/10-llms-from-scratch/11-quantization/quiz.json",
    "phases/10-llms-from-scratch/12-inference-optimization/quiz.json",
    "phases/11-llm-engineering/01-prompt-engineering/quiz.json",
    "phases/11-llm-engineering/02-few-shot-cot/quiz.json",
    "phases/11-llm-engineering/03-structured-outputs/quiz.json",
    "phases/11-llm-engineering/04-embeddings/quiz.json",
    "phases/11-llm-engineering/05-context-engineering/quiz.json",
    "phases/11-llm-engineering/06-rag/quiz.json",
    "phases/11-llm-engineering/07-advanced-rag/quiz.json",
    "phases/11-llm-engineering/08-fine-tuning-lora/quiz.json",
    "phases/11-llm-engineering/09-function-calling/quiz.json",
    "phases/11-llm-engineering/10-evaluation/quiz.json",
    "phases/11-llm-engineering/11-caching-cost/quiz.json",
    "phases/11-llm-engineering/12-guardrails/quiz.json",
    "phases/11-llm-engineering/13-production-app/quiz.json",
    "phases/11-llm-engineering/14-model-context-protocol/quiz.json",
    "phases/11-llm-engineering/15-prompt-caching/quiz.json",
    "phases/11-llm-engineering/16-langgraph-state-machines/quiz.json",
    "phases/11-llm-engineering/17-agent-framework-tradeoffs/quiz.json",
    "phases/14-agent-engineering/01-the-agent-loop/quiz.json",
    "phases/14-agent-engineering/02-rewoo-plan-and-execute/quiz.json",
    "phases/14-agent-engineering/03-reflexion-verbal-rl/quiz.json",
    "phases/14-agent-engineering/04-tree-of-thoughts-lats/quiz.json",
    "phases/14-agent-engineering/05-self-refine-and-critic/quiz.json",
    "phases/14-agent-engineering/06-tool-use-and-function-calling/quiz.json",
    "phases/14-agent-engineering/07-memory-virtual-context-memgpt/quiz.json",
    "phases/14-agent-engineering/08-memory-blocks-sleep-time-compute/quiz.json",
    "phases/14-agent-engineering/09-hybrid-memory-mem0/quiz.json",
    "phases/14-agent-engineering/10-skill-libraries-voyager/quiz.json",
    "phases/14-agent-engineering/11-planning-htn-and-evolutionary/quiz.json",
    "phases/14-agent-engineering/12-anthropic-workflow-patterns/quiz.json",
    "phases/14-agent-engineering/13-langgraph-stateful-graphs/quiz.json",
    "phases/14-agent-engineering/14-autogen-actor-model/quiz.json",
    "phases/14-agent-engineering/15-crewai-role-based-crews/quiz.json",
    "phases/14-agent-engineering/16-openai-agents-sdk/quiz.json",
    "phases/14-agent-engineering/17-claude-agent-sdk/quiz.json",
    "phases/14-agent-engineering/18-agno-and-mastra-runtimes/quiz.json",
    "phases/14-agent-engineering/19-benchmarks-swebench-gaia/quiz.json",
    "phases/14-agent-engineering/20-benchmarks-webarena-osworld/quiz.json",
    "phases/14-agent-engineering/21-computer-use-agents/quiz.json",
    "phases/14-agent-engineering/22-voice-agents-pipecat-livekit/quiz.json",
    "phases/14-agent-engineering/23-otel-genai-conventions/quiz.json",
    "phases/14-agent-engineering/24-agent-observability-platforms/quiz.json",
    "phases/14-agent-engineering/25-multi-agent-debate/quiz.json",
    "phases/14-agent-engineering/26-failure-modes-agentic/quiz.json",
    "phases/14-agent-engineering/27-prompt-injection-defense/quiz.json",
    "phases/14-agent-engineering/28-orchestration-patterns/quiz.json",
    "phases/14-agent-engineering/29-production-runtimes/quiz.json",
    "phases/14-agent-engineering/30-eval-driven-agent-development/quiz.json",
    "phases/14-agent-engineering/31-agent-workbench-why-models-fail/quiz.json",
    "phases/14-agent-engineering/32-minimal-agent-workbench/quiz.json",
    "phases/14-agent-engineering/33-instructions-as-executable-constraints/quiz.json",
    "phases/14-agent-engineering/34-repo-memory-and-state/quiz.json",
    "phases/14-agent-engineering/35-initialization-scripts/quiz.json",
    "phases/14-agent-engineering/36-scope-contracts/quiz.json",
    "phases/14-agent-engineering/37-runtime-feedback-loops/quiz.json",
    "phases/14-agent-engineering/38-verification-gates/quiz.json",
    "phases/14-agent-engineering/39-reviewer-agent/quiz.json",
    "phases/14-agent-engineering/40-multi-session-handoff/quiz.json",
    "phases/14-agent-engineering/41-workbench-for-real-repos/quiz.json",
    "phases/14-agent-engineering/42-agent-workbench-capstone/quiz.json",
    "phases/16-multi-agent-and-swarms/01-why-multi-agent/quiz.json",
    "phases/16-multi-agent-and-swarms/03-communication-protocols/quiz.json",
    "phases/17-infrastructure-and-production/01-managed-llm-platforms/quiz.json",
    "phases/17-infrastructure-and-production/02-inference-platform-economics/quiz.json",
    "phases/17-infrastructure-and-production/03-gpu-autoscaling-kubernetes/quiz.json",
    "phases/17-infrastructure-and-production/04-vllm-serving-internals/quiz.json",
    "phases/17-infrastructure-and-production/05-eagle3-speculative-decoding/quiz.json",
    "phases/17-infrastructure-and-production/06-sglang-radixattention/quiz.json",
    "phases/17-infrastructure-and-production/07-tensorrt-llm-blackwell/quiz.json",
    "phases/17-infrastructure-and-production/08-inference-metrics-goodput/quiz.json",
    "phases/17-infrastructure-and-production/09-production-quantization/quiz.json",
    "phases/17-infrastructure-and-production/10-cold-start-mitigation/quiz.json",
    "phases/17-infrastructure-and-production/11-multi-region-kv-locality/quiz.json",
    "phases/17-infrastructure-and-production/12-edge-inference/quiz.json",
    "phases/17-infrastructure-and-production/13-llm-observability/quiz.json",
    "phases/17-infrastructure-and-production/14-prompt-semantic-caching/quiz.json",
    "phases/17-infrastructure-and-production/15-batch-apis/quiz.json",
    "phases/17-infrastructure-and-production/16-model-routing/quiz.json",
    "phases/17-infrastructure-and-production/17-disaggregated-prefill-decode/quiz.json",
    "phases/17-infrastructure-and-production/18-vllm-production-stack-lmcache/quiz.json",
    "phases/17-infrastructure-and-production/19-ai-gateways/quiz.json",
    "phases/17-infrastructure-and-production/20-shadow-canary-progressive/quiz.json",
    "phases/17-infrastructure-and-production/21-ab-testing-llm-features/quiz.json",
    "phases/17-infrastructure-and-production/22-load-testing-llm-apis/quiz.json",
    "phases/17-infrastructure-and-production/23-sre-for-ai/quiz.json",
    "phases/17-infrastructure-and-production/24-chaos-engineering-llm/quiz.json",
    "phases/17-infrastructure-and-production/25-security-secrets-audit/quiz.json",
    "phases/17-infrastructure-and-production/26-compliance-frameworks/quiz.json",
    "phases/17-infrastructure-and-production/27-finops-llms/quiz.json",
    "phases/17-infrastructure-and-production/28-self-hosted-serving-selection/quiz.json",
    "phases/18-ethics-safety-alignment/01-instruction-following-alignment-signal/quiz.json",
    "phases/18-ethics-safety-alignment/02-reward-hacking-goodhart/quiz.json",
    "phases/18-ethics-safety-alignment/03-direct-preference-optimization-family/quiz.json",
    "phases/18-ethics-safety-alignment/04-sycophancy-rlhf-amplification/quiz.json",
    "phases/18-ethics-safety-alignment/05-constitutional-ai-rlaif/quiz.json",
    "phases/18-ethics-safety-alignment/06-mesa-optimization-deceptive-alignment/quiz.json",
    "phases/18-ethics-safety-alignment/07-sleeper-agents-persistent-deception/quiz.json",
    "phases/18-ethics-safety-alignment/08-in-context-scheming-frontier-models/quiz.json",
    "phases/18-ethics-safety-alignment/09-alignment-faking/quiz.json",
    "phases/18-ethics-safety-alignment/10-ai-control-subversion/quiz.json",
    "phases/18-ethics-safety-alignment/11-scalable-oversight-weak-to-strong/quiz.json",
    "phases/18-ethics-safety-alignment/12-red-teaming-pair-automated-attacks/quiz.json",
    "phases/18-ethics-safety-alignment/13-many-shot-jailbreaking/quiz.json",
    "phases/18-ethics-safety-alignment/14-ascii-art-visual-jailbreaks/quiz.json",
    "phases/18-ethics-safety-alignment/15-indirect-prompt-injection/quiz.json",
    "phases/18-ethics-safety-alignment/16-red-team-tooling-garak-llamaguard-pyrit/quiz.json",
    "phases/18-ethics-safety-alignment/17-wmdp-dual-use-evaluation/quiz.json",
    "phases/18-ethics-safety-alignment/18-frontier-safety-frameworks-rsp-pf-fsf/quiz.json",
    "phases/18-ethics-safety-alignment/19-model-welfare-research/quiz.json",
    "phases/18-ethics-safety-alignment/20-bias-representational-harm/quiz.json",
    "phases/18-ethics-safety-alignment/21-fairness-criteria-group-individual-counterfactual/quiz.json",
    "phases/18-ethics-safety-alignment/22-differential-privacy-for-llms/quiz.json",
    "phases/18-ethics-safety-alignment/23-watermarking-synthid-stable-signature-c2pa/quiz.json",
    "phases/18-ethics-safety-alignment/24-regulatory-frameworks-eu-us-uk-korea/quiz.json",
    "phases/18-ethics-safety-alignment/25-echoleak-cves-for-ai/quiz.json",
    "phases/18-ethics-safety-alignment/26-model-system-dataset-cards/quiz.json",
    "phases/18-ethics-safety-alignment/27-data-provenance-training-governance/quiz.json",
    "phases/18-ethics-safety-alignment/28-alignment-research-ecosystem/quiz.json",
    "phases/18-ethics-safety-alignment/29-moderation-systems-openai-perspective-llamaguard/quiz.json",
    "phases/18-ethics-safety-alignment/30-dual-use-risk-cyber-bio-chem-nuclear/quiz.json",
    "phases/19-capstone-projects/01-terminal-native-coding-agent/quiz.json",
    "phases/19-capstone-projects/02-rag-over-codebase/quiz.json",
    "phases/19-capstone-projects/03-realtime-voice-assistant/quiz.json",
    "phases/19-capstone-projects/04-multimodal-document-qa/quiz.json",
    "phases/19-capstone-projects/05-autonomous-research-agent/quiz.json",
    "phases/19-capstone-projects/06-devops-troubleshooting-agent/quiz.json",
    "phases/19-capstone-projects/07-end-to-end-fine-tuning-pipeline/quiz.json",
    "phases/19-capstone-projects/08-production-rag-chatbot/quiz.json",
    "phases/19-capstone-projects/09-code-migration-agent/quiz.json",
    "phases/19-capstone-projects/10-multi-agent-software-team/quiz.json",
    "phases/19-capstone-projects/11-llm-observability-dashboard/quiz.json",
    "phases/19-capstone-projects/12-video-understanding-pipeline/quiz.json",
    "phases/19-capstone-projects/13-mcp-server-with-registry/quiz.json",
    "phases/19-capstone-projects/14-speculative-decoding-server/quiz.json",
    "phases/19-capstone-projects/15-constitutional-safety-harness/quiz.json",
    "phases/19-capstone-projects/16-github-issue-to-pr-agent/quiz.json",
    "phases/19-capstone-projects/17-personal-ai-tutor/quiz.json",
]


def translate_text(text: str) -> str:
    """Placeholder for translation - just returns original for now."""
    # This will be replaced with actual API calls
    return text


def translate_quiz_data(data):
    """Translate quiz data structure."""
    if isinstance(data, list):
        # Direct array of questions
        for item in data:
            translate_question(item)
        return data
    elif isinstance(data, dict) and "questions" in data:
        # Object with questions array
        for item in data["questions"]:
            translate_question(item)
        return data
    else:
        raise ValueError(f"Unknown quiz format: {type(data)}")


def translate_question(item: dict):
    """Translate a single question item."""
    if "question" in item:
        item["question"] = translate_text(item["question"])
    if "options" in item and isinstance(item["options"], list):
        item["options"] = [translate_text(opt) for opt in item["options"]]
    if "explanation" in item:
        item["explanation"] = translate_text(item["explanation"])


def process_file(filepath: str):
    """Process a single quiz file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Translate
        translated = translate_quiz_data(data)

        # Write output
        output_path = filepath.replace('.json', '.zh.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(translated, f, ensure_ascii=False, indent=2)

        return True, output_path
    except Exception as e:
        return False, str(e)


def main():
    """Main entry point."""
    success_count = 0
    error_count = 0
    errors = []

    print(f"Processing {len(FILES_TO_TRANSLATE)} quiz files...")

    for filepath in FILES_TO_TRANSLATE:
        success, result = process_file(filepath)
        if success:
            success_count += 1
            print(f"✓ {filepath}")
        else:
            error_count += 1
            errors.append(f"✗ {filepath}: {result}")
            print(f"✗ {filepath}: {result}")

    print(f"\n{'='*60}")
    print(f"Completed: {success_count} success, {error_count} errors")

    if errors:
        print("\nErrors:")
        for err in errors:
            print(f"  {err}")

    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
