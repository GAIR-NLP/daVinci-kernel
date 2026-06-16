# daVinci-kernel: Co-Evolving Skill Selection, Summarization, and Utilization via RL for GPU Kernel Optimization

[![Paper](https://img.shields.io/badge/arXiv-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2606.16497)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-yellow)](https://huggingface.co/datasets/GAIR/daVinci-kernel-sft/tree/main)
[![Models](https://img.shields.io/badge/🤗%20Models-yellow)](https://huggingface.co/GAIR-NLP/daVinci-kernel-14B-RL)

**daVinci-kernel** is a reinforcement learning framework for GPU kernel optimization that couples skill discovery with skill exploitation through a dynamically evolving skill library. daVinci-kernel jointly trains three agents sharing one LLM backbone: a **Skill Selection Agent** that retrieves relevant techniques via BM25 and LLM reranking, a **Policy Agent** that generates multi-turn CUDA/Triton kernels conditioned on selected skills, and a **Skill Summary Agent** that distills successful rollouts into reusable skills. Candidate skills are added only after execution-based verification confirms reproducible speedups.

> **daVinci-kernel: Co-Evolving Skill Selection, Summarization, and Utilization via RL for GPU Kernel Optimization**
>
> See [davinci-kernel/](davinci-kernel/) for training implementation and scripts.

On [KernelBench](https://github.com/ScalingIntelligence/KernelBench), **daVinci-kernel-14B achieves 37.2%, 70.6%, and 32.2% on Level 1, 2, and 3** under the Fast₁ threshold, outperforming Dr.Kernel-14B by up to 46% on Level 3.

---

## Repository Structure

```
daVinci-kernel/
├── README.md                  # This file
├── requirements.txt           # KernelGYM dependencies
├── setup.sh                   # KernelGYM installation script
├── start_all_with_monitor.sh  # Single-node KernelGYM launcher
├── start_worker_multinode.sh  # Multi-node worker launcher
├── stop_all.sh                # Stop all KernelGYM services
├── kernelgym/                 # KernelGYM evaluation environment (Python package)
├── davinci-kernel/                     # daVinci-kernel training implementation
│   ├── README.md              # Training documentation
│   ├── setup.sh               # Training environment setup
│   ├── verl/                  # VERL framework (submodule)
│   ├── verl_patch/            # Custom VERL patches
│   └── kernel/                # Core training code
│       ├── main_kernel.py     # RL training entry point
│       ├── main_grading.py    # Evaluation entry point
│       ├── kernel_trainer.py  # daVinci-kernel trainer implementation
│       ├── config/            # Training configurations
│       ├── scripts/           # Training, eval, and SFT scripts
│       ├── skill/             # Skill library implementation
│       ├── rewards/           # Reward functions
│       ├── workers/           # Rollout workers (vLLM-based)
│       └── metrics/           # Evaluation metrics
└── scripts/                   # SFT data construction pipeline
    ├── multi_turn_kernel_sampling.py   # Cold-start trajectory collection
    ├── skill_generator.py             # Phase 1: Skill summarization from seed trajectories
    ├── generate_skill_data.py         # Two-phase skill data generation
    ├── run_skill_inference.py         # Skill inference (with/without BM25 retrieval)
    ├── selected_skill_data.py         # Selection SFT data construction
    ├── extract_skills.py              # Extract accepted skills into library
    ├── auto_configure.sh              # KernelGYM auto-configuration
    └── skill_data/
        ├── generate_skill_injected_sft.py  # Skill-injected policy SFT data
        ├── merge_and_shuffle.py            # Merge and shuffle datasets
        └── extract_type_samples.py         # Sample by type
```

---

## Overview

Training LLMs to write optimized GPU kernels presents a moving capability frontier: knowledge that helps early in training becomes internalized as the model improves, while more nuanced optimizations emerge as new bottlenecks. daVinci-kernel addresses this by treating skill maintenance as part of the learning problem itself.

### Three Co-Evolving Agents

All three agents share a single LLM backbone and are jointly trained end-to-end with reinforcement learning:

| Agent | Role |
|-------|------|
| **Skill Selection Agent** | Retrieves task-relevant skills from the library via BM25 recall (top-20) followed by LLM reranking (top-3) |
| **Policy Agent** | Generates multi-turn CUDA/Triton kernels, optionally conditioned on injected skill context |
| **Skill Summary Agent** | Distills successful rollouts into reusable skills via tool-call interface; new skills are accepted only after execution-based verification |

### Skill Verification

A new skill is accepted into the library only if it satisfies:
- Speedup vs. original rollout ≥ 1.2×
- Absolute speedup ≥ 1.2× (no weak-skill pollution)

### Training Algorithm

daVinci-kernel builds on [Dr.Kernel](https://arxiv.org/abs/2602.05885)'s **TRLOO** (Turn-level REINFORCE Leave-One-Out) training recipe:
- Each task produces k+1=4 parallel skill schemes (k=3 Selection outputs + 1 null scheme)
- Each scheme generates n=4 policy rollouts for advantage normalization
- Per-agent loss weights: Selection = 0.3, Summary = 0.5
- Framework: [VERL](https://github.com/volcengine/verl) distributed RL

---

## Step 0: Start KernelGYM (Required for All Training & Evaluation)

KernelGYM is a GPU-distributed evaluation environment that handles kernel compilation, correctness checking, and performance measurement. All training and evaluation steps require a running KernelGYM server.

### Installation

```bash
# Clone repository (with submodules)
git clone --recursive https://github.com/GAIR-NLP/daVinci-kernel.git
cd daVinci-kernel

# Install KernelGYM dependencies
bash setup.sh
```

### Start KernelGYM

```bash
# Single node: auto-configure and start
bash scripts/auto_configure.sh
./start_all_with_monitor.sh

# Verify
curl http://localhost:10907/health

# Export server URL (needed by all training/eval scripts)
export KERNELGYM_SERVER_URL="http://<your-host>:10907"
```

For multi-node deployment, see [KernelGYM Multi-Node Setup](#multi-node-deployment).

---

## Step 1: SFT Cold Start Data Construction

daVinci-kernel's SFT cold start initializes all three agents. The data pipeline has three phases:

### Phase 0: Collect Multi-Turn Kernel Trajectories

We use the [data](https://huggingface.co/datasets/hkust-nlp/drkernel-coldstart-8k) from Dr.Kernel as the initial codestart data.

### Phase 1: Generate Initial Skill Library

Run the Summary Agent on high-speedup trajectories to extract the initial skills:

```bash
python scripts/skill_generator.py \
    --input data/cold_start_trajectories.jsonl \
    --output data/skill/generated_skills.jsonl \
    --speedup_threshold 1.5 \
    --api_base <openai_compatible_api> \
    --model <summarization_model>

# Extract accepted skills into skill_library.jsonl
python scripts/extract_skills.py \
    --input data/skill/generated_skills.jsonl \
    --output data/skill/skill_library.jsonl
```

### Phase 2: Generate Summarization & Selection SFT Data

```bash
# Two-phase skill data generation (BM25 retrieval + skill injection)
python scripts/generate_skill_data.py \
    --input data/cold_start_trajectories.jsonl \
    --skill_library data/skill/skill_library.jsonl \
    --output_dir data/skill/ \
    --api_base <api> --model <model>

# Selection SFT data (BM25 top-20 → LLM top-3 tool calls)
python scripts/selected_skill_data.py \
    --input data/cold_start_trajectories.jsonl \
    --skill_library data/skill/skill_library.jsonl \
    --output data/skill/selected_skill_data.jsonl

# Skill-injected policy SFT data
python scripts/skill_data/generate_skill_injected_sft.py \
    --cold_start hkust-nlp/drkernel-coldstart-8k \
    --skill_library data/skill/skill_library.jsonl \
    --output data/skill/skill_injected_policy_sft.parquet

# Merge and shuffle all SFT data
python scripts/skill_data/merge_and_shuffle.py \
    --inputs data/skill/generated_skills_with_retrieval.jsonl \
               data/skill/selected_skill_data.jsonl \
               data/skill/skill_injected_policy_sft.parquet \
    --output data/skill/combined_sft_v2.parquet
```

Pre-built SFT datasets are also available on HuggingFace (see [Datasets](#datasets)).

### Phase 3: Run SFT Cold Start Training

```bash
cd davinci-kernel/kernel/scripts/sft

# 8B model
bash 8b-skill-combined.sh

# 14B model
bash 14b-skill-combined.sh
```

---

## Step 2: daVinci-kernel RL Training

After SFT cold start, run joint RL training of all three agents:

```bash
cd davinci-kernel/kernel/scripts/rl

# daVinci-kernel-8B
bash 8b_trloo_mrs_pr_prs_skill.sh

# daVinci-kernel-14B
bash 14b_trloo_mrs_pr_prs_skill.sh
```

**Key configuration variables** (set at top of each script):

```bash
export KERNELGYM_SERVER_URL="http://<server>:10907"  # KernelGYM server
MODEL_PATH="path/to/sft_checkpoint"                  # SFT warm-start model
export HDFS_CHECKPOINT_PATH="/path/to/save/ckpts"    # Checkpoint output dir
TRAIN_DATASET="hkust-nlp/drkernel-rl-data"           # RL training data, we use drkernel's original data
VALID_DATASET="hkust-nlp/drkernel-validation-data"           # Validation data
```

See [davinci-kernel/README.md](davinci-kernel/README.md) for complete training documentation.

---

## Step 3: Evaluation

```bash
cd davinci-kernel/kernel/scripts/eval

# Evaluate daVinci-kernel-14B (3 turns)
bash drkernel-14b-maxturns3-skill.sh

# Evaluate with sequential test-time scaling (5 turns × 10 iterations)
bash drkernel-14b-maxturns5-maxiter10-skill.sh

# Evaluate Dr.Kernel baseline (no skill)
bash drkernel-14b-maxturns3.sh

# Evaluate with OpenAI-compatible APIs (e.g., Claude, GPT)
bash claude-4.5-sonnet-level2.sh
```


---

## Models

We release four checkpoints — the SFT cold-start and final RL models at both 8B and 14B scales:

| Model | HuggingFace | Description |
|-------|-------------|-------------|
| `GAIR-NLP/daVinci-kernel-14B-RL`  | [link](https://huggingface.co/GAIR-NLP/daVinci-kernel-14B-RL)  | daVinci-kernel-14B final RL model (Qwen3-14B-Base) |
| `GAIR-NLP/daVinci-kernel-14B-SFT` | [link](https://huggingface.co/GAIR-NLP/daVinci-kernel-14B-SFT) | daVinci-kernel-14B SFT cold-start model (Qwen3-14B-Base) |
| `GAIR-NLP/daVinci-kernel-8B-RL`   | [link](https://huggingface.co/GAIR-NLP/daVinci-kernel-8B-RL)   | daVinci-kernel-8B final RL model (Qwen3-8B-Base) |
| `GAIR-NLP/daVinci-kernel-8B-SFT`  | [link](https://huggingface.co/GAIR-NLP/daVinci-kernel-8B-SFT)  | daVinci-kernel-8B SFT cold-start model (Qwen3-8B-Base) |

The RL checkpoints (`*-RL`) ship with the co-evolved skill library bundled as `skill_library.jsonl` inside the model directory. If you use our RL models for evaluation, remember to load this skill library from the checkpoint.

### Datasets

The SFT cold-start data used to train these models is also released in [this](https://huggingface.co/datasets/GAIR/daVinci-kernel-sft/tree/main) link:

| Dataset | Description |
|---------|-------------|
| `daVinci-kernel-sft-w-downsampling.parquet`  | SFT cold-start data (with downsampling) |
| `daVinci-kernel-sft-wo-downsampling.parquet` | SFT cold-start data (without downsampling) |

---

## Multi-Node Deployment

For training across multiple nodes, each node needs a KernelGYM worker pointing to a shared Redis server:

**Main node** (Redis + API server + worker monitor):
```bash
redis-server --bind 0.0.0.0
python -m kernelgym.server.api.server

# Start the worker monitor — required to track worker liveness across nodes
python -m kernelgym.worker.worker_monitor --persistent

# Create .env pointing to main node
cat > .env << EOF
REDIS_HOST=<main_node_ip>
REDIS_PORT=6379
API_HOST=<main_node_ip>
API_PORT=10907
GPU_DEVICES=[0,1,2,3,4,5,6,7]
EOF
```

**Worker nodes**:
```bash
./start_worker_multinode.sh
```

---

## Citation

If you use daVinci-kernel in your research, please cite:

```bibtex
@misc{fu2026davincikernelcoevolvingskillselection,
      title={daVinci-kernel: Co-Evolving Skill Selection, Summarization, and Utilization via RL for GPU Kernel Optimization}, 
      author={Dayuan Fu and Mohan Jiang and Tongyu Wang and Dian Yang and Jiarui Hu and Liming Liu and Jinlong Hou and Pengfei Li},
      year={2026},
      eprint={2606.16497},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.16497}, 
}
```


## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Acknowledgements

This repo builds on [VERL](https://github.com/verl-project/verl), [KernelBench](https://github.com/ScalingIntelligence/KernelBench), and [Dr.Kernel](https://arxiv.org/abs/2602.05885).
