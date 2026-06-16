# daVinci-kernel Training

This directory contains the training recipe and evaluation scripts for **daVinci-kernel**, building on top of [Dr.Kernel](https://arxiv.org/abs/2602.05885)'s RL framework with a co-evolving skill library.


daVinci-kernel-14B achieves 37.2%, 70.6%, and 32.2% on KernelBench Level 1/2/3 under Fast₁, outperforming Dr.Kernel-14B by up to 46% on Level 3.

## Overview

daVinci-kernel extends the Dr.Kernel RL framework with three jointly-trained agents:

- **Skill Selection Agent**: BM25 recall (top-20) + LLM reranking (top-3) to retrieve task-relevant skills
- **Policy Agent**: Multi-turn CUDA/Triton kernel generation conditioned on injected skills
- **Skill Summary Agent**: Distills successful rollouts into reusable skills via `update_skill_library` tool call

All three agents share one LLM backbone and are trained end-to-end with **TRLOO** (Turn-level REINFORCE Leave-One-Out) from the Dr.Kernel paper.

The training pipeline builds on the [VERL](https://github.com/volcengine/verl) framework for distributed RL.

## Prerequisites

### 1. Install Dependencies

```bash
cd davinci-kernel
bash setup.sh
```

This will:
- Initialize the VERL submodule
- Install VERL and its dependencies
- Install additional packages (vLLM, flash-attention, etc.)

### 2. Start KernelGYM

All training and evaluation requires a running KernelGYM server:

```bash
# From the daVinci-kernel root directory
cd ..
./start_all_with_monitor.sh

export KERNELGYM_SERVER_URL="http://<your-host>:10907"
```

### 3. Checklist Before Running

- `KERNELGYM_SERVER_URL` points to a healthy KernelGYM service
- `MODEL_PATH` is set to a valid SFT checkpoint or HF model ID
- `HDFS_CHECKPOINT_PATH` is set to a writable checkpoint directory
- `TRAIN_DATASET` / `VALID_DATASET` point to valid parquet files or HF datasets
- For non-8-GPU nodes, adjust `NNODES` / `GPUS_PER_NODE` in scripts

## Directory Structure

```
davinci-kernel/
├── kernel/
│   ├── main_kernel.py              # RL training entry point
│   ├── main_grading.py             # Evaluation entry point
│   ├── kernel_trainer.py           # daVinci-kernel trainer (TRLOO + skill co-evolution)
│   ├── fsdp_sft_trainer.py         # FSDP SFT trainer for cold start
│   ├── constant.py                 # Global constants
│   ├── config/
│   │   ├── kernel_trainer.yaml     # Dr.Kernel RL config (no-skill baseline)
│   │   ├── skill_memory.yaml       # daVinci-kernel skill config (all hyperparameters)
│   │   ├── kernel_grading.yaml     # Evaluation config
│   │   ├── skill_prompts.yaml      # Agent system prompts
│   │   └── prompt_config/
│   │       └── multi_turn_kernel.yaml
│   ├── scripts/
│   │   ├── rl/                     # RL training scripts
│   │   │   ├── train_rl_common.sh          # Shared training logic
│   │   │   ├── 14b_trloo_mrs_pr_prs.sh     # Dr.Kernel-14B (no skill)
│   │   │   ├── 14b_trloo_mrs_pr_prs_skill.sh  # daVinci-kernel-14B
│   │   │   ├── 8b_trloo_mrs_pr_prs.sh      # Dr.Kernel-8B (no skill)
│   │   │   └── 8b_trloo_mrs_pr_prs_skill.sh   # daVinci-kernel-8B
│   │   ├── eval/                   # Evaluation scripts
│   │   │   ├── grading_common.sh           # Shared evaluation logic
│   │   │   ├── drkernel-14b-maxturns3.sh   # Dr.Kernel-14B baseline
│   │   │   ├── drkernel-14b-maxturns3-skill.sh  # daVinci-kernel-14B
│   │   │   ├── drkernel-14b-maxturns5-maxiter10.sh  # Sequential scaling
│   │   │   ├── drkernel-14b-maxturns5-maxiter10-skill.sh
│   │   │   ├── claude-4.5-sonnet-level2.sh        # API model eval
│   │   │   └── claude-4.5-sonnet-level2-compile.sh
│   │   ├── sft/                    # SFT cold-start scripts
│   │   │   ├── 8b-coldstart.sh             # Dr.Kernel 8B cold start
│   │   │   ├── 14b-coldstart.sh            # Dr.Kernel 14B cold start
│   │   │   ├── 8b-skill-combined.sh        # daVinci-kernel 8B combined SFT
│   │   │   └── 14b-skill-combined.sh       # daVinci-kernel 14B combined SFT
│   │   └── preprocess/             # Dataset push/pull utilities
│   ├── skill/                      # Skill library implementation
│   │   ├── skill_library.py        # Library management (JSONL + BM25 index)
│   │   ├── skill_adv.py            # Skill advantage estimation
│   │   ├── skill_prompt_builder.py # Prompt construction for agents
│   │   ├── skill_summary_env.py    # Summary agent environment
│   │   └── skill_data_flow.md      # Data flow documentation
│   ├── rewards/
│   │   ├── kernel_reward.py        # Reward function
│   │   ├── reward_client.py        # KernelGYM reward client
│   │   └── coverage_helper.py      # Profiling coverage utilities
│   ├── workers/
│   │   ├── rollout/
│   │   │   ├── async_server.py                    # Rollout async server
│   │   │   └── vllm_rollout/
│   │   │       ├── vllm_async_engine.py            # Standard vLLM engine
│   │   │       ├── vllm_async_engine_multi_iter.py # Multi-iteration engine
│   │   │       ├── vllm_async_engine_skill.py      # Skill-aware engine (daVinci-kernel)
│   │   │       └── openai_async_engine_multi_iter.py
│   │   ├── agent/
│   │   │   └── kernel_agent.py     # Evaluation agent loop
│   │   └── reward_manager/
│   │       └── kernel_async.py     # Async reward manager
│   ├── metrics/
│   │   ├── kernel_multi_turn_metrics.py
│   │   └── mismatch_quality_metrics.py
│   └── trainer/
│       └── ppo/
│           └── core_algos.py       # PPO/TRLOO core algorithms
├── verl/                           # VERL framework (submodule)
├── verl_patch/                     # Custom VERL patches
└── setup.sh                        # Installation script
```

## Training

### SFT Cold Start

daVinci-kernel requires a structured SFT warm-up that initializes all three agents:

```bash
cd kernel/scripts/sft

# daVinci-kernel 8B (policy + selection + summary SFT data)
bash 8b-skill-combined.sh

# daVinci-kernel 14B
bash 14b-skill-combined.sh
```

Configure in the script:
```bash
MODEL_PATH="Qwen3-14B-Base"           # or local path
export HDFS_CHECKPOINT_PATH="/path/to/ckpts"
TRAIN_DATA_PATH="" # or local path to combined_sft_v2.parquet
```

For Dr.Kernel baseline (no skill) cold start:
```bash
bash 8b-coldstart.sh   # or 14b-coldstart.sh
# Uses: hkust-nlp/drkernel-coldstart-8k
```

### daVinci-kernel RL Training

Full daVinci-kernel training with skill co-evolution:

```bash
cd kernel/scripts/rl

# daVinci-kernel-14B: TRLOO + MRS + PR + PRS + Skill
bash 14b_trloo_mrs_pr_prs_skill.sh

# daVinci-kernel-8B
bash 8b_trloo_mrs_pr_prs_skill.sh
```

For the Dr.Kernel baseline (no skill):
```bash
bash 14b_trloo_mrs_pr_prs.sh
bash 8b_trloo_mrs_pr_prs.sh
```

**Required configuration** (edit at top of script):
```bash
export KERNELGYM_SERVER_URL="http://<server>:10907"
MODEL_PATH="path/to/sft_checkpoint"          # SFT warm-start checkpoint
export HDFS_CHECKPOINT_PATH="/path/to/ckpts" # Where to save checkpoints
TRAIN_DATASET="hkust-nlp/drkernel-rl-data"
VALID_DATASET="hkust-nlp/drkernel-validation-data"
```

**Key daVinci-kernel skill parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SKILL_ENABLE` | `True` | Enable skill co-evolution |
| `SKILL_K` | `3` | Number of selection inferences per task |
| `SKILL_TOP_BM25` | `20` | BM25 recall count |
| `SKILL_TOP_K_SELECT` | `3` | Skills LLM picks from BM25 candidates |
| `SKILL_SELECTION_WEIGHT` | `0.3` | Selection agent loss weight |
| `SKILL_SUMMARY_WEIGHT` | `0.5` | Summary agent loss weight |
| `SKILL_VERIFY_MIN_ABSOLUTE_SPEEDUP` | `1.2` | Minimum speedup for skill acceptance |
| `SKILL_SAVE_FREQ` | `1` | Skill library flush frequency (steps) |
| `SKILL_LIBRARY_ROOT` | `${HDFS_CHECKPOINT_PATH}/skill_library` | Skill library directory |

**Key training parameters** (shared with Dr.Kernel):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ALGORITHM` | `trloo` | RL algorithm |
| `MAX_TURN` | `3` | Maximum turns per episode |
| `ROLLOUT_N` | `4` | Rollouts per skill scheme (4 schemes × 4 = 16 total) |
| `LEARNING_RATE` | `1e-6` | Learning rate |
| `ROLLOUT_RS` | `geometric` | Rejection sampling strategy |
| `COVERAGE_RS` | `turn` | Coverage-based rejection sampling |

## Evaluation

### Evaluate daVinci-kernel Models

```bash
cd kernel/scripts/eval

# daVinci-kernel-14B, 3-turn evaluation
bash daVinci-kernel-14b-maxturns3-skill.sh

# Sequential test-time scaling (5 turns × 10 iterations)
bash daVinci-kernel-14b-maxturns5-maxiter10-skill.sh

# Dr.Kernel baseline (no skill)
bash daVinci-kernel-14b-maxturns3.sh
```

Configure in the script:
```bash
export KERNELGYM_SERVER_URL="http://<server>:10907"
HF_MODEL_PATH=""    # or local checkpoint path
EVAL_DATASET=""
```

### Evaluate with OpenAI-Compatible APIs

```bash
bash claude-4.5-sonnet-level2.sh
bash claude-4.5-sonnet-level2-compile.sh  # torch.compile reference
```

Configure:
```bash
BACKEND="openai"
OPENAI_MODEL="anthropic/claude-sonnet-4-5"
OPENAI_API_KEY="your-api-key"
OPENAI_BASE_URL="https://api.openai.com/v1"
```

## Key Environment Variables

| Variable | Description |
|----------|-------------|
| `KERNELGYM_SERVER_URL` | KernelGYM evaluation server URL |
| `MODEL_PATH` | Explicit model path or HF model ID |
| `HDFS_CHECKPOINT_PATH` | Checkpoint output directory |
| `HDFS_DATA_PATH` | Base directory for parquet datasets |
| `WANDB_API_KEY` | (Optional) Weights & Biases API key |

## Troubleshooting

**KernelGYM connection failed:**
```bash
curl http://<server>:10907/health
```

**CUDA OOM during training:**
- Reduce `TRAIN_BATCH_SIZE` or `ROLLOUT_N`
- Enable `ACTOR_OPTIMIZER_OFFLOAD=True`
- Reduce `ROLLOUT_GPU_MEMORY_UTIL`

**vLLM errors:**
- Try `ENFORCE_EAGER=True` for debugging
- Check CUDA compatibility with installed vLLM version

## Citation

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
