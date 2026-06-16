#!/bin/bash
# Standalone grading script for drkernel-14b with maxturns5 / maxiter10 + skill injection.
# Does NOT source grading_common.sh — fully self-contained.
#
# Usage:
#   bash drkernel-14b-maxturns5-maxiter10-skill.sh \
#     --run_name my-skill-eval \
#     --eval_dataset .../level1.parquet \
#     --model_path .../global_step_60/actor/huggingface \
#     --library_file .../skill_library/global_step_60.jsonl

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../../setup_env.sh"

# =============================================================================
# Default Configuration
# =============================================================================

FSDP_SIZE=-1
PROJECT_NAME="kernel-grading-skill"
RUN_NAME="drkernel-14b-maxturns5-maxiter10-skill"

HDFS_RUNS_PATH="${HDFS_RUNS_PATH:-./runs}"
EVAL_DATASET="hkust-nlp/drkernel-validation-data"
HF_MODEL_PATH="hkust-nlp/drkernel-14b"

# Skill
SKILL_ENABLE="True"
SKILL_LIBRARY_FILE=""
SKILL_LIBRARY_ROOT=""

# Multi-turn
MULTI_TURN=True
MAX_USER_TURNS=5

# Multi-iteration
MULTI_ITERATION=True
MAX_ITERATIONS=10
REMAIN_TURNS=4
ITERATION_METHOD="best"
BEST_SELECTION_METRIC="reward"

# Gradio
GRADIO_VISUALIZATION=True
GRADIO_SHARE=True
VISUALIZE_ONLY=False

# Lengths
MAX_PROMPT_LENGTH=20480
MAX_RESPONSE_LENGTH=8192

# Generation
N_SAMPLES=8
BATCH_SIZE=128
TEMPERATURE=1.0
TOP_P=0.95
TOP_K=-1
MIN_P=0.0
DO_SAMPLE=True
APPLY_CHAT_TEMPLATE=True

# Rollout
ROLLOUT_MODE="async_vllm"
ROLLOUT_GPU_MEMORY_UTIL=0.5
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=1
ROLLOUT_ENFORCE_EAGER=False
BACKEND="vllm"

# Evaluation
SOLVE_THRESHOLD=0.99
PASS_AT_K=1

# Reward
REWARD_SERVER_URL="${REWARD_SERVER_URL:-${KERNELGYM_SERVER_URL:-""}}"
REWARD_MANAGER="kernel_async"
REWARD_FUNC_NAME="calculate_reward_speedup"
REWARD_WEIGHTS="0.3_0.4_0.3"
REWARD_ENHANCED=True
REWARD_USE_SANDBOX_RATE_LIMIT=True
REWARD_RATE_LIMIT=128
REWARD_ACQUIRE_TIMEOUT=2400
REWARD_MAX_CONCURRENT=128
REWARD_TIMEOUT=2400
REWARD_MAX_RETRIES=3
REWARD_TASK_TIMEOUT=1200
REWARD_TASK_TIMEOUT_CLIENT=2400
REWARD_PRINT_STATUS=True
NUM_PERF_TRIALS=10
NUM_CORRECT_TRIALS=5
SPEEDUP_REWARD_UPPER_BOUND=3.0
REWARD_PENALTY_SCORE=0.0
REWARD_PENALTY_COMPILATION=-0.5
REWARD_PENALTY_CORRECTNESS=-0.3
REWARD_PENALTY_PERF_DEGRADE=-0.1

CUSTOM_REWARD_PATH="kernel/rewards/kernel_reward.py"
CUSTOM_REWARD_NAME="compute_kernel_reward_batch"

NNODES=1
N_GPUS_PER_NODE=8
FIX_QWEN3_CHAT_TEMPLATE=False
MAX_NUM_WORKERS=32
REFERENCE_BACKEND="pytorch"

# OpenAI (unused by default but accepted as args)
OPENAI_MODEL=""
OPENAI_THINKING_MODE=False
OPENAI_API_KEY=""
OPENAI_BASE_URL=""
OPENAI_TIMEOUT=120
OPENAI_MAX_RETRIES=3
OPENAI_MAX_CONCURRENCY=256

# DataProto / extra output paths
DATAPROTO_PATH=""

# =============================================================================
# Argument Parsing
# =============================================================================

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --help|-h)
      echo "Usage: $0 [OPTIONS]"
      echo "  --run_name NAME               Experiment run name"
      echo "  --eval_dataset PATH           Input dataset (parquet)"
      echo "  --model_path PATH             Model checkpoint path"
      echo "  --library_file PATH           Skill library snapshot (.jsonl)  [required for skill]"
      echo "  --skill_enable BOOL           Enable skill injection (default: True)"
      echo "  --skill_library_root PATH     Root dir for skill library (auto-derived when omitted)"
      echo "  --max_user_turns N            Max multi-turn user turns (default: 5)"
      echo "  --max_iterations N            Max multi-iteration count (default: 10)"
      echo "  --remain_turns N              Remain turns for iteration (default: 4)"
      echo "  --n_samples N                 Samples per prompt (default: 8)"
      echo "  --batch_size N                Batch size (default: 128)"
      echo "  --temperature T               Sampling temperature (default: 1.0)"
      echo "  --top_p V                     Top-p sampling (default: 0.95)"
      echo "  --rollout_mode MODE           async_vllm|sync|standalone_vllm (default: async_vllm)"
      echo "  --reward_server_url URL       Kernel reward server URL"
      echo "  --nnodes N                    Number of nodes (default: 1)"
      echo "  --n_gpus_per_node N           GPUs per node (default: 8)"
      exit 0
      ;;
    --run_name)             RUN_NAME="$2";                        shift 2 ;;
    --eval_dataset)         EVAL_DATASET="$2";                    shift 2 ;;
    --model_path)           HF_MODEL_PATH="$2";                   shift 2 ;;
    --library_file)         SKILL_LIBRARY_FILE="$2";              shift 2 ;;
    --skill_enable)         SKILL_ENABLE="$2";                    shift 2 ;;
    --skill_library_root)   SKILL_LIBRARY_ROOT="$2";              shift 2 ;;
    --max_user_turns)       MAX_USER_TURNS="$2";                  shift 2 ;;
    --max_iterations)       MAX_ITERATIONS="$2";                  shift 2 ;;
    --remain_turns)         REMAIN_TURNS="$2";                    shift 2 ;;
    --iteration_method)     ITERATION_METHOD="$2";                shift 2 ;;
    --best_selection_metric) BEST_SELECTION_METRIC="$2";          shift 2 ;;
    --n_samples)            N_SAMPLES="$2";                       shift 2 ;;
    --batch_size)           BATCH_SIZE="$2";                      shift 2 ;;
    --temperature)          TEMPERATURE="$2";                     shift 2 ;;
    --top_p)                TOP_P="$2";                           shift 2 ;;
    --top_k)                TOP_K="$2";                           shift 2 ;;
    --min_p)                MIN_P="$2";                           shift 2 ;;
    --do_sample)            DO_SAMPLE="$2";                       shift 2 ;;
    --apply_chat_template)  APPLY_CHAT_TEMPLATE="$2";             shift 2 ;;
    --max_prompt_length)    MAX_PROMPT_LENGTH="$2";               shift 2 ;;
    --max_response_length)  MAX_RESPONSE_LENGTH="$2";             shift 2 ;;
    --rollout_mode)         ROLLOUT_MODE="$2";                    shift 2 ;;
    --rollout_tp)           ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE="$2"; shift 2 ;;
    --rollout_gpu_memory_util) ROLLOUT_GPU_MEMORY_UTIL="$2";      shift 2 ;;
    --rollout_enforce_eager) ROLLOUT_ENFORCE_EAGER="$2";          shift 2 ;;
    --backend)              BACKEND="$2";                         shift 2 ;;
    --solve_threshold)      SOLVE_THRESHOLD="$2";                 shift 2 ;;
    --pass_at_k)            PASS_AT_K="$2";                       shift 2 ;;
    --reward_server_url)    REWARD_SERVER_URL="$2";               shift 2 ;;
    --reward_manager)       REWARD_MANAGER="$2";                  shift 2 ;;
    --reward_func_name)     REWARD_FUNC_NAME="$2";                shift 2 ;;
    --reward_weights)       REWARD_WEIGHTS="$2";                  shift 2 ;;
    --reward_rate_limit)    REWARD_RATE_LIMIT="$2";               shift 2 ;;
    --reward_max_concurrent) REWARD_MAX_CONCURRENT="$2";          shift 2 ;;
    --reward_timeout)       REWARD_TIMEOUT="$2";                  shift 2 ;;
    --reward_task_timeout)  REWARD_TASK_TIMEOUT="$2";             shift 2 ;;
    --num_perf_trials)      NUM_PERF_TRIALS="$2";                 shift 2 ;;
    --num_correct_trials)   NUM_CORRECT_TRIALS="$2";              shift 2 ;;
    --speedup_reward_upper_bound) SPEEDUP_REWARD_UPPER_BOUND="$2"; shift 2 ;;
    --nnodes)               NNODES="$2";                          shift 2 ;;
    --n_gpus_per_node)      N_GPUS_PER_NODE="$2";                 shift 2 ;;
    --fix_qwen3_chat_template) FIX_QWEN3_CHAT_TEMPLATE="$2";     shift 2 ;;
    --gradio_visualization) GRADIO_VISUALIZATION="$2";            shift 2 ;;
    --gradio_share)         GRADIO_SHARE="$2";                    shift 2 ;;
    --visualize_only)       VISUALIZE_ONLY="$2";                  shift 2 ;;
    --project_name)         PROJECT_NAME="$2";                    shift 2 ;;
    --dataproto_path)       DATAPROTO_PATH="$2";                  shift 2 ;;
    --openai_model)         OPENAI_MODEL="$2";                    shift 2 ;;
    --openai_api_key)       OPENAI_API_KEY="$2";                  shift 2 ;;
    --openai_base_url)      OPENAI_BASE_URL="$2";                 shift 2 ;;
    *)
      echo "[ERROR] Unknown option: $1"
      echo "Use --help for usage information."
      exit 1
      ;;
  esac
done

# =============================================================================
# Derived Paths
# =============================================================================

MODEL_NAME="${HF_MODEL_PATH}"
MODEL_PATH="${HF_MODEL_PATH}"
EXPERIMENT_NAME="${RUN_NAME}"

OUTPUT_DIR="${HDFS_RUNS_PATH}/${RUN_NAME}/grading_results"
OUTPUT_PATH="${OUTPUT_DIR}/graded_results.parquet"
METRICS_OUTPUT_PATH="${OUTPUT_DIR}/metrics.json"
RAW_RESPONSE_PATH="${OUTPUT_DIR}/raw_responses.jsonl"
mkdir -p "${OUTPUT_DIR}"

MAX_NUM_BATCHED_TOKENS=$(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1000 ))

# Reward weights
IFS='_' read -r REWARD_WEIGHT_COMPILATION REWARD_WEIGHT_CORRECTNESS REWARD_WEIGHT_PERFORMANCE <<< "${REWARD_WEIGHTS}"

# =============================================================================
# Validation
# =============================================================================

if [[ "${SKILL_ENABLE}" == "True" || "${SKILL_ENABLE}" == "true" ]]; then
  if [[ -z "${SKILL_LIBRARY_FILE}" || ! -f "${SKILL_LIBRARY_FILE}" ]]; then
    echo "[ERROR] SKILL_ENABLE=True but --library_file not set or file not found: '${SKILL_LIBRARY_FILE}'"
    exit 1
  fi
fi

# =============================================================================
# Summary
# =============================================================================

echo "=========================================="
echo "Kernel Grading with Skill Injection (maxturns5 maxiter10)"
echo "  RUN_NAME       : ${RUN_NAME}"
echo "  MODEL_PATH     : ${MODEL_PATH}"
echo "  EVAL_DATASET   : ${EVAL_DATASET}"
echo "  OUTPUT_DIR     : ${OUTPUT_DIR}"
echo "  SKILL_ENABLE   : ${SKILL_ENABLE}"
echo "  LIBRARY_FILE   : ${SKILL_LIBRARY_FILE:-<not set>}"
echo "  MAX_USER_TURNS : ${MAX_USER_TURNS}"
echo "  MAX_ITERATIONS : ${MAX_ITERATIONS}"
echo "=========================================="

# =============================================================================
# Build Skill Args
# =============================================================================

SKILL_ARGS=""
if [[ "${SKILL_ENABLE}" == "True" || "${SKILL_ENABLE}" == "true" ]]; then
  _SKILL_ROOT="${SKILL_LIBRARY_ROOT:-$(dirname "${SKILL_LIBRARY_FILE}")}"
  SKILL_ARGS="skill.enable=True skill.library_file=${SKILL_LIBRARY_FILE} skill.library_root=${_SKILL_ROOT}"
fi

# =============================================================================
# Run Grading
# =============================================================================

sleep 1

PYTHONUNBUFFERED=1 python -m kernel.main_grading \
    data.path="${EVAL_DATASET}" \
    data.output_path="${OUTPUT_PATH}" \
    data.raw_response_path="${RAW_RESPONSE_PATH}" \
    ${DATAPROTO_PATH:+data.dataproto_path="${DATAPROTO_PATH}"} \
    data.metrics_output_path="${METRICS_OUTPUT_PATH}" \
    data.n_samples="${N_SAMPLES}" \
    data.batch_size="${BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.solve_threshold="${SOLVE_THRESHOLD}" \
    data.pass_at_k="${PASS_AT_K}" \
    data.do_sample="${DO_SAMPLE}" \
    data.apply_chat_template="${APPLY_CHAT_TEMPLATE}" \
    model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.rollout.mode="${ROLLOUT_MODE}" \
    actor_rollout_ref.rollout.temperature="${TEMPERATURE}" \
    actor_rollout_ref.rollout.top_p="${TOP_P}" \
    actor_rollout_ref.rollout.top_k="${TOP_K}" \
    actor_rollout_ref.rollout.min_p="${MIN_P}" \
    actor_rollout_ref.rollout.val_kwargs.temperature="${TEMPERATURE}" \
    actor_rollout_ref.rollout.val_kwargs.top_p="${TOP_P}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTIL}" \
    actor_rollout_ref.rollout.enforce_eager="${ROLLOUT_ENFORCE_EAGER}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS}" \
    actor_rollout_ref.rollout.multi_turn.enable="${MULTI_TURN}" \
    actor_rollout_ref.rollout.multi_turn.max_user_turns="${MAX_USER_TURNS}" \
    actor_rollout_ref.rollout.multi_turn.multi_iteration.enable="${MULTI_ITERATION}" \
    actor_rollout_ref.rollout.multi_turn.multi_iteration.max_iterations="${MAX_ITERATIONS}" \
    actor_rollout_ref.rollout.multi_turn.multi_iteration.remain_turns="${REMAIN_TURNS}" \
    actor_rollout_ref.rollout.multi_turn.multi_iteration.iteration_method="${ITERATION_METHOD}" \
    actor_rollout_ref.rollout.multi_turn.multi_iteration.best_selection_metric="${BEST_SELECTION_METRIC}" \
    actor_rollout_ref.actor.fsdp_config.fsdp_size="${FSDP_SIZE}" \
    actor_rollout_ref.rollout.backend="${BACKEND}" \
    actor_rollout_ref.rollout.openai.model="${OPENAI_MODEL}" \
    actor_rollout_ref.rollout.openai.thinking_mode="${OPENAI_THINKING_MODE}" \
    actor_rollout_ref.rollout.openai.api_key="${OPENAI_API_KEY}" \
    actor_rollout_ref.rollout.openai.base_url="${OPENAI_BASE_URL}" \
    actor_rollout_ref.rollout.openai.timeout="${OPENAI_TIMEOUT}" \
    actor_rollout_ref.rollout.openai.max_retries="${OPENAI_MAX_RETRIES}" \
    actor_rollout_ref.rollout.openai.max_concurrency="${OPENAI_MAX_CONCURRENCY}" \
    actor_rollout_ref.rollout.agent.num_workers="${MAX_NUM_WORKERS}" \
    actor_rollout_ref.rollout.free_cache_engine=true \
    reward_model.reward_manager="${REWARD_MANAGER}" \
    reward_model.reference_backend="${REFERENCE_BACKEND}" \
    "reward_model.server_url=${REWARD_SERVER_URL}" \
    reward_model.reward_func_name="${REWARD_FUNC_NAME}" \
    reward_model.enhanced="${REWARD_ENHANCED}" \
    reward_model.use_sandbox_rate_limit="${REWARD_USE_SANDBOX_RATE_LIMIT}" \
    reward_model.rate_limit="${REWARD_RATE_LIMIT}" \
    reward_model.acquire_timeout="${REWARD_ACQUIRE_TIMEOUT}" \
    reward_model.max_concurrent="${REWARD_MAX_CONCURRENT}" \
    reward_model.timeout="${REWARD_TIMEOUT}" \
    reward_model.max_retries="${REWARD_MAX_RETRIES}" \
    reward_model.task_timeout="${REWARD_TASK_TIMEOUT}" \
    reward_model.task_timeout_in_client="${REWARD_TASK_TIMEOUT_CLIENT}" \
    reward_model.print_status="${REWARD_PRINT_STATUS}" \
    reward_model.num_perf_trials="${NUM_PERF_TRIALS}" \
    reward_model.num_correct_trials="${NUM_CORRECT_TRIALS}" \
    reward_model.speedup_reward_upper_bound="${SPEEDUP_REWARD_UPPER_BOUND}" \
    reward_model.reward_weights.compilation="${REWARD_WEIGHT_COMPILATION}" \
    reward_model.reward_weights.correctness="${REWARD_WEIGHT_CORRECTNESS}" \
    reward_model.reward_weights.performance="${REWARD_WEIGHT_PERFORMANCE}" \
    reward_model.reward_policy.penalties.penalty_score="${REWARD_PENALTY_SCORE}" \
    reward_model.reward_policy.penalties.compilation_fail="${REWARD_PENALTY_COMPILATION}" \
    reward_model.reward_policy.penalties.correctness_fail="${REWARD_PENALTY_CORRECTNESS}" \
    reward_model.reward_policy.penalties.perf_degrade="${REWARD_PENALTY_PERF_DEGRADE}" \
    custom_reward_function.path="${CUSTOM_REWARD_PATH}" \
    custom_reward_function.name="${CUSTOM_REWARD_NAME}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.fix_qwen3_chat_template="${FIX_QWEN3_CHAT_TEMPLATE}" \
    gradio="${GRADIO_VISUALIZATION}" \
    gradio_share="${GRADIO_SHARE}" \
    visualize_only="${VISUALIZE_ONLY}" \
    ${SKILL_ARGS}
