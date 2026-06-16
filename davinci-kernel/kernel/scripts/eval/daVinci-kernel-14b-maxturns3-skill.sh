#!/bin/bash
# Same as drkernel-14b-maxturns3.sh but with skill injection.
# Produces metrics.json with val_with_skill/ keys directly.
# Extra args: --library_file <path-to-snapshot.jsonl>
#
# Usage:
#   bash drkernel-14b-maxturns3-skill.sh \
#     --run_name my-skill-eval \
#     --eval_dataset .../level1.parquet \
#     --model_path .../global_step_20/actor/huggingface \
#     --library_file .../skill_library/global_step_9.jsonl

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/grading_common.sh"
pkill -f ray::AsyncActorRolloutRefWorker
pkill -f ray::AsyncActorRolloutRefWorker

FSDP_SIZE=-1
PROJECT_NAME="kernel-grading-skill"
RUN_NAME="drkernel-14b-maxturns3-skill"
EXPERIMENT_NAME=${RUN_NAME}

HDFS_RUNS_PATH="${HDFS_RUNS_PATH:-./runs}"
EVAL_DATASET="hkust-nlp/drkernel-validation-data"
HF_MODEL_PATH="hkust-nlp/drkernel-14b"
LIBRARY_FILE=""

# Pre-parse --run_name, --model_path, --eval_dataset, --library_file
_ARGS=("$@")
PASS_ARGS=()
_IDX=0
while [[ $_IDX -lt ${#_ARGS[@]} ]]; do
  case "${_ARGS[$_IDX]}" in
    --run_name)
      RUN_NAME="${_ARGS[$((_IDX+1))]}";   _IDX=$((_IDX+2)) ;;
    --model_path)
      HF_MODEL_PATH="${_ARGS[$((_IDX+1))]}"; PASS_ARGS+=("--model_path" "${_ARGS[$((_IDX+1))]}"); _IDX=$((_IDX+2)) ;;
    --eval_dataset)
      EVAL_DATASET="${_ARGS[$((_IDX+1))]}"; _IDX=$((_IDX+2)) ;;
    --library_file)
      LIBRARY_FILE="${_ARGS[$((_IDX+1))]}"; _IDX=$((_IDX+2)) ;;
    *)
      PASS_ARGS+=("${_ARGS[$_IDX]}"); _IDX=$((_IDX+1)) ;;
  esac
done
unset _ARGS _IDX

if [[ -z "$LIBRARY_FILE" || ! -f "$LIBRARY_FILE" ]]; then
  echo "[ERROR] --library_file required and must be a .jsonl file: '$LIBRARY_FILE'"
  exit 1
fi

MODEL_NAME="${HF_MODEL_PATH}"
MODEL_PATH="${MODEL_NAME}"
EXPERIMENT_NAME=${RUN_NAME}

MULTI_TURN=True
MAX_USER_TURNS=3

GRADIO_VISUALIZATION=False
GRADIO_SHARE=False
VISUALIZE_ONLY=False

MAX_PROMPT_LENGTH=20480
MAX_RESPONSE_LENGTH=8192

OUTPUT_DIR="${HDFS_RUNS_PATH}/${RUN_NAME}/grading_results"
OUTPUT_PATH="${OUTPUT_DIR}/graded_results.parquet"
METRICS_OUTPUT_PATH="${OUTPUT_DIR}/metrics.json"
RAW_RESPONSE_PATH="${OUTPUT_DIR}/raw_responses.jsonl"

N_SAMPLES=8
BATCH_SIZE=128
TEMPERATURE=1.0
TOP_P=0.95
DO_SAMPLE=True

ROLLOUT_MODE="async_vllm"
ROLLOUT_GPU_MEMORY_UTIL=0.5
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=1

SOLVE_THRESHOLD=0.99
PASS_AT_K=1

REWARD_SERVER_URL="${REWARD_SERVER_URL:-${KERNELGYM_SERVER_URL:-""}}"
REWARD_MANAGER="kernel_async"
REWARD_FUNC_NAME="calculate_reward_speedup"
REWARD_WEIGHTS="0.3_0.4_0.3"
REWARD_ENHANCED=True
REWARD_USE_SANDBOX_RATE_LIMIT=True
REWARD_RATE_LIMIT=256
REWARD_ACQUIRE_TIMEOUT=2400
REWARD_MAX_CONCURRENT=256
REWARD_TIMEOUT=2400
REWARD_MAX_RETRIES=3
REWARD_TASK_TIMEOUT=600
REWARD_TASK_TIMEOUT_CLIENT=2400
REWARD_PRINT_STATUS=True
NUM_PERF_TRIALS=10
NUM_CORRECT_TRIALS=5
SPEEDUP_REWARD_UPPER_BOUND=3.0

CUSTOM_REWARD_PATH="kernel/rewards/kernel_reward.py"
CUSTOM_REWARD_NAME="compute_kernel_reward_batch"

NNODES=1
N_GPUS_PER_NODE=8
FIX_QWEN3_CHAT_TEMPLATE=False

# multi_iteration must be True so async_server upgrades to SkillAwareMultiIterAsyncvLLMEngine
MULTI_ITERATION=True
MAX_ITERATIONS=1
REMAIN_TURNS=2

export PROJECT_NAME RUN_NAME EVAL_DATASET OUTPUT_PATH METRICS_OUTPUT_PATH RAW_RESPONSE_PATH
export MODEL_NAME MODEL_PATH
export N_SAMPLES BATCH_SIZE TEMPERATURE TOP_P DO_SAMPLE
export ROLLOUT_MODE ROLLOUT_GPU_MEMORY_UTIL ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE
export SOLVE_THRESHOLD PASS_AT_K
export REWARD_SERVER_URL REWARD_MANAGER REWARD_FUNC_NAME REWARD_WEIGHTS
export REWARD_ENHANCED REWARD_USE_SANDBOX_RATE_LIMIT REWARD_RATE_LIMIT REWARD_ACQUIRE_TIMEOUT
export REWARD_MAX_CONCURRENT REWARD_TIMEOUT REWARD_MAX_RETRIES REWARD_TASK_TIMEOUT
export REWARD_PRINT_STATUS NUM_PERF_TRIALS NUM_CORRECT_TRIALS SPEEDUP_REWARD_UPPER_BOUND
export CUSTOM_REWARD_PATH CUSTOM_REWARD_NAME
export NNODES N_GPUS_PER_NODE FIX_QWEN3_CHAT_TEMPLATE
export MULTI_ITERATION MAX_ITERATIONS REMAIN_TURNS

echo "=========================================="
echo "Kernel Grading with Skill Injection"
echo "  MODEL     : $MODEL_PATH"
echo "  SKILL LIB : $LIBRARY_FILE"
echo "  DATASET   : $EVAL_DATASET"
echo "  OUTPUT    : $OUTPUT_DIR"
echo "=========================================="

LIBRARY_ROOT="$(dirname "$LIBRARY_FILE")"

START_TIME=$(date +%s)
parse_reward_weights "$REWARD_WEIGHTS"
mkdir -p "$OUTPUT_DIR"

# skill.enable=True → main_grading.py 内部自动将 val/ → val_with_skill/ 写入 metrics.json
PYTHONUNBUFFERED=1 python -m kernel.main_grading \
    data.path="$EVAL_DATASET" \
    data.output_path="$OUTPUT_PATH" \
    data.raw_response_path="$RAW_RESPONSE_PATH" \
    data.metrics_output_path="$METRICS_OUTPUT_PATH" \
    data.n_samples=$N_SAMPLES \
    data.batch_size=$BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.solve_threshold=$SOLVE_THRESHOLD \
    data.pass_at_k=$PASS_AT_K \
    data.do_sample=$DO_SAMPLE \
    data.apply_chat_template=True \
    model.path="$MODEL_PATH" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.rollout.mode=$ROLLOUT_MODE \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.top_p=$TOP_P \
    actor_rollout_ref.rollout.val_kwargs.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.val_kwargs.top_p=$TOP_P \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTIL \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1000)) \
    actor_rollout_ref.rollout.multi_turn.enable=$MULTI_TURN \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$MAX_USER_TURNS \
    actor_rollout_ref.rollout.multi_turn.multi_iteration.enable=$MULTI_ITERATION \
    actor_rollout_ref.rollout.multi_turn.multi_iteration.max_iterations=$MAX_ITERATIONS \
    actor_rollout_ref.rollout.multi_turn.multi_iteration.remain_turns=$REMAIN_TURNS \
    actor_rollout_ref.rollout.agent.num_workers=32 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=$FSDP_SIZE \
    reward_model.reward_manager=$REWARD_MANAGER \
    reward_model.server_url='"'"$REWARD_SERVER_URL"'"' \
    reward_model.reward_func_name=$REWARD_FUNC_NAME \
    reward_model.enhanced=$REWARD_ENHANCED \
    reward_model.use_sandbox_rate_limit=$REWARD_USE_SANDBOX_RATE_LIMIT \
    reward_model.rate_limit=$REWARD_RATE_LIMIT \
    reward_model.acquire_timeout=$REWARD_ACQUIRE_TIMEOUT \
    reward_model.max_concurrent=$REWARD_MAX_CONCURRENT \
    reward_model.timeout=$REWARD_TIMEOUT \
    reward_model.max_retries=$REWARD_MAX_RETRIES \
    reward_model.task_timeout=$REWARD_TASK_TIMEOUT \
    reward_model.task_timeout_in_client=$REWARD_TASK_TIMEOUT_CLIENT \
    reward_model.print_status=$REWARD_PRINT_STATUS \
    reward_model.num_perf_trials=$NUM_PERF_TRIALS \
    reward_model.num_correct_trials=$NUM_CORRECT_TRIALS \
    reward_model.speedup_reward_upper_bound=$SPEEDUP_REWARD_UPPER_BOUND \
    reward_model.reward_weights.compilation=$REWARD_WEIGHT_COMPILATION \
    reward_model.reward_weights.correctness=$REWARD_WEIGHT_CORRECTNESS \
    reward_model.reward_weights.performance=$REWARD_WEIGHT_PERFORMANCE \
    custom_reward_function.path=$CUSTOM_REWARD_PATH \
    custom_reward_function.name=$CUSTOM_REWARD_NAME \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.nnodes=$NNODES \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.fix_qwen3_chat_template=$FIX_QWEN3_CHAT_TEMPLATE \
    gradio=False \
    skill.enable=True \
    skill.library_file="$LIBRARY_FILE" \
    "skill.library_root=$LIBRARY_ROOT"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
printf "\n=========================================="
printf "\nTotal time: %02dh %02dm %02ds\n" $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60))
printf "  metrics → %s\n" "$METRICS_OUTPUT_PATH"
printf "==========================================\n"
