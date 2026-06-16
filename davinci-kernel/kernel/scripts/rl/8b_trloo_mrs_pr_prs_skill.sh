#!/bin/bash
# Skill Memory RL training — 8B TRLOO + MRS + PRS
# Same hyperparameters as 8b_trloo_mrs_pr_prs.sh, adds skill.* config
cd "$(dirname "$0")/../../../../"
pkill -f ray
ray stop
sleep 5
ray stop
export PYTHONPATH=$(pwd):$PYTHONPATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd "$(dirname "$0")/../../../../"

conda activate davinci-kernel_env

# 4499000


TRAIN_DATASET=("hkust-nlp/drkernel-rl-data")
VALID_DATASET=("hkust-nlp/drkernel-validation-data")
KERNELGYM_SERVER_URL="${KERNELGYM_SERVER_URL:-"http://10.246.235.138:10907"}"
if [ -z "$KERNELGYM_SERVER_URL" ]; then
    echo "[ERROR] KERNELGYM_SERVER_URL is not set. Please export it before running this script."
    echo "  e.g.: export KERNELGYM_SERVER_URL=http://<host>:<port>"
    exit 1
fi
echo "[INFO] KERNELGYM_SERVER_URL=${KERNELGYM_SERVER_URL}"
MODEL_NAME=hkust-nlp/drkernel-8b
# export HDFS_CHECKPOINT_PATH=/path/to/checkpoints  # set your own checkpoint path
# export HDFS_CHECKPOINT_PATH=/path/to/checkpoints  # set your own checkpoint path

# RUN_NAME="drkernel-8b-skill-debug"
REWARD_MANAGER=kernel_async
REWARD_FUNC_NAME="calculate_reward_speedup"

ALGORITHM="trloo"

SPEEDUP_REWARD_UPPER_BOUND=6.0
SPEEDUP_REWARD_LOWER_BOUND=0.0

ROLLOUT_RS="geometric"
ROLLOUT_TOKEN_VETO_THRESHOLD=1e-4
ROLLOUT_RS_KWARGS="{lower:0.999,upper:1.001}"

COVERAGE_RS="turn"
COVERAGE_RS_THRESHOLD=0.3
COVERAGE_RS_FACTOR=0.1
COVERAGE_RS_KEY="time_coverage"

COVERAGE_REWARD_TYPE="time_coverage"
COVERAGE_REWARD_WEIGHT=0.5
COVERAGE_REWARD_ENABLE=True

REWARD_TASK_TIMEOUT=300
REWARD_TIMEOUT=1800
REWARD_ACQUIRE_TIMEOUT=2400
REWARD_MAX_CONCURRENT=128
REWARD_MAX_RETRIES=3
REWARD_PRINT_STATUS=True
NUM_PERF_TRIALS=100
REWARD_TASK_TIMEOUT_CLIENT=2400

VAL_BEFORE_TRAIN=True
IS_GET_LAST_TURN=True

ENABLE_MULTI_TURN=True
MAX_TURN=3
N_VAL=8
ACTOR_OPTIMIZER_OFFLOAD=True
ACTOR_PARAMETER_OFFLOAD=True
LEARNING_RATE=1e-6

# TRAIN_BATCH_SIZE must be divisible by rollout_dp_size (= num_gpus / TP_size = 8 / 1 = 8).
# With skill: total effective prompts per step = TRAIN_BATCH_SIZE * (k+1) * n * max_turns.
# Set TRAIN_BATCH_SIZE=8 (min valid value) so chunk(8) divides evenly.
# Total sequences per step = 8 * 4 * 4 * 3 = 384  (vs original 16 * 16 * 3 = 768, ~half).
TRAIN_BATCH_SIZE=16
PPO_MINI_BATCH_SIZE=16

AUTOMATIC_OVERSAMPLING=False
REJECTION_SAMPLE=True

PPO_MICRO_TOKEN=null
CLIP_RATIO=0.2_0.28
ENTROPY_CLIP_RATE=0.0
GRAD_CLIP=1.0
VLLM_IS_THRESHOLD=2.0
EXTREME_RISK_PROB_THRESHOLD=null
KL_LOSS_COEF=0.0
ENTROPY_COEFFIENT=0.0
KL_LOSS_TYPE="low_var_kl"
TEMPERATURE=1.0
MIN_P=0.0
TOP_P=1.0
TOP_K=-1
# With skill enabled, vLLM sees B*(k+1)*n*T concurrent sequences.
# k=3 → 4× expansion vs baseline. Set n=4 so total stays ≈ baseline n=16.
# Formula: ROLLOUT_N = baseline_n / (k+1)  →  16 / 4 = 4
# Adjust if you change SKILL_K.
ROLLOUT_N=4
KL_COEF=0.0
TOTAL_EPOCHS=1000
ROLLOUT_GPU_MEMORY_UTIL=0.75

SAVE_FREQ=10         # model checkpoint save frequency (steps)
SKILL_SAVE_FREQ=1    # skill library flush frequency (steps); must divide SAVE_FREQ
TEST_FREQ=10
# Per-step training data dump: saves rollout results, log_probs, rewards to JSONL.
# Set DEBUG_BATCH_SAVE_DIR="" to disable entirely.
# DEBUG_BATCH_SAVE_FREQ: save every N steps (aligned with SAVE_FREQ is recommended).
# Set to 0 to disable even when DIR is set.
DEBUG_BATCH_SAVE_DIR="${HDFS_CHECKPOINT_PATH}/rollout_data"
DEBUG_BATCH_SAVE_FREQ=1   # default: same cadence as model checkpoints
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=1
SP_SIZE=2
NUM_PERF_TRIALS=100
APPLY_CHAT_TEMPLATE=True
FREE_CACHE_ENGINE=True
ENFORCE_EAGER=False
NNODES=$PET_NNODES
GPUS_PER_NODE=$GPUS_PER_NODE
if [ -z "$GPUS_PER_NODE" ]; then
    GPUS_PER_NODE=8
fi

MAX_PROMPT_LENGTH=24576
MAX_RESPONSE_LENGTH=8192
PROMPT_OVERSAMPLING_FACTOR=2.0
SAMPLE_OVERSAMPLING_FACTOR=1.0
SAMPLE_SELECTION_STRATEGY=efficiency_stochastic
MAX_SKIP_STEPS=5

# ── Skill memory configuration ──────────────────────────────────────────────
# multi_iteration must be enabled; the engine upgrades automatically
MULTI_ITERATION_ENABLE=True
MULTI_ITERATION_MAX_ITERATIONS=1
MULTI_ITERATION_REMAIN_TURNS=2

SKILL_ENABLE=True
# When use_bm25_direct=True: no LLM selection inference; BM25 top_k_select skills are
# injected directly into all (k+1) copies of each prompt.  All n*(k+1) rollouts of the
# same task form one unified advantage group (no null scheme, no selection trajectories).
# ROLLOUT_N stays unchanged; the (k+1) copies provide the capacity expansion.
SKILL_USE_BM25_DIRECT=False
# k: how many independent selection inferences per task (scheme diversity)
SKILL_K=3
# top_bm25: BM25 recall count — how many candidate skills shown to selection agent
SKILL_TOP_BM25=20
# top_k_select: how many skills LLM picks from BM25 candidates in one selection call
SKILL_TOP_K_SELECT=3
SKILL_N_PER_SCHEME=2          # same as ROLLOUT_N; each scheme gets n rollouts
# Skill library lives as a subdirectory under the same checkpoint root.
# Model checkpoints (saved every SAVE_FREQ steps) and skill snapshots
# (flushed after each successful update_actor that stages new skills) are
# independent — their step numbers do NOT need to align.
#   ${HDFS_CHECKPOINT_PATH}/
#     global_step_10/          ← model checkpoint (every SAVE_FREQ steps)
#     global_step_20/
#     ...
#     skill_library/           ← SKILL_LIBRARY_ROOT
#       global_step_0.jsonl    ← initial empty snapshot
#       global_step_7.jsonl    ← skill flush (whenever new skills are staged)
#       global_step_23.jsonl
#       ...
# Checkpoint restart: load_for_step(N) finds max(step) ≤ N automatically,
# so model ckpt and skill snapshot do not need to be at the same step.
SKILL_LIBRARY_ROOT="${HDFS_CHECKPOINT_PATH}/skill_library"
SKILL_SPEEDUP_IMPROVE_THRESH=1.2
SKILL_SPEEDUP_VS_BASELINE_THRESH=1.2
SKILL_SUMMARY_PARALLEL_S=8
SKILL_MAX_NEW_SKILLS_PER_STEP=2
SKILL_SKILL_VERIFY_SPEEDUP_THRESH=1.2
SKILL_VERIFY_MIN_ABSOLUTE_SPEEDUP=1.2
SKILL_SUMMARY_REQUIRE_TURN1_CORRECT=True
SKILL_SELECTION_TEMPERATURE=1.0
SKILL_SUMMARY_TEMPERATURE=1.0
SKILL_SELECTION_MAX_TOKENS=1024
SKILL_SUMMARY_MAX_TOKENS=2048
SKILL_ADV_CROSS_SCHEME=True
SKILL_SELECTION_WEIGHT=0.3
SKILL_SUMMARY_WEIGHT=0.5
SKILL_TRAIN_SELECTION=False  # MUST be False when SKILL_USE_BM25_DIRECT=True
SKILL_TRAIN_SUMMARY=False
SKILL_SELECTION_REWARD_AGGREGATION=mean
SKILL_SELECTION_MAX_SKILLS_SHOWN=30  # max skills in file_tree when BM25 finds nothing
# ────────────────────────────────────────────────────────────────────────────

# Load common script
source "$(dirname "$0")/train_rl_common.sh"

# Override run_training to inject skill config
run_training() {
  sleep 3

  # ── Initialize skill library directory and baseline snapshot ─────────────
  # The trainer determines skill_start_step by checking whether the RL output
  # directory already contains global_step_* sub-folders:
  #   - Fresh RL start (no RL ckpt):  skill_start_step = 0  → global_step_0.jsonl
  #   - RL resume from step N:        skill_start_step = N  → global_step_N.jsonl
  # We create the initial empty snapshot to match what the trainer expects.
  mkdir -p "${SKILL_LIBRARY_ROOT}"
  # Determine the RL checkpoint subdirectory (same logic as trainer.default_local_dir)
  RL_CKPT_DIR="${HDFS_CHECKPOINT_PATH}/${RUN_NAME}"
  SKILL_START_STEP=0
  if [ -d "$RL_CKPT_DIR" ]; then
    LATEST_RL_STEP=$(ls -d "${RL_CKPT_DIR}"/global_step_* 2>/dev/null | sed 's/.*global_step_//' | sort -n | tail -1)
    if [ -n "$LATEST_RL_STEP" ]; then
      SKILL_START_STEP="$LATEST_RL_STEP"
    fi
  fi
  SKILL_LIB_INIT="${SKILL_LIBRARY_ROOT%/}/global_step_${SKILL_START_STEP}.jsonl"
  if [ ! -f "$SKILL_LIB_INIT" ]; then
    # Check if any snapshot already exists (resume case — skill lib already seeded)
    N_EXISTING=$(ls "${SKILL_LIBRARY_ROOT}"/global_step_*.jsonl 2>/dev/null | wc -l)
    if [ "$N_EXISTING" -eq 0 ]; then
      touch "$SKILL_LIB_INIT"
      echo "[Skill] Initialized empty skill library at ${SKILL_LIB_INIT}  (start_step=${SKILL_START_STEP})"
    else
      echo "[Skill] Existing skill library snapshots found in ${SKILL_LIBRARY_ROOT} (count=${N_EXISTING}), skipping init"
    fi
  else
    N_SKILLS=$(wc -l < "$SKILL_LIB_INIT")
    echo "[Skill] Existing skill library found at ${SKILL_LIB_INIT}  skills=${N_SKILLS}"
  fi

  echo "[Skill] Config summary:"
  echo "[Skill]   enable=${SKILL_ENABLE}  k=${SKILL_K}  top_bm25=${SKILL_TOP_BM25}  top_k_select=${SKILL_TOP_K_SELECT}"
  echo "[Skill]   library_root=${SKILL_LIBRARY_ROOT}"
  echo "[Skill]   speedup_improve_thresh=${SKILL_SPEEDUP_IMPROVE_THRESH}  vs_baseline_thresh=${SKILL_SPEEDUP_VS_BASELINE_THRESH}"
  echo "[Skill]   summary_parallel_s=${SKILL_SUMMARY_PARALLEL_S}"
  echo "[Skill]   verify_thresh=${SKILL_SKILL_VERIFY_SPEEDUP_THRESH}  max_new_skills=${SKILL_MAX_NEW_SKILLS_PER_STEP}"
  echo "[Skill]   selection_weight=${SKILL_SELECTION_WEIGHT}  summary_weight=${SKILL_SUMMARY_WEIGHT}"
  echo "[Skill]   selection_temp=${SKILL_SELECTION_TEMPERATURE}  summary_temp=${SKILL_SUMMARY_TEMPERATURE}"

  if [ "${NNODES:-1}" -gt 1 ]; then
      if [ "${NODE_RANK:-0}" -eq 0 ]; then
          ray start --head \
              --node-ip-address="$RAY_MASTER_ADDR" \
              --port="$RAY_MASTER_PORT" \
              --num-gpus="$N_GPUS_PER_NODE" \
              --block &
          RAY_HEAD_PID=$!
          sleep 10
      else
          ray start \
              --address="$RAY_MASTER_ADDR:$RAY_MASTER_PORT" \
              --num-gpus="$N_GPUS_PER_NODE" \
              --block &
          RAY_WORKER_PID=$!
          wait $RAY_WORKER_PID
          exit 0
      fi
  fi



export WANDB_MODE="offline"
export WANDB_DIR="${WANDB_DIR:-./wandb}"
export WANDB_API_KEY=""
WANDB_MODE="offline"
REFERENCE_BACKEND=${REFERENCE_BACKEND:-"pytorch"}

   HYDRA_FULL_ERROR=1 WANDB_MODE="offline" PYTHONUNBUFFERED=1 python -m kernel.main_kernel \
      trainer.val_before_train=$VAL_BEFORE_TRAIN \
      algorithm.adv_estimator=$ALGORITHM \
      algorithm.is_get_last_turn=$IS_GET_LAST_TURN \
      data.train_files=$TRAIN_FILES \
      data.val_files=$VALID_FILES \
      data.return_raw_chat=$RETURN_RAW_CHAT \
      data.train_batch_size=$TRAIN_BATCH_SIZE \
      data.val_sample_size=$VAL_SAMPLE_SIZE \
      data.max_prompt_length=$MAX_PROMPT_LENGTH \
      data.max_response_length=$MAX_RESPONSE_LENGTH \
      data.apply_chat_template=$APPLY_CHAT_TEMPLATE \
      data.use_prioritized_sampling=$USE_PRIORITIZED_SAMPLING \
      data.update_success_rates_every=1 \
      data.prompt_oversampling_factor=$PROMPT_OVERSAMPLING_FACTOR \
      data.sample_oversampling_factor=$SAMPLE_OVERSAMPLING_FACTOR \
      data.sample_selection_strategy=$SAMPLE_SELECTION_STRATEGY \
      data.automatic_oversampling=$AUTOMATIC_OVERSAMPLING \
      data.use_moderate_sampling=$USE_MODERATE_SAMPLING \
      data.use_refresh_sampling=$USE_REFRESH_SAMPLING \
      data.solverate_low=$SOLVERATE_LOW \
      data.solverate_high=$SOLVERATE_HIGH \
      data.solverate_mean=$SOLVERATE_MEAN \
      data.solverate_std=$SOLVERATE_STD \
      trainer.fix_qwen3_chat_template=$FIX_QWEN3_CHAT_TEMPLATE \
      +algorithm.rollout_is_kwargs=$ROLLOUT_IS_KWARGS \
      +algorithm.rollout_rs_kwargs=$ROLLOUT_RS_KWARGS \
      algorithm.rollout_rs=$ROLLOUT_RS \
      algorithm.rollout_token_veto_threshold=$ROLLOUT_TOKEN_VETO_THRESHOLD \
      actor_rollout_ref.rollout.multi_turn.enable=$ENABLE_MULTI_TURN \
      actor_rollout_ref.rollout.multi_turn.max_user_turns=$MAX_TURN \
      actor_rollout_ref.rollout.multi_turn.multi_iteration.enable=$MULTI_ITERATION_ENABLE \
      actor_rollout_ref.rollout.multi_turn.multi_iteration.max_iterations=$MULTI_ITERATION_MAX_ITERATIONS \
      actor_rollout_ref.rollout.multi_turn.multi_iteration.remain_turns=$MULTI_ITERATION_REMAIN_TURNS \
      actor_rollout_ref.model.path=$MODEL_PATH_RESOLVED \
      actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
      actor_rollout_ref.model.use_remove_padding=True \
      actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
      actor_rollout_ref.actor.use_dynamic_bsz=True \
      actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MICRO_TOKEN \
      actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS \
      actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
      actor_rollout_ref.actor.kl_loss_type=$KL_LOSS_TYPE \
      actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFFIENT \
      actor_rollout_ref.actor.clip_ratio_high=$CLIP_RATIO_HIGH \
      actor_rollout_ref.actor.clip_ratio_low=$CLIP_RATIO_LOW \
      actor_rollout_ref.actor.entropy_clip_rate=$ENTROPY_CLIP_RATE \
      actor_rollout_ref.actor.loss_agg_mode=$LOSS_AGG_MODE \
      actor_rollout_ref.actor.loss_scale_factor=$LOSS_SCALE_FACTOR \
      actor_rollout_ref.actor.extreme_risk_prob_threshold=$EXTREME_RISK_PROB_THRESHOLD \
      actor_rollout_ref.actor.grad_clip=$GRAD_CLIP \
      actor_rollout_ref.model.enable_gradient_checkpointing=True \
      actor_rollout_ref.actor.fsdp_config.param_offload=$ACTOR_PARAMETER_OFFLOAD \
      actor_rollout_ref.actor.fsdp_config.optimizer_offload=$ACTOR_OPTIMIZER_OFFLOAD \
      actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP_SIZE \
      actor_rollout_ref.rollout.enforce_eager=$ENFORCE_EAGER \
      actor_rollout_ref.rollout.free_cache_engine=$FREE_CACHE_ENGINE \
      actor_rollout_ref.rollout.temperature=$TEMPERATURE \
      actor_rollout_ref.rollout.top_p=$TOP_P \
      actor_rollout_ref.rollout.top_k=$TOP_K \
      actor_rollout_ref.rollout.min_p=$MIN_P \
      actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$LOG_PROB_MICRO_TOKEN \
      actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE \
      actor_rollout_ref.rollout.name=vllm \
      actor_rollout_ref.rollout.mode=$ROLLOUT_MODE \
      actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTIL \
      actor_rollout_ref.rollout.n=$ROLLOUT_N \
      actor_rollout_ref.rollout.val_kwargs.n=$N_VAL \
      actor_rollout_ref.rollout.val_kwargs.do_sample=$VAL_DO_SAMPLE \
      actor_rollout_ref.rollout.val_kwargs.temperature=$VAL_TEMPERATURE \
      actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
      actor_rollout_ref.rollout.val_kwargs.max_user_turns=$VAL_MAX_TURN \
      actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
      actor_rollout_ref.rollout.calculate_log_probs=$CALCULATE_LOG_PROBS \
      actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$LOG_PROB_MICRO_TOKEN \
      actor_rollout_ref.ref.fsdp_config.param_offload=True \
      actor_rollout_ref.ref.ulysses_sequence_parallel_size=$SP_SIZE \
      actor_rollout_ref.rollout.agent.num_workers=32 \
      reward_model.enable=False \
      reward_model.reward_manager=$REWARD_MANAGER \
      reward_model.enhanced=$REWARD_ENHANCED \
      reward_model.reference_backend=$REFERENCE_BACKEND \
      reward_model.use_sandbox_rate_limit=$REWARD_USE_SANDBOX_RATE_LIMIT \
      reward_model.server_url='"'$REWARD_SERVER_URL'"' \
      reward_model.rate_limit=$REWARD_RATE_LIMIT \
      reward_model.acquire_timeout=$REWARD_ACQUIRE_TIMEOUT \
      reward_model.max_concurrent=$REWARD_MAX_CONCURRENT \
      reward_model.timeout=$REWARD_TIMEOUT \
      reward_model.task_timeout_in_client=$REWARD_TASK_TIMEOUT_CLIENT \
      reward_model.max_retries=$REWARD_MAX_RETRIES \
      reward_model.task_timeout=$REWARD_TASK_TIMEOUT \
      reward_model.num_perf_trials=$NUM_PERF_TRIALS \
      reward_model.print_status=$REWARD_PRINT_STATUS \
      reward_model.reward_func_name=$REWARD_FUNC_NAME \
      reward_model.speedup_reward_upper_bound=$SPEEDUP_REWARD_UPPER_BOUND \
      reward_model.speedup_reward_lower_bound=$SPEEDUP_REWARD_LOWER_BOUND \
      reward_model.coverage_reward.reward_type=$COVERAGE_REWARD_TYPE \
      reward_model.coverage_reward.weight=$COVERAGE_REWARD_WEIGHT \
      reward_model.coverage_reward.enable=$COVERAGE_REWARD_ENABLE \
      reward_model.coverage_rs=$COVERAGE_RS \
      reward_model.coverage_rs_threshold=$COVERAGE_RS_THRESHOLD \
      reward_model.coverage_rs_factor=$COVERAGE_RS_FACTOR \
      reward_model.coverage_rs_key=$COVERAGE_RS_KEY \
      reward_model.speedup_threshold=$SPEEDUP_THRESHOLD \
      reward_model.detect_decoy_kernel=$DETECT_DECOY_KERNEL \
      algorithm.reward_shaping=$REWARD_SHAPING \
      algorithm.unbiased_shaping=$UNBIASED_SHAPING \
      algorithm.adv_estimator=${ALGORITHM:-grpo} \
      algorithm.use_kl_in_reward=$USE_KL_COEF \
      algorithm.kl_ctrl.kl_coef=$KL_COEF \
      algorithm.batch_std=${BATCH_STD:-False} \
      algorithm.adv_by_last_turn=$ADV_BY_LAST_TURN \
      algorithm.use_final_reward=$USE_FINAL_REWARD \
      algorithm.gamma=$GAMMA \
      critic.ppo_micro_batch_size_per_gpu=4 \
      trainer.critic_warmup=0 \
      trainer.logger=['console','wandb'] \
      trainer.rejection_sample=$REJECTION_SAMPLE \
      trainer.project_name=$PROJECT_NAME \
      trainer.experiment_name=$RUN_NAME \
      trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
      trainer.nnodes=$NNODES \
      trainer.remove_clip=$REMOVE_CLIP \
      trainer.rollout_data_dir=$ROLLOUT_DATA_DIR \
      trainer.validation_data_dir=$VALIDATION_DATA_DIR \
      trainer.debug_batch_save_dir=$DEBUG_BATCH_SAVE_DIR \
      trainer.debug_batch_save_freq=$DEBUG_BATCH_SAVE_FREQ \
      trainer.log_val_generations=10 \
      trainer.save_freq=$SAVE_FREQ \
      trainer.test_freq=$TEST_FREQ \
      trainer.default_local_dir=$CHECKPOINT_DIR \
      trainer.total_epochs=$TOTAL_EPOCHS \
      trainer.val_only=$VAL_ONLY \
      trainer.max_skip_steps=$MAX_SKIP_STEPS \
      ${_WANDB_PROXY_ARG:+"$_WANDB_PROXY_ARG"} \
      rejection_sampling.enable_two_gate_filter=$ENABLE_TWO_GATE_FILTER \
      rejection_sampling.gate1.enabled=$GATE1_ENABLED \
      rejection_sampling.gate1.bias_epsilon=$GATE1_BIAS_EPSILON \
      rejection_sampling.gate2.enabled=$GATE2_ENABLED \
      rejection_sampling.gate2.instability_threshold=$GATE2_INSTABILITY_THRESHOLD \
      rejection_sampling.log_rejected_samples=$LOG_REJECTED_SAMPLES \
      rejection_sampling.save_rejection_stats=$SAVE_REJECTION_STATS \
      skill.enable=$SKILL_ENABLE \
      skill.k=$SKILL_K \
      skill.top_bm25=$SKILL_TOP_BM25 \
      skill.top_k_select=$SKILL_TOP_K_SELECT \
      skill.selection_max_skills_shown=$SKILL_SELECTION_MAX_SKILLS_SHOWN \
      skill.library_root=$SKILL_LIBRARY_ROOT \
      skill.speedup_improve_thresh=$SKILL_SPEEDUP_IMPROVE_THRESH \
      skill.speedup_vs_baseline_thresh=$SKILL_SPEEDUP_VS_BASELINE_THRESH \
      skill.summary_parallel_s=$SKILL_SUMMARY_PARALLEL_S \
      skill.max_new_skills_per_step=$SKILL_MAX_NEW_SKILLS_PER_STEP \
      skill.skill_verify_speedup_thresh=$SKILL_SKILL_VERIFY_SPEEDUP_THRESH \
      skill.verify_min_absolute_speedup=$SKILL_VERIFY_MIN_ABSOLUTE_SPEEDUP \
      skill.summary_require_turn1_correct=$SKILL_SUMMARY_REQUIRE_TURN1_CORRECT \
      skill.selection_temperature=$SKILL_SELECTION_TEMPERATURE \
      skill.summary_temperature=$SKILL_SUMMARY_TEMPERATURE \
      skill.selection_max_tokens=$SKILL_SELECTION_MAX_TOKENS \
      skill.summary_max_tokens=$SKILL_SUMMARY_MAX_TOKENS \
      skill.adv_cross_scheme=$SKILL_ADV_CROSS_SCHEME \
      skill.selection_weight=$SKILL_SELECTION_WEIGHT \
      skill.summary_weight=$SKILL_SUMMARY_WEIGHT \
      skill.train_selection=$SKILL_TRAIN_SELECTION \
      skill.train_summary=$SKILL_TRAIN_SUMMARY \
      skill.selection_reward_aggregation=$SKILL_SELECTION_REWARD_AGGREGATION \
      skill.use_bm25_direct=$SKILL_USE_BM25_DIRECT \
      skill.save_freq=$SKILL_SAVE_FREQ

  if [ -n "${RAY_HEAD_PID:-}" ]; then
      ray stop
  fi
}

main "$@"
