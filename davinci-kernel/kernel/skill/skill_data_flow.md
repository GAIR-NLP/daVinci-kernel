# Skill Memory Training — Data Structure & Flow Documentation

> Updated: 2026-03-18
> Codebase: `drkernel/kernel/`
> Plan: see `skill_memory_training_plan_v5.md`

---

## 1. Notation

| Symbol | Meaning |
|--------|---------|
| B | Dataloader batch size (`data.train_batch_size`) |
| k | Number of skill-selection schemes per task (`skill.k`, default 3) |
| k+1 | Total schemes: k distinct + 1 null (no skill injection) |
| n | Policy rollout count per prompt (`actor_rollout_ref.rollout.n`) |
| T | Max policy turns (`multi_turn.max_user_turns`) |
| s | Parallel summary trajectories (`skill.summary_parallel_s`, default 2) |
| P | Prompt token length (`rollout.prompt_length`) |
| L | Response token length (`rollout.response_length`) |

---

## 2. Three-Agent Prompt Design

Each agent sees a different view of the skill library:

| Agent | Skill information shown | Rationale |
|-------|------------------------|-----------|
| **Selection** | Numbered list: name, description, scope, tags (no body text) | Needs metadata to choose which skills to use |
| **Policy** | Full skill body content injected into system prompt | Needs the actual techniques to write code |
| **Summary** | Same full skill content that policy received (`injected_skill_content`) | Must know what policy already had to propose complementary new skills |
| **Verify step** | Original skill content + proposed new skill content (both injected together) | Tests whether the new skill provides *incremental* benefit on top of existing skills |

### 2.1 Selection agent prompt

```
[SYSTEM]
You are an expert CUDA/Triton kernel optimization engineer.
...identify the top __top_k_select__ skills...
When calling select_skills, first fill in `think`...
Then fill in `selected_skills` with the names and one-sentence reasons.
Call select_skills exactly once.

[USER]
## Task
__task_description__        ← first user message (≤1000 chars)

---

## Candidate Skills (from the library)
__file_tree__               ← BM25 top-20 or full library; headers only
                             format:  N. **name**  [scope=...]  [tags: ...]
                                          description

Now call select_skills to choose the top __top_k_select__ most relevant skills for this task.
```

Tool: `select_skills(think: str, selected_skills: [{name: str, reason: str}])`
- `think`: key bottleneck + why chosen skills address it
- `selected_skills`: `[{name, reason}]` — `name` is the snake_case skill name;
  the engine resolves `name → skill content` via `SkillLibrary.get_rel_path_by_name()`

The selection agent is called **k times independently** per task (same prompt, temperature > 0).
Each call produces one `SelectionTrajectory` with its own `(prompt_ids, response_ids, logprobs)`.
Diversity comes from temperature sampling.  Plus one null scheme (no inference) → total k+1 trajectories per task.

BM25 two-stage retrieval:
1. `retrieve_bm25(task_description, top_k=top_bm25)` → up to 20 candidate names
2. Selection agent picks `top_k_select` (≤ 3) from those candidates

### 2.2 Policy agent prompt (each scheme)

Original task messages with skill content injected into system prompt:

```
[SYSTEM]
<original system message, if any>

---
## Relevant Optimization Techniques

<full content of selected skill files — one block per skill, separated by ---->
---

[USER]
<original task description / problem>

[ASSISTANT] → turn 1 code
[USER]      → env feedback (speedup + correctness + error messages)
[ASSISTANT] → turn 2 code
...
```

Null scheme (j == k): no injection, system prompt is unchanged.

The environment (`KernelEnv`) returns per-turn:
- `performance` (speedup ratio vs PyTorch baseline)
- `time_coverage` (fraction of op execution time covered by kernel)
- Compilation success/failure + error messages (shown as feedback in next user turn)

### 2.3 Summary agent prompt (single-turn)

Aligned with `run_skill_inference.py`.  The model makes **one inference** and calls
`update_skill_library(think, new_skills)` exactly once.  No ReAct, no `read_skill_files`.

```
[SYSTEM]
You are an expert CUDA/Triton kernel optimization engineer.
You have observed a successful multi-turn kernel optimization trajectory...
Your task: extract at most __max_skills__ GENERAL, REUSABLE optimization skills...
...quality criteria, what to AVOID, common pitfalls...
When calling update_skill_library, first fill in the `think` field...
Call update_skill_library exactly once.

[USER]
## Task
__task_messages_formatted__      ← [SYSTEM] + [USER] turns policy received
                                    (4000 char limit, system+user roles only)

## Skills Already Given to the Policy Agent
__injected_skill_content__        ← exact same text that was injected into
                                    policy's system prompt for the winning scheme;
                                    "(no skills were injected — null scheme was used)" if null

## Turn 1 Result (speedup: __turn1_speedup__x)
```python
__turn1_code__
```

## Best Turn Result (speedup: __best_turn_speedup__x)
```python
__best_turn_code__
```

Now call update_skill_library with at most __max_skills__ new, general, reusable skills...
```

Tool: `update_skill_library(think: str, new_skills: [{name, description, scope, tags, content}])`

### 2.4 Verification step prompt

After summary agent proposes a new skill:

```
[SYSTEM]
<original system>

---
## Relevant Optimization Techniques

<original injected skill content>    ← same as policy's winning scheme
                                       (omitted if null scheme)

<new proposed skill content>         ← appended on top
---

[USER]
<original task>
```

Compare result speedup against `best_traj["turn1_speedup"]` (policy's turn-1 with original skills).
New skill accepted if `verify_speedup >= turn1_speedup × skill_verify_speedup_thresh`.

---

## 3. DataProto Structure (unified mixed batch)

After `generate_sequences()` returns, one `DataProto` holds all three agent types.
All rows share identical field names; type is in `non_tensor_batch["agent_type"]`.

### 3.1 Tensor fields — `batch` (TensorDict)

| Field | Shape | dtype | Description |
|-------|-------|-------|-------------|
| `prompts` | `(N, P)` | int64 | Left-padded prompt token ids |
| `responses` | `(N, L)` | int64 | Right-padded response token ids |
| `input_ids` | `(N, P+L)` | int64 | `concat(prompts, responses)` |
| `attention_mask` | `(N, P+L)` | int64 | 1=real token, 0=pad |
| `position_ids` | `(N, P+L)` | int64 | `cumsum(attention_mask) - 1` |
| `response_mask` | `(N, L)` | int64 | 1=real response token, 0=pad |
| `rollout_log_probs` | `(N, L)` | float32 | Per-token log-prob at rollout time; -1.0 for pad |
| `token_level_scores` | `(N, L)` | float32 | Reward at last real token of reward-bearing turns; 0 elsewhere |
| `token_level_rewards` | `(N, L)` | float32 | Same as scores pre-KL; overwritten if KL penalty applied |
| `turn_indices` | `(N,)` | int64 | Policy: 1..T, -1=padding turn; Selection: 1; Summary: 1 |
| `sample_indices` | `(N,)` | int64 | Which sample in the batch (for grouping turns) |
| `loss_mask` | `(N,)` | int64 | 1=enters loss; 0=padding turn or tool-result turn |

Fields added during trainer processing (absent from rollout output):

| Field | Added when | Description |
|-------|-----------|-------------|
| `old_log_probs` | Policy: `compute_log_prob` or bypass; Sel/Sum: after skill_extra concat | Log-prob for IS ratio. For sel/sum rows: set to `rollout_log_probs` (IS=1, same-step generation) |
| `advantages` | After `compute_advantage` | GRPO/TRLOO advantage per token |
| `returns` | After `compute_advantage` | Return estimate per token |
| `rollout_is_weights` | After `compute_rollout_correction` | IS weight for MRS/PRS (policy rows only) |

### 3.2 Non-tensor fields — `non_tensor_batch` (numpy arrays, all shape `(N,)`)

| Field | dtype | policy | selection | summary |
|-------|-------|--------|-----------|---------|
| `uid` | object | `"{orig_uid}_scheme{j}"` | `"{orig_uid}_sel_scheme{j}"` | `"{orig_uid}_summary_{idx}"` |
| `original_uid` | object | `"{orig_uid}"` | same | same |
| `agent_type` | object | `"policy"` | `"selection"` | `"summary"` |
| `skill_scheme_idx` | int32 | j = 0..k | j = 0..k | -1 |
| `summary_group_id` | object | `""` | `""` | `"{orig_uid}_summary"` |
| `skill_staged` | bool | False | False | True if trajectory produced a verified staged skill |
| `num_turns` | int32 | actual turns | 1 | 1 |
| `contain_void_turn` | int32 | 0 or 1 | 0 | 0 |
| `finish_reasons` | object | `"stop"`/`"length"`/... | `"stop"` | `"stop"` |
| `multiturn_messages` | object | full messages (first turn only) / None | None | None |
| `global_turn_indices` | int32 | chronological turn index | 1 | 1 |
| `reward_extra_info` | object | `{speedup, correctness, code, ...}` | `{}` | `{}` |

### 3.3 Row counts

```
N_policy    = B × (k+1) × n × T    includes turn_index == -1 padding rows
N_selection = B × k'               k' = non-null, non-failed selection inferences
                                    (null scheme has no response → not packed)
N_summary   = N_triggered × s      0 if no trigger; s parallel single-turn summaries
N_total     = N_policy + N_selection + N_summary

Why no null row in N_selection:
  The null scheme has no model output → nothing to train on.
  Its reward still enters the LOO computation via the policy row reward
  for scheme k (policy did run on the null prompt and got a real reward).
```

---

## 4. Data Flow (per training step)

```
┌───────────────────────────────────────────────────────────────────────────┐
│ Dataloader  DataProto (B,)                                                │
│   batch["input_ids"]       (B, P+L)   prompt tokens only                 │
│   non_tensor["uid"]        (B,)       unique per step                    │
│   non_tensor["raw_prompt"] (B,)       List[dict] messages                │
│   non_tensor["reward_model"](B,)      ground_truth, entry_point          │
└───────────────────────────┬───────────────────────────────────────────────┘
                            │ batch.pop(prompt fields) → gen_batch (B,)
                            │ gen_batch.meta_info["n"] = n
                            ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ SkillAwareMultiIterAsyncvLLMEngine.generate_sequences(gen_batch)          │
│                                                                           │
│  ── Step 1: Selection  (B×k tasks, async parallel) ──────────────────  │
│    BM25: retrieve_bm25(task_description, top_bm25=20) → candidate names  │
│    Run k INDEPENDENT single-turn inferences per task (temperature > 0)   │
│    Tool: select_skills(think, selected_skills: [{name, reason}])         │
│    Resolve name → skill content via SkillLibrary.get_rel_path_by_name()  │
│    Plus one implicit null scheme (no inference needed)                   │
│                                                                           │
│    Returns:                                                               │
│      skill_contexts_per_task[B][k+1]                                     │
│        [b][j]  = full skill content text for scheme j of task b          │
│        [b][k]  = None  (null scheme)                                     │
│      selection_trajs_per_task[B][k+1]                                    │
│        Each SelectionTrajectory:                                          │
│          .original_uid     str                                            │
│          .scheme_idx       int  0..k-1 (real), k (null)                  │
│          .prompt_ids       List[int]                                      │
│          .response_token_ids List[int]  (≤ selection_max_tokens)         │
│          .logprobs         List[float]  ← per-token logprob              │
│          .selected_paths   List[str]  resolved skill names               │
│          .skill_context    str|None   joined skill body content           │
│          .reward           0.0  (filled in Step 4)                       │
│    Null scheme traj: empty prompt_ids/response_ids/logprobs (no training) │
│                                                                           │
│  ── Step 2: Expand prompts (B → B×(k+1)) ─────────────────────────────  │
│    Each scheme j in 0..k gets its own policy rollout.                    │
│    uid "{orig}_scheme{j}"                                                 │
│                                                                           │
│  ── Step 3: Policy rollout  (parent class) ─────────────────────────────  │
│    Input : B×(k+1) prompts, n rollouts each                              │
│    policy_output: DataProto (B×(k+1)×n×T,)                              │
│                                                                           │
│  ── Step 4: Assign selection rewards ───────────────────────────────────  │
│    Per scheme j: aggregate n rollout returns for uid "{orig}_scheme{j}"   │
│    Duplicate skill-set recipes share their pooled returns.               │
│    traj.reward = aggregated(scheme_returns[(orig_uid, j)])               │
│                                                                           │
│  ── Step 5: Conditional summary ────────────────────────────────────────  │
│    Trigger condition (checked per rollout uid):                           │
│      best_turn_speedup >= turn1_speedup × speedup_improve_thresh (1.1)  │
│      AND best_turn_speedup >= speedup_vs_baseline_thresh (1.1)           │
│                                                                           │
│    For each triggered task, build best_traj:                              │
│      {original_uid, task_messages, task_description,                     │
│       turn1_speedup, turn1_code,                                         │
│       best_turn_speedup, best_turn_code,                                 │
│       ground_truth, entry_point,                                         │
│       injected_skill_content}   ← skill content the winning scheme used  │
│                                   (None for null scheme)                  │
│                                                                           │
│    Run s parallel SINGLE-TURN summary inferences per triggered task:     │
│      Prompt: [summary_system] + [summary_user]                           │
│              user contains: task_messages_formatted                      │
│                           + injected_skill_content  (what policy had)   │
│                           + turn1_code, best_turn_code                   │
│      Tool:   update_skill_library(think, new_skills) — called once       │
│      Parse:  parse_summary_response() → List[NewSkill]                   │
│                                                                           │
│    Verify each proposed skill:                                            │
│      Inject: original_injected_skills + new_skill → turn-1 prompt        │
│      Run greedy inference, call reward_fn                                 │
│      Accept if verify_speedup >= turn1_speedup × verify_thresh           │
│      Stage accepted skills in skill_library._staged                      │
│                                                                           │
│    summary_output: DataProto (N_triggered×s,)                            │
│      .batch["token_level_scores"]  verify_speedup at last token           │
│      .non_tensor["skill_staged"]   True if trajectory staged a skill     │
│                                                                           │
│  ── Step 6: Pack ───────────────────────────────────────────────────────  │
│    Tag policy rows: agent_type="policy", original_uid, skill_staged=F    │
│    Pack selection: one row per real inference (null scheme = no row)      │
│    Pack summary:   (N_triggered×s,)  agent_type="summary"               │
│    DataProto.concat → (N_total,)                                         │
│    All rows share the same non_tensor_batch key set (_NTB_KEYS)          │
└───────────────────────────┬───────────────────────────────────────────────┘
                            │ gen_batch_output: DataProto (N_total,)
                            ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ kernel_trainer.py — main training loop                                    │
│                                                                           │
│  A. Split gen_batch_output  ← BEFORE repeat                              │
│     gen_batch_output[agent_type=="policy"]   → policy_output (N_policy,) │
│     gen_batch_output[agent_type!="policy"]   → skill_extra_batch (N_sel+N_sum,)│
│                                                                           │
│  B–H. (standard policy buffer, log_probs, IS weights, rejection sampling) │
│                                                                           │
│  I. apply_loss_mask_to_rewards(batch)  ← policy only                     │
│                                                                           │
│  J. Concat skill_extra back                                               │
│     all_skill_extra = concat(skill_extra_buffer, skill_extra_batch)      │
│     batch = concat([policy_batch, aligned_extra])                        │
│                                                                           │
│  K. compute_advantage  (full mixed batch)                                 │
│     agent_type=="policy"    → TRLOO  (multi-turn, grouped by uid_scheme) │
│     agent_type=="selection" → GRPO LOO  (grouped by original_uid)       │
│     agent_type=="summary"   → GRPO LOO  (grouped by summary_group_id)   │
│                                                                           │
│  L–M. filter, pad, optional train switch filter                           │
│                                                                           │
│  N. update_actor(batch)                                                   │
│                                                                           │
│  O. flush_skill_library(step)  ← only here, after successful update      │
│     skill_library._staged → appended to new global_step_{step}.jsonl    │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Reward placement

Reward is placed at the **last real token** of the reward-bearing turn.
All other positions are 0.

| Agent | Reward value | When |
|-------|-------------|------|
| Policy | `performance` (speedup) from KernelEnv | Each turn that gets evaluated |
| Selection | aggregated policy return for scheme j (mean or max over n rollouts) | Single turn, last token |
| Summary | `verify_speedup` of the best verified skill in this trajectory | Last token |

---

## 6. Advantage computation grouping

```
Policy  (TRLOO, multi-turn):
  Group key: uid_with_scheme  e.g. "{orig}_scheme2"
  n rollouts share same group key → LOO across n rollouts per turn_idx

Selection  (GRPO LOO, single-turn):
  Group key: original_uid  e.g. "{orig}"
  Group members: k independent inference rows (each has distinct tokens)
  A_j = R_j - mean(R_{j'≠j})

Summary  (GRPO LOO, single-turn):
  Group key: summary_group_id  e.g. "{orig}_summary"
  Group size: s (parallel single-turn summaries for same trigger)
  A_i = R_i - mean(R_{j≠i})
  Advantage broadcast to all loss_mask=1 token positions in trajectory
```

---

## 7. Buffer mechanics

```
buffer_batch       (DataProto) — policy rows
skill_extra_buffer (DataProto) — selection + summary rows

Per rollout step:
  if policy_valid_count < train_batch_size:
    buffer_batch       ← concat(buffer_batch,       new_policy_rows)
    skill_extra_buffer ← concat(skill_extra_buffer, new_sel+sum_rows)
    skill_library._staged accumulates  (NOT flushed yet)
    ← continue

  else (train):
    compute_advantage; update_actor
    flush_skill_library(global_step)   ← write staged skills to JSONL
    buffer_batch = None; skill_extra_buffer = None
```

---

## 8. Skill library versioning (JSONL format)

```
skill_library/
  global_step_0.jsonl      initial snapshot (empty = no skills yet)
  global_step_50.jsonl     snapshot after flush at step 50
  global_step_100.jsonl
  ...

Each .jsonl file: one JSON object per line:
  {"name": "skill_name", "description": "...", "scope": "general",
   "tags": [...], "content": "...", "verify_speedup": 1.5}

At generate_sequences(global_step=N):
  → load_for_step(N): picks snapshot with max(step) ≤ N
  → supports checkpoint restart: always resumes the correct version
  → shared filesystem; all ranks read the same file; no sync barrier needed

flush_staged_skills(step):
  → copy all skills from current snapshot
  → append staged skills (dedup names with _1, _2 … suffix)
  → atomic write: .tmp file then os.replace → global_step_{step}.jsonl
  → reload in-memory state from new file
```

Skill format (get_skill_content() output, same as old .md format):

```markdown
---
name: snake_case_skill_name
description: One-sentence description for the selection agent
scope: general                     # or task_specific/<task_type>
tags: [shared_memory, latency_hiding]
---

## Motivation   — why this technique matters and when to use it
## Key Idea     — the core mechanism and how to implement it
## Example      — a short self-contained code snippet
```

---

## 9. Configuration reference (`skill_prompts.yaml` + training script)

### Hyperparameters

| Key | Default | Description |
|-----|---------|-------------|
| `skill.enable` | `false` | Master switch |
| `skill.k` | 3 | Schemes per task (plus 1 null scheme) |
| `skill.top_bm25` | 20 | BM25 recall count for selection |
| `skill.top_k_select` | 3 | Max skills LLM picks from BM25 candidates |
| `skill.selection_max_skills_shown` | 50 | Fallback: skills shown when BM25 returns nothing |
| `skill.selection_temperature` | 0.8 | Selection agent temperature |
| `skill.selection_max_tokens` | 1024 | Selection max output tokens |
| `skill.speedup_improve_thresh` | 1.1 | Trigger: best ≥ turn1 × this |
| `skill.speedup_vs_baseline_thresh` | 1.1 | Trigger: best ≥ baseline × this |
| `skill.summary_parallel_s` | 2 | Parallel summary trajectories per trigger |
| `skill.summary_temperature` | 0.9 | Summary agent temperature |
| `skill.summary_max_tokens` | 2048 | Summary max output tokens |
| `skill.max_new_skills_per_step` | 2 | Max new_skills proposed per summary call |
| `skill.skill_verify_speedup_thresh` | 1.08 | Accept: verify_speedup ≥ turn1 × this |
| `skill.adv_cross_scheme` | false | Normalize policy ADV across schemes |
| `skill.selection_weight` | 0.3 | Selection loss weight in mixed batch |
| `skill.summary_weight` | 0.5 | Summary loss weight in mixed batch |
| `skill.train_selection` | true | Include selection rows in update_actor |
| `skill.train_summary` | true | Include summary rows in update_actor |
| `skill.library_root` | `"skill_library/"` | Root directory for JSONL snapshots |
| `skill.global_step_prefix` | `"global_step_"` | Snapshot file prefix |

### Prompt placeholders (`skill_prompts.yaml`)

| Template | Placeholders |
|----------|-------------|
| `selection_system` | `__top_k_select__` |
| `selection_user` | `__top_k_select__`, `__task_description__`, `__file_tree__` |
| `summary_system` | `__max_skills__` |
| `summary_user` | `__task_messages_formatted__`, `__injected_skill_content__`, `__turn1_speedup__`, `__turn1_code__`, `__best_turn_speedup__`, `__best_turn_code__`, `__max_skills__` |
| `skill_injection_header` | (none) |
| `skill_injection_footer` | (none) |

Substitution uses `str.replace("__name__", value)`, NOT `.format()`.

---

## 10. Compatibility guarantees

| Concern | Mechanism |
|---------|-----------|
| `skill.enable=false` | `generate_sequences` calls `super()` unchanged; no `agent_type` field; all skill branches are no-ops |
| `old_log_probs` for sel/sum | Set to `rollout_log_probs` at concat time (same-step model → IS ratio = 1) |
| Two `pad_dataproto_to_divisor` calls | Pad #1: policy-only before old_log_prob; Pad #2: mixed batch after advantage |
| Oversampling `valid_query_size` | Counts only policy rows; sel/sum rows never trigger `continue` |
| MRS/PRS rejection sampling | Runs before skill_extra concat; only affects policy rows |
| `train_selection` / `train_summary` | Rows removed after advantage computation but before `update_actor` |
| Buffer across steps | policy → `buffer_batch`, sel/sum → `skill_extra_buffer`; both cleared on same training step |
| Skill library flush | Only on successful `update_actor`; staged skills accumulate in `skill_library._staged` across buffer steps |
| Checkpoint restart | `load_for_step(N)` finds snapshot with `max(step) ≤ N`; if resuming from ckpt step M, global_step=M is passed → correct snapshot automatically loaded |

---

## 11. 实现进度（2026-03-21）

### 已完成

| 文件 | 说明 |
|------|------|
| `kernel/skill/__init__.py` | 模块入口 |
| `kernel/skill/skill_library.py` | JSONL 格式存储；版本管理；BM25 检索；stage/flush 原子写入；断点重启 load_for_step；`flush_staged_skills` 即使 staged 为空也写文件（保证时间线完整）；返回 `{skill/library_size, skill/new_this_step, skill/flush_skipped}` dict 供 wandb |
| `kernel/skill/skill_prompt_builder.py` | 所有 agent prompt 构建；工具 schema 从 YAML 加载；对齐数据采集脚本；`build_summary_initial_messages(best_traj, max_skills, ...)` — 单轮无 ReAct，无 read_skill_files |
| `kernel/skill/skill_summary_env.py` | Summary 单轮 response 解析器（`parse_summary_response`）；lazy filter；`_parse_tool_call` 加 `ast.literal_eval` fallback（修复 Qwen3 Python list 单引号格式） |
| `kernel/skill/skill_adv.py` | 三类 agent advantage 计算（policy TRLOO / selection LOO / summary LOO）；summary 单轮注释已修正 |
| `kernel/config/skill_memory.yaml` | 所有超参的 OmegaConf struct 定义，包含 `top_bm25`、`top_k_select`、`selection_max_skills_shown`、`save_freq`；移除 `summary_max_turns` |
| `kernel/config/skill_prompts.yaml` | 所有 prompt 模板 + tool schema（selection/summary）；无 read_skill_files |
| `kernel/workers/rollout/vllm_rollout/vllm_async_engine_skill.py` | 完整 RL 引擎；selection BM25 两阶段；summary 单轮；`flush_skill_library` 返回 stats dict 或 (stats, skills) tuple；Step6 将 `skill_step_stats` 写入 `meta_info`（tasks_with_skill / policy_rows_with_skill / selection_rows / summary_rows / summary_triggered / staged_pending / library_size）；`_verify_skill` 同时检查 correctness + speedup |
| `kernel/scripts/rl/14b_trloo_mrs_pr_prs_skill.sh` | Skill 库初始化从 **RL 输出目录**中检测 global_step 子文件夹来确定 start_step（而非从 MODEL_PATH 提取 SFT step）；`SKILL_SAVE_FREQ` 须整除 `SAVE_FREQ`；`DEBUG_BATCH_SAVE_DIR` 每 step 保存 rollout 数据 |
| `kernel/kernel_trainer.py` | `_skill_start_step` / `meta_info` 注入；`skill_extra_buffer` 完整 split/accumulate/concat/clear 流程；`flush_skill_library` via Ray remote 并收集返回 stats；从 `gen_batch_output.meta_info["skill_step_stats"]` 读 rollout 期 skill 使用统计；所有 `train/skill/*` metrics 合并到 wandb log；**metric 计算后 selection/summary 行重新拼回 batch**（`_non_policy_batch` 保留再 concat） |

### Skill library 保存规则（2026-03-19 修订）

| 规则 | 实现位置 |
|------|---------|
| 从 RL 实际 start step 开始存（trainer 检查 RL 输出目录是否已有 global_step_* 子文件夹；无则 start_step=0，有则取当前 global_steps）| `kernel_trainer.py` `_skill_start_step`；shell 脚本从 **RL 输出目录**检测 start_step（不再读 MODEL_PATH） |
| 即使没有新 skill 也写文件（保证每个 save step 都有完整记录） | `skill_library.py` `flush_staged_skills`（移除 `if not self._staged: return`） |
| skill save step 必须整除 ckpt save_freq | `vllm_async_engine_skill.py` `__init__` 中懒校验（首次 generate_sequences 时从 meta_info 读 ckpt_save_freq） |
| flush 只在 update_actor 之后，不在 generate_sequences 内 | `kernel_trainer.py` L4011；engine 内无 auto-flush（避免 buffer 导致 staged skill 与 update_actor 错位） |

### 每 step 训练数据保存

`trainer.debug_batch_save_dir` 参数控制（脚本中为 `${HDFS_CHECKPOINT_PATH}/rollout_data`）：
- **Point 1** `step{N:06d}_1_after_rollout.jsonl` — generate_sequences 返回后的原始 DataProto（包含 rollout token ids、rollout_log_probs、reward_extra_info）
- **Point 2** `step{N:06d}_2_pre_update.jsonl` — update_actor 前的完整 batch（包含 old_log_probs、advantages、returns、token_level_rewards）

实现：`kernel/utils/batch_debug_save.py` `save_batch_to_jsonl()`，大 tensor 以 summary（shape/min/max/mean）形式存储。

### 端到端运行状态（step=1 实测，library 为空）

```
[Skill] Step1 selection  elapsed=0.0s   ← 库为空，k次推理全部跳过
[Skill] Step2 expanded   2 → 8 prompts  ← B=2, k+1=4 (全null scheme)
[Skill] Step3 policy     elapsed=216~361s  ← 主瓶颈：4×提示词量
[Skill] Step5 summary triggered (1次出现)  turn1=0.00x → best=1.65x
```

### 已知瓶颈：库为空时的冗余展开（不修复，接受现状）

**现象**：库为空时 k 次 selection 推理均返回 null，导致 k+1 个方案全部相同（都是 null scheme），policy rollout 跑 k+1 份完全相同的 prompt（4× overhead）。

**决策**：不做方案去重（`_expand_prompts_with_schemes` 不改），接受训练初期的额外开销。库中积累 skill 后自然消失。

### 仍待完成

| 任务 | 优先级 | 文件 |
|------|--------|------|
| 空行问题：打印 message 内容时每行都经 Ray PID 前缀，导致日志碎片化 | LOW（可读性） | — |

详细说明见 [`skill_memory_training_plan_v5.md`](../../../skill_memory_training_plan_v5.md)。
