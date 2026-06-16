#!/usr/bin/env python3
"""
multi_turn_kernel_sampling.py

Multi-turn kernel optimization data collection script.

Loads seed tasks from an SFT parquet dataset, optionally enriches the first prompt
with BM25-retrieved similar examples, then runs up to MAX_TURNS=5 rounds of:
    model generation → KernelGym environment evaluation → feedback

Supports two data formats:
  - drkernel-coldstart-8k.parquet:
      columns: messages, original_python_code, entry_point, uuid, ...
  - sft_cuda_llm_r1.parquet (cudaLLM):
      columns: messages, py_code (or original_python_code),
               reward_model (dict: {ground_truth, entry_point}), ...

Output: JSONL (append mode). Each line contains:
    - messages: list of 10 dicts (5 user/assistant pairs, len=10)
    - metadata: uuid, entry_point, original_python_code, round_speedups, etc.
    Format is aligned with drkernel-coldstart-8k SFT data.

Usage:
    python multi_turn_kernel_sampling.py \\
        --data   ../data/drkernel-coldstart-8k/drkernel-coldstart-8k.parquet \\
        --output ../data/sampled_multiturn.jsonl \\
        --server http://localhost:8000 \\
        --model  gpt-4o \\
        --base-url http://your-api-endpoint/v1 \\
        --api-key "" \\
        --bm25-top-k 3 \\
        --workers 4 \\
        --resume

    # Using cudaLLM data:
    python multi_turn_kernel_sampling.py \
        --data   ../data/drkernel-coldstart-8k/drkernel-coldstart-8k.parquet \
        --output ../data/sampled_multiturn_cudallm_short.jsonl \
        --server    http://10.246.235.138:10907 \
        --proxy-url http://localhost:8101 \
        --model  gpt-5.4 \
        --base-url https://apicz.boyuerichdata.com/v1 \
        --api-key "sk-OgrTdF7xxs6HvxVw47Qg0gsohiZdVOlFP3J3wRdQFfPYjMlT" \
        --bm25-top-k 3 \
        --workers 50 \
        --resume \
        --subset 2000
"""

import argparse
import json
import os
import re
import string
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional
from uuid import uuid4

import numpy as np
import pandas as pd
import requests
from openai import OpenAI

try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

MAX_TURNS = 5

# ──────────────────────────────────────────────────────────────────────────────
# System prompt (hardcoded)
# ──────────────────────────────────────────────────────────────────────────────

# SYSTEM_PROMPT = """\
# You write custom Triton kernels to replace the pytorch operators in the given architecture to get speedups.

#     You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom Triton kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.


#         Here's an example to show you the syntax of inline embedding custom Triton kernels in torch: The example given architecture is:

#         ```

#         import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class Model(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()

#     def forward(self, a, b):
#         return a + b


# def get_inputs():
#     # randomly generate input tensors based on the model architecture
#     a = torch.randn(1, 128).cuda()
#     b = torch.randn(1, 128).cuda()
#     return [a, b]


# def get_init_inputs():
#     # randomly generate tensors required for initialization based on the model architecture
#     return []


#         ```

#         The example new arch with custom Triton kernels looks like this:

#         ```
#         import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import triton
# import triton.language as tl


# @triton.jit
# def add_kernel(
#     x_ptr,  # Pointer to first input
#     y_ptr,  # Pointer to second input
#     out_ptr,  # Pointer to output
#     n_elements,  # Total number of elements in input/output
#     BLOCK_SIZE: tl.constexpr,
# ):
#     # Each program handles a contiguous block of data of size BLOCK_SIZE
#     block_start = tl.program_id(0) * BLOCK_SIZE
#     # Create a range of offsets [0..BLOCK_SIZE-1]
#     offsets = block_start + tl.arange(0, BLOCK_SIZE)
#     # Mask to ensure we don't go out of bounds
#     mask = offsets < n_elements
#     # Load input values
#     x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
#     y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
#     # Perform the elementwise addition
#     out = x + y
#     # Store the result
#     tl.store(out_ptr + offsets, out, mask=mask)


# def triton_add(x: torch.Tensor, y: torch.Tensor):
#     assert x.is_cuda and y.is_cuda, "Tensors must be on CUDA."
#     x = x.contiguous()
#     y = y.contiguous()
#     out = torch.empty_like(x)
#     n_elements = x.numel()
#     BLOCK_SIZE = 128
#     grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
#     add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
#     return out


# class ModelNew(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()

#     def forward(self, a, b):
#         return triton_add(a, b)
#         ```
# """


SYSTEM_PROMPT = None
# ──────────────────────────────────────────────────────────────────────────────
# Message templates
# ──────────────────────────────────────────────────────────────────────────────

TASK_PROMPT_TEMPLATE = """\
You are given the following architecture:
    ```
    {original_python_code}
    ```

Optimize the architecture named Model with custom Triton operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Let's think step by step."""

FEEDBACK_PROMPT_TEMPLATE = """\
Now you have received the server feedback for your last implementation. Based on that and all your previous responses, improve the implementation.

Here is the server feedback. Please refer to this feedback to improve the implementation:
Server feedback (status/metrics/errors):
{feedback_json}

Return an improved Triton implementation named `ModelNew` as a single ```python``` block. Let's think step by step."""


TASK_PROMPT_TEMPLATE = """\
You are looking at this PyTorch code and thinking it could be optimized with Triton.  

Here's the PyTorch code:  

```python 
{original_python_code} 
```

Here is the skill that might be relevant to the code:
{skill_content}

You need to create a Triton version with the entry point called `ModelNew`.  

Please firstly analyze this code and think hard how you can optimize it, considering the skills you have. 
You also should think about the whether the skill is useful to improve the code and how to use it properly if it is useful.

**Please output and show your thinking, plan, analysis etc. in a markdown format, before your coding, which should be as more as possible.**"""



FEEDBACK_PROMPT_TEMPLATE = """\
Server feedback from the evaluation environment for your last implementation: 
{feedback_json} 

Based on the above server feedback, please improve the implementation: 
- If there are errors/crashes/illegal memory access: identify the root cause and fix it; prevent recurrence. 
- If there is no speedup or performance regresses: optimize the bottlenecks to achieve a clear speedup. 
- If there is already a speedup: further improve performance without degrading correctness. 
- If the suggested skill has already been fully applied and speedup is no longer improving or even regresses, the remaining gains likely require going beyond it — consider mathematical equivalences shortcut, algorithmic redesign, or any insight the skill does not cover.
- Please output your thinking, plan, analysis, and the final code."""
# ──────────────────────────────────────────────────────────────────────────────
# BM25 skill retriever  (indexes skill_library.jsonl)
# ──────────────────────────────────────────────────────────────────────────────

# Skill injection constants — aligned with kernel_trainer/skill_prompt_builder.py
_SKILL_INJECTION_HEADER = "\n\n---\n## [Skill Library] Potentially Relevant Optimization Techniques\n\n"
_SKILL_INJECTION_FOOTER = "\n\n---"
# Sentinel used to find the insertion point in the first user message
_FINAL_INSTRUCTION_PREFIX = "Optimize the architecture named Model"


def _tokenize(text: str) -> list:
    text = text.lower()
    text = re.sub(r"[_\-/]", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return [t for t in text.split() if t]


def load_skill_library(path: str) -> list:
    """Load skills from a JSONL skill library file."""
    skills = []
    if not os.path.exists(path):
        print(f"[warn] Skill library not found: {path}")
        return skills
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    skills.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return skills


class SkillBM25Retriever:
    """
    BM25 retriever over a skill library (skill_library.jsonl).

    Index corpus: name + description + tags + content for each skill.
    Query: original_python_code of the task being processed.

    Aligns with the BM25 retrieval pattern in kernel_trainer / vllm_async_engine_skill.py.
    """

    def __init__(self, skills: list):
        if not _HAS_BM25:
            raise ImportError("rank_bm25 required. Install with: pip install rank-bm25")
        self.skills = skills
        corpus = []
        for s in skills:
            doc = " ".join([
                s.get("name", ""),
                s.get("description", ""),
                " ".join(s.get("tags", [])),
                s.get("content", ""),
            ])
            corpus.append(_tokenize(doc))
        self._bm25 = BM25Okapi(corpus)

    def retrieve(self, query: str, top_k: int = 3) -> list:
        """Return top_k skills most relevant to query (by BM25 score > 0)."""
        if not self.skills:
            return []
        q_tokens = _tokenize(query)
        scores = self._bm25.get_scores(q_tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [
            self.skills[i]
            for i in ranked[:top_k]
            if scores[i] > 0
        ]


def _format_skill_content(skills: list) -> str:
    """
    Concatenate retrieved skill contents for injection.
    Each skill contributes:  ### <name>\n<content>
    """
    parts = []
    for s in skills:
        name = s.get("name", "")
        content = s.get("content", "").strip()
        parts.append(f"### {name}\n{content}")
    return "\n\n".join(parts)


def _insert_skill_block_into_text(text: str, skill_block: str) -> str:
    """
    Insert skill_block immediately before "Optimize the architecture named Model".
    Falls back to appending if the sentinel is not found.
    Mirrors skill_prompt_builder._insert_skill_into_text().
    """
    idx = text.rfind(_FINAL_INSTRUCTION_PREFIX)
    if idx >= 0:
        return text[:idx] + skill_block + "\n\n" + text[idx:]
    return text + skill_block


def inject_skills_into_messages(messages: list, skills: list) -> list:
    """
    Inject retrieved skill contents into the conversation messages.

    Mirrors kernel_trainer / skill_prompt_builder.inject_skill_into_messages():
    - If a system message exists: append skill block to system content.
    - Otherwise: insert into the first user message before the final instruction.

    Returns a new list (does not mutate the original).
    """
    if not skills:
        return messages

    skill_content = _format_skill_content(skills)
    skill_block = _SKILL_INJECTION_HEADER + skill_content + _SKILL_INJECTION_FOOTER

    import copy
    messages = copy.deepcopy(messages)

    for msg in messages:
        if msg.get("role") == "system":
            msg["content"] = msg["content"] + skill_block
            return messages

    # No system turn — inject into first user message before the final instruction
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        part["text"] = _insert_skill_block_into_text(part["text"], skill_block)
                        return messages
                content.append({"type": "text", "text": skill_block.strip()})
            else:
                msg["content"] = _insert_skill_block_into_text(content, skill_block)
            return messages

    # Fallback: prepend system message
    messages.insert(0, {"role": "system", "content": skill_block.strip()})
    return messages


# ──────────────────────────────────────────────────────────────────────────────
# Kernel code extractor
# ──────────────────────────────────────────────────────────────────────────────

_KERNEL_MARKERS = [
    re.compile(r"#\s*Kernel\s+Implementation\s*\n(.*?)(?=#\s*End\b|$)", re.IGNORECASE | re.DOTALL),
    re.compile(r"```python\s*#\s*Kernel\s*\n(.*?)```", re.IGNORECASE | re.DOTALL),
    re.compile(r"#\s*Your\s+implementation:\s*\n(.*?)(?=#\s*End\b|$)", re.IGNORECASE | re.DOTALL),
    re.compile(r"#\s*Generated\s+kernel:\s*\n(.*?)(?=#\s*End\b|$)", re.IGNORECASE | re.DOTALL),
]
_ANSWER_BLOCK_RE = re.compile(
    r"```answer[ \t]*(?:\r?\n)?(?P<code>.*?)(?:\r?\n)?```",
    re.IGNORECASE | re.DOTALL,
)
_GENERIC_CODE_RE = re.compile(r"```(?:[\w+-]+)?\s*\n?(.*?)```", re.DOTALL)


def extract_kernel_code(response: str) -> Optional[str]:
    """Extract kernel code from model response (mirrors KernelAgent logic)."""
    # 1. ```answer block
    m = _ANSWER_BLOCK_RE.search(response)
    if m:
        return m.group("code").strip()

    # 2. Specific kernel markers
    for pattern in _KERNEL_MARKERS:
        m = pattern.search(response)
        if m:
            return m.group(1).strip()

    # 3. Generic python code blocks — prefer last block containing ModelNew,
    #    fall back to last block overall
    blocks = _GENERIC_CODE_RE.findall(response)
    if blocks:
        for block in reversed(blocks):
            if "ModelNew" in block:
                return block.strip()
        return blocks[-1].strip()

    return None


# ──────────────────────────────────────────────────────────────────────────────
# KernelGym environment client
# ──────────────────────────────────────────────────────────────────────────────

class KernelEnvClient:
    """Simple synchronous HTTP client for KernelGym server (submit/poll/result)."""

    def __init__(
        self,
        server_url: str,
        proxy_url: Optional[str] = None,
        timeout: int = 300,
        poll_interval: float = 2.0,
    ):
        """
        Args:
            server_url:   Actual KernelGym GPU server URL (e.g. http://10.x.x.x:10907).
            proxy_url:    Local GPFS proxy client URL (e.g. http://localhost:8090).
                          When set, all requests go through the proxy:
                            POST {proxy_url}/evaluate  +  X-TARGET-HOST: {server_url}
                          The proxy client writes to shared FS; the GPU-side proxy server
                          picks it up and forwards to server_url.
                          When None, requests are sent directly to server_url.
            timeout:      Server-side evaluation timeout (seconds).
            poll_interval: Status polling interval (seconds).
        """
        self.server_url = server_url.rstrip("/")
        self.proxy_url = proxy_url.rstrip("/") if proxy_url else None
        self.timeout = timeout
        self.poll_interval = poll_interval

    def _request_url(self, path: str) -> str:
        """Return full URL for a given endpoint path, routing via proxy if configured."""
        base = self.proxy_url if self.proxy_url else self.server_url
        return f"{base}/{path}"

    def _extra_headers(self) -> dict:
        """Return extra headers needed for proxy routing (X-TARGET-HOST)."""
        if self.proxy_url:
            return {"X-TARGET-HOST": self.server_url}
        return {}

    def evaluate(
        self,
        reference_code: str,
        kernel_code: str,
        entry_point: str = "Model",
        task_id: Optional[str] = None,
        num_correct_trials: int = 5,
        num_perf_trials: int = 100,
    ) -> dict:
        """Submit kernel for evaluation and poll until done. Returns result dict."""
        if task_id is None:
            task_id = f"sampling_{uuid4().hex[:12]}"

        payload = {
            "task_id": task_id,
            "reference_code": reference_code,
            "kernel_code": kernel_code,
            "backend": "triton",
            "num_correct_trials": num_correct_trials,
            "num_perf_trials": num_perf_trials,
            "timeout": self.timeout,
            "priority": "normal",
            "entry_point": entry_point,
            "verbose_errors": True,
            "enable_profiling": True,
        }

        headers = self._extra_headers()

        # Submit
        try_times = 0
        while True:
            try:
                resp = requests.post(
                    self._request_url("evaluate"),
                    json=payload,
                    headers=headers,
                    timeout=180,
                )
                resp.raise_for_status()
                break
            except Exception as e:
                try_times += 1
                if try_times >= 5:
                    return {"status": "failed", "error_message": f"Submit failed: {e}"}
                time.sleep(3 * (try_times + 1))
        # Poll status
        start = time.time()
        while time.time() - start < self.timeout + 60:
            try:
                s = requests.get(
                    self._request_url(f"status/{task_id}"),
                    headers=headers,
                    timeout=10,
                )
                if s.status_code == 200:
                    status_data = s.json()
                    status = status_data.get("status", "unknown")
                    if status == "completed":
                        r = requests.get(
                            self._request_url(f"results/{task_id}"),
                            headers=headers,
                            timeout=10,
                        )
                        if r.status_code == 200:
                            result = r.json()
                            result["status"] = "completed"
                            result["task_id"] = task_id
                            return result
                        return {"status": "failed", "error_message": "Failed to fetch results", "task_id": task_id}
                    elif status in ("failed", "timeout", "cancelled"):
                        return {
                            "status": status,
                            "error_message": status_data.get("error_message", f"Task {status}"),
                            "task_id": task_id,
                        }
            except Exception:
                pass
            time.sleep(self.poll_interval)

        return {"status": "timeout", "error_message": f"Client-side timeout after {self.timeout}s", "task_id": task_id}

    def format_feedback(self, result: dict) -> str:
        """Format server result as feedback message for next turn."""
        # Build a clean feedback dict (match training data format)
        feedback = {
            "task_id": result.get("task_id", ""),
            "status": result.get("status", "unknown"),
            "compiled": result.get("compiled", False),
            "correctness": result.get("correctness", False),
            "decoy_kernel": result.get("decoy_kernel", False),
            "reference_runtime": result.get("reference_runtime", 0.0),
            "kernel_runtime": result.get("kernel_runtime", 0.0),
            "speedup": result.get("speedup", 0.0),
            "metadata": result.get("metadata", {}),
            "error_message": result.get("error_message", None),
            "error_code": result.get("error_code", None),
        }
        feedback_json = json.dumps(feedback, indent=2)
        return FEEDBACK_PROMPT_TEMPLATE.format(feedback_json=feedback_json)


# ──────────────────────────────────────────────────────────────────────────────
# Multi-turn conversation runner
# ──────────────────────────────────────────────────────────────────────────────

def run_multiturn(
    record: dict,
    client: OpenAI,
    env_client: KernelEnvClient,
    model: str,
    retriever: Optional[SkillBM25Retriever],
    bm25_top_k: int,
    max_turns: int,
    temperature: float,
    max_tokens: int,
    max_retries: int,
) -> dict:
    """Run multi-turn kernel optimization for one record. Returns result dict."""
    uuid = record.get("uuid", str(uuid4()))
    original_code = record.get("original_python_code", "")
    entry_point = record.get("entry_point", "Model")

    # ── Build initial user message ──────────────────────────────────────────
    # task_content = TASK_PROMPT_TEMPLATE.format(original_python_code=original_code)

    # Base messages (no system role in saved output — system is always passed separately)
    # messages: list = [{"role": "user", "content": task_content}]

    # ── BM25 skill injection ────────────────────────────────────────────────
    # Query the skill library with the task's original Python code,
    # then inject retrieved skills into the first user message —
    # exactly as kernel_trainer / skill_prompt_builder.inject_skill_into_messages().
    retrieved_skills: list = []
    if retriever is not None and bm25_top_k > 0:
        retrieved_skills = retriever.retrieve(original_code, top_k=bm25_top_k)
        if retrieved_skills:
            skill_content = _format_skill_content(retrieved_skills)
            task_content = TASK_PROMPT_TEMPLATE.format(original_python_code=original_code, skill_content=skill_content)
            messages = [{"role": "user", "content": task_content}]
            # messages = inject_skills_into_messages(messages, retrieved_skills)

    # The OpenAI call always prepends the system message
    def _call_api(conversation_messages: list) -> Optional[str]:
        if SYSTEM_PROMPT is not None:
            api_messages = [{"role": "system", "content": SYSTEM_PROMPT.format(reference_code=original_code, entry_point=entry_point)}] + conversation_messages
        else:
            api_messages = conversation_messages
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=api_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                # 两种方式分析OpenAI返回的token数，一个是直接用返回的resp.usage，一个用tokenizer实际编码response字符串计算
                response_str = resp.choices[0].message.content or ""

                print("response_str: ", response_str)
                print("real resp length from OpenAI API (resp.usage.completion_tokens):", resp.usage.completion_tokens)
                
                try:
                    from openai import OpenAI
                    # 直接从当前client获取tokenizer，如果有
                    if hasattr(client, 'tokenizer'):
                        num_tokens_tokenizer = len(client.tokenizer.encode(response_str))
                    else:
                        # openai官方tokenizer需安装 tiktoken
                        try:
                            import tiktoken
                            # Could not automatically map gzy/gpt-5.4 to a tokeniser. Please use `tiktoken.get_encoding` to explicitly get the tokeniser you expect.
                            enc = tiktoken.get_encoding("o200k_base")
                            num_tokens_tokenizer = len(enc.encode(response_str))
                        except Exception as e_tok:
                            print("[warn] failed to use tiktoken, fallback to len(str.split()):", e_tok)
                            num_tokens_tokenizer = len(response_str.split())
                    print("token count from tokenizer analysis :", num_tokens_tokenizer)
                except Exception as e:
                    print("Tokenize post-analysis error:", str(e))
                return resp.choices[0].message.content or ""
            except Exception as e:
                print(f"  [warn] API error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(3 * (attempt + 1))
        return None

    round_speedups: list = []
    round_results: list = []


    for turn in range(max_turns):
        # ── Model generation ──────────────────────────────────────────────
        response_text = _call_api(messages)
        if response_text is None:
            print(f"  [warn] uuid={uuid} turn={turn+1}: API call failed, stopping.")
            messages.append({"role": "assistant", "content": ""})
            break

        messages.append({"role": "assistant", "content": response_text})
        print("response_text: ", response_text)

        # ── Extract kernel code ───────────────────────────────────────────
        kernel_code = extract_kernel_code(response_text)
        if kernel_code is None:
            # No code found — give an error feedback
            feedback_result = {
                "task_id": f"sampling_{uuid}_{turn+1}",
                "status": "failed",
                "compiled": False,
                "correctness": False,
                "decoy_kernel": False,
                "reference_runtime": 0.0,
                "kernel_runtime": 0.0,
                "speedup": 0.0,
                "error_message": "No kernel code found in model response. Please provide a complete Python code block with the ModelNew class.",
                "error_code": "NO_CODE",
            }
            round_speedups.append(0.0)
            round_results.append(feedback_result)
        else:
            # ── Environment interaction ───────────────────────────────────
            task_id = f"sampling_{uuid}_{turn+1}_{uuid4().hex[:8]}"
            env_result = env_client.evaluate(
                reference_code=original_code,
                kernel_code=kernel_code,
                entry_point=entry_point,
                task_id=task_id,
            )
            speedup = env_result.get("speedup", 0.0) or 0.0
            round_speedups.append(speedup)
            round_results.append(env_result)
            print(
                f"  [env] uuid={uuid} turn={turn+1} "
                f"compiled={env_result.get('compiled')} "
                f"correct={env_result.get('correctness')} "
                f"speedup={speedup:.4f}x"
            )

        # Last turn: env already evaluated, no feedback message needed
        if turn == max_turns - 1:
            break

        # ── Append feedback as next user message ──────────────────────────
        feedback_content = env_client.format_feedback(round_results[-1])
        print("feedback_content: ", feedback_content)
        print("="*100)
        messages.append({"role": "user", "content": feedback_content})

    return {
        "uuid": uuid,
        "messages": messages,
        "entry_point": entry_point,
        "original_python_code": original_code,
        "round_speedups": round_speedups,
        "final_speedup": max(round_speedups) if round_speedups else 0.0,
        "num_rounds": len(round_speedups),
        "source_uuid": record.get("uuid"),
        "injected_skills": [s.get("name") for s in retrieved_skills],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Resume helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_done_uuids(paths: list) -> set:
    done = set()
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "source_uuid" in obj:
                        done.add(obj["source_uuid"])
                    elif "uuid" in obj:
                        done.add(obj["uuid"])
                except json.JSONDecodeError:
                    pass
        print(f"[resume] loaded done uuids from {path}: {len(done)} total so far")
    return done


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Multi-turn kernel optimization sampling.")
    p.add_argument("--data", "-d",
                   default="../data/drkernel-coldstart-8k/drkernel-coldstart-8k.parquet",
                   help="Path to SFT parquet file.")
    p.add_argument("--output", "-o",
                   default="../data/sampled_multiturn.jsonl",
                   help="Output JSONL path (append mode).")
    p.add_argument("--server",
                   default="http://localhost:8000",
                   help="Actual KernelGym GPU server URL (used as X-TARGET-HOST when --proxy-url is set).")
    p.add_argument("--proxy-url", default=None,
                   help="Local GPFS proxy client URL (e.g. http://localhost:8090). "
                        "When set, all env requests go through this proxy with "
                        "X-TARGET-HOST pointing to --server. "
                        "Leave unset to contact --server directly.")
    p.add_argument("--model", "-m", default="gpt-4o")
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key", default="")
    p.add_argument("--skill-lib",
                   default="../data/skill/skill_library.jsonl",
                   help="Skill library JSONL for BM25 retrieval (disable with --bm25-top-k 0).")
    p.add_argument("--bm25-top-k", type=int, default=3,
                   help="Number of skills to inject per task via BM25 (0 = disable).")
    p.add_argument("--subset", "-n", type=int, default=None,
                   help="Only process first N records.")
    p.add_argument("--offset", type=int, default=0,
                   help="Skip first N records.")
    p.add_argument("--workers", "-w", type=int, default=1,
                   help="Parallel workers.")
    p.add_argument("--max-turns", type=int, default=MAX_TURNS,
                   help="Maximum conversation turns (default 5).")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--env-timeout", type=int, default=300,
                   help="KernelGym server-side timeout per evaluation (seconds).")
    p.add_argument("--resume", action="store_true",
                   help="Skip already-done source_uuids found in --output.")
    p.add_argument("--resume-from", nargs="*", default=[],
                   help="Additional JSONL files to load done uuids from.")
    return p.parse_args()


def main():
    args = parse_args()

    # ── OpenAI client ─────────────────────────────────────────────────────
    client_kwargs: dict = {"api_key": args.api_key or "EMPTY"}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    # ── KernelGym env client ──────────────────────────────────────────────
    env_client = KernelEnvClient(
        server_url=args.server,
        proxy_url=args.proxy_url,
        timeout=args.env_timeout,
    )
    if args.proxy_url:
        print(f"Proxy mode: requests → {args.proxy_url}  (X-TARGET-HOST: {args.server})")
    else:
        print(f"Direct mode: requests → {args.server}")

    # ── Load data ─────────────────────────────────────────────────────────
    print(f"Loading data from {args.data} ...")
    df = pd.read_parquet(args.data)
    print(f"Loaded {len(df)} records.")

    # Convert to list of dicts, normalizing field names across data formats
    records = []
    for _, row in df.iterrows():
        rec = {}
        for col in df.columns:
            val = row[col]
            # Convert numpy arrays to Python lists/dicts
            if isinstance(val, np.ndarray):
                val = val.tolist()
            elif hasattr(val, "item"):
                val = val.item()
            rec[col] = val

        # ── Normalize: original_python_code ──────────────────────────────
        # Support: original_python_code | py_code | reward_model.ground_truth
        if "original_python_code" not in rec or not rec["original_python_code"]:
            if "py_code" in rec and rec["py_code"]:
                rec["original_python_code"] = rec["py_code"]
            elif "reward_model" in rec:
                rm = rec["reward_model"]
                if isinstance(rm, str):
                    try:
                        rm = json.loads(rm)
                    except Exception:
                        rm = {}
                if isinstance(rm, dict) and rm.get("ground_truth"):
                    rec["original_python_code"] = rm["ground_truth"]

        # ── Normalize: entry_point ────────────────────────────────────────
        # Support: entry_point | reward_model.entry_point | default "Model"
        if "entry_point" not in rec or not rec["entry_point"]:
            rm = rec.get("reward_model", {})
            if isinstance(rm, str):
                try:
                    rm = json.loads(rm)
                except Exception:
                    rm = {}
            rec["entry_point"] = (rm.get("entry_point") if isinstance(rm, dict) else None) or "Model"

        # ── Normalize: uuid ───────────────────────────────────────────────
        if "uuid" not in rec or rec["uuid"] is None:
            rec["uuid"] = uuid4().hex

        records.append(rec)

    records = records[args.offset:]
    if args.subset is not None:
        records = records[:args.subset]
    print(f"Records after offset/subset: {len(records)}")

    # ── Resume ────────────────────────────────────────────────────────────
    if args.resume or args.resume_from:
        resume_paths = list(args.resume_from or [])
        if args.resume:
            resume_paths.append(args.output)
        done_uuids = load_done_uuids(resume_paths)
        before = len(records)
        records = [r for r in records if r.get("uuid") not in done_uuids]
        print(f"Resume: skipped {before - len(records)}, {len(records)} remaining.")

    if not records:
        print("Nothing to do.")
        return

    # ── BM25 skill retriever ──────────────────────────────────────────────
    # Load skill library and build BM25 index over skill content.
    # Query at inference time uses original_python_code of the task.
    retriever: Optional[SkillBM25Retriever] = None
    if args.bm25_top_k > 0:
        if not _HAS_BM25:
            print("[warn] rank_bm25 not installed — BM25 disabled. Run: pip install rank-bm25")
        else:
            skills = load_skill_library(args.skill_lib)
            if skills:
                retriever = SkillBM25Retriever(skills)
                print(f"BM25 index built over {len(skills)} skills from {args.skill_lib}, top_k={args.bm25_top_k}.")
            else:
                print(f"[warn] No skills loaded from {args.skill_lib} — BM25 disabled.")

    # ── Output ────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out_mode = "a" if (args.resume or args.resume_from) else "w"

    stats = {"ok": 0, "error": 0, "total_turns": 0}

    def _process(rec):
        return run_multiturn(
            record=rec,
            client=client,
            env_client=env_client,
            model=args.model,
            retriever=retriever,
            bm25_top_k=args.bm25_top_k,
            max_turns=args.max_turns,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
        )

    print(
        f"\nStarting sampling: model={args.model} "
        f"max_turns={args.max_turns} workers={args.workers} "
        f"bm25_top_k={args.bm25_top_k}\n"
    )

    with open(args.output, out_mode) as fout:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_process, r): r for r in records}
            done_count = 0
            for future in as_completed(futures):
                done_count += 1
                rec = futures[future]
                source_uuid = rec.get("uuid", "?")
                try:
                    result = future.result()
                except Exception:
                    tb = traceback.format_exc()
                    print(f"  [error] uuid={source_uuid}:\n{tb}")
                    result = None

                if result is None:
                    stats["error"] += 1
                    continue

                msgs = result.get("messages", [])
                num_msgs = len(msgs)
                speedups = result.get("round_speedups", [])
                best_speedup = max(speedups) if speedups else 0.0

                # Only save if we have the expected 10 messages (5 full turns)
                if num_msgs == 2 * args.max_turns:
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    fout.flush()
                    stats["ok"] += 1
                    stats["total_turns"] += len(speedups)
                    print(
                        f"  [{done_count}/{len(records)}] uuid={source_uuid} "
                        f"msgs={num_msgs} turns={len(speedups)} "
                        f"best_speedup={best_speedup:.4f}x  [saved]"
                    )
                else:
                    # Partial conversation — save anyway with a flag
                    result["partial"] = True
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    fout.flush()
                    stats["ok"] += 1
                    print(
                        f"  [{done_count}/{len(records)}] uuid={source_uuid} "
                        f"msgs={num_msgs} (partial) turns={len(speedups)} "
                        f"best_speedup={best_speedup:.4f}x  [saved-partial]"
                    )

    print("\n=== Summary ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
