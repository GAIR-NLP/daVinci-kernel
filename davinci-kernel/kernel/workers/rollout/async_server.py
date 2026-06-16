# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import copy
import heapq
import importlib
import logging
import os
import random
import socket
import threading
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
from urllib.parse import urlparse
from uuid import uuid4

import aiohttp
import fastapi
import httpx
import numpy as np
import ray
import torch
import uvicorn
from cachetools import LRUCache
from omegaconf import DictConfig
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from starlette.requests import Request
from verl.protocol import DataProto
from verl.single_controller.ray.base import RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local

from kernel.workers.rollout.vllm_rollout.vllm_async_engine import (
    AsyncvLLMEngine,
    MultiTurnAsyncvLLMEngine,
)

from kernel.workers.rollout.vllm_rollout.vllm_async_engine_multi_iter import (
    MultiIterAsyncvLLMEngine
)

# Skill memory rollout engine (imported lazily to avoid overhead when disabled)
_SkillAwareEngine = None
def _get_skill_aware_engine():
    global _SkillAwareEngine
    if _SkillAwareEngine is None:
        from kernel.workers.rollout.vllm_rollout.vllm_async_engine_skill import (
            SkillAwareMultiIterAsyncvLLMEngine,
        )
        _SkillAwareEngine = SkillAwareMultiIterAsyncvLLMEngine
    return _SkillAwareEngine



logger = logging.getLogger(__file__)


class AsyncLLMEngineManager:
    """AsyncLLMEngineManager manage a group of vllm instances, i.e AsyncvLLMEngine."""

    def __init__(
        self,
        config: DictConfig,
        worker_group: RayWorkerGroup,
        tokenizer,
        reward_fn=None,
        val_reward_fn=None,
        *,
        scheduler_kwargs: Dict[str, Any] = None,
        skill_config=None,   # top-level skill config node (OmegaConf); None = disabled
    ):
        """Initialize AsyncLLMEngineManager.

        Args:
            config: DictConfig, actor_rollout_ref config.
            worker_group: RayWorkerGroup, worker group of AsyncActorRolloutRefWorker.
            scheduler_kwargs: Dict[str, Any], kwargs for chat scheduler.
            skill_config: Optional top-level skill config; passed explicitly because
                          `config` is actor_rollout_ref (a sub-node, not the root config).
        """
        self.config = config
        self.worker_group = worker_group
        self.scheduler_kwargs = scheduler_kwargs if scheduler_kwargs else {}
        self.tokenizer = tokenizer
        self.rollout_tp_size = self.config.rollout.tensor_model_parallel_size
        self.rollout_dp_size = self.worker_group.world_size // self.rollout_tp_size

        # Simple configuration for overall safety timeout
        self.max_timeout = 86400  # Maximum timeout in seconds (~24 hours) as safety net

        # Simple configuration for overall safety timeout
        self.max_timeout = 86400  # Maximum timeout in seconds (~24 hours) as safety net

        workers_info = ray.get(
            [
                worker.__ray_call__.remote(lambda self: ray.get_runtime_context().get_node_id())
                for worker in self.worker_group.workers
            ]
        )
        assert len(workers_info) == self.worker_group.world_size

        self.async_llm_servers = [None] * self.rollout_dp_size

        rollout_backend = self.config.rollout.get("backend", "vllm")
        if rollout_backend in ("openai", "openai_sdk"):
            from kernel.workers.rollout.vllm_rollout.openai_async_engine_multi_iter import (
                AsyncvLLMEngine as OpenAIAsyncEngine,
                MultiIterAsyncvLLMEngine as OpenAIMultiIterEngine,
            )

            engine_class = OpenAIMultiIterEngine if self.config.rollout.multi_turn.enable else OpenAIAsyncEngine
        else:
            engine_class = MultiTurnAsyncvLLMEngine if self.config.rollout.multi_turn.enable else AsyncvLLMEngine
            if self.config.rollout.multi_turn.multi_iteration.enable:
                engine_class = MultiIterAsyncvLLMEngine

        # Skill memory: upgrade to SkillAwareMultiIterAsyncvLLMEngine when enabled.
        # Requires vllm backend + multi_iteration (inherits MultiIterAsyncvLLMEngine).
        # skill_config is passed explicitly from the trainer because `config` here is
        # actor_rollout_ref (a sub-node), not the full config that holds `skill`.
        _skill_enabled = (
            skill_config is not None
            and (
                skill_config.get("enable", False)
                if hasattr(skill_config, "get")
                else getattr(skill_config, "enable", False)
            )
        )
        if _skill_enabled and engine_class is MultiIterAsyncvLLMEngine:
            engine_class = _get_skill_aware_engine()
            print(f"[Skill] AsyncLLMEngineManager: upgraded engine to SkillAwareMultiIterAsyncvLLMEngine")
            # Inject skill config into the actor_rollout_ref config so the engine's
            # __init__ can find it as self.config.skill.
            # (config here is actor_rollout_ref, skill lives at the top level)
            try:
                from omegaconf import open_dict
                with open_dict(config):
                    config.skill = skill_config
                print(f"[Skill] skill config injected into actor_rollout_ref: enable={config.skill.get('enable', False)}")
            except Exception as _e:
                print(f"[Skill] Warning: could not inject skill config into actor_rollout_ref config: {_e}")
                # Fallback: monkey-patch the config object directly
                object.__setattr__(config, "skill", skill_config)

        config.rollout.max_model_len = (
            config.rollout.max_model_len
            if config.rollout.max_model_len
            else config.rollout.prompt_length + config.rollout.response_length
        )

        # Start all server instances, restart if address already in use.
        unready_dp_ranks = set(range(self.rollout_dp_size))
        while len(unready_dp_ranks) > 0:
            servers = {
                rollout_dp_rank: engine_class.options(
                    # make sure AsyncvLLMEngine colocates with its corresponding workers
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=workers_info[rollout_dp_rank * self.rollout_tp_size],
                        soft=False,
                    ),
                    name=f"async_llm_server_{rollout_dp_rank}",
                ).remote(
                    config,
                    self.rollout_dp_size,
                    rollout_dp_rank,
                    self.worker_group.name_prefix,
                    self.tokenizer,
                    reward_fn,
                    val_reward_fn,
                )
                for rollout_dp_rank in unready_dp_ranks
            }

            for rollout_dp_rank, server in servers.items():
                try:
                    # address = ray.get(server.get_server_address.remote())
                    self.async_llm_servers[rollout_dp_rank] = server
                    unready_dp_ranks.remove(rollout_dp_rank)
                except Exception:
                    ray.kill(server)
                    print(f"rollout server {rollout_dp_rank} failed, maybe address already in use, restarting...")

        # All server instances are ready, init AsyncLLM engine.
        ray.get([server.init_engine.remote() for server in self.async_llm_servers])

        assert self.config.rollout.free_cache_engine, "Only free cache engine is supported for now."
        if self.config.rollout.free_cache_engine:
            self.sleep()

    def wake_up(self):
        """Wake up all vllm instances."""
        ray.get([server.wake_up.remote() for server in self.async_llm_servers])

    def sleep(self):
        """Sleep all vllm instances."""
        ray.get([server.sleep.remote() for server in self.async_llm_servers])

    def get_skill_library_size(self) -> int:
        """Return the number of skills in the library (queries rank-0 worker)."""
        if not self.async_llm_servers:
            return 0
        try:
            return int(ray.get(self.async_llm_servers[0].get_skill_library_size.remote()))
        except Exception:
            return 0

    def flush_skill_library(self, global_step: int, start_step: int = 0) -> dict:
        """Collect staged skills from all DP workers, flush on rank-0, then
        broadcast the updated skill list back to every worker in memory.
        Returns metrics dict for trainer logging.
        """
        if not self.async_llm_servers:
            return {}
        try:
            # Step 1: drain staged skills from every worker
            drain_refs = [s.drain_staged_skills.remote()
                          for s in self.async_llm_servers]
            all_staged_lists = ray.get(drain_refs)
            extra = []
            for skills in all_staged_lists:
                extra.extend(skills)

            # Step 2: flush on rank-0 → returns (metrics, updated_skills)
            result = ray.get(
                self.async_llm_servers[0].flush_skill_library.remote(
                    global_step, start_step, extra
                )
            )
            if isinstance(result, tuple):
                metrics, updated_skills = result
            else:
                # flush was skipped (returned metrics-only dict)
                metrics, updated_skills = result, None

            # Step 3: broadcast updated skill list to all other workers
            if updated_skills is not None and len(self.async_llm_servers) > 1:
                broadcast_refs = [
                    s.set_skill_library.remote(updated_skills, global_step)
                    for s in self.async_llm_servers[1:]
                ]
                ray.get(broadcast_refs)

            return metrics or {}
        except Exception as e:
            print(f"[Skill] AsyncLLMEngineManager.flush_skill_library failed: {e}")
            return {}

    def generate_sequences(self, prompts: DataProto, **sampling_params) -> DataProto:
        """Generate multiple sequences in parallel via chat scheduler."""

        assert self.config.rollout.free_cache_engine, "Only free cache engine is supported for now."
        if self.config.rollout.free_cache_engine:
            self.wake_up()

        chunkes = prompts.chunk(len(self.async_llm_servers))
        outputs = ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.async_llm_servers, chunkes, strict=True)
            ]
        )
        # filter out output which is None
        outputs = [output for output in outputs if output is not None]
        if len(outputs) == 0:
            return None

        # Align tensor keys across DP workers before concat.
        # Workers that had skill selection/summary rows include extra keys
        # (e.g. token_level_rewards) that pure-policy workers don't have.
        if len(outputs) > 1:
            import torch
            all_keys: set = set()
            for op in outputs:
                if op.batch is not None:
                    all_keys.update(op.batch.keys())
            for op in outputs:
                if op.batch is None:
                    continue
                for key in all_keys:
                    if key not in op.batch:
                        ref = next(
                            (q.batch[key] for q in outputs
                             if q.batch is not None and key in q.batch),
                            None,
                        )
                        if ref is None:
                            continue
                        n = len(op)
                        if ref.dim() == 1:
                            op.batch[key] = torch.zeros(
                                n, dtype=ref.dtype, device=ref.device)
                        else:
                            op.batch[key] = torch.zeros(
                                n, *ref.shape[1:], dtype=ref.dtype, device=ref.device)

        output = DataProto.concat(outputs)
        if self.config.rollout.free_cache_engine:
            self.sleep()
        return output


class StandaloneVLLMEngineManager:
    """Standalone vLLM manager that does not rely on FSDP rollout workers."""

    def __init__(
        self,
        config: DictConfig,
        tokenizer,
        reward_fn=None,
        val_reward_fn=None,
        *,
        total_gpus: int,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.rollout_tp_size = self.config.rollout.tensor_model_parallel_size
        if total_gpus < 1:
            raise ValueError("standalone_vllm requires total_gpus >= 1")
        if total_gpus % self.rollout_tp_size != 0:
            raise ValueError(
                f"standalone_vllm requires total_gpus ({total_gpus}) divisible by tensor_model_parallel_size "
                f"({self.rollout_tp_size})"
            )
        self.rollout_dp_size = total_gpus // self.rollout_tp_size
        self.world_size = self.rollout_dp_size * self.rollout_tp_size

        rollout_backend = self.config.rollout.get("backend", "vllm")
        if rollout_backend in ("openai", "openai_sdk"):
            from kernel.workers.rollout.vllm_rollout.openai_async_engine_multi_iter import (
                AsyncvLLMEngine as OpenAIAsyncEngine,
                MultiIterAsyncvLLMEngine as OpenAIMultiIterEngine,
            )

            engine_class = OpenAIMultiIterEngine if self.config.rollout.multi_turn.enable else OpenAIAsyncEngine
        else:
            engine_class = MultiTurnAsyncvLLMEngine if self.config.rollout.multi_turn.enable else AsyncvLLMEngine
            if self.config.rollout.multi_turn.multi_iteration.enable:
                engine_class = MultiIterAsyncvLLMEngine

        # Skill memory: upgrade to SkillAwareMultiIterAsyncvLLMEngine when enabled.
        skill_cfg = self.config.get("skill", None)
        if (
            skill_cfg is not None
            and skill_cfg.get("enable", False)
            and engine_class is MultiIterAsyncvLLMEngine
        ):
            engine_class = _get_skill_aware_engine()

        self.config.rollout.max_model_len = (
            self.config.rollout.max_model_len
            if self.config.rollout.max_model_len
            else self.config.rollout.prompt_length + self.config.rollout.response_length
        )

        user = os.environ.get("USER", "user")
        cache_root = f"/tmp/{user}"
        runtime_env = {
            "env_vars": {
                "VERL_VLLM_DISTRIBUTED_BACKEND": "local",
                "XDG_CACHE_HOME": f"{cache_root}/.cache",
                "TORCHINDUCTOR_CACHE_DIR": f"{cache_root}/torchinductor",
            }
        }
        self.async_llm_servers = [
            engine_class.options(
                num_gpus=self.rollout_tp_size,
                runtime_env=runtime_env,
                name=f"standalone_async_llm_server_{rollout_dp_rank}",
            ).remote(
                self.config,
                self.rollout_dp_size,
                rollout_dp_rank,
                "standalone_vllm",
                self.tokenizer,
                reward_fn,
                val_reward_fn,
            )
            for rollout_dp_rank in range(self.rollout_dp_size)
        ]

        ray.get([server.init_engine.remote() for server in self.async_llm_servers])

        assert self.config.rollout.free_cache_engine, "Only free cache engine is supported for now."
        if self.config.rollout.free_cache_engine:
            self.sleep()

    def wake_up(self):
        """Wake up all vllm instances."""
        ray.get([server.wake_up.remote() for server in self.async_llm_servers])

    def sleep(self):
        """Sleep all vllm instances."""
        ray.get([server.sleep.remote() for server in self.async_llm_servers])

    def generate_sequences(self, prompts: DataProto, **sampling_params) -> DataProto:
        """Generate multiple sequences in parallel via chat scheduler."""
        assert self.config.rollout.free_cache_engine, "Only free cache engine is supported for now."
        if self.config.rollout.free_cache_engine:
            self.wake_up()

        chunkes = prompts.chunk(len(self.async_llm_servers))
        outputs = ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.async_llm_servers, chunkes, strict=True)
            ]
        )
        outputs = [output for output in outputs if output is not None]
        if len(outputs) == 0:
            return None

        output = DataProto.concat(outputs)
        if self.config.rollout.free_cache_engine:
            self.sleep()
        return output
