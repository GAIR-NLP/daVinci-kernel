"""KernelBench model loading helpers (toolkit layer)."""

from __future__ import annotations

import importlib.util
import os
import tempfile
from typing import Tuple

import torch
import torch.nn as nn


def load_original_model_and_inputs(
    model_original_src: str, context: dict, entry_point: str = "Model"
) -> Tuple[nn.Module, callable, callable]:
    # Write to a real file so torch.compile can find the source via inspect.getsource
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
        tmp_file.write(model_original_src)
        tempfile_path = tmp_file.name
    try:
        spec = importlib.util.spec_from_file_location("original_module", tempfile_path)
        temp_module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(temp_module)
        except Exception as e:
            print(f"Error in executing original code {e}")
            return None
        get_init_inputs_fn = getattr(temp_module, "get_init_inputs", None)
        get_inputs_fn = getattr(temp_module, "get_inputs", None)
        Model = getattr(temp_module, entry_point, None)
        # also populate context for callers that read from it
        context.update(vars(temp_module))
        return (Model, get_init_inputs_fn, get_inputs_fn)
    finally:
        os.remove(tempfile_path)


def load_custom_model_with_tempfile(model_custom_src: str, entry_point: str = "ModelNew"):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
        tmp_file.write(model_custom_src)
        tempfile_path = tmp_file.name
        temp_file = tmp_file

    spec = importlib.util.spec_from_file_location("temp_module", tempfile_path)
    temp_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(temp_module)

    ModelNew = getattr(temp_module, entry_point)

    return ModelNew, temp_file


def load_custom_model(
    model_custom_src: str, context: dict, build_directory: str = None
) -> nn.Module:
    if build_directory:
        context["BUILD_DIRECTORY"] = build_directory
        model_custom_src = (
            "import os\n" f"os.environ['TORCH_EXTENSIONS_DIR'] = '{build_directory}'\n"
        ) + model_custom_src

    try:
        compile(model_custom_src, "<string>", "exec")
        exec(model_custom_src, context)
    except SyntaxError as e:
        print(f"Syntax Error in custom generated code or Compilation Error {e}")
        return None

    ModelNew = context.get("ModelNew")
    return ModelNew


def graceful_eval_cleanup(
    curr_context: dict,
    device: torch.device,
    tempfile: tempfile.NamedTemporaryFile = None,
):
    del curr_context
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=device)
        torch.cuda.synchronize(device=device)
    if tempfile:
        tempfile.close()
        os.remove(tempfile.name)
