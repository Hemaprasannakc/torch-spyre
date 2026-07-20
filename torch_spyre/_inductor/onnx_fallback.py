# Copyright 2026 The Torch-Spyre Authors.
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

"""Compile ops diverted to ONNX (see passes.divert_unsupported_ops_to_onnx)
via zdlc/onnx-mlir, and execute the result in place of the plain CPU eager
fallback in ops/fallbacks.py.

Gated end-to-end by config.onnx_fallback_enabled (TORCH_SPYRE_ONNX_FALLBACK=1).
With that unset, none of this module is exercised and behavior is identical
to the existing CPU-only fallback path.

Scope: only single-node partitions are registered for runtime redirect.
Multi-node fused partitions are still exported to ONNX (so the artifact
exists and the fusion is visible), but a fused .so expects all of the
cluster's original inputs at once, while ops/fallbacks.py's hook fires once
per individual op dispatch - redirecting those needs replacing the whole
cluster with a single graph node, which this does not yet do.
"""

import os
import subprocess
import sys
import threading

import torch

from . import config
from .logging_utils import get_inductor_logger

logger = get_inductor_logger("onnx_fallback")

_lock = threading.RLock()
_so_by_key: dict[tuple, str] = {}
_sessions: dict[str, object] = {}
_pyruntime_ready = False


def _subprocess_env() -> dict:
    env = os.environ.copy()
    home = config.zdlc_home
    env["PATH"] = home + os.pathsep + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = (
        os.path.join(home, "lib64") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    )
    return env


def _ensure_pyruntime_importable() -> None:
    global _pyruntime_ready
    if _pyruntime_ready:
        return
    if config.zdlc_home not in sys.path:
        sys.path.insert(0, config.zdlc_home)
    lib64 = os.path.join(config.zdlc_home, "lib64")
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if lib64 not in ld_path.split(os.pathsep):
        # Only affects dlopen calls made from this point on in this process;
        # PyRuntimeC's own deps (checked in the POC) don't actually need
        # this, but set it defensively in case a future build does.
        os.environ["LD_LIBRARY_PATH"] = lib64 + os.pathsep + ld_path
    _pyruntime_ready = True


def compile_onnx_to_so(onnx_path: str, name: str, out_dir: str) -> str:
    """Compile onnx_path with zdlc, returning the resulting .so path.

    Raises subprocess.CalledProcessError if compilation fails - callers
    should let this surface rather than silently falling back, since a
    compile failure under an opt-in flag is a bug to fix, not a case to
    quietly paper over.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_base = os.path.join(out_dir, name)
    zdlc_bin = os.path.join(config.zdlc_home, "zdlc")
    cmd = [zdlc_bin, "--O3", "-march=z17", "--mtriple=s390x-ibm-loz", "-o", out_base, onnx_path]
    logger.info("onnx_fallback: compiling %s -> %s.so", onnx_path, out_base)
    subprocess.run(cmd, env=_subprocess_env(), check=True, capture_output=True)
    return out_base + ".so"


def _key_part(a):
    """Reduce one call arg to a hashable, comparable fragment of the cache
    key. Tensors and FX Nodes carrying a FakeTensor both become their
    (shape, dtype) - Nodes so registration (which only has the traced graph,
    not real tensors) produces the same shape of key as a real runtime call.
    Everything else (e.g. `dim=0`) is used as-is, since a baked-in scalar
    arg is part of what the compiled .so is specialized for - two calls that
    only differ in `dim` must not collide on the same artifact.
    """
    if isinstance(a, torch.Tensor):
        return (tuple(a.shape), str(a.dtype))
    if isinstance(a, torch.fx.Node):
        fake = a.meta.get("val")
        if isinstance(fake, torch.Tensor):
            return (tuple(fake.shape), str(fake.dtype))
        return None  # non-tensor node value - can't key on this reliably
    return a


def _cache_key(op, args, kwargs=()) -> tuple:
    kwargs = kwargs.items() if hasattr(kwargs, "items") else kwargs
    arg_parts = tuple(_key_part(a) for a in args)
    kwarg_parts = tuple(sorted((k, _key_part(v)) for k, v in kwargs))
    return (op.name(), arg_parts, kwarg_parts)


def register_compiled_artifact(node, so_path: str) -> None:
    """Record that `so_path` implements `node`'s op for this exact
    shape/dtype/scalar-arg signature. Called by
    passes.divert_unsupported_ops_to_onnx after a successful single-node
    compile, passing the *original* FX node (not the fused submodule's
    placeholder) so the key is built from its real args, including any
    non-tensor ones baked into the traced graph (e.g. `dim`).
    """
    key = _cache_key(node.target, node.args, node.kwargs)
    with _lock:
        _so_by_key[key] = so_path
    logger.info("onnx_fallback: registered %s for %s", so_path, key)


def _get_session(so_path: str):
    session = _sessions.get(so_path)
    if session is None:
        _ensure_pyruntime_importable()
        from PyRuntime import OMExecutionSession

        session = OMExecutionSession(so_path)
        _sessions[so_path] = session
    return session


def try_run(op, args, kwargs):
    import traceback
    print(f"onnx_fallback: try_run CALLED, enabled={config.onnx_fallback_enabled}, op={op}")
    if not config.onnx_fallback_enabled:
        return None
    try:
        key = _cache_key(op, args, kwargs)
        print(f"onnx_fallback: key computed = {key!r}")
        so_path = _so_by_key.get(key)
        if so_path is None:
            print(f"onnx_fallback: MISS; registered keys: {list(_so_by_key.keys())!r}")
            return None

        session = _get_session(so_path)
        print(f"onnx_fallback: EXECUTING via compiled artifact: {so_path}")
        np_inputs = [
            a.detach().cpu().contiguous().numpy()
            for a in args
            if isinstance(a, torch.Tensor)
        ]
        np_outputs = session.run(np_inputs)
        result = tuple(torch.from_numpy(o) for o in np_outputs)
        return result[0] if len(result) == 1 else result
    except Exception:
        print("onnx_fallback: EXCEPTION in try_run:")
        traceback.print_exc()
        return None
