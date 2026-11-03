"""Log every compute_coordinates call while compiling the miscompiling shape.

Run under two trees and diff: tip+pivot (correct) vs #5804+pivot (wrong values).
"""

import torch
import torch._inductor.config as ind_config

# compile in-process, uncached, or the instrumentation never sees the calls
ind_config.compile_threads = 1
ind_config.force_disable_caches = True

import torch_spyre  # noqa: F401,E402
from torch_spyre._inductor import views  # noqa: E402

torch.manual_seed(0)

_orig = views.compute_coordinates
_seen = []


def traced(size, stride, var_ranges, index, *a, **k):
    try:
        out = _orig(size, stride, var_ranges, index, *a, **k)
    except Exception as e:
        _seen.append((list(size), list(stride), index, f"RAISED {type(e).__name__}: {e}"))
        raise
    _seen.append((list(size), list(stride), index, out))
    return out


views.compute_coordinates = traced
# these import it by name, so the module-level binding must be replaced too
import importlib

for mod_name in (
    "torch_spyre._inductor.pass_utils",
    "torch_spyre._inductor.propagate_layouts",
    "torch_spyre._inductor.padding",
):
    m = importlib.import_module(mod_name)
    if getattr(m, "compute_coordinates", None) is _orig:
        m.compute_coordinates = traced
        print(f"patched {mod_name}")

x = torch.randn(1, 2, 4, 8, 4, 7, dtype=torch.float16)
ref = (x.float().reshape(1, 8, 224) * 1.0)
try:
    f = torch.compile(lambda t: t.reshape(1, 8, 224) * 1.0, dynamic=False)
    out = f(x.to("spyre"))
    err = (out.cpu().float() - ref).abs().max().item()
    print(f"RESULT rel_err={err:.3e}  {'OK' if err < 1e-2 else 'WRONG VALUES'}")
except Exception as e:
    print(f"RESULT raised {type(e).__name__}: {str(e).splitlines()[0][:120]}")

print(f"--- {len(_seen)} compute_coordinates calls ---")
for i, (size, stride, index, out) in enumerate(_seen):
    print(f"[{i}] size={size}")
    print(f"     stride={stride}")
    print(f"     index={index}")
    print(f"     coords={out}")
