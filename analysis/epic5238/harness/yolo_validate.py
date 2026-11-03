#!/usr/bin/env python3
"""Full YOLO26s on Spyre: does it compile, and are the numbers right?

Also records every reshape our pass rewrote, and which path each took,
so we can say exactly what the fix is doing in a real model.
"""
import logging
import sys
import traceback

import torch
import torch_spyre  # noqa: F401
from ultralytics import YOLO

REWRITES = []


def install_probe():
    try:
        from torch_spyre._inductor import stick_reshape_pivot as srp
    except Exception as e:
        print(f"[probe] pass not present ({e}); this is the baseline tree")
        return False

    class H(logging.Handler):
        def emit(self, record):
            msg = record.getMessage()
            if "stick_reshape_pivot" in msg:
                REWRITES.append(msg)

    srp.logger.addHandler(H())
    srp.logger.setLevel(logging.INFO)
    print("[probe] installed on stick_reshape_pivot.logger")
    return True


@torch.inference_mode()
def main():
    have_pass = install_probe()
    torch.manual_seed(0)
    x_cpu = torch.rand(1, 3, 640, 640)

    model = YOLO("yolo26s.pt", task="detect").model.cpu().float().eval()
    model.fuse()
    head = model.model[-1]
    head.export = True
    head.dynamic = False
    ref = model(x_cpu)                      # fp32 CPU reference, also builds anchors
    ref_t = ref[0] if isinstance(ref, (tuple, list)) else ref

    model = model.half().to("spyre")
    for module in model.modules():
        for name in ("anchors", "strides", "stride"):
            v = getattr(module, name, None)
            if isinstance(v, torch.Tensor) and v.numel():
                setattr(module, name, v.to(device="spyre", dtype=torch.float16))

    x = x_cpu.half().to("spyre")
    compiled = torch.compile(model, backend="inductor", dynamic=False, fullgraph=True)

    rc = 0
    try:
        out = compiled(x)
        out_t = out[0] if isinstance(out, (tuple, list)) else out
        out_t = out_t.cpu().float()
        print(f"RESULT compiled OK, output shape {tuple(out_t.shape)}")
        finite = bool(torch.isfinite(out_t).all())
        print(f"RESULT finite={finite}")
        if out_t.shape == ref_t.shape:
            denom = max(1.0, ref_t.abs().max().item())
            err = (out_t - ref_t).abs().max().item() / denom
            print(f"RESULT vs CPU fp32 reference: rel_err={err:.3e} "
                  f"({'MATCH' if err < 5e-2 else 'MISMATCH'})")
        else:
            print(f"RESULT shape differs from reference {tuple(ref_t.shape)}; "
                  "skipping value comparison")
    except Exception:
        rc = 1
        print("RESULT compile/run FAILED")
        tb = traceback.format_exc()
        print("---- full exception ----")
        for ln in tb.splitlines():
            print("   " + ln[:200])
        print("---- end exception ----")

    print(f"\n=== reshapes our pass handled: {len(REWRITES)} ===")
    piv = [m for m in REWRITES if "pivot on dim" in m]
    cpu = [m for m in REWRITES if "reshape_via_cpu" in m]
    print(f"    on device (pivot): {len(piv)}")
    print(f"    host fallback    : {len(cpu)}")
    seen = {}
    for m in REWRITES:
        key = m.split("stick_reshape_pivot: ", 1)[-1]
        key = " ".join(key.split()[1:])      # drop the node name
        seen[key] = seen.get(key, 0) + 1
    for k, n in sorted(seen.items(), key=lambda kv: -kv[1]):
        print(f"    {n:3d} x  {k}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
