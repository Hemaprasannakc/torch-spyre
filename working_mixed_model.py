import os
import time
import torch
import torch.nn.functional as F

torch.manual_seed(0)

BATCH, SEQ, IN_DIM, HIDDEN = 1, 128, 256, 256
DTYPE = torch.float16
WARMUP = 3
ITERS = 10


@torch.compile(backend="inductor", fullgraph=True)
def mixed_model(x, w, b):
    # --- native Spyre path ---
    h = F.linear(x, w, b)      # native
    h = F.gelu(h)                # native

    # --- CPU fallback path (redirected via zDLC when the flag is on) ---
    running_sum = torch.cumsum(h, dim=1)   # fallback
    phase = torch.sin(h)                     # fallback

    # --- recombine: native op consuming both fallback outputs ---
    # deliberately does NOT reuse h a third time here - that specific
    # pattern hits a separate, still-open torch-spyre compiler bug
    out = running_sum + phase
    return out


def eager_reference(x, w, b):
    h = F.linear(x, w, b)
    h = F.gelu(h)
    running_sum = torch.cumsum(h, dim=1)
    phase = torch.sin(h)
    return running_sum + phase


x = torch.randn(BATCH, SEQ, IN_DIM, dtype=DTYPE)
w = torch.randn(HIDDEN, IN_DIM, dtype=DTYPE) * 0.05
b = torch.randn(HIDDEN, dtype=DTYPE) * 0.05

expected = eager_reference(x, w, b)

x_s, w_s, b_s = (t.to("spyre") for t in (x, w, b))

t0 = time.perf_counter()
out = mixed_model(x_s, w_s, b_s)
first_call_ms = (time.perf_counter() - t0) * 1000

ok = torch.allclose(out.cpu().float(), expected.float(), rtol=1e-2, atol=1e-2)
max_diff = (out.cpu().float() - expected.float()).abs().max().item()

for _ in range(WARMUP):
    mixed_model(x_s, w_s, b_s)

times = []
for _ in range(ITERS):
    t0 = time.perf_counter()
    mixed_model(x_s, w_s, b_s)
    times.append((time.perf_counter() - t0) * 1000)
times = torch.tensor(times)

flag = os.environ.get("TORCH_SPYRE_ONNX_FALLBACK", "0")
print(f"\n=== mixed_model: native Linear+GELU -> fallback cumsum+sin -> native combine ===")
print(f"ONNX fallback flag: {flag}")
print(f"Shapes: x{tuple(x.shape)} -> h({BATCH},{SEQ},{HIDDEN}) -> out{tuple(expected.shape)}")
print(f"Correctness vs eager reference: {'MATCH' if ok else 'MISMATCH'} (max abs diff: {max_diff:.6f})")
print(f"First call: {first_call_ms:.3f} ms")
print(f"Warm median latency: {times.median():.3f} ms")
print(f"Warm mean latency:   {times.mean():.3f} ms")
