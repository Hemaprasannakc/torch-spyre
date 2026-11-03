"""Do the offset (cat/split) cases work with a manual pivot + rebase_copy?"""
import torch, torch_spyre  # noqa
torch.manual_seed(0)
rb = torch.ops.spyre.rebase_copy

def run(name, f, ref_fn, *xs, dtype):
    torch._dynamo.reset()
    xs = [x.to(dtype) for x in xs]
    ref = ref_fn(*[x.float() for x in xs])
    try:
        out = torch.compile(f, dynamic=False)(*[x.to("spyre") for x in xs]).cpu().float()
        rel = (out - ref).abs().max().item() / max(1.0, ref.abs().max().item())
        print(f"{'OK ' if rel < 1e-2 else 'BAD'} {name} [{str(dtype)[6:]}]: rel={rel:.1e}", flush=True)
    except Exception as e:
        print(f"ERR {name} [{str(dtype)[6:]}]: {type(e).__name__}: {[l for l in str(e).splitlines() if l.strip()][:1]}", flush=True)

for dt in (torch.float16, torch.float32):
    # #5196: cat two (32,8,16,1) on the stick dim
    c196 = lambda a, b: torch.cat([a, b], -1) * 1.0
    p196 = lambda a, b: rb(torch.cat([a.transpose(-1, -2).contiguous(),
                                      b.transpose(-1, -2).contiguous()], -2)).transpose(-1, -2).contiguous() * 1.0
    A, B = torch.randn(32, 8, 16, 1), torch.randn(32, 8, 16, 1)
    run("#5196 cat plain", c196, c196, A, B, dtype=dt)
    run("#5196 cat pivot+rebase", p196, c196, A, B, dtype=dt)
    # #5216: cat (32,33)x2 then sum
    c216 = lambda a, b: torch.cat([a, b], -1).sum(-1)
    p216 = lambda a, b: rb(torch.cat([a.t().contiguous(), b.t().contiguous()], 0)).t().contiguous().sum(-1)
    A, B = torch.randn(32, 33), torch.randn(32, 33)
    run("#5216 cat plain", c216, c216, A, B, dtype=dt)
    run("#5216 cat pivot+rebase", p216, c216, A, B, dtype=dt)
    # #5383 (compiled form): split at 65 on the stick dim, second chunk
    c383 = lambda t: torch.split(t, 65, dim=-1)[1] * 1.0
    p383 = lambda t: rb(t.t().contiguous()[65:, :]).t().contiguous() * 1.0
    T = torch.randn(8, 130)
    run("#5383 split compiled plain", c383, c383, T, dtype=dt)
    run("#5383 split pivot+rebase", p383, c383, T, dtype=dt)
