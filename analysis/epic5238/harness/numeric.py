"""Numerical sub-issues of #5238: tip behaviour with each issue's own tolerances.
Uses several seeds, because these errors depend on the data."""
import torch, torch_spyre  # noqa
ATOL = {torch.float16: 1e-2, torch.bfloat16: 2e-2, torch.float32: 1e-5}
def run(i, name, f, shape, dtypes, mode):
    for dt in dtypes:
        bad = 0; worst = 0.0; err = None; rat = 0.0
        for seed in range(8):
            torch._dynamo.reset(); torch.manual_seed(seed)
            x = torch.randn(shape).to(dt)
            ref = f(x.float()) ; 
            try:
                g = torch.compile(f, dynamic=False) if mode == "c" else f
                o = g(x.to("spyre")).cpu().float()
                d = (o - ref).abs()
                tol = ATOL[dt] + 1e-2 * ref.abs()
                worst = max(worst, d.max().item()); rat = max(rat, (d / tol).max().item())
                bad += int((d > tol).any())
            except Exception as e:
                err = type(e).__name__ + ": " + str(e).splitlines()[0][:90]; break
        print(f"#{i} {name} [{str(dt)[6:]} {mode}] " + (f"ERR {err}" if err else f"seeds_failing={bad}/8 worst_abs={worst:.4g} x_tol={rat:.2f}"), flush=True)
S = (torch.float16, torch.bfloat16)
A3 = S + (torch.float32,)
for m in "ce":
    run(5299, "narrow(3,0,65).sum", lambda t: t.narrow(3,0,65).sum(-1), (1,16,32,256), S, m)
    run(5300, "cat dim1 sum", lambda t: torch.cat([t,t+1],1).sum(-1), (1,33,256), S, m)
    run(5302, "(65,256).t().sum", lambda t: t.t().sum(-1), (65,256), S, m)
    run(5302, "(33,70).t().sum", lambda t: t.t().sum(-1), (33,70), S, m)
    run(5317, "unsqueeze(2).sum", lambda t: t.unsqueeze(2).sum(-1), (2,8,32,65), S, m)
    run(5319, "(4,32,70).transpose(1,2).sum", lambda t: t.transpose(1,2).sum(-1), (4,32,70), S, m)
    run(5320, "reshape(1,32,1032)*1", lambda t: t.reshape(1,32,1032)*1.0, (1,32,8,129), A3, m)
    run(5320, "reshape(1,32,1032).sum", lambda t: t.reshape(1,32,1032).sum(-1), (1,32,8,129), A3, m)
    run(5321, "(1,8,1,256)->(8,256).sum", lambda t: t.reshape(8,256).sum(-1), (1,8,1,256), S, m)
    run(5322, "(1,16,32,65).select(1,0).sum", lambda t: t.select(1,0).sum(-1), (1,16,32,65), S, m)
    run(5374, "squeeze.sum", lambda t: t.squeeze().sum(-1), (2,1,32,1,256), S, m)
    run(5377, "(1,32,128).narrow(2,0,70).sum", lambda t: t.narrow(2,0,70).sum(-1), (1,32,128), S, m)
    run(5378, "split(2,dim=2) stack sum", lambda t: torch.stack([c.sum(-1) for c in t.split(2,dim=2)]), (1,2,4,8,256), S, m)
print("DONE")
