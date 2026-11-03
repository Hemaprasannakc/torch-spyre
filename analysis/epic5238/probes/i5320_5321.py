"""Repros for #5320 and #5321, both modes, to see exactly what our PR covers."""
import torch, torch_spyre  # noqa
TOL = {torch.float16:(1e-2,1e-2), torch.bfloat16:(2e-2,1e-2), torch.float32:(1e-5,1e-2)}

def check(tag, fn, x, dtype, mode):
    ref = fn(x)
    try:
        g = torch.compile(fn, dynamic=False) if mode=="compiled" else fn
        got = g(x.to("spyre")).cpu()
        atol, rtol = TOL[dtype]
        try:
            torch.testing.assert_close(got, ref, atol=atol, rtol=rtol, equal_nan=True)
            return "OK "
        except AssertionError:
            return "BAD"
    except Exception:
        return "ERR"

print("=== #5320: (1,32,8,129) -> (1,32,1032) ===")
for dtype in (torch.float16, torch.bfloat16, torch.float32):
    for cname, fn in (("*1", lambda t: t.reshape(1,32,1032)*1.0),
                      ("sum(-1)", lambda t: t.reshape(1,32,1032).sum(-1))):
        for mode in ("compiled","eager"):
            torch._dynamo.reset(); torch.manual_seed(0)
            x = torch.randn(1,32,8,129, dtype=dtype)
            print(f"  {check('',fn,x,dtype,mode)} {str(dtype)[6:]:<9} {cname:<8} {mode}")

print("=== #5321: its two cases, 6 seeds ===")
for in_shape, new_shape in (((1,8,1,256),(8,256)), ((2,8,32,1),(2,8,32))):
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        res = {}
        for seed in range(6):
            for mode in ("compiled","eager"):
                torch._dynamo.reset(); torch.manual_seed(seed)
                x = torch.randn(in_shape, dtype=dtype)
                fn = lambda t, s=new_shape: t.reshape(s).sum(dim=-1)
                r = check("", fn, x, dtype, mode)
                res[(mode,r)] = res.get((mode,r),0)+1
        summary = "  ".join(f"{m}:{r}x{c}" for (m,r),c in sorted(res.items()))
        print(f"  {in_shape}->{new_shape} {str(dtype)[6:]:<9} {summary}")
