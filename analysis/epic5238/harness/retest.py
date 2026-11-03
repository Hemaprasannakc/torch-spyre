"""Retest of epic #5238 sub-issues using the cases written in each issue.

Each case runs on Spyre and is compared to an fp32 CPU reference.
Prints OK / BAD (wrong values) / ERR (exception, first line).
"""

import sys

import torch
import torch_spyre  # noqa: F401

torch.manual_seed(0)
DTYPES = (torch.float16, torch.bfloat16, torch.float32)
ONLY = set(sys.argv[1:])


def run(issue, name, f, *args, dtype, compiled=True):
    if ONLY and str(issue) not in ONLY:
        return
    torch._dynamo.reset()
    xs = [a.to(dtype) for a in args]
    ref = f(*[x.float() for x in xs])
    mode = "compiled" if compiled else "eager"
    label = f"#{issue} {name} [{str(dtype)[6:]} {mode}]"
    try:
        g = torch.compile(f, dynamic=False) if compiled else f
        out = g(*[x.to("spyre") for x in xs])
        outs = out if isinstance(out, (tuple, list)) else (out,)
        refs = ref if isinstance(ref, (tuple, list)) else (ref,)
        err = max(
            (o.cpu().float() - r).abs().max().item() / max(1.0, r.abs().max().item())
            for o, r in zip(outs, refs)
        )
        tag = "OK " if err < 1e-2 else "BAD"
        print(f"{tag} {label}: rel={err:.1e}", flush=True)
    except Exception as e:
        first = [ln for ln in str(e).splitlines() if ln.strip()][:1]
        msg = first[0][:170] if first else ""
        print(f"ERR {label}: {type(e).__name__}: {msg}", flush=True)


def pw(t):
    return t * 1.0


def last(t):
    return t.sum(dim=-1)


CONSUMERS = (("*1", pw), ("sum", last))
r = torch.randn

for dt in DTYPES:
    for cn, c in CONSUMERS:
        # 5197: flatten, both modes
        for mode in (True, False):
            run(5197, f"(8,65).reshape(-1) {cn}", lambda t: c(t.reshape(-1)),
                r(8, 65), dtype=dt, compiled=mode)
        run(5196, f"cat((32,8,16,1)x2) {cn}",
            lambda a, b: c(torch.cat([a, b], dim=-1)),
            r(32, 8, 16, 1), r(32, 8, 16, 1), dtype=dt)
        # 5198 is the eager form of the same reshape as 5214
        run(5198, f"(1,2,4,8,7)->(1,8,56) {cn}", lambda t: c(t.reshape(1, 8, 56)),
            r(1, 2, 4, 8, 7), dtype=dt, compiled=False)
        run(5214, f"(1,2,4,8,7)->(1,8,56) {cn}", lambda t: c(t.reshape(1, 8, 56)),
            r(1, 2, 4, 8, 7), dtype=dt)
        run(5214, f"(1,2,4,8,4,7)->(1,8,224) {cn}", lambda t: c(t.reshape(1, 8, 224)),
            r(1, 2, 4, 8, 4, 7), dtype=dt)
        for mode in (True, False):
            run(5216, f"cat((32,33)x2) {cn}", lambda a, b: c(torch.cat([a, b], dim=-1)),
                r(32, 33), r(32, 33), dtype=dt, compiled=mode)
        for shape, n in (((32, 66), 33), ((4, 32, 66), 33), ((2, 4, 32, 66), 33)):
            run(5216, f"{shape}.split({n}) {cn}",
                lambda t, n=n: tuple(c(p) for p in t.split(n, dim=-1)),
                r(*shape), dtype=dt)
        for shape, n in (((8, 130), 65), ((4, 32, 140), 70),
                         ((1, 16, 32, 130), 65), ((4, 32, 2), 1)):
            run(5383, f"{shape}.split({n}) {cn}",
                lambda t, n=n: tuple(c(p) for p in t.split(n, dim=-1)),
                r(*shape), dtype=dt, compiled=False)
    for mode in (True, False):
        run(5216, "transpose(4,256,65).clone", lambda t: t.transpose(1, 2).clone(),
            r(4, 256, 65), dtype=dt, compiled=mode)
    run(5216, "(256,1408).split(704) stack transpose",
        lambda t: torch.stack([p.transpose(0, 1) for p in t.split(704, dim=-1)]) * 1.0,
        r(256, 1408), dtype=dt)
    for shape in ((1, 16, 32, 33), (1, 2, 4, 8, 16, 13)):
        run(5216, f"cat({shape}x2) *1", lambda a, b: torch.cat([a, b], dim=-1) * 1.0,
            r(*shape), r(*shape), dtype=dt)
    for shape, d0, d1 in (((2, 32, 8, 70), 1, 2), ((1, 8, 1, 32, 65), 1, 3),
                          ((1, 2, 4, 8, 16, 13), 1, 4)):
        run(5214, f"{shape}.transpose({d0},{d1}).sum",
            lambda t, a=d0, b=d1: t.transpose(a, b).sum(dim=-1), r(*shape), dtype=dt)
    run(5219, "scalar->(1,16,32,70).sum", lambda t: t.expand(1, 16, 32, 70).sum(dim=-1),
        torch.tensor(1.0), dtype=dt)
    run(5219, "(16,1,4,16,1)->(16,8,4,16,256).sum",
        lambda t: t.expand(16, 8, 4, 16, 256).sum(dim=-1), r(16, 1, 4, 16, 1), dtype=dt)
    for src, dst in (((70,), (2, 32, 70)), ((1, 33), (4, 33)),
                     ((1, 32, 33), (4, 32, 33)),
                     ((1, 8, 1, 32, 65), (1, 8, 2, 32, 65)),
                     ((1, 1, 33, 33), (2, 8, 33, 33)),
                     ((1, 1, 1, 256), (1, 16, 32, 256)),
                     ((1, 1, 1, 4, 1, 256), (2, 8, 4, 4, 16, 256)),
                     ((1, 2, 1, 4, 1, 256), (8, 2, 4, 4, 16, 256))):
        run(5237, f"{src}->{dst}.sum", lambda t, d=dst: t.expand(*d).sum(dim=-1),
            r(*src), dtype=dt)

    # 1353: the YOLO split (flat -> H,W off a stick boundary) and a stick-dim stride
    for src, dst in (((1, 64, 25600), (1, 64, 160, 160)), ((1, 128, 6400), (1, 128, 80, 80))):
        run(1353, f"{src}->{dst} *1", lambda t, d=dst: t.reshape(*d) * 1.0, r(*src), dtype=dt)
    run(1353, "(1,64,160,160)->flat *1", lambda t: t.reshape(1, 64, 25600) * 1.0,
        r(1, 64, 160, 160), dtype=dt)
    run(1353, "x[..., ::3]*1", lambda t: t[..., ::3] * 1.0, r(4, 192), dtype=dt)
    run(5388, "x[:, ::3, :]+1", lambda t: t[:, ::3, :] + 1, r(2, 97, 256), dtype=dt)
    run(5224, "(1,32,8,129)->(1,32,1032).sum", lambda t: t.reshape(1, 32, 1032).sum(dim=-1),
        r(1, 32, 8, 129), dtype=dt)

print("DONE", flush=True)
