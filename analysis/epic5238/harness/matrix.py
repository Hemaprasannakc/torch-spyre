"""Coverage matrix for the stick-reshape pivot.

Runs every case on Spyre against an fp32 CPU reference and prints one line each:
  OK  <id> ...   values match
  BAD <id> ...   compiled but WRONG VALUES
  ERR <id> ...   raised (first line of the exception)

Run against plain tip to get the baseline, then against the pivot tree.
Stick size is dtype-dependent: 64 elements for fp16/bf16, 32 for fp32, so the
same shape can be unaligned in one dtype and aligned in another.
"""

import sys

import torch
import torch_spyre  # noqa: F401

torch.manual_seed(0)
ONLY = set(sys.argv[1:])
r = torch.randn

F16 = (torch.float16,)
ALL3 = (torch.float16, torch.bfloat16, torch.float32)


def run(cid, cat, desc, f, *args, dtype, compiled=True, note=""):
    if ONLY and cid not in ONLY and cat not in ONLY:
        return
    torch._dynamo.reset()
    xs = [a.to(dtype) if a.is_floating_point() else a for a in args]
    ref = f(*[x.float() if x.is_floating_point() else x for x in xs])
    mode = "cmp" if compiled else "eag"
    label = f"{cid:<5} {cat:<2} {str(dtype)[6:]:<9} {mode} {desc}"
    try:
        g = torch.compile(f, dynamic=False) if compiled else f
        out = g(*[x.to("spyre") for x in xs])
        outs = out if isinstance(out, (tuple, list)) else (out,)
        refs = ref if isinstance(ref, (tuple, list)) else (ref,)
        err = max(
            (o.cpu().float() - rr.float()).abs().max().item()
            / max(1.0, rr.float().abs().max().item())
            for o, rr in zip(outs, refs)
        )
        print(f"{'OK ' if err < 1e-2 else 'BAD'} {label} rel={err:.1e} {note}", flush=True)
    except Exception as e:
        first = [ln for ln in str(e).splitlines() if ln.strip()][:1]
        print(f"ERR {label} {type(e).__name__}: {first[0][:110] if first else ''}", flush=True)


def pw(t):
    return t * 1.0


# ---------------------------------------------------------------- A. split
SPLIT = [
    ("A1", (1, 64, 25600), (1, 64, 160, 160), "W=160 2.5 sticks [YOLO #1353]"),
    ("A2", (1, 128, 6400), (1, 128, 80, 80), "W=80 1.25 sticks"),
    ("A3", (1, 32, 4225), (1, 32, 65, 65), "W=65 just over a stick"),
    ("A4", (2, 16, 961), (2, 16, 31, 31), "W=31 prime, sub-stick"),
    ("A5", (4, 8, 289), (4, 8, 17, 17), "W=17 prime"),
    ("A6", (1, 8, 16129), (1, 8, 127, 127), "W=127 prime ~2 sticks"),
    ("A7", (2, 4, 529), (2, 4, 23, 23), "W=23 prime"),
    ("A8", (3, 1001), (3, 7, 11, 13), "7*11*13 primes, rank+2"),
    ("A9", (2, 6, 210), (2, 6, 5, 6, 7), "rank+2, inner 7"),
]
for cid, src, dst, desc in SPLIT:
    for dt in (ALL3 if cid in ("A1", "A4") else F16):
        run(cid, "A", f"{src}->{dst} *1 {desc}",
            lambda t, d=dst: pw(t.reshape(*d)), r(*src), dtype=dt)

# ---------------------------------------------------------------- B. merge
MERGE = [
    ("B1", (1, 64, 160, 160), (1, 64, 25600), "merge W=160 [silent wrong #5320]"),
    ("B2", (1, 32, 8, 131), (1, 32, 1048), "131 prime"),
    ("B3", (1, 2, 4, 8, 7), (1, 8, 56), "[#5214/#5198]"),
    ("B4", (1, 2, 4, 8, 4, 7), (1, 8, 224), "6-dim [exposed #5804 bug]"),
    ("B5", (2, 16, 31, 31), (2, 16, 961), "31 prime"),
    ("B6", (4, 8, 17, 17), (4, 8, 289), "17 prime"),
    ("B7", (3, 7, 11, 13), (3, 1001), "primes -> rank 2"),
]
for cid, src, dst, desc in MERGE:
    for dt in (ALL3 if cid in ("B1", "B4") else F16):
        run(cid, "B", f"{src}->{dst} *1 {desc}",
            lambda t, d=dst: pw(t.reshape(*d)), r(*src), dtype=dt)

# ------------------------------------------- C. must be left alone (guard)
LEAVE = [
    ("C1", (1, 32, 16384), (1, 32, 128, 128), "128 = 2 sticks"),
    ("C2", (1, 32, 128, 128), (1, 32, 16384), "aligned merge"),
    ("C3", (1, 64, 4096), (1, 64, 64, 64), "64 = 1 stick"),
    ("C4", (4, 8, 160), (32, 160), "stick dim untouched"),
    ("C5", (64,), (8, 8), "one stick, tiles evenly"),
    ("C6", (8, 8), (64,), "reverse of C5"),
]
for cid, src, dst, desc in LEAVE:
    for dt in (ALL3 if cid in ("C1",) else F16):
        run(cid, "C", f"{src}->{dst} *1 {desc}",
            lambda t, d=dst: pw(t.reshape(*d)), r(*src), dtype=dt)
# C7: identical shape to A1 but fp32 stick is 32, so 160 IS aligned
run("C7", "C", "(1,32,25600)->(1,32,160,160) *1 fp32: 160 = 5 sticks, ALIGNED",
    lambda t: pw(t.reshape(1, 32, 160, 160)), r(1, 32, 25600), dtype=torch.float32)

# ------------------------------------------------- D. no spare dim -> CPU
NOSPARE = [
    ("D1", (8, 65), (520,), "flatten all [#5197]"),
    ("D2", (520,), (8, 65), "split all"),
    ("D3", (7, 13), (91,), "primes, flatten all"),
    ("D4", (1, 3, 7), (1, 21), "only unit dim spare"),
]
for cid, src, dst, desc in NOSPARE:
    run(cid, "D", f"{src}->{dst} *1 {desc}",
        lambda t, d=dst: pw(t.reshape(*d)), r(*src), dtype=torch.float16)

# ------------------------------------------------- E. pivot dim selection
PIVOTSEL = [
    ("E1", (100, 3, 7), (100, 21), "pivot = dim0 (100)"),
    ("E2", (2, 100, 3, 7), (2, 100, 21), "pivot = dim1, largest"),
    ("E3", (8, 8, 7), (8, 56), "tie -> innermost"),
]
for cid, src, dst, desc in PIVOTSEL:
    run(cid, "E", f"{src}->{dst} *1 {desc}",
        lambda t, d=dst: pw(t.reshape(*d)), r(*src), dtype=torch.float16)

# ---------------------------------------------------------------- F. dtypes
for dt in (torch.int32, torch.int64):
    run("F4", "F", f"(1,64,25600)->(1,64,160,160) +1 int path",
        lambda t: t.reshape(1, 64, 160, 160) + 1,
        torch.randint(0, 100, (1, 64, 25600), dtype=dt), dtype=dt)
# fp32 widths that are NOT multiples of the 32-element fp32 stick
for cid, src, dst, desc in (
    ("F6", (1, 32, 1600), (1, 32, 40, 40), "fp32 W=40, 1.25 fp32 sticks"),
    ("F7", (1, 32, 4225), (1, 32, 65, 65), "fp32 W=65, 2.03 fp32 sticks"),
):
    run(cid, "F", f"{src}->{dst} *1 {desc}",
        lambda t, d=dst: pw(t.reshape(*d)), r(*src), dtype=torch.float32)

# ------------------------------------------------------------- G. consumers
A1S, A1D = (1, 64, 25600), (1, 64, 160, 160)
run("G1", "G", "A1 then sum(-1)", lambda t: t.reshape(*A1D).sum(-1), r(*A1S), dtype=torch.float16)
run("G2", "G", "A1 then sum(1) non-last dim", lambda t: t.reshape(*A1D).sum(1), r(*A1S), dtype=torch.float16)
run("G3", "G", "A1 as direct graph output", lambda t: t.reshape(*A1D), r(*A1S), dtype=torch.float16)
run("G4", "G", "chained reshape->reshape",
    lambda t: pw(t.reshape(*A1D).reshape(1, 64, 25600)), r(*A1S), dtype=torch.float16)
run("G5", "G", "B3 then cat on last dim",
    lambda a, b: torch.cat([a.reshape(1, 8, 56), b.reshape(1, 8, 56)], -1) * 1.0,
    r(1, 2, 4, 8, 7), r(1, 2, 4, 8, 7), dtype=torch.float16)
run("G6", "G", "reshape feeding matmul",
    lambda a, b: torch.matmul(a.reshape(8, 56), b), r(1, 2, 4, 8, 7), r(56, 16),
    dtype=torch.float16)

# ----------------------------------------------------------- H. op spelling
run("H1", "H", "A1 via .view()", lambda t: pw(t.view(*A1D)), r(*A1S), dtype=torch.float16)
run("H2", "H", "B1 via .flatten(2)", lambda t: pw(t.flatten(2)), r(*A1D), dtype=torch.float16)
run("H3", "H", "A1 via .unflatten(-1,(160,160))",
    lambda t: pw(t.unflatten(-1, (160, 160))), r(*A1S), dtype=torch.float16)
run("H4", "H", "torch.reshape(x, ...)", lambda t: pw(torch.reshape(t, A1D)), r(*A1S), dtype=torch.float16)
run("H5", "H", "B1 via reshape(-1) inferred", lambda t: pw(t.reshape(1, 64, -1)), r(*A1D), dtype=torch.float16)

# -------------------------------------------------------- I. must be skipped
run("I1", "I", "non-contiguous: transpose then reshape",
    lambda t: pw(t.transpose(1, 2).reshape(1, 64, 25600)), r(1, 64, 160, 160), dtype=torch.float16)
run("I2", "I", "A1 in EAGER mode", lambda t: pw(t.reshape(*A1D)), r(*A1S),
    dtype=torch.float16, compiled=False)
run("I3", "I", "B3 in EAGER mode", lambda t: pw(t.reshape(1, 8, 56)), r(1, 2, 4, 8, 7),
    dtype=torch.float16, compiled=False)

print("MATRIX DONE", flush=True)
