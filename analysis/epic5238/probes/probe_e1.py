"""E1 hypothesis: the pivot dim itself must be stick-friendly.

Pivot dim P becomes the stick dim of the intermediate. Vary P while holding
the rest of the shape fixed, and see which P give wrong values.
Stick = 64 for fp16.
"""
import torch, torch_spyre  # noqa
torch.manual_seed(0)

def t(name, f, shape, dtype=torch.float16):
    torch._dynamo.reset()
    x = torch.randn(*shape, dtype=dtype); ref = f(x.float())
    try:
        out = torch.compile(f, dynamic=False)(x.to("spyre"))
        e = (out.cpu().float()-ref).abs().max().item()/max(1.0, ref.abs().max().item())
        print(f"{'OK ' if e<1e-2 else 'BAD'} {name:<40} rel={e:.1e}")
    except Exception as ex:
        m=[l for l in str(ex).splitlines() if l.strip()][:1]
        print(f"ERR {name:<40} {m[0][:70] if m else ''}")

print("--- vary pivot dim P in (P,3,7)->(P,21), p=0, rank-2 intermediate ---")
for P in (8, 21, 32, 63, 64, 65, 100, 127, 128, 129, 192, 200):
    kind = "aligned" if P % 64 == 0 else ("sub-stick" if P < 64 else "UNALIGNED >1 stick")
    t(f"({P},3,7)->({P},21)  P={P} {kind}", lambda x: x.reshape(x.shape[0], 21)*1.0, (P,3,7))

print("--- same P but with a leading dim, so p=1 (E2 shape) ---")
for P in (100, 65, 200):
    t(f"(2,{P},3,7)->(2,{P},21)  p=1", lambda x,P=P: x.reshape(2, P, 21)*1.0, (2,P,3,7))
