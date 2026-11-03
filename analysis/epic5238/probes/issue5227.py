"""Verbatim repro from torch-spyre issue #5227, plus the W<64 and W=80/96 variants it names."""
import torch, torch_spyre, torch.nn.functional as F  # noqa
torch.manual_seed(0)

def t(name, f, shape):
    torch._dynamo.reset()
    x = torch.randn(*shape, dtype=torch.float16)
    ref = f(x.float())
    try:
        out = torch.compile(f, dynamic=False)(x.to("spyre"))
        e = (out.cpu().float()-ref).abs().max().item()/max(1.0, ref.abs().max().item())
        print(f"{'OK ' if e<1e-2 else 'BAD'} {name:<52} rel={e:.1e}")
    except Exception as ex:
        m=[l for l in str(ex).splitlines() if l.strip()][:1]
        print(f"ERR {name:<52} {m[0][:72] if m else ''}")

C = 32
print("--- the issue's own two lines ---")
t("reshape alone (issue says this works)",
  lambda s: s.reshape(1, C, 160, 160), (1, C, 160*160))
t("silu(reshape)  (issue says 5*c1/2)",
  lambda s: F.silu(s.reshape(1, C, 160, 160)), (1, C, 160*160))
print("--- the other widths #5227 names: 96, 80 (>64) and 40 (<64) ---")
for W in (96, 80, 40):
    t(f"silu(reshape) W={W}", lambda s, W=W: F.silu(s.reshape(1, C, W, W)), (1, C, W*W))
