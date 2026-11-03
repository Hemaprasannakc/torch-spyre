"""Does the host path make a last-dim cat of a reshape work?"""
import torch, torch_spyre  # noqa
from torch_spyre._inductor import config as sc
torch.manual_seed(0)
def t(name, f, *shapes, dtype=torch.float16, on_device=True):
    torch._dynamo.reset()
    xs=[torch.randn(*s, dtype=dtype) for s in shapes]
    ref=f(*[x.float() for x in xs])
    try:
        with sc.patch(stick_reshape_on_device=on_device):
            out=torch.compile(f, dynamic=False)(*[x.to("spyre") for x in xs])
        e=(out.cpu().float()-ref).abs().max().item()/max(1.0, ref.abs().max().item())
        print(f"{'OK ' if e<1e-2 else 'BAD'} {name:<44} rel={e:.1e}")
    except Exception as ex:
        m=[l for l in str(ex).splitlines() if l.strip()][:1]
        print(f"ERR {name:<44} {m[0][:68] if m else ''}")

g5    = lambda a,b: torch.cat([a.reshape(1,8,56), b.reshape(1,8,56)],-1)*1.0
mixed = lambda a,b: torch.cat([a.reshape(1,8,56), b],-1)*1.0
print("--- G5 with HOST path (stick_reshape_on_device=False) ---")
t("cat(2x reshape,-1)  host", g5, (1,2,4,8,7),(1,2,4,8,7), on_device=False)
t("cat(reshape,plain,-1) host", mixed, (1,2,4,8,7),(1,8,56), on_device=False)
print("--- same, ON DEVICE (for contrast) ---")
t("cat(2x reshape,-1)  device", g5, (1,2,4,8,7),(1,2,4,8,7), on_device=True)
t("cat(reshape,plain,-1) device", mixed, (1,2,4,8,7),(1,8,56), on_device=True)
print("--- does an ALIGNED reshape + last-dim cat work on device? ---")
t("cat(2x aligned reshape,-1)",
  lambda a,b: torch.cat([a.reshape(1,8,128), b.reshape(1,8,128)],-1)*1.0,
  (1,2,4,8,16),(1,2,4,8,16))
