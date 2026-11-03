"""Is a size-1 leading dim enough to avoid the rank-2-intermediate bug?"""
import torch, torch_spyre  # noqa
from torch_spyre._inductor import config as sc
torch.manual_seed(0)

def t(name, f, shape, dtype=torch.float16, on_device=True):
    torch._dynamo.reset()
    x = torch.randn(*shape, dtype=dtype); ref = f(x.float())
    try:
        with sc.patch(stick_reshape_on_device=on_device):
            out = torch.compile(f, dynamic=False)(x.to("spyre"))
        e = (out.cpu().float()-ref).abs().max().item()/max(1.0, ref.abs().max().item())
        print(f"{'OK ' if e<1e-2 else 'BAD'} {name:<46} rel={e:.1e}")
    except Exception as ex:
        m=[l for l in str(ex).splitlines() if l.strip()][:1]
        print(f"ERR {name:<46} {m[0][:65] if m else ''}")

print("--- does a UNIT leading dim avoid it? (p=1, rest=[1]) ---")
for P in (65, 100, 128, 200):
    t(f"(1,{P},3,7)->(1,{P},21)", lambda x,P=P: x.reshape(1,P,21)*1.0, (1,P,3,7))
print("--- confirm the CPU path is correct for the same shapes ---")
for P in (100, 128):
    t(f"({P},3,7)->({P},21)  CPU path", lambda x,P=P: x.reshape(P,21)*1.0, (P,3,7),
      on_device=False)
print("--- split direction, same p=0 multi-stick pivot ---")
for P in (64, 100):
    t(f"({P},21)->({P},3,7)  split, p=0", lambda x,P=P: x.reshape(P,3,7)*1.0, (P,21))
print("--- is it the transpose itself? plain permute+clone, no reshape ---")
for P in (64, 100):
    t(f"(21,{P}) transpose+clone  P={P}", lambda x: x.transpose(0,1).clone()*1.0, (21,P))
