"""Which reshapes entirely INSIDE one stick work today? Decides the skip rule."""
import torch, torch_spyre  # noqa
torch.manual_seed(0)
def t(name, f, shape, dtype=torch.float16):
    torch._dynamo.reset()
    x = torch.randn(*shape, dtype=dtype); ref = f(x.float())
    try:
        out = torch.compile(f, dynamic=False)(x.to("spyre"))
        e = (out.cpu().float()-ref).abs().max().item()/max(1.0, ref.abs().max().item())
        print(f"{'OK ' if e<1e-2 else 'BAD'} {name}")
    except Exception as ex:
        m=[l for l in str(ex).splitlines() if l.strip()][:1]
        print(f"ERR {name:<28} {m[0][:80] if m else ''}")
print("--- split 64 (one fp16 stick) every way, row divides the stick ---")
for a,b in ((2,32),(4,16),(8,8),(16,4),(32,2)):
    t(f"(64,)->({a},{b})", lambda x,a=a,b=b: x.reshape(a,b)*1.0, (64,))
print("--- merge back ---")
for a,b in ((2,32),(4,16),(8,8),(16,4),(32,2)):
    t(f"({a},{b})->(64,)", lambda x: x.reshape(64)*1.0, (a,b))
print("--- row does NOT divide the stick (already known broken) ---")
t("(56,)->(8,7)", lambda x: x.reshape(8,7)*1.0, (56,))
print("--- sub-stick but with a spare dim in front ---")
for a,b in ((2,32),(8,8)):
    t(f"(4,64)->(4,{a},{b})", lambda x,a=a,b=b: x.reshape(4,a,b)*1.0, (4,64))
