"""Nail the rule: broken iff (no non-unit dim besides the pivot) AND (pivot > stick)."""
import torch, torch_spyre  # noqa
torch.manual_seed(0)
def t(name, f, shape, dtype=torch.float16):
    torch._dynamo.reset()
    x = torch.randn(*shape, dtype=dtype); ref = f(x.float())
    try:
        out = torch.compile(f, dynamic=False)(x.to("spyre"))
        e = (out.cpu().float()-ref).abs().max().item()/max(1.0, ref.abs().max().item())
        print(f"{'OK ' if e<1e-2 else 'BAD'} {name:<44} rel={e:.1e}")
    except Exception as ex:
        m=[l for l in str(ex).splitlines() if l.strip()][:1]
        print(f"ERR {name:<44} {m[0][:60] if m else ''}")

print("--- pivot>stick, NO other non-unit dim  => predict BAD ---")
t("(100,3,5)->(100,15)",     lambda x: x.reshape(100,15)*1.0, (100,3,5))
t("(100,8,7)->(100,56)",     lambda x: x.reshape(100,56)*1.0, (100,8,7))
t("(1,1,100,3,7)->(1,1,100,21)", lambda x: x.reshape(1,1,100,21)*1.0, (1,1,100,3,7))
print("--- pivot>stick, WITH another non-unit dim => predict OK ---")
t("(3,100,3,7)->(3,100,21)", lambda x: x.reshape(3,100,21)*1.0, (3,100,3,7))
t("(100,3,3,7)->(100,3,21)", lambda x: x.reshape(100,3,21)*1.0, (100,3,3,7))
print("--- pivot<=stick, no other non-unit dim  => predict OK ---")
t("(64,8,7)->(64,56)",       lambda x: x.reshape(64,56)*1.0, (64,8,7))
t("(50,3,7)->(50,21)",       lambda x: x.reshape(50,21)*1.0, (50,3,7))
