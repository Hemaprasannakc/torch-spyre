"""Is G5 our bug, or is cat-on-last-dim independently broken (#5196/#5216)?"""
import torch, torch_spyre  # noqa
torch.manual_seed(0)
def t(name, f, *shapes, dtype=torch.float16):
    torch._dynamo.reset()
    xs=[torch.randn(*s, dtype=dtype) for s in shapes]
    ref=f(*[x.float() for x in xs])
    try:
        out=torch.compile(f, dynamic=False)(*[x.to("spyre") for x in xs])
        e=(out.cpu().float()-ref).abs().max().item()/max(1.0, ref.abs().max().item())
        print(f"{'OK ' if e<1e-2 else 'BAD'} {name:<46} rel={e:.1e}")
    except Exception as ex:
        m=[l for l in str(ex).splitlines() if l.strip()][:1]
        print(f"ERR {name:<46} {m[0][:72] if m else ''}")

print("--- cat alone, NO reshape: is 56-wide cat supported at all? ---")
t("cat((1,8,56)x2,-1)      no reshape", lambda a,b: torch.cat([a,b],-1)*1.0, (1,8,56),(1,8,56))
t("cat((1,8,64)x2,-1)      aligned, no reshape", lambda a,b: torch.cat([a,b],-1)*1.0, (1,8,64),(1,8,64))
t("cat((1,8,128)x2,-1)     aligned, no reshape", lambda a,b: torch.cat([a,b],-1)*1.0, (1,8,128),(1,8,128))
print("--- one pivoted reshape + an ordinary tensor ---")
t("cat(reshape, plain) -1", lambda a,b: torch.cat([a.reshape(1,8,56), b],-1)*1.0,
  (1,2,4,8,7),(1,8,56))
print("--- two pivoted reshapes, cat on a NON-last dim ---")
t("cat(2x reshape, dim=1)", lambda a,b: torch.cat([a.reshape(1,8,56), b.reshape(1,8,56)],1)*1.0,
  (1,2,4,8,7),(1,2,4,8,7))
print("--- two pivoted reshapes, no cat (just add) ---")
t("reshape + reshape (add)", lambda a,b: a.reshape(1,8,56)+b.reshape(1,8,56),
  (1,2,4,8,7),(1,2,4,8,7))
print("--- the G5 case itself ---")
t("cat(2x reshape, -1)  [G5]", lambda a,b: torch.cat([a.reshape(1,8,56), b.reshape(1,8,56)],-1)*1.0,
  (1,2,4,8,7),(1,2,4,8,7))
