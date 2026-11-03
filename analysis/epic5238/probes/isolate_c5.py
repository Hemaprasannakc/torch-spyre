"""Is C5/C6 a reshape bug, or are narrow rank-2 tensors unsupported anyway?"""
import torch, torch_spyre  # noqa
torch.manual_seed(0)

def t(name, f, *shapes, dtype=torch.float16):
    torch._dynamo.reset()
    xs = [torch.randn(*s, dtype=dtype) for s in shapes]
    ref = f(*[x.float() for x in xs])
    try:
        out = torch.compile(f, dynamic=False)(*[x.to("spyre") for x in xs])
        e = (out.cpu().float()-ref).abs().max().item()/max(1.0, ref.abs().max().item())
        print(f"{'OK ' if e<1e-2 else 'BAD'} {name:<42} rel={e:.1e}")
    except Exception as ex:
        m = [l for l in str(ex).splitlines() if l.strip()][:1]
        print(f"ERR {name:<42} {m[0][:95] if m else ''}")

print("--- NO reshape at all: is the shape itself supported? ---")
t("(8,8) * 1.0        no reshape", lambda a: a*1.0, (8,8))
t("(64,) * 1.0        no reshape", lambda a: a*1.0, (64,))
t("(16,16) * 1.0      no reshape", lambda a: a*1.0, (16,16))
t("(8,64) * 1.0       no reshape", lambda a: a*1.0, (8,64))
t("(64,64) * 1.0      no reshape", lambda a: a*1.0, (64,64))
t("(128,) * 1.0       no reshape", lambda a: a*1.0, (128,))
print("--- the C5/C6 reshapes themselves ---")
t("(64,)->(8,8) *1", lambda a: a.reshape(8,8)*1.0, (64,))
t("(8,8)->(64,) *1", lambda a: a.reshape(64)*1.0, (8,8))
print("--- same reshape, but wider rows (should be fine) ---")
t("(4096,)->(64,64) *1", lambda a: a.reshape(64,64)*1.0, (4096,))
t("(64,64)->(4096,) *1", lambda a: a.reshape(4096)*1.0, (64,64))
print("--- with a batch dim in front ---")
t("(4,64)->(4,8,8) *1", lambda a: a.reshape(4,8,8)*1.0, (4,64))
t("(4,8,8)->(4,64) *1", lambda a: a.reshape(4,64)*1.0, (4,8,8))
