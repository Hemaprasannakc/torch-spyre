"""Control: does a plain fp16 sum over 1032 elements meet #5321-style tolerance at all?
No reshape involved -> isolates accumulation error from any layout bug."""
import torch, torch_spyre  # noqa
for dtype, atol in ((torch.float16,1e-2), (torch.bfloat16,2e-2), (torch.float32,1e-5)):
    torch._dynamo.reset(); torch.manual_seed(0)
    x = torch.randn(1, 32, 1032, dtype=dtype)          # ALREADY flat, no reshape
    ref = x.sum(dim=-1)
    got = torch.compile(lambda t: t.sum(dim=-1), dynamic=False)(x.to("spyre")).cpu()
    try:
        torch.testing.assert_close(got, ref, atol=atol, rtol=1e-2)
        print(f"OK  plain sum(-1), no reshape, {str(dtype)[6:]:<9} meets atol={atol}")
    except AssertionError as e:
        line = [l for l in str(e).splitlines() if "Greatest absolute" in l]
        print(f"BAD plain sum(-1), no reshape, {str(dtype)[6:]:<9} {line[0][:85] if line else ''}")
