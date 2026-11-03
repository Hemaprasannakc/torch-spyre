"""Does a plain sum, with no view op, miss the same tolerance?"""
import torch,torch_spyre
for shape in ((2,32,256),(32,65),(4,70,32),(1,8,32,256)):
  for dt,atol in ((torch.float16,1e-2),(torch.bfloat16,2e-2)):
    bad=0;rat=0
    for seed in range(8):
        torch.manual_seed(seed); x=torch.randn(shape).to(dt); ref=x.float().sum(-1)
        cpu16=x.sum(-1).float()
        o=(x.to("spyre").clone().sum(-1)).cpu().float()
        d=(o-ref).abs(); tol=atol+1e-2*ref.abs(); bad+=int((d>tol).any()); rat=max(rat,(d/tol).max().item())
    print(shape,str(dt)[6:],f"plain_sum seeds_failing={bad}/8 x_tol={rat:.2f}",flush=True)
