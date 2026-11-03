import sys,torch,torch_spyre
s=eval(sys.argv[1]); 
x=torch.randn(*s).to(torch.float32)
f=lambda t: tuple(p*1.0 for p in t.split(33,dim=-1))
o=torch.compile(f,dynamic=False)(x.to("spyre")); r=f(x)
print("OK" if all(torch.allclose(a.cpu(),b,atol=1e-4) for a,b in zip(o,r)) else "BAD", s)
