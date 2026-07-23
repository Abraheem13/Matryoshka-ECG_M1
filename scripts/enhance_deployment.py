import os,sys,yaml,time,pickle,numpy as np,torch
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.mrl_ecg_model import MatryoshkaECGModel

DEVICES={'Smartwatch':{'desc':'ARM Cortex-M4','mflops':32,'power_mw':15,'dim':16},'Fitness Band':{'desc':'ARM Cortex-M7','mflops':240,'power_mw':100,'dim':32},'Smartphone':{'desc':'Snapdragon 888','mflops':50000,'power_mw':5000,'dim':128},'Edge Gateway':{'desc':'Jetson Nano','mflops':472000,'power_mw':10000,'dim':256},'Clinical Server':{'desc':'NVIDIA T4','mflops':8100000,'power_mw':70000,'dim':512}}

with open('configs/mrl_resnet1d.yaml') as f: config=yaml.safe_load(f)
device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
nesting_dims=config['mrl']['nesting_dims']
model=MatryoshkaECGModel(config).to(device);model.eval()
import glob
ckpt=torch.load(glob.glob('results/checkpoints/*mrl*_best.pt')[0],map_location=device,weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
x=torch.randn(1,12,1000).to(device)
with torch.no_grad():
    for _ in range(10): model.get_embedding(x)
times=[]
with torch.no_grad():
    for _ in range(50):
        t0=time.perf_counter();emb=model.get_embedding(x)
        if device.type=='mps':torch.mps.synchronize()
        times.append(time.perf_counter()-t0)
bb_ms=np.mean(times)*1000
dim_ms={}
emb=model.get_embedding(x)
for dim in nesting_dims:
    ts=[]
    with torch.no_grad():
        for _ in range(50):
            t0=time.perf_counter();_=model.head.get_predictions(emb,dim=dim)
            if device.type=='mps':torch.mps.synchronize()
            ts.append(time.perf_counter()-t0)
    dim_ms[dim]=np.mean(ts)*1000

params=34071630;model_mb=params*4/1e6
print(f'\n{"="*80}')
print(f'  TABLE VII: Computational Complexity (per inference)')
print(f'{"="*80}')
print(f'  Backbone: xresnet1d101, {params:,} params, {model_mb:.1f} MB')
print(f'  Backbone latency ({device}): {bb_ms:.1f} ms\n')
print(f'  {"Dim":>5} | {"Emb Bytes":>10} | {"Clf Params":>11} | {"Clf (ms)":>9} | {"Total (ms)":>11} | {"Compression":>12}')
print(f'  {"-"*70}')
for dim in nesting_dims:
    eb=dim*4;cp=dim*5+5;ct=dim_ms[dim];total=bb_ms+ct;ratio=f'{512/dim:.0f}x'
    print(f'  {dim:5d} | {eb:>8} B | {cp:>11,} | {ct:>7.2f} ms | {total:>9.1f} ms | {ratio:>12}')

print(f'\n{"="*80}')
print(f'  TABLE VIII: Device Deployment Feasibility')
print(f'{"="*80}')
ref_mflops=50000
print(f'\n  {"Device":<20} | {"Target d":>9} | {"Storage":>8} | {"Est Latency":>12} | {"Energy/Inf":>11}')
print(f'  {"-"*70}')
results={}
for name,p in DEVICES.items():
    dim=p['dim'];eb=dim*4;scale=ref_mflops/p['mflops']
    lat=(bb_ms+dim_ms.get(dim,0.1))*scale;energy=p['power_mw']*lat/1000
    results[name]={'dim':dim,'latency_ms':lat,'energy_mj':energy}
    print(f'  {name:<20} | d={dim:<6} | {eb:>6} B | {lat:>10.0f} ms | {energy:>9.1f} mJ')

print(f'\n{"="*80}')
print(f'  TABLE IX: Storage - MRL vs Separate Models')
print(f'{"="*80}')
print(f'  MRL (ours):        1 model,  {params:>12,} params, {model_mb:>7.1f} MB')
print(f'  Fixed-dim:         6 models, {params*6:>12,} params, {model_mb*6:>7.1f} MB')
print(f'  Reduction:         6x fewer models, 6x less storage')
pickle.dump({'backbone_ms':bb_ms,'dim_ms':dim_ms,'devices':results},open('results/deployment_analysis.pkl','wb'))
print(f'\n  Saved to results/deployment_analysis.pkl')
