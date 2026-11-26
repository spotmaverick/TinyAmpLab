"""统一评测：数字精度 + 器件仿真 + 残差幅度诊断"""
import glob, json, sys
from pathlib import Path
import numpy as np, soundfile as sf, torch
sys.path.insert(0, "/root/autodl-tmp")
import analog_sim_generic as A
from analog_sim_generic import DevSim, load_ckpt, sim_forward
from nam.models.losses import esr as official_esr

DEV="cuda" if torch.cuda.is_available() else "cpu"
x,_=sf.read("/root/autodl-tmp/data/tsa15_test_input.wav",dtype="float32")
yt,_=sf.read("/root/autodl-tmp/data/tsa15_test_target.wav",dtype="float32")
x=torch.from_numpy(x[:48000*3]).to(DEV); yt=torch.from_numpy(yt[:48000*3]).to(DEV)

def find_ckpt(run):
    ck=sorted(glob.glob(f"{run}/*/lightning_logs/version_0/checkpoints/epoch=*.ckpt"))
    return ck[-1] if ck else None

def eval_run(run, vsat=16.0, mc=2):
    ck=find_ckpt(run)
    if not ck: return None
    cfg_p=str(Path(ck).parent.parent.parent.parent/"config_model.json")
    cfg,sd=load_ckpt(ck,cfg_p)
    base=dict(A.P)
    # 数字（无失真）
    A.P["V_SAT"]=1e9; A.P["BITS"]=64; A.P["SIGMA_MISMATCH"]=0.0; A.P["DRIFT"]=0.0
    A.P["BETA_H3"]=0.0; A.P["SIGMA_OFFSET"]=0.0; A.P["SNR_DB"]=1e9
    dev=DevSim(sd, torch.Generator(device=DEV).manual_seed(7))
    with torch.no_grad(): pr=sim_forward(dev,sd,cfg,x)
    off=yt.numel()-pr.numel(); t=yt[off:]
    es_dig=float(official_esr(pr.float().reshape(1,-1),t.float().reshape(1,-1)))
    rms_dig=float(pr.float().pow(2).mean().sqrt())
    # 器件
    A.P.clear(); A.P.update(base); A.P["V_SAT"]=vsat
    es_dev=[]
    for i in range(mc):
        d2=DevSim(sd, torch.Generator(device=DEV).manual_seed(100+i))
        with torch.no_grad(): p2=sim_forward(d2,sd,cfg,x)
        es_dev.append(float(official_esr(p2.float().reshape(1,-1),t.float().reshape(1,-1))))
    A.P.clear(); A.P.update(base)
    e_dev=float(np.mean(es_dev))
    npar=sum(v.numel() for v in sd.values())
    lc=cfg["layers_configs"][0]
    return dict(run=Path(run).name, nrms_dig=float(np.sqrt(es_dig)),
                nrms_dev=float(np.sqrt(max(e_dev,0))),
                ratio=float(np.sqrt(max(e_dev,0)/max(es_dig,1e-12))),
                rms_out=rms_dig, params=npar, ch=lc["channels"], L=len(lc["dilations"]),
                g_in=lc.get("groups_input",1), alpha=cfg.get("residual_scale"),
                gains=cfg.get("layer_gains"))

runs=sys.argv[1:]
out=[]
for r in runs:
    try:
        res=eval_run(r)
        if res:
            out.append(res)
            print(json.dumps(res,ensure_ascii=False),flush=True)
    except Exception as e:
        print(json.dumps(dict(run=Path(r).name,error=str(e)[:200])),flush=True)
json.dump(out,open("/root/autodl-tmp/eval_v3.json","w"),indent=1,ensure_ascii=False)
