# 三大诊断实验 (DCU): 分离协议红利 / 架构差距 / 正交组合
# 用法: python diag_three.py <exp> <epochs>
#   exp ∈ {d1_official_standard_ourproto, d2_lgw_rf10, d3_dual_p3}
# 统一口径: TSA15, 我方协议(纯MSE + batch192000 + lr3e-3 + 余弦退火), 评价 NRMS/corr/esr
import sys, time, json, importlib.util
from pathlib import Path
import numpy as np, soundfile as sf, torch
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path("/work/home/spotmaverick/nam_project")
DATA = ROOT / "data"
OUT = ROOT / "diag3"; OUT.mkdir(exist_ok=True)
SR, BATCH, LR, NY = 48000, 4 * 48000, 3e-3, 8192

def load(p):
    x, sr = sf.read(str(p)); assert sr == 48000
    return torch.from_numpy(x.astype(np.float32))

def metrics(pred, ref):
    p, r = pred.double() - pred.double().mean(), ref.double() - ref.double().mean()
    corr = float((p * r).sum() / (p.norm() * r.norm() + 1e-12))
    nrms = float((pred.double() - ref.double()).norm() / ref.double().norm())
    return corr, nrms

# 动态导入两个模型定义
def imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m
_lgw, _wn = None, None
def get_lgw():
    global _lgw
    if _lgw is None: _lgw = imp("lgw", ROOT / "lgw_dcu2.py")
    return _lgw
def get_wn():
    global _wn
    if _wn is None: _wn = imp("wn", ROOT / "e1" / "wavenet_official_dcu.py")
    return _wn

def build(exp):
    if exp == "d1_official_standard_ourproto":
        # 诊断1: 官方 standard 13801 参数 + 我方协议 (唯一变量=架构)
        m = get_wn().build("standard")
        return m, "官方 standard 架构 @ 我方协议"
    if exp == "d2_lgw_rf10":
        # 诊断2: 冠军等参 RF 对照, ch8/L12/none/dil_exp10 (RF 11.3ms→85ms, 参数不变)
        m = get_lgw().LGNet(8, 12, "none", False, True, 10, False)
        return m, "LGW 等参 RF10 @ 我方协议"
    if exp == "d3_dual_p3":
        # 诊断3: dual 门控 + P3 协议 (dual 从没享受过 300ep 余弦)
        m = get_lgw().LGNet(8, 12, "dual", True, True, 10, False)
        return m, "dual 门控 + d10 @ P3 协议"
    raise ValueError(exp)

def main():
    exp = sys.argv[1]
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    t0 = time.time()
    model, desc = build(exp)
    model = model.to(DEV)
    nparam = sum(p.numel() for p in model.parameters())
    rf = getattr(model, "receptive_field", None)
    if rf is None:
        dils = [2 ** (i % 11) for i in range(12)] if exp != "d3_dual_p3" else [2 ** (i % 11) for i in range(12)]
        rf = 1 + 2 * sum(dils)
    print(f"=== {desc} | exp={exp} params={nparam} RF={rf}({rf/SR*1000:.1f}ms) dev={DEV} epochs={epochs} ===", flush=True)

    dx = load(DATA / "tsa15_train_input.wav").to(DEV)
    dy = load(DATA / "tsa15_train_target.wav").to(DEV)
    vx = load(DATA / "tsa15_val_input.wav").to(DEV)
    vy = load(DATA / "tsa15_val_target.wav").to(DEV)
    tx = load(DATA / "tsa15_test_input.wav").to(DEV)
    ty = load(DATA / "tsa15_test_target.wav").to(DEV)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)
    nbatch = len(dx) // BATCH
    best_val, best_state = 1e9, None

    for ep in range(epochs):
        model.train()
        starts = torch.randint(0, len(dx) - BATCH, (nbatch,), device=DEV)
        tot = 0.0
        for s in starts:
            s = int(s)
            opt.zero_grad()
            if exp == "d1_official_standard_ourproto":
                # WaveNet 输出 (B,1,L-RF), 取尾 ny 对齐
                pred = model(dx[s:s + BATCH + rf][None, None])[0, 0]
                target = dy[s:s + BATCH]
                L = min(len(pred), len(target))
                loss = F.mse_loss(pred[-L:], target[-L:])
            else:
                pred = model(dx[s:s + BATCH])
                target = dy[s:s + BATCH]
                L = min(len(pred), len(target))
                loss = F.mse_loss(pred[:L], target[:L])
            loss.backward(); opt.step()
            tot += loss.item()
        sched.step()
        model.eval()
        with torch.no_grad():
            if exp == "d1_official_standard_ourproto":
                o = model(vx[None, None])[0, 0]
                L = min(len(o), len(vy))
                vc, vn = metrics(o[-L:], vy[-L:])
            else:
                vc, vn = metrics(model(vx), vy)
        if vn < best_val:
            best_val = vn
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"ep {ep:3d} loss {tot/nbatch:.6f} val_NRMS {vn:.4f} corr {vc:.4f} lr {sched.get_last_lr()[0]:.6f} ({time.time()-t0:.0f}s)", flush=True)

    model.load_state_dict(best_state)
    model.eval()
    res = {"exp": exp, "desc": desc, "params": nparam, "rf": rf, "rf_ms": round(rf/SR*1000, 2),
           "epochs": epochs, "protocol": "our: MSE+batch192000+lr3e-3+cosine",
           "train_seconds": round(time.time() - t0, 1), "best_val_NRMS": round(best_val, 6)}
    with torch.no_grad():
        for tag, x, y in [("train", dx, dy), ("val", vx, vy), ("test", tx, ty)]:
            if exp == "d1_official_standard_ourproto":
                o = model(x[None, None])[0, 0]
                L = min(len(o), len(y))
                c, n = metrics(o[-L:], y[-L:])
            else:
                c, n = metrics(model(x), y)
            res[f"{tag}_NRMS"] = round(n, 6)
            res[f"{tag}_corr"] = round(c, 6)
    res["test_esr"] = round(res["test_NRMS"] ** 2, 6)
    json.dump(res, open(OUT / f"{exp}.json", "w"), indent=2)
    torch.save(best_state, OUT / f"{exp}.pt")
    print("RESULT " + json.dumps(res), flush=True)

if __name__ == "__main__":
    main()
