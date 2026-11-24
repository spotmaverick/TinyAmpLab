"""WaveNet 架构族系统搜索 v2 (DCU, 我方协议)

背景:
  d1 证明官方 standard @ 我方协议 = 0.0653, 远优于 LGW 冠军 0.1736;
  d2/d3 证明拉长 RF 与 dual 门控在 LGW 族均为负收益.
  本轮转向 WaveNet 架构族, 搜索参数效率前沿与短 RF 可行性.

用法:
  python arch_search_v2.py <variant> <epochs> [seed] [dataset]

  dataset 默认 tsa15; 其他如 marshall_jvm410h_chod1 / marshall_jvm410h_chod1_spk

协议 (与 d1 完全一致, 保证可比):
  纯 MSE, batch=4*48000, lr=3e-3, CosineAnnealingLR(eta_min=1e-4)
  官方 GridDataset 对齐: x[i:i+rf+ny-1], y[i+rf-1:i+rf-1+ny]
  评价: 全段推理 NRMS/corr, 与历史 NRMS 口径一致 (NRMS=sqrt(ESR))

变体命名: <arch> 或 <arch>_rf<scale>
  arch ∈ {nano, feather, lite, standard}
  rf<scale>: dilation 整体缩放, 如 rf50 = 原 dilation 减半 (RF 约减半)
"""
import sys, time, json, importlib.util
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path("/work/home/spotmaverick/nam_project")
DATA = ROOT / "data"
OUT = ROOT / "arch_v2"
OUT.mkdir(exist_ok=True)
SR, BATCH, LR, ETA_MIN = 48000, 4 * 48000, 3e-3, 1e-4


def imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_wn = imp("wn_official", ROOT / "e1" / "wavenet_official_dcu.py")


def scale_dilations(dils, scale_pct):
    """按比例缩放 dilation (向下取整, 最小 1). scale_pct=50 表示减半."""
    if scale_pct == 100:
        return dils
    return [max(1, round(d * scale_pct / 100)) for d in dils]


def build(variant):
    """构造模型, 返回 (model, desc, scale_pct)."""
    parts = variant.split("_rf")
    arch = parts[0]
    scale = int(parts[1]) if len(parts) > 1 else 100
    assert arch in _wn.OFFICIAL_ARCHS, f"未知架构: {arch}"
    assert 10 <= scale <= 100, f"RF 缩放比例需在 10~100: {scale}"

    groups = []
    for input_size, ch, dils, head_out, head_bias in _wn.OFFICIAL_ARCHS[arch]:
        groups.append((input_size, ch, scale_dilations(dils, scale), head_out, head_bias))
    model = _wn.OfficialWaveNet(groups)
    desc = f"{arch} @ 我方协议" + (f", RF缩放{scale}%" if scale != 100 else "")
    return model, desc, scale


def load(p):
    x, sr = sf.read(str(p))
    assert sr == SR
    return torch.from_numpy(x.astype(np.float32))


def metrics(pred, ref):
    """与 diag_three.py / 历史 NRMS 口径一致."""
    pred = pred.reshape(-1).double()
    ref = ref.reshape(-1).double()
    p, r = pred - pred.mean(), ref - ref.mean()
    corr = float((p * r).sum() / (p.norm() * r.norm() + 1e-12))
    nrms = float((pred - ref).norm() / ref.norm())
    return corr, nrms


@torch.no_grad()
def eval_full(model, x, y):
    """官方 GridDataset 对齐: out[j] ↔ y[RF-1+j]."""
    o = model(x[None, None])[0, 0].reshape(-1)
    off = y.numel() - o.numel()
    return metrics(o, y[off:])


def main():
    variant = sys.argv[1]
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    dataset = sys.argv[4] if len(sys.argv) > 4 else "tsa15"
    t0 = time.time()
    torch.manual_seed(seed)

    model, desc, scale = build(variant)
    model = model.to(DEV)
    nparam = sum(p.numel() for p in model.parameters())
    rf = model.receptive_field
    print(f"=== {desc} | variant={variant} dataset={dataset} params={nparam} "
          f"RF={rf}({rf/SR*1000:.1f}ms) seed={seed} dev={DEV} epochs={epochs} ===", flush=True)

    dx = load(DATA / f"{dataset}_train_input.wav").to(DEV)
    dy = load(DATA / f"{dataset}_train_target.wav").to(DEV)
    vx = load(DATA / f"{dataset}_val_input.wav").to(DEV)
    vy = load(DATA / f"{dataset}_val_target.wav").to(DEV)
    tx = load(DATA / f"{dataset}_test_input.wav").to(DEV)
    ty = load(DATA / f"{dataset}_test_target.wav").to(DEV)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=ETA_MIN)
    win = rf + BATCH - 1
    nbatch = len(dx) // BATCH
    best_val, best_state = 1e9, None

    for ep in range(epochs):
        model.train()
        starts = torch.randint(0, len(dx) - win, (nbatch,), device=DEV)
        tot = 0.0
        for s in starts:
            s = int(s)
            opt.zero_grad()
            pred = model(dx[s:s + win][None, None])[0, 0].reshape(-1)
            target = dy[s + rf - 1:s + rf - 1 + BATCH]
            loss = F.mse_loss(pred, target)
            loss.backward()
            opt.step()
            tot += loss.item()
        sched.step()
        model.eval()
        with torch.no_grad():
            vc, vn = eval_full(model, vx, vy)
        if vn < best_val:
            best_val = vn
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"ep {ep:3d} loss {tot/nbatch:.6f} val_NRMS {vn:.4f} corr {vc:.4f} "
                  f"lr {sched.get_last_lr()[0]:.6f} ({time.time()-t0:.0f}s)", flush=True)

    model.load_state_dict(best_state)
    model.eval()
    res = {
        "variant": variant, "desc": desc, "arch": variant.split("_rf")[0],
        "rf_scale_pct": scale, "params": nparam, "rf": rf, "rf_ms": round(rf / SR * 1000, 2),
        "epochs": epochs, "seed": seed, "dataset": dataset,
        "protocol": "our: MSE+batch192000+lr3e-3+cosine+GridDataset-align",
        "train_seconds": round(time.time() - t0, 1), "best_val_NRMS": round(best_val, 6),
    }
    with torch.no_grad():
        for tag, x, y in [("train", dx, dy), ("val", vx, vy), ("test", tx, ty)]:
            c, n = eval_full(model, x, y)
            res[f"{tag}_NRMS"] = round(n, 6)
            res[f"{tag}_corr"] = round(c, 6)
    res["test_esr"] = round(res["test_NRMS"] ** 2, 6)
    stem = f"{variant}_s{seed}" if dataset == "tsa15" else f"{variant}_{dataset}_s{seed}"
    json.dump(res, open(OUT / f"{stem}.json", "w"), indent=2)
    torch.save(best_state, OUT / f"{stem}.pt")
    print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
