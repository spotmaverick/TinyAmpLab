"""A2 对标训练/评测脚本（4090 / DCU / z100 通用）

协议与 arch_search_v2.py 完全一致，保证与历史结果可比：
  MSE, lr 3e-3, cosine → 1e-4, batch 192000, 官方 GridDataset 对齐
  评价：全段推理 NRMS = ||pred-target||/||target||，corr = 去均值 Pearson

用法：
  python train_a2.py <arch> <epochs> [seed] [dataset] [batch]
  arch 见 a2_model.build_arch：a2 / a2_ch3 / a2_dw / a2_dw_ch24 / a2_rf50 ...

输出：a2_runs/<arch>_<dataset>_s<seed>[_b<batch>k].json
"""
import sys, os, time, json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a2_model import build_arch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "a2_runs"
OUT.mkdir(exist_ok=True)

SR = 48000
BATCH = int(os.environ.get("NAMBATCH", 4 * 48000))
LR, ETA_MIN = 3e-3, 1e-4


def load(p):
    x, sr = sf.read(str(p))
    assert sr == SR, f"{p}: sr={sr}"
    if x.ndim > 1:
        x = x[:, 0]
    return torch.from_numpy(x.astype(np.float32))


def metrics(pred, ref):
    pred = pred.reshape(-1).double()
    ref = ref.reshape(-1).double()
    p, r = pred - pred.mean(), ref - ref.mean()
    corr = float((p * r).sum() / (p.norm() * r.norm() + 1e-12))
    nrms = float((pred - ref).norm() / ref.norm())
    return corr, nrms


@torch.no_grad()
def eval_full(model, x, y):
    """官方 GridDataset 对齐：out[j] ↔ y[RF-1+j]。"""
    model.eval()
    o = model(x[None, None].to(DEV))[0, 0].reshape(-1)
    off = y.numel() - o.numel()
    return metrics(o.cpu(), y[off:])


def main():
    arch = sys.argv[1] if len(sys.argv) > 1 else "a2"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    dataset = sys.argv[4] if len(sys.argv) > 4 else "tsa15"
    batch = int(sys.argv[5]) if len(sys.argv) > 5 else BATCH

    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)

    model, desc = build_arch(arch)
    model = model.to(DEV)
    nparam = sum(p.numel() for p in model.parameters())
    rf = model.receptive_field
    print(f"=== {desc} | arch={arch} ds={dataset} params={nparam} "
          f"RF={rf}({rf/SR*1000:.1f}ms) seed={seed} batch={batch} dev={DEV} "
          f"epochs={epochs} ===", flush=True)

    dx = load(DATA / f"{dataset}_train_input.wav").to(DEV)
    dy = load(DATA / f"{dataset}_train_target.wav").to(DEV)
    vx = load(DATA / f"{dataset}_val_input.wav")
    vy = load(DATA / f"{dataset}_val_target.wav")
    tx = load(DATA / f"{dataset}_test_input.wav")
    ty = load(DATA / f"{dataset}_test_target.wav")

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=ETA_MIN)
    win = rf + batch - 1
    nbatch = max(1, len(dx) // batch)
    best_val, best_state, best_ep = 1e9, None, -1

    for ep in range(epochs):
        model.train()
        starts = torch.randint(0, max(1, len(dx) - win), (nbatch,), device=DEV)
        tot = 0.0
        for s in starts:
            s = int(s)
            opt.zero_grad()
            pred = model(dx[s:s + win][None, None])[0, 0].reshape(-1)
            target = dy[s + rf - 1:s + rf - 1 + batch]
            n = min(pred.numel(), target.numel())
            loss = F.mse_loss(pred[:n], target[:n])
            loss.backward()
            opt.step()
            tot += float(loss)
        sched.step()
        vc, vn = eval_full(model, vx, vy)
        if vn < best_val:
            best_val, best_ep = vn, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep == 0 or (ep + 1) % 25 == 0 or ep == epochs - 1:
            print(f"ep{ep+1:4d} loss={tot/max(1,nbatch):.6f} valNRMS={vn:.5f} "
                  f"best={best_val:.5f}@{best_ep+1} {time.time()-t0:.0f}s", flush=True)

    if best_state:
        model.load_state_dict(best_state)
    tc, tn = eval_full(model, tx, ty)
    el = time.time() - t0
    res = dict(arch=arch, desc=desc, dataset=dataset, seed=seed, epochs=epochs,
               batch=batch, params=nparam, rf=rf, rf_ms=round(rf / SR * 1000, 1),
               best_val_NRMS=round(best_val, 6), best_ep=best_ep + 1,
               val_corr=round(vc, 6), val_NRMS=round(vn, 6),
               test_corr=round(tc, 6), test_NRMS=round(tn, 6),
               elapsed_s=round(el, 1),
               protocol=f"A2: MSE+batch{batch}+lr3e-3+cosine+GridDataset-align")
    tag = f"{arch}_{dataset}_s{seed}" + ("" if batch == 4 * 48000 else f"_b{batch//1000}k")
    (OUT / f"{tag}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"### DONE {arch} {dataset} s{seed} test_NRMS={tn:.6f} corr={tc:.4f} "
          f"params={nparam} {el:.0f}s", flush=True)


if __name__ == "__main__":
    main()
