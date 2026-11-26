"""A2 slimmable 双档训练 —— 与官方协议等价的核心训练器

关键差异（相比 train_a2.py 的单档训练）：
  每步随机选一个 allowed_channels 档位（如 3 或 12）训练，
  训练完在**每个档位分别评测**，得到 Lite 与 Full 两套指标。
  这才是 A2 的真实形态：一套权重，两个档位都能用。

用法：
  python train_a2_slim.py <arch> <epochs> [seed] [dataset] [batch]
  arch: a2_slim / a2_dw_slim
输出：a2_runs/<arch>_<ds>_s<seed>.json，含 per-tier 指标
"""
import sys, os, time, json, random
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a2_slim import build, active_params

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
    assert sr == SR
    if x.ndim > 1:
        x = x[:, 0]
    return torch.from_numpy(x.astype(np.float32))


def metrics(pred, ref):
    pred, ref = pred.reshape(-1).double(), ref.reshape(-1).double()
    p, r = pred - pred.mean(), ref - ref.mean()
    corr = float((p * r).sum() / (p.norm() * r.norm() + 1e-12))
    return corr, float((pred - ref).norm() / ref.norm())


@torch.no_grad()
def eval_at(model, x, y, k):
    model.eval()
    o = model(x[None, None].to(DEV), k=k)[0, 0].reshape(-1)
    off = y.numel() - o.numel()
    return metrics(o.cpu(), y[off:])


def main():
    arch = sys.argv[1] if len(sys.argv) > 1 else "a2_slim"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    dataset = sys.argv[4] if len(sys.argv) > 4 else "tsa15"
    batch = int(sys.argv[5]) if len(sys.argv) > 5 else BATCH

    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, desc = build(arch)
    model = model.to(DEV)
    total = sum(p.numel() for p in model.parameters())
    rf = model.receptive_field
    tiers = model.allowed_channels
    print(f"=== {desc} | arch={arch} ds={dataset} total_params={total} "
          f"tiers={tiers} RF={rf}({rf/SR*1000:.1f}ms) seed={seed} batch={batch} "
          f"epochs={epochs} ===", flush=True)
    print("档位激活参数: " + ", ".join(f"k={k}:{active_params(model,k)}" for k in tiers),
          flush=True)

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

    best = {k: (1e9, None, -1) for k in tiers}  # tier -> (val_nrms, state, ep)

    for ep in range(epochs):
        model.train()
        starts = torch.randint(0, max(1, len(dx) - win), (nbatch,), device=DEV)
        tot = 0.0
        for s in starts:
            s = int(s)
            k = random.choice(tiers)          # ★ 随机档位：A2 核心
            opt.zero_grad()
            pred = model(dx[s:s + win][None, None], k=k)[0, 0].reshape(-1)
            target = dy[s + rf - 1:s + rf - 1 + batch]
            n = min(pred.numel(), target.numel())
            loss = F.mse_loss(pred[:n], target[:n])
            loss.backward()
            opt.step()
            tot += loss.item()
        sched.step()

        for k in tiers:
            _, vn = eval_at(model, vx, vy, k)
            if vn < best[k][0]:
                best[k] = (vn, {kk: v.detach().clone()
                                for kk, v in model.state_dict().items()}, ep)

        if ep == 0 or (ep + 1) % 25 == 0 or ep == epochs - 1:
            msg = " ".join(f"k{k}={best[k][0]:.5f}" for k in tiers)
            print(f"ep{ep+1:4d} loss={tot/max(1,nbatch):.6f} best_val: {msg} "
                  f"{time.time()-t0:.0f}s", flush=True)

    res = dict(arch=arch, desc=desc, dataset=dataset, seed=seed, epochs=epochs,
               batch=batch, total_params=total, rf=rf,
               rf_ms=round(rf / SR * 1000, 1), tiers=tiers,
               elapsed_s=round(time.time() - t0, 1),
               protocol=f"A2slim: MSE+batch{batch}+lr3e-3+cosine+random-tier+GridAlign",
               per_tier={})
    for k in tiers:
        vn, st, bep = best[k]
        model.load_state_dict(st)
        tc, tn = eval_at(model, tx, ty, k)
        vc, _ = eval_at(model, vx, vy, k)
        res["per_tier"][str(k)] = dict(
            active_params=active_params(model, k),
            best_val_NRMS=round(vn, 6), best_ep=bep + 1,
            val_corr=round(vc, 6), test_corr=round(tc, 6),
            test_NRMS=round(tn, 6))
        print(f"### tier k={k:2d} params={active_params(model,k):6d} "
              f"test_NRMS={tn:.6f} corr={tc:.4f}", flush=True)

    tag = f"{arch}_{dataset}_s{seed}" + ("" if batch == 4*48000 else f"_b{batch//1000}k")
    (OUT / f"{tag}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"### DONE {arch} {dataset} s{seed} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
