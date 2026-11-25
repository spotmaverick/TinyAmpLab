"""模拟友好拓扑的判定性实验 (DCU, 我方协议)

数学动机 (见 src/tools/math_audit.py / param_breakdown.py):
  1. A2 Lite 的 6553 参数中 68% 在 3-tap 全连接 dilated conv, 带来 6228 MACs/sample
     和 85ms 延迟线 —— 这是模拟实现的三大成本来源。
  2. depthwise separable conv 把 conv 成本降 ch 倍 (12x/6x), 且 MACs 降一个数量级。
  3. 官方 WaveNet 的 tanh 激活天然匹配模拟差分对 —— 应保留。
  4. 此前 depthwise 复现 (nam_wavenet_baseline.py, 0.2207) 与官方 (0.0852) 的
     巨大差距, 从未在严格等协议下归因。本实验分离这个差距。

判定性假设 (可证伪):
  H1: depthwise conv 是精度损失的主因, 而非门的缺失或训练。
      → 若 dw-full 与 full 差距 << full 与 baseline0.22 差距, 则 H1 成立。
  H2: depthwise + gated tanh + 中等 RF 能达到 A2 Lite 量级 (<0.12)。
      → 决定模拟可行性的天花板。

变体设计 (严格单变量, seed 7 + 11 双种子, 300ep 我方协议):
  full_ch12     : 官方 lite 全连接 conv (基线, 6553 参数, RF 85ms)
  dw_ch12       : 仅把 conv 换成 depthwise, 其余全同 (等 RF 等协议)
  dw_ch24       : depthwise 但通道翻倍 (补回参数, 测"宽度换 conv 类型")
  dw_ch12_rf50  : depthwise + 半 RF (测延迟线减半的代价)
  dw_ch24_rf50  : 上式 + 宽通道

用法: python arch_analog_friendly.py <variant> <epochs> <seed>
"""
import sys, time, json, importlib.util
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "arch_analog"
OUT.mkdir(exist_ok=True)
SR, BATCH, LR, ETA_MIN = 48000, 4 * 48000, 3e-3, 1e-4


def group_rf(dils):
    return 2 * sum(dils) + 1


class AFLayer(nn.Module):
    """模拟友好层: depthwise dilated conv + tanh + 1x1 残差.

    mix_rank: 若给定，layer1x1 由 ch->ch 改为低秩 ch->r->ch。
              动机见 src/tools/param_flow.py：1x1 的 ch² 是模拟实现的真瓶颈
              （cw_ch12 里 1x1 占 62.7% 参数、ch24 加宽后交叉开关反超 lite），
              低秩把它降到 2·ch·r。
    """
    def __init__(self, ch, dilation, depthwise, mix_rank=None):
        super().__init__()
        groups = ch if depthwise else 1
        self.conv = nn.Conv1d(ch, ch, 3, dilation=dilation, groups=groups, bias=True)
        self.input_mixer = nn.Conv1d(1, ch, 1, bias=False)
        if mix_rank and mix_rank < ch:
            # 低秩分解 ch -> r -> ch（两段都不带 bias，末端补一个 bias 保持与原层等价）
            self.layer1x1 = nn.Sequential(
                nn.Conv1d(ch, mix_rank, 1, bias=False),
                nn.Conv1d(mix_rank, ch, 1, bias=True),
            )
            self.low_rank = True
        else:
            self.layer1x1 = nn.Conv1d(ch, ch, 1, bias=True)
            self.low_rank = False

    def forward(self, y, c):
        z = self.conv(y)
        L = z.shape[-1]
        z = torch.tanh(z + self.input_mixer(c)[:, :, -L:])
        resid = y[:, :, -L:] + self.layer1x1(z)
        return resid, z


class AFGroup(nn.Module):
    def __init__(self, in_size, ch, dils, head_out, head_bias, depthwise,
                 mix_rank=None):
        super().__init__()
        self.rechannel = nn.Conv1d(in_size, ch, 1, bias=False)
        self.layers = nn.ModuleList([AFLayer(ch, d, depthwise, mix_rank)
                                     for d in dils])
        self.head_rechannel = nn.Conv1d(ch, head_out, 1, bias=head_bias)
        self.dilations = dils
        self.rf = group_rf(dils)

    def forward(self, y, c, head_input):
        y = self.rechannel(y)
        out_len = y.shape[-1] - self.rf + 1
        for layer in self.layers:
            y, term = layer(y, c)
            term = term[:, :, -out_len:]
            head_input = term if head_input is None else head_input[:, :, -out_len:] + term
        return self.head_rechannel(head_input), y[:, :, -out_len:]


class AFNet(nn.Module):
    def __init__(self, groups, depthwise=True, head_scale=0.02, mix_rank=None):
        super().__init__()
        self.groups = nn.ModuleList([AFGroup(*g, depthwise, mix_rank)
                                     for g in groups])
        self.head_scale = head_scale

    def forward(self, x):
        c = x
        head_input, y = None, x
        for g in self.groups:
            head_input, y = g(y, c, head_input)
        return self.head_scale * head_input

    @property
    def receptive_field(self):
        return sum(g.rf for g in self.groups) - (len(self.groups) - 1)


def build(variant):
    """解析变体名 -> (model, desc)

    变体名语法：  [dw|full|wd]_ch<N>[_rf<PCT>][_r<R>]
      dw/full  : depthwise 或全连接 dilated conv
      wd       : 宽 depthwise + 低秩 1x1（必须带 _r<R>）
      ch<N>    : 通道数
      rf<PCT>  : RF 缩放百分比
      r<R>     : 1x1 低秩瓶颈维数
    """
    parts = variant.split("_")
    head = parts[0]
    depthwise = head in ("dw", "wd")
    ch = 12
    for p in parts:
        if p.startswith("ch"):
            ch = int(p[2:])
    rf_pct = 100
    for p in parts:
        if p.startswith("rf"):
            rf_pct = int(p[2:])
    mix_rank = None
    for p in parts:
        if p.startswith("r") and not p.startswith("rf") and p[1:].isdigit():
            mix_rank = int(p[1:])
    head_ch = max(1, ch // 2)
    d1 = [1, 2, 4, 8, 16, 32, 64]
    d2 = [128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    if rf_pct != 100:
        d1 = [max(1, round(d * rf_pct / 100)) for d in d1]
        d2 = [max(1, round(d * rf_pct / 100)) for d in d2]
    groups = [(1, ch, d1, head_ch, False),
              (ch, head_ch, d2, 1, True)]
    model = AFNet(groups, depthwise=depthwise, mix_rank=mix_rank)
    kind = "depthwise" if depthwise else "full"
    if mix_rank:
        kind += f"+lowrank1x1(r={mix_rank})"
    desc = f"{variant}: {kind} ch={ch} rf={rf_pct}%"
    return model, desc


def load(p):
    x, sr = sf.read(str(p))
    assert sr == SR
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

    model, desc = build(variant)
    model = model.to(DEV)
    nparam = sum(p.numel() for p in model.parameters())
    rf = model.receptive_field
    # 模拟成本: MACs 与延迟线
    macs = 0
    taps = 0
    xbar = 0
    for g in model.groups:
        macs += g.rechannel.weight.numel()
        xbar += g.rechannel.weight.numel()
        macs += sum(l.conv.weight.numel() for l in g.layers)
        macs += sum(l.input_mixer.weight.numel() for l in g.layers)
        for l in g.layers:
            if getattr(l, "low_rank", False):
                # 低秩 1x1：两段权重都要算；交叉开关按两段的实际矩阵规模
                for m in l.layer1x1:
                    macs += m.weight.numel()
                    xbar += m.weight.numel()
            else:
                macs += l.layer1x1.weight.numel()
                xbar += l.layer1x1.weight.numel()
            grp = l.conv.groups
            if grp == 1:
                # 全连接 conv: ch×ch×3 交叉开关
                xbar += l.conv.weight.numel() - l.conv.bias.numel()
        macs += g.head_rechannel.weight.numel()
        xbar += g.head_rechannel.weight.numel()
        taps += sum(2 * d for d in g.dilations)
    print(f"=== {desc} | params={nparam} RF={rf}({rf/SR*1000:.1f}ms) "
          f"MACs={macs} xbar={xbar} taps={taps} seed={seed} epochs={epochs} "
          f"dataset={dataset} ===", flush=True)

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
                  f"({time.time()-t0:.0f}s)", flush=True)

    model.load_state_dict(best_state)
    model.eval()
    res = {
        "variant": variant, "desc": desc, "params": nparam, "rf": rf,
        "rf_ms": round(rf / SR * 1000, 2), "macs": macs, "delay_taps": taps,
        "epochs": epochs, "seed": seed, "dataset": dataset, "depthwise": bool(
            model.groups[0].layers[0].conv.groups > 1),
        "protocol": "our: MSE+batch192000+lr3e-3+cosine",
        "train_seconds": round(time.time() - t0, 1), "best_val_NRMS": round(best_val, 6),
    }
    with torch.no_grad():
        for tag, x, y in [("train", dx, dy), ("val", vx, vy), ("test", tx, ty)]:
            c, n = eval_full(model, x, y)
            res[f"{tag}_NRMS"] = round(n, 6)
            res[f"{tag}_corr"] = round(c, 6)
    stem = f"{variant}_s{seed}"
    json.dump(res, open(OUT / f"{stem}.json", "w"), indent=2)
    torch.save(best_state, OUT / f"{stem}.pt")
    print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
