# 官方 NAM WaveNet 架构 DCU 移植版 (torch 1.10 / DTK 22.04 兼容)
# 结构还原自 neural-amp-modeler 0.13.0 (nam/models/wavenet/_wavenet.py + _layer_array.py)
# 还原正确性: nano 架构参数量验算 841=841, RF=4093 与官方一致
#
# 用法:
#   对拍(加载官方 .nam 权重评测):  python wavenet_official_dcu.py --eval <model.nam> <dataset>
#   训练:                          python wavenet_official_dcu.py <arch> <dataset> [epochs]
#     arch ∈ {nano, feather, lite, standard}
#
# 官方协议 (0.13 nam_full_configs):
#   采样: 固定网格窗口 i=idx*ny, x 窗 RF+ny-1, y 窗 ny=8192, DataLoader shuffle idx
#   loss: 纯 MSE; optimizer: Adam(lr=0.004); scheduler: ExponentialLR(gamma=0.993, per-epoch)
#   batch 16, drop_last, epochs 100
#   评价: 官方 esr = mean(mean((p-t)^2) / mean(t^2)) 全段推理
import sys, json, time, os
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

DATA = Path("/work/home/spotmaverick/nam_project/data")
E1 = Path("/work/home/spotmaverick/nam_project/e1")
OUT = Path("/work/home/spotmaverick/nam_project/e1_dcu")
NY = 8192

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # DCU 的 HIP 后端映射为 cuda API

# ---------------- 官方四级架构 (v0.8.0 core.py 转译) ----------------
OFFICIAL_ARCHS = {
    "nano":     [(1, 4,  [1,2,4,8,16,32,64], 2, False),
                 (4, 2,  [128,256,512,1,2,4,8,16,32,64,128,256,512], 1, True)],
    "feather":  [(1, 8,  [1,2,4,8,16,32,64], 4, False),
                 (8, 4,  [128,256,512,1,2,4,8,16,32,64,128,256,512], 1, True)],
    "lite":     [(1, 12, [1,2,4,8,16,32,64], 6, False),
                 (12, 6, [128,256,512,1,2,4,8,16,32,64,128,256,512], 1, True)],
    "standard": [(1, 16, [1,2,4,8,16,32,64,128,256,512], 8, False),
                 (16, 8, [1,2,4,8,16,32,64,128,256,512], 1, True)],
}

# ---------------- 模型 ----------------
class _Layer(nn.Module):
    def __init__(self, ch, dilation, cond=1):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, 3, dilation=dilation, bias=True)
        self.input_mixer = nn.Conv1d(cond, ch, 1, bias=False)
        self.layer1x1 = nn.Conv1d(ch, ch, 1, bias=True)
        self.dilation = dilation

    def forward(self, y, c):
        z = self.conv(y)  # (B,ch,L-2d)
        L = z.shape[-1]
        z = torch.tanh(z + self.input_mixer(c)[:, :, -L:])
        resid = y[:, :, -L:] + self.layer1x1(z)
        return resid, z  # (residual, head_term=raw activation)

class _LayerArray(nn.Module):
    def __init__(self, input_size, ch, dilations, head_out, head_bias, cond=1):
        super().__init__()
        self.rechannel = nn.Conv1d(input_size, ch, 1, bias=False)
        self.layers = nn.ModuleList([_Layer(ch, d, cond) for d in dilations])
        self.head_rechannel = nn.Conv1d(ch, head_out, 1, bias=head_bias)
        self.dilations = dilations
        self.rf = 2 * sum(dilations) + 1

    def forward(self, y, c, head_input):
        y = self.rechannel(y)
        out_len = y.shape[-1] - self.rf + 1
        for layer in self.layers:
            y, term = layer(y, c)
            term = term[:, :, -out_len:]
            head_input = term if head_input is None else head_input[:, :, -out_len:] + term
        head_out = self.head_rechannel(head_input)
        return head_out, y[:, :, -out_len:]

class OfficialWaveNet(nn.Module):
    def __init__(self, groups, head_scale=0.02):
        super().__init__()
        self.layer_arrays = nn.ModuleList(
            [_LayerArray(i, ch, d, ho, hb) for (i, ch, d, ho, hb) in groups])
        self.head_scale = head_scale

    def forward(self, x):
        # x: (B,1,L); condition c = x (官方 _wavenet.forward: c = x if no condition_dsp)
        c = x
        head_input, y = None, x
        for g in self.layer_arrays:
            head_input, y = g(y, c, head_input)
        return self.head_scale * head_input

    @property
    def receptive_field(self):
        return sum(g.rf for g in self.layer_arrays) - (len(self.layer_arrays) - 1)

def build(arch):
    return OfficialWaveNet(OFFICIAL_ARCHS[arch])

# ---------------- 官方 .nam 扁平权重解析 ----------------
def load_nam_weights(model, weights):
    w = weights if isinstance(weights, torch.Tensor) else torch.tensor(weights, dtype=torch.float32)
    i = 0
    with torch.no_grad():
        for g in model.layer_arrays:
            in_c, ch = g.rechannel.weight.shape[1], g.rechannel.weight.shape[0]
            n = in_c * ch
            g.rechannel.weight.copy_(w[i:i + n].reshape(ch, in_c, 1)); i += n
            for layer in g.layers:
                cch = layer.conv.weight.shape[0]
                n = cch * cch * 3
                layer.conv.weight.copy_(w[i:i + n].reshape(cch, cch, 3)); i += n
                layer.conv.bias.copy_(w[i:i + cch]); i += cch
                cond = layer.input_mixer.weight.shape[1]
                n = cch * cond
                layer.input_mixer.weight.copy_(w[i:i + n].reshape(cch, cond, 1)); i += n
                n = cch * cch
                layer.layer1x1.weight.copy_(w[i:i + n].reshape(cch, cch, 1)); i += n
                layer.layer1x1.bias.copy_(w[i:i + cch]); i += cch
            ho = g.head_rechannel.weight.shape[0]
            n = ho * ch
            g.head_rechannel.weight.copy_(w[i:i + n].reshape(ho, ch, 1)); i += n
            if g.head_rechannel.bias is not None:
                g.head_rechannel.bias.copy_(w[i:i + ho]); i += ho
    model.head_scale = float(w[i]); i += 1
    return i

# ---------------- 数据 (官方固定网格采样) ----------------
def load_wav(path):
    x, sr = sf.read(path)
    assert sr == 48000, f"{path}: sr={sr}"
    return x.astype(np.float32)

class GridDataset(torch.utils.data.Dataset):
    """官方 Dataset 协议: i = idx*ny; x 窗 [i, i+nx+ny-1); y 窗 [i+nx-1, +ny)"""
    def __init__(self, x, y, nx, ny):
        self.x, self.y = x, y
        self.nx, self.ny = nx, ny
    def __getitem__(self, idx):
        i = idx * self.ny
        xw = self.x[i: i + self.nx + self.ny - 1]
        yw = self.y[i + self.nx - 1: i + self.nx - 1 + self.ny]
        return torch.from_numpy(xw), torch.from_numpy(yw)
    def __len__(self):
        return (len(self.x) - self.nx + 1) // self.ny

# ---------------- 官方 esr ----------------
def esr(preds, targets):
    return torch.mean(
        torch.mean(torch.square(preds - targets), dim=1)
        / torch.mean(torch.square(targets), dim=1)).item()

# ---------------- 全段推理评价 ----------------
@torch.no_grad()
def evaluate(model, split, dataset):
    x = load_wav(DATA / f"{dataset}_{split}_input.wav")
    y = load_wav(DATA / f"{dataset}_{split}_target.wav")
    xt = torch.from_numpy(x)[None, None].to(DEVICE)
    out = model(xt)[0, 0].cpu()          # (L-RF+1,) 尾部对齐
    yt = torch.from_numpy(y[-out.shape[0]:])[None]
    return esr(out[None], yt)

# ---------------- 对拍模式 ----------------
def eval_nam(nam_path, dataset):
    d = json.load(open(nam_path))
    assert d["architecture"] == "WaveNet"
    cfgs = []
    for lc in d["config"]["layers"]:
        cfgs.append((lc["input_size"], lc["channels"], lc["dilations"],
                     lc["head"]["out_channels"] if "head" in lc else lc.get("head_size", 1),
                     bool(lc["head"]["bias"]) if "head" in lc else bool(lc.get("head_bias", True))))
    model = OfficialWaveNet(cfgs).to(DEVICE)
    n = load_nam_weights(model, d["weights"])
    assert n == len(d["weights"]), f"权重解析不匹配: 解析 {n} vs 文件 {len(d['weights'])}"
    model.eval()
    res = {"nam": str(nam_path), "params": sum(p.numel() for p in model.parameters()),
           "rf": model.receptive_field}
    for split in ["train", "val", "test"]:
        res[f"esr_{split}"] = round(evaluate(model, split, dataset), 6)
    print("RESULT " + json.dumps(res))
    return res

# ---------------- 训练 (官方协议) ----------------
def train(arch, dataset, epochs=100):
    t0 = time.time()
    outdir = OUT / f"{arch}_{dataset}_dcu"; outdir.mkdir(parents=True, exist_ok=True)
    model = build(arch).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    rf = model.receptive_field
    print(f"[{arch}] params={n_params} RF={rf} device={DEVICE}")

    x = load_wav(DATA / f"{dataset}_train_input.wav")
    y = load_wav(DATA / f"{dataset}_train_target.wav")
    ds = GridDataset(x, y, rf, NY)
    dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True, drop_last=True,
                                     num_workers=2)
    print(f"train windows={len(ds)} steps/epoch={len(dl)}")

    opt = torch.optim.Adam(model.parameters(), lr=0.004)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.993)
    log = open(outdir / "train.log", "w")
    best_val, best_state = None, None
    xval = torch.from_numpy(load_wav(DATA / f"{dataset}_val_input.wav"))[None, None].to(DEVICE)
    yval = torch.from_numpy(load_wav(DATA / f"{dataset}_val_target.wav"))
    for ep in range(epochs):
        model.train()
        ep_loss, nb = 0.0, 0
        for xb, yb in dl:
            xb = xb[:, None].to(DEVICE)          # (B,1,nx+ny-1)
            yb = yb.to(DEVICE)                    # (B,ny)
            out = model(xb)[:, 0]                 # (B,1,ny) -> (B,ny)
            loss = torch.nn.functional.mse_loss(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += float(loss); nb += 1
        sched.step()
        model.eval()
        with torch.no_grad():
            o = model(xval)[0, 0].cpu()
            v = esr(o[None], yval[-o.shape[0]:][None])
        line = f"epoch {ep} loss={ep_loss/max(nb,1):.6f} val_esr={v:.6f} lr={sched.get_last_lr()[0]:.6f}"
        print(line); log.write(line + "\n"); log.flush()
        if best_val is None or v < best_val:
            best_val = v
            best_state = {k: vv.detach().cpu().clone() for k, vv in model.state_dict().items()}
    torch.save({"state_dict": best_state, "arch": arch, "params": n_params, "rf": rf,
                "dataset": dataset, "best_val_esr": best_val, "protocol": "official-0.13"},
               outdir / "weights.pt")
    res = {"arch": arch, "dataset": dataset, "params": n_params, "rf": rf,
           "train_seconds": round(time.time() - t0, 1), "best_val_esr": round(best_val, 6)}
    for split in ["train", "val", "test"]:
        model.load_state_dict(best_state)
        res[f"esr_{split}"] = round(evaluate(model, split, dataset), 6)
    json.dump(res, open(outdir / "result.json", "w"), indent=2)
    print("RESULT " + json.dumps(res))

if __name__ == "__main__":
    if sys.argv[1] == "--eval":
        eval_nam(sys.argv[2], sys.argv[3])
    else:
        arch, dataset = sys.argv[1], sys.argv[2]
        ep = int(sys.argv[3]) if len(sys.argv) > 3 else 100
        train(arch, dataset, ep)
