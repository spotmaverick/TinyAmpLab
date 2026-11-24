"""NAM WaveNet 基线 - DCU 版。数据读取用标准库(struct)，零外部依赖。"""
import sys, os, wave, struct, time
import torch
import torch.nn as nn
import torch.nn.functional as F

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 7
CHANNELS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
LAYERS = int(sys.argv[4]) if len(sys.argv) > 4 else 10
BATCH = 4 * 48000
KERNEL = 3
LR = 3e-3
torch.manual_seed(SEED)


def load_wav_float(path):
    """手动解析 RIFF，读取 32-bit float wav (fmt tag=3) -> torch.float32"""
    with open(path, 'rb') as f:
        f.read(12)  # RIFF....WAVE
        while True:
            tag = f.read(4)
            if len(tag) < 4:
                raise ValueError('no data chunk in ' + path)
            size = struct.unpack('<I', f.read(4))[0]
            if tag == b'data':
                raw = f.read(size)
                n = size // 4
                x = struct.unpack('<%df' % n, raw[:n*4])
                return torch.tensor(x, dtype=torch.float32)
            f.read(size + (size & 1))  # 跳过并对齐偶数字节


class DilatedBlock(nn.Module):
    def __init__(self, ch, dil):
        super().__init__()
        self.pad = (KERNEL - 1) * dil
        self.dw = nn.Conv1d(ch, ch, KERNEL, dilation=dil, groups=ch, padding=0)
        self.expand = nn.Conv1d(ch, 2*ch, 1)
        self.collapse = nn.Conv1d(ch, ch, 1)

    def forward(self, x):
        h = self.dw(F.pad(x, (self.pad, 0)))
        g, f = self.expand(h).chunk(2, dim=1)
        h = self.collapse(torch.tanh(f) * torch.sigmoid(g))
        return x + h, h


class NamWaveNet(nn.Module):
    def __init__(self, ch, layers):
        super().__init__()
        self.inp = nn.Conv1d(1, ch, 1)
        self.blocks = nn.ModuleList([DilatedBlock(ch, 2**(i % 8)) for i in range(layers)])
        self.head = nn.Sequential(nn.ReLU(), nn.Conv1d(ch, 1, 1))

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        h = self.inp(x.unsqueeze(1))
        skip = 0
        for b in self.blocks:
            h, s = b(h)
            skip = skip + s
        return self.head(skip).squeeze(1)


def metrics(pred, ref):
    p, r = pred.double() - pred.double().mean(), ref.double() - ref.double().mean()
    corr = float((p*r).sum() / (p.norm()*r.norm()+1e-12))
    nrms = float((pred.double()-ref.double()).norm() / ref.double().norm())
    return corr, nrms


if __name__ == '__main__':
    dev = 'cuda'
    t0 = time.time()
    dx = load_wav_float('data/tsa15_train_input.wav').to(dev)
    dy = load_wav_float('data/tsa15_train_target.wav').to(dev)
    vx = load_wav_float('data/tsa15_val_input.wav').to(dev)
    vy = load_wav_float('data/tsa15_val_target.wav').to(dev)
    tx = load_wav_float('data/tsa15_test_input.wav').to(dev)
    ty = load_wav_float('data/tsa15_test_target.wav').to(dev)
    print('data loaded to DCU in %.1fs' % (time.time()-t0), flush=True)

    model = NamWaveNet(CHANNELS, LAYERS).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    nbatch = len(dx)//BATCH
    print('WaveNet ch=%d layers=%d params=%d batches/ep=%d ep=%d seed=%d' % (
        CHANNELS, LAYERS, sum(p.numel() for p in model.parameters()), nbatch, EPOCHS, SEED), flush=True)

    for ep in range(EPOCHS):
        starts = torch.randint(0, len(dx)-BATCH, (nbatch,), device=dev)
        tot = 0.0
        for s in starts:
            s = int(s)
            opt.zero_grad()
            pred = model(dx[s:s+BATCH])
            loss = F.mse_loss(pred, dy[s:s+BATCH])
            loss.backward()
            opt.step()
            tot += loss.item()
        if ep % 10 == 0 or ep == EPOCHS-1:
            model.eval()
            with torch.no_grad():
                vg, vref = model(vx), vy
            vc, vn = metrics(vg, vref)
            model.train()
            print('ep %3d loss %.6f val_corr %.4f val_NRMS %.4f (%.0fs)' % (
                ep, tot/nbatch, vc, vn, time.time()-t0), flush=True)

    print('=== FINAL ===', flush=True)
    model.eval()
    with torch.no_grad():
        for tag, x, y in [('train',dx,dy),('val',vx,vy),('test',tx,ty)]:
            c, n = metrics(model(x), y)
            print('[%s] corr %+.4f NRMS %.4f' % (tag, c, n), flush=True)

    # DCU 推理交付：保存权重 + 生成 test 段全量输出 wav
    torch.save(model.state_dict(), 'nam_wavenet_tsa15.pt')
    with torch.no_grad():
        pred = model(tx).cpu().numpy()
    import soundfile as _sf
    _sf.write('nam_wavenet_test_out.wav', pred, 48000, subtype='FLOAT')
    print('saved nam_wavenet_tsa15.pt / nam_wavenet_test_out.wav', flush=True)
