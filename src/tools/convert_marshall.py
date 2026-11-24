"""Marshall JVM410H 数据集校验 + 转换 (本地 Mac 执行)

前置: pdl.sh 下载完成, 合并出 Marshall-JVM410H-ChOD1.zip
流程:
  1. md5 校验 (Zenodo API 期望 05ee998404f44c3a449c4ad6a8160cb6, size 2755382616)
  2. 解压到 marshal_extract/
  3. 探测内部目录结构, 打印供人工确认
  4. 转换为 48kHz 单声道 float32, 命名 marshall_jvm410h_{train,val,test}_{input,target}.wav
  5. 输出到项目 data/ 并生成上传清单

用法:
  python convert_marshall.py --verify    # 仅校验+解压+探测结构
  python convert_marshall.py --convert   # 执行转换 (需先确认结构)
"""
import sys, json, hashlib, zipfile, urllib.request
from pathlib import Path
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset_zenodo"
ZIP = DATASET / "Marshall-JVM410H-ChOD1.zip"
EXTRACT = DATASET / "marshall_extract"
OUT = ROOT / "data"
RECORD_ID = "10892012"
EXPECT_MD5 = "05ee998404f44c3a449c4ad6a8160cb6"
EXPECT_SIZE = 2755382616


def sanity():
    if not ZIP.exists():
        print(f"[FAIL] 文件不存在: {ZIP}")
        return False
    sz = ZIP.stat().st_size
    print(f"size: {sz} / {EXPECT_SIZE} ({'OK' if sz == EXPECT_SIZE else 'MISMATCH'})")
    if sz != EXPECT_SIZE:
        return False
    h = hashlib.md5()
    with open(ZIP, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    got = h.hexdigest()
    print(f"md5:  {got} / {EXPECT_MD5} ({'OK' if got == EXPECT_MD5 else 'MISMATCH'})")
    return got == EXPECT_MD5


def extract():
    """解压; 已解压则跳过 (幂等, 避免重复 mkdir 在沙箱下报 EEXIST)."""
    marker = EXTRACT / "Marshall-JVM410H-ChOD1-trimmed"
    if marker.exists():
        print(f"[SKIP] 已解压: {EXTRACT}")
        return
    EXTRACT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP) as z:
        names = z.namelist()
        print(f"zip 条目数: {len(names)}")
        # 安全性: 拒绝路径穿越
        bad = [n for n in names if n.startswith("/") or ".." in Path(n).parts]
        if bad:
            raise ValueError(f"zip 含不安全路径: {bad[:3]}")
        z.extractall(EXTRACT)
    print(f"解压到: {EXTRACT}")


def probe():
    """打印目录树与 wav 清单, 供确定 target 映射."""
    print("=== 目录结构 (3层) ===")
    for d in sorted(EXTRACT.rglob("*"))[:200]:
        if d.is_dir():
            print("  ", d.relative_to(EXTRACT))
    wavs = sorted(EXTRACT.rglob("*.wav"))
    print(f"=== wav 文件 {len(wavs)} 个 (前40) ===")
    for w in wavs[:40]:
        print("  ", w.relative_to(EXTRACT), w.stat().st_size)
    if len(wavs) > 40:
        print(f"  ... 其余 {len(wavs)-40} 个")
    return wavs


def conv48(src, dst):
    x, sr = sf.read(str(src))
    if x.ndim > 1:
        x = x[:, 0]
    n = int(len(x) * 48000 / sr)
    y = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    sf.write(str(dst), y, 48000, subtype="FLOAT")
    return round(n / 48000, 1)


def conv48_slice(src, dst, t0, t1):
    """只取 [t0, t1) 秒区间写盘, 避免整段读入后再裁的低效与内存峰值."""
    sr = sf.info(str(src)).samplerate  # 帧偏移必须按源采样率算, 不能用目标 48k
    x, sr = sf.read(str(src), start=int(t0 * sr), frames=int((t1 - t0) * sr))
    if x.ndim > 1:
        x = x[:, 0]
    n = int(len(x) * 48000 / sr)
    if n <= 0:
        raise ValueError(f"切片为空: {src} [{t0},{t1})")
    y = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    sf.write(str(dst), y, 48000, subtype="FLOAT")
    return round(n / 48000, 1)


def convert_set(tag, out_dir, setting, capture):
    """转换一个 (采集点, 设置) 组合成 train/val/test 三段数据集.

    切分策略 (本数据集标准设置无独立 test 录音, 见下):
      trainval 单文件 300s 按时序切 70/10/20 —— train [0,70%), val [70%,80%), test [80%,100%)
    test 段取末尾, 与 train 尽量远, 且三段均来自同一次连续录音 (输入信号同源, DRY 只有一份).
    说明: test/ 目录下的 9 个录音都是其他旋钮组合, 与该设置的模型不是同一个任务, 不能用作 test.
    """
    base = EXTRACT / "Marshall-JVM410H-ChOD1-trimmed"
    x_tr_v = base / capture / "trainval" / setting / f"{setting}.trainval.target.wav"
    dry_tr_v = base / "DRY" / "trainval" / "trainval.input.wav"

    for p in (x_tr_v, dry_tr_v):
        if not p.exists():
            print(f"[SKIP] {tag}: 缺文件 {p}")
            return None

    dur = sf.info(str(dry_tr_v)).duration
    t_train_end = round(dur * 0.7, 3)
    t_val_end = round(dur * 0.8, 3)

    out_dir.mkdir(parents=True, exist_ok=True)
    rec = {}
    jobs = [
        ("train", dry_tr_v, x_tr_v, 0.0, t_train_end),
        ("val", dry_tr_v, x_tr_v, t_train_end, t_val_end),
        ("test", dry_tr_v, x_tr_v, t_val_end, None),
    ]
    for split, xi, yi, t0, t1 in jobs:
        if t1 is None:
            a = conv48_slice(xi, out_dir / f"{tag}_{split}_input.wav", t0, dur)
            b = conv48_slice(yi, out_dir / f"{tag}_{split}_target.wav", t0, dur)
        else:
            a = conv48_slice(xi, out_dir / f"{tag}_{split}_input.wav", t0, t1)
            b = conv48_slice(yi, out_dir / f"{tag}_{split}_target.wav", t0, t1)
        rec[split] = {"input_sec": a, "target_sec": b, "src": [str(xi.name), str(yi.name)]}
        print(f"  {tag}_{split}: {a}s in / {b}s out  (src {xi.name})")
    return rec


def convert_all():
    """生成 Marshall JVM410H OD1 数据集: 前级 + 全链路两个采集点, 各取标准中间设置."""
    base = EXTRACT / "Marshall-JVM410H-ChOD1-trimmed"
    # 文件名形如 B050_M050_T050_G050/B050_M050_T050_G050.trainval.target.wav —— 设置是子目录名
    avail = sorted(
        d.name
        for d in (base / "PreampOut" / "trainval").iterdir()
        if d.is_dir() and (d / f"{d.name}.trainval.target.wav").exists()
    )
    print(f"可用 trainval 设置 {len(avail)} 个")

    targets = []
    std = "B050_M050_T050_G050"
    if std in avail:
        targets = [
            ("marshall_jvm410h_chod1", "PreampOut", std),
            ("marshall_jvm410h_chod1_spk", "SpeakerOut", std),
        ]
    else:
        print(f"[WARN] 标准设置 {std} 不在清单, 回退到清单首项 {avail[0]}")
        targets = [
            ("marshall_jvm410h_chod1", "PreampOut", avail[0]),
            ("marshall_jvm410h_chod1_spk", "SpeakerOut", avail[0]),
        ]

    manifest = {"record": RECORD_ID, "zip_md5": EXPECT_MD5,
                "splits": "trainval 300s 时序切 70/10/20 (test 取末尾 60s, 无独立 test 录音)",
                "datasets": {}}
    for tag, cap, setting in targets:
        print(f"[{tag}] {cap} / {setting}")
        rec = convert_set(tag, OUT, setting, cap)
        if rec:
            manifest["datasets"][tag] = {"capture": cap, "setting": setting, "splits": rec}

    json.dump(manifest, open(DATASET / "marshall_manifest.json", "w"), indent=2)
    print("manifest ->", DATASET / "marshall_manifest.json")
    return manifest


def convert(mapping):
    """mapping: {'train': (input_rel, target_rel), ...} 相对 EXTRACT."""
    OUT.mkdir(exist_ok=True)
    manifest = {}
    for split, (xi, yi) in mapping.items():
        xp, yp = EXTRACT / xi, EXTRACT / yi
        if not xp.exists() or not yp.exists():
            print(f"[SKIP] {split}: 缺文件 {xp if not xp.exists() else yp}")
            continue
        a = conv48(xp, OUT / f"marshall_jvm410h_{split}_input.wav")
        b = conv48(yp, OUT / f"marshall_jvm410h_{split}_target.wav")
        manifest[split] = {"input": str(xi), "target": str(yi), "input_sec": a, "target_sec": b}
        print(f"marshall_jvm410h_{split}: {a}s input / {b}s target")
    json.dump(manifest, open(DATASET / "marshall_manifest.json", "w"), indent=2)
    print("manifest ->", DATASET / "marshall_manifest.json")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--verify"
    if mode == "--verify":
        if sanity():
            extract()
            probe()
        else:
            print("[ABORT] 校验失败, 不要解压/转换")
    elif mode == "--convert":
        if not sanity():
            print("[ABORT] md5/size 不符, 拒绝转换")
            sys.exit(1)
        extract()
        convert_all()
    else:
        print(__doc__)
