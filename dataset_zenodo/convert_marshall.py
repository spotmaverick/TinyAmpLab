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

HERE = Path(__file__).resolve().parent
ZIP = HERE / "Marshall-JVM410H-ChOD1.zip"
EXTRACT = HERE / "marshall_extract"
OUT = HERE.parent / "data"
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
    EXTRACT.mkdir(exist_ok=True)
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
    json.dump(manifest, open(HERE / "marshall_manifest.json", "w"), indent=2)
    print("manifest ->", HERE / "marshall_manifest.json")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--verify"
    if mode == "--verify":
        if sanity():
            extract()
            probe()
            print("\n下一步: 根据上面 wav 清单, 用 --convert 并传入映射 (见脚本 convert() 用法)")
        else:
            print("[ABORT] 校验失败, 不要解压/转换")
    elif mode == "--convert":
        # 映射在确认结构后填写; 默认按 ToneTwist 常见 DRY/trainval 结构尝试
        mapping = {
            "train": ("DRY/trainval/train.input.wav", "Marshall-JVM410H-ChOD1/trainval/G050/G050.train.target.wav"),
            "val": ("DRY/trainval/val.input.wav", "Marshall-JVM410H-ChOD1/trainval/G050/G050.val.target.wav"),
            "test": ("DRY/test/test.input.wav", "Marshall-JVM410H-ChOD1/test/G050/G050.test.target.wav"),
        }
        convert(mapping)
    else:
        print(__doc__)
