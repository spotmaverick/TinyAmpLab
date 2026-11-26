"""官方 nam-full 的种子可控入口（族 A/B 通用）

官方 `nam/train/full.py` 模块导入时硬编码 `torch.manual_seed(0)`，且训练过程不再设种子，
因此 `nam-full` 命令行无法做多种子复现。本脚本在**调用官方 main 之前**重设种子，
其余流程（数据集、LightningModule、loss、scheduler、导出）完全是官方的。

用法：
    python train_seeded.py <data_cfg> <model_cfg> <learning_cfg> <outdir> <seed> [--no-show] [--no-plots]
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from nam.models import factory as _factory  # noqa: E402

# 族 B 可选注册（若配置里用到）
try:
    import analog_ssm  # noqa: E402

    _factory.register("DiagSSM", analog_ssm.DiagSSM.init_from_config, overwrite=True)
    _factory.register("GatedIIR", analog_ssm.GatedIIR.init_from_config, overwrite=True)
except Exception:  # noqa: BLE001
    pass

try:
    import analog_rec  # noqa: E402

    _factory.register(
        "RecurrentNet", analog_rec.RecurrentNet.init_from_config, overwrite=True
    )
except Exception:  # noqa: BLE001
    pass

try:
    import analog_scaled_wavenet  # noqa: E402

    _factory.register(
        "ScaledWaveNet",
        analog_scaled_wavenet.ScaledWaveNet.init_from_config,
        overwrite=True,
    )
except Exception:  # noqa: BLE001
    pass

import nam.train.full as _full  # noqa: E402  确保模块已导入（其内部的 manual_seed(0) 已执行）
from nam.util import timestamp as _timestamp  # noqa: E402


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 训练用确定性算子会显著变慢，这里只固定种子（与官方行为一致，不额外开 deterministic）


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    if len(argv) < 5:
        print(__doc__)
        sys.exit(1)
    data_p, model_p, learn_p, outdir, seed = argv[:5]
    seed = int(seed)

    outdir = Path(outdir, _timestamp())
    outdir.mkdir(parents=True, exist_ok=False)

    data_config = json.load(open(data_p))
    model_config = json.load(open(model_p))
    learning_config = json.load(open(learn_p))

    # 关键：在官方 main 之前重设种子
    set_seed(seed)
    print(f"[train_seeded] seed={seed} outdir={outdir}", flush=True)

    _full.main(
        data_config,
        model_config,
        learning_config,
        outdir,
        "--no-show" in flags,
        make_plots="--no-plots" not in flags,
    )


if __name__ == "__main__":
    main()
