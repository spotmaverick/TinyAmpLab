"""族 B 训练入口：先向官方 factory 注册自定义架构，再调用官方 nam_full 主流程。

用法（与 nam-full 完全相同的参数）：
    python train_analog.py <data_cfg> <model_cfg> <learning_cfg> <outdir> [--no-show] [--no-plots]

这样族 B 复用的是**官方训练器本体**（同一 LightningModule、同一 loss、同一调度），
我们只多加一步注册，保证与 A2 的公平性。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nam.models import factory as _factory  # noqa: E402

import analog_ssm  # noqa: E402

_factory.register("DiagSSM", analog_ssm.DiagSSM.init_from_config, overwrite=True)
_factory.register("GatedIIR", analog_ssm.GatedIIR.init_from_config, overwrite=True)

from nam.train.full import main as _nam_full  # noqa: E402
from nam.util import timestamp as _timestamp  # noqa: E402


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    if len(argv) < 4:
        print(__doc__)
        sys.exit(1)
    data_p, model_p, learn_p, outdir = argv[:4]
    outdir = Path(outdir, _timestamp())
    outdir.mkdir(parents=True, exist_ok=False)

    data_config = json.load(open(data_p))
    model_config = json.load(open(model_p))
    learning_config = json.load(open(learn_p))

    _nam_full(
        data_config,
        model_config,
        learning_config,
        outdir,
        "--no-show" in flags,
        make_plots="--no-plots" not in flags,
    )


if __name__ == "__main__":
    main()
