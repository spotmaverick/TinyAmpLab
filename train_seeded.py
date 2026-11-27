"""官方 nam-full 的种子可控入口（族 A/B 通用），支持 checkpoint 续训中断恢复。

官方 `nam/train/full.py` 模块导入时硬编码 `torch.manual_seed(0)`，且训练过程不再设种子，
因此 `nam-full` 命令行无法做多种子复现。本脚本在**调用官方 main 之前**重设种子，
其余流程（数据集、LightningModule、loss、scheduler、导出）完全是官方的。

官方代码一行不改，续训通过 `trainer_fit_kwargs={"ckpt_path": ...}` 注入
（`full.main` 内部本来就是 `trainer.fit(..., **learning_config.get("trainer_fit_kwargs", {}))`）。

用法：
    # 从零训练
    python train_seeded.py <data_cfg> <model_cfg> <learning_cfg> <outdir> <seed> [--no-show] [--no-plots]

    # 续训：ckpt 可以从 `--resume-from` 自动定位 run 目录，不必再传 outdir
    python train_seeded.py <data_cfg> <model_cfg> <learning_cfg> --resume-from <ckpt> <seed> [--max-epochs N]

续训前提（已实测确认）：
- `checkpoint_epoch_*.ckpt` 含 `optimizer_states` / `lr_schedulers` / `epoch` / `global_step`，不是空壳；
- `Dataset.__getitem__` 是 `i = idx * ny` 的确定性切分，无随机采样，换机后数据侧一致；
- `train_dataloader.shuffle=True` 的 batch 顺序**不在** ckpt 里，续训后该 epoch 的 shuffle 序列
  与"不中断连跑"不同。这是统计学上的微小差异，不是 bug，但严格逐 batch 复现做不到。
"""
import argparse
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


def _find_run_dir_from_ckpt(ckpt: Path) -> Path:
    """从 ckpt 反推官方 main 需要的 outdir。

    Lightning 的目录约定是 <outdir>/<timestamp>/lightning_logs/version_N/checkpoints/*.ckpt，
    官方 main 的 outdir 其实是那个 <timestamp> 层（它在里面写 config_*.json）。
    这里不靠固定层级硬猜，改为向上找到第一个**含 config_learning.json** 的目录——
    config_*.json 正是 full.main 自己写进 outdir 的，用它做锚点最可靠。
    """
    for p in ckpt.resolve().parents:
        if (p / "config_learning.json").is_file():
            return p
    raise SystemExit(
        f"[train_seeded] 无法从 {ckpt} 向上定位 run 目录（找不到 config_learning.json）。"
        f"该 ckpt 可能不是 full.main 产出的，请显式传 outdir。"
    )


def _parse_args(argv_raw: list[str]) -> argparse.Namespace:
    """手写解析：位置参数 4 或 5 个，flag 分"无值开关"和"带值选项"两类。

    不能用朴素 `[a for a in argv if not a.startswith("--")]`——那样 `--max-epochs 120`
    里的 `120` 会被当成位置参数，直接把位置参数个数搞乱。
    """
    value_opts = {"resume-from", "max-epochs"}
    flags: set[str] = set()
    opts: dict[str, str] = {}
    positional: list[str] = []

    i = 0
    while i < len(argv_raw):
        a = argv_raw[i]
        if a.startswith("--"):
            key = a[2:]
            if key in value_opts:
                if i + 1 >= len(argv_raw):
                    raise SystemExit(f"[train_seeded] --{key} 缺少取值")
                opts[key] = argv_raw[i + 1]
                i += 2
                continue
            flags.add(key)
            i += 1
            continue
        positional.append(a)
        i += 1

    return argparse.Namespace(
        positional=positional, flags=flags, opts=opts
    )


def main():
    args = _parse_args(sys.argv[1:])
    positional, flags, opts = args.positional, args.flags, args.opts

    resume_ckpt = opts.get("resume-from")
    max_epochs = opts.get("max-epochs")

    if resume_ckpt:
        ckpt = Path(resume_ckpt)
        if not ckpt.is_file():
            raise SystemExit(f"[train_seeded] ckpt 不存在：{ckpt}")
        if len(positional) < 4:
            print(__doc__)
            raise SystemExit(1)
        # 续训：位置参数是 cfg/cfg/cfg/seed（outdir 从 ckpt 反推）
        data_p, model_p, learn_p, seed = positional[:4]
        outdir = _find_run_dir_from_ckpt(ckpt)
        if len(positional) >= 5:
            given = Path(positional[4])
            if given.resolve() != outdir.resolve():
                raise SystemExit(
                    f"[train_seeded] 显式 outdir({given}) 与 ckpt 反推的 run 目录({outdir}) 不一致，"
                    f"续训只能写回原 run 目录，否则 checkpoint 路径分裂。"
                )
    else:
        if len(positional) < 5:
            print(__doc__)
            raise SystemExit(1)
        data_p, model_p, learn_p, outdir, seed = positional[:5]
        outdir = Path(outdir)

    seed = int(seed)
    data_config = json.load(open(data_p))
    model_config = json.load(open(model_p))
    learning_config = json.load(open(learn_p))

    if resume_ckpt:
        # 已有 run 目录，不能 exist_ok=False（这正是之前续训被堵死的真正原因）
        outdir = outdir.resolve()
        print(f"[train_seeded] RESUME from {ckpt.name} -> outdir={outdir}", flush=True)
        learning_config.setdefault("trainer_fit_kwargs", {})
        learning_config["trainer_fit_kwargs"]["ckpt_path"] = str(ckpt.resolve())
        # 不覆盖 official 已写好的 config_*.json，但把本次意图记下来便于回溯
        with open(Path(outdir, "config_resume.json"), "w") as fp:
            json.dump(
                {"ckpt": str(ckpt.resolve()), "seed": seed, "max_epochs": max_epochs},
                fp,
                indent=2,
            )
        if max_epochs is not None:
            learning_config["trainer"]["max_epochs"] = int(max_epochs)
        print(
            f"[train_seeded] max_epochs={learning_config['trainer']['max_epochs']} "
            f"ckpt_path={learning_config['trainer_fit_kwargs']['ckpt_path']}",
            flush=True,
        )
    else:
        outdir = Path(outdir, _timestamp())
        outdir.mkdir(parents=True, exist_ok=False)
        if max_epochs is not None:
            learning_config["trainer"]["max_epochs"] = int(max_epochs)
        print(f"[train_seeded] seed={seed} outdir={outdir}", flush=True)

    # 关键：在官方 main 之前重设种子（续训同样要做，否则续训段的随机性不受控）
    set_seed(seed)

    _full.main(
        data_config,
        model_config,
        learning_config,
        outdir,
        "no-show" in flags,          # flags 里存的是去掉 "--" 后的名字，不能写成 "--no-show"
        make_plots="no-plots" not in flags,
    )


if __name__ == "__main__":
    main()
