# 两机训练依赖安装记录（2026-11-27）

目标：在 z100 与 bw2 两台 DCU 机器上补齐训练依赖 `pytorch_lightning` / `auraloss`。
原则：不动已有 torch / numpy ABI；服务器只跑任务，临时安装介质用完即删。

## 安装结果

| 机器 | Python | 目标环境 | 已装版本 | 验证作业 |
|---|---|---|---|---|
| z100 | 3.10.13 | `/work/home/spotmaverick/torch21_env` | torch 2.1.0, pytorch-lightning 2.5.6, torchmetrics 1.9.0, lightning-utilities 0.15.3, auraloss 0.4.0 | 50081023（COMPLETED, DCU True Z100SM） |
| bw2 | 3.11.5 | `/public/home/spotmaverick/t29`（经 `~/p26/bw2env.sh`） | torch 2.9.0 (BW/gfx936), pytorch-lightning 2.5.6, torchmetrics 1.9.0, lightning-utilities 0.15.3, auraloss 0.4.0 | 1256726（COMPLETED, DCU True BW, `randn(64,64,device="cuda").sum()=17.217`） |

附带安装：fsspec[http] 依赖链 `aiohttp 3.14.5 / aiohappyeyeballs 2.7.1 / aiosignal 1.4.0 /
async-timeout 5.0.1 / attrs 26.1.0 / frozenlist 1.8.0 / idna 3.20 / multidict 7.1.0 /
propcache 0.5.4 / yarl 1.25.1`；z100 上 PyYAML 升至 6.0.4、新增 tqdm 4.70.1。

## 执行路径

1. **z100（有外网）**：直接在 `torch21_env` 中
   `pip install "pytorch_lightning>=2.2,<2.6" auraloss`。安装前先 `--dry-run` 确认不会动 torch/numpy。
2. **bw2（物理断网）**：在 z100 上以 py3.11 目标平台下载 wheel：
   ```
   pip download --no-deps --dest <wheels> \
     --python-version 3.11 --implementation cp --abi cp311 \
     --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64 \
     --platform manylinux1_x86_64 --platform manylinux_2_28_x86_64 \
     --only-binary=:all: <14 个包==锁定版本>
   ```
   共 14 个 wheel / 4.8 MB，经本地 Mac 用 `dcu.sh -m z100 pull` → `dcu.sh -m bw2 push` 中转。
   在 bw2 计算节点作业中执行
   `pip install --no-deps --target=$BW2_PYLIBS <wheels>/*.whl`。
3. 两机均用 Slurm 作业验证 import + `device="cuda"` 实算；临时 wheel 目录与安装脚本已从两机删除。

## 关键约束与坑

- **必须用 `--no-deps` 装进 bw2 的 t29**：t29 是扁平 PYTHONPATH 目录，
  依赖解析会升级 numpy / typing_extensions，破坏 torch 2.9 的 numpy 1.x ABI。
  安装前已确认 t29 实际加载版本为 numpy 1.26.4 / pydantic 2.12.5 /
  typing_extensions 4.7.1 / fsspec 2026.3.0 / packaging 23.1 / PyYAML 6.0 / tqdm 4.65.0，
  全部满足 PL 2.5.6 的 `Requires-Dist`。
- `pip list` 在 bw2 上显示的是 conda base 的陈旧元数据（numpy 1.24.3 等），
  **不要以它为准**；以 `import module; print(module.__version__, module.__file__)` 为准。
- z100 登录节点 `import torch` 会因 `libmpi.so.40` 失败（已知坑），验证必须提交到计算节点。
- bw2 登录节点缺 `libmsgpackc.so.2`，同样必须在计算节点验证。
- auraloss 0.4.0 的硬依赖只有 numpy + torch，不引入 librosa；`auraloss.freq.STFTLoss` 已在两机导入通过
  （bw2 上验证；z100 上 `import auraloss` 通过）。

## 本地留存

`/tmp/bw2_traindeps_wheels/bw2_traindeps_wheels/` —— 14 个 wheel，bw2 后续重装环境时可直接复用，
无需再走 z100 下载。
