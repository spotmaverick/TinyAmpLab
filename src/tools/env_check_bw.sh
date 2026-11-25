#!/usr/bin/env bash
# BW 系列（海光 DCU）新机器环境验收与指纹采集
#
# 用途: 迁到新服务器后，第一步先跑这个，把环境事实全部打出来再决定是否开工。
#       不改动系统，只读。
#
# 用法:
#   bash src/tools/env_check_bw.sh            # 环境指纹
#   bash src/tools/env_check_bw.sh --full     # 额外做 torch 前向/反向冒烟测试
#
# 判读: 见 docs/BW_ENV_SELECTION.md 第三节硬约束表。

set -uo pipefail

FULL=0
[ "${1:-}" = "--full" ] && FULL=1

hr() { printf '\n\033[1m===== %s =====\033[0m\n' "$1"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$1"; }
bad()  { printf '  \033[31m✗\033[0m %s\n' "$1"; }

hr "1. 主机与系统"
hostname 2>/dev/null
date
uname -a
[ -r /etc/os-release ] && grep -E '^(PRETTY_NAME|VERSION_ID)=' /etc/os-release

hr "2. DCU 设备与驱动"
if command -v hy-smi >/dev/null 2>&1; then
  hy-smi --showproductname 2>/dev/null | head -20 || hy-smi 2>/dev/null | head -20
else
  warn "未找到 hy-smi（可能不在计算节点，或路径未加载）"
fi
echo "--- 驱动模块 ---"
lsmod 2>/dev/null | grep -E 'hycu|hydcu|hyttm|hykcl' || warn "未见 hycu/hydcu 模块（登录节点正常）"
echo "--- lspci 设备 ---"
lspci 2>/dev/null | grep -i "co-processor" | head -10 || echo "(无 co-processor 记录)"

hr "3. DTK 版本（决定 torch 该用哪个）"
DTK_FOUND=""
for d in /opt/dtk /opt/dtk-*; do
  [ -d "$d" ] || continue
  v=$(cat "$d/.info/version" 2>/dev/null | head -1)
  echo "  目录: $d   版本: ${v:-未知}"
  [ -n "$v" ] && DTK_FOUND="$v"
done
[ -z "$DTK_FOUND" ] && warn "未找到 /opt/dtk*，请确认 DTK 是否已安装"
if command -v hipcc >/dev/null 2>&1; then
  echo "--- hipcc ---"; hipcc --version 2>/dev/null | head -5
fi
if command -v rocminfo >/dev/null 2>&1; then
  echo "--- rocminfo 架构 ---"
  rocminfo 2>/dev/null | grep -E 'Name:.*gfx|Marketing Name' | head -8
fi

hr "4. Python 与预装包（关键）"
PY=$(command -v python3 || command -v python)
echo "  python: ${PY:-未找到}"
echo "  VIRTUAL_ENV=${VIRTUAL_ENV:-<无>}"
echo "  CONDA_PREFIX=${CONDA_PREFIX:-<无>}"
if [ -n "${PY:-}" ]; then
  "$PY" - <<'PY' 2>&1 | grep -v "HIP initialization\|return torch"
import sys, importlib.metadata as m
print("  python version:", sys.version.split()[0])

def show(name):
    try:
        print(f"  {name:16s} {m.version(name)}")
    except Exception:
        print(f"  {name:16s} <未安装>")

for n in ("torch", "numpy", "soundfile", "scipy", "pytest"):
    show(n)

# torch 详细
try:
    import torch
    print("  --- torch 详情 ---")
    print("  torch.__version__ :", torch.__version__)
    print("  torch.version.hip :", getattr(torch.version, "hip", None))
    print("  cuda.is_available :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("  device_count      :", torch.cuda.device_count())
        print("  device_name(0)    :", torch.cuda.get_device_name(0))
    v = torch.__version__
    if "dtk" in v:
        print("  [OK] 版本带 dtk 后缀 = 海光适配版")
        import re
        mm = re.search(r"dtk(\d+)", v)
        if mm:
            print("  版本内 dtk 代号:", mm.group(1), "(应与 /opt/dtk 的版本对应)")
    else:
        print("  [WARN] 版本不带 dtk 后缀，可能被社区公版覆盖，DCU 可能不可用！")
    if not torch.cuda.is_available():
        print("  [WARN] cuda 不可用：若在登录节点属正常；若在计算节点需排查")
except Exception as e:
    print("  [ERROR] import torch 失败:", e)
PY
fi

hr "5. 本项目依赖是否齐备"
MISSING=0
if [ -n "${PY:-}" ]; then
  for m in numpy soundfile torch; do
    if "$PY" -c "import $m" 2>/dev/null; then
      ok "$m 可用"
    else
      bad "$m 缺失"; MISSING=1
    fi
  done
fi
echo "--- 说明: 本项目不需要 nam / pytorch_lightning / pydantic ---"

hr "6. Slurm 资源"
if command -v sinfo >/dev/null 2>&1; then
  sinfo -o "%.14P %.8T %.6D %.12C %.20G" 2>/dev/null | head -8
  echo "--- 我的配额 ---"
  sacctmgr show assoc where user="$USER" format=Account%18,Partition%14,GrpTRES%30 2>/dev/null | head -5
else
  warn "无 sinfo（可能不是 Slurm 集群）"
fi

if [ "$FULL" = "1" ] && [ -n "${PY:-}" ]; then
  hr "7. torch 前向/反向冒烟测试"
  "$PY" - <<'PY' 2>&1 | grep -v "HIP initialization\|return torch"
import torch
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("  使用设备:", dev)
try:
    import torch.nn as nn
    import torch.nn.functional as F
    # 覆盖本项目实际用到的算子: conv1d(带 dilation) / pad / tanh / sigmoid / matmul
    m = nn.Conv1d(12, 12, 3, dilation=64, padding=128).to(dev)
    x = torch.randn(2, 12, 4096, device=dev, requires_grad=True)
    y = torch.tanh(m(x)) + F.pad(x, (2, 0))[:, :, :4096]
    z = torch.sigmoid(y)
    loss = (z ** 2).mean()
    loss.backward()
    print("  [OK] conv1d(dilation) + tanh + sigmoid + backward 通过")
    print("  loss =", float(loss))
    # 设备端 RNG（物理仿真用）
    try:
        g = torch.Generator(device=dev).manual_seed(42)
        t = torch.empty(8, device=dev).normal_(0, 1, generator=g)
        print("  [OK] 设备端 Generator 可用（物理仿真蒙特卡洛依赖此项）")
    except Exception as e:
        print("  [WARN] 设备端 Generator 不可用:", e)
        print("         => 物理仿真需改用 CPU 生成器，见 BW_ENV_SELECTION.md 五-风险2")
    # float64（物理仿真里 diag 用 double 计算相关性）
    try:
        a = torch.randn(16, device=dev, dtype=torch.float64)
        print("  [OK] float64 可用")
    except Exception as e:
        print("  [WARN] float64 不可用:", e)
except Exception as e:
    print("  [ERROR] 算子测试失败:", repr(e))
PY
fi

hr "汇总提示"
echo "  1) torch 版本必须带 '+das.opt1.dtkNNNN' 后缀且 dtkNNNN 与 /opt/dtk 一致"
echo "  2) 若 cuda.is_available()=False 且当前在登录节点 -> 正常，需 sbatch 到计算节点"
echo "  3) 环境 OK 后第一件事：重跑锚点实验 full_ch12 s7，应复现 test NRMS≈0.0785"
echo "     sbatch sbatch_analog.sh full_ch12 300 7 tsa15"
echo "  4) 再跑物理仿真 sanity，必须 PASS："
echo "     python src/analog/analog_sim_wavenet.py --sanity"
echo "     python src/analog/analog_sim_dw.py --variant dw_ch12 --sanity"
echo
