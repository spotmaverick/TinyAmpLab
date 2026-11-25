#!/bin/bash
# 第三轮审计后的最小判定集批量提交
#
# 依据：docs/reports/AUDIT_ROUND3_STATS_AND_COST.md 第五节
# 目标：用最少的作业把三条结论从「观察」升级为「有误差棒的判定」
#
# 用法（在智算服务器项目根目录执行）：
#   bash submit_round3.sh            # 只打印将要提交的作业（dry run）
#   bash submit_round3.sh --go       # 真正提交
#
# 预计：6 个作业 × 约 3h ≈ 18h 单卡时间；8 卡并行约 3h。

set -u
GO=0
LOCAL=0
for arg in "$@"; do
  case "$arg" in
    --go)    GO=1 ;;
    --local) LOCAL=1 ;;
  esac
done

PROJ=/work/home/spotmaverick/nam_project
[ "$LOCAL" -eq 1 ] && PROJ=$(cd "$(dirname "$0")" && pwd)
CANCEL_ON_FAIL=1

# ------------------------------------------------------------------ 作业清单
# 格式: 变体 种子 用途 脚本
# 脚本: analog = arch_analog_friendly.py（dw_* / full_* / wd_* 系列）
#       v2     = arch_search_v2.py（官方四档 nano/feather/lite/standard）
JOBS=(
  # ★★★ H2 唯一候选点，目前只有 s7，补 s11 给出误差棒
  "dw_ch24 11 H2-误差棒 analog"

  # ★★★ H3 目前 dw_ch12_rf50 只有 s7，而 lite_rf50 有 s7/s11
  #       补这一个才能做真正的配对比较
  "dw_ch12_rf50 11 H3-配对补齐 analog"

  # ★★★ 新方向：宽 depthwise + 低秩 1x1（6445 参数，交叉开关为 lite 的 74%）
  #       预登记判据见 param_flow.py
  "wd_ch24_r6 7 lowrank-A-s7 analog"
  "wd_ch24_r6 11 lowrank-A-s11 analog"

  # ★★ 补全 b_local 的容量趋势（当前只有 s7）
  #    ⚠ standard 是官方 13801 参数架构，必须用 arch_search_v2.py，
  #      不能用 arch_analog_friendly.py（后者只按 ch 缩放，无法表达 standard）
  "standard_rf50 11 b_local趋势 v2"

  # ★★ 上界探针（1.61× lite 参数，非等参数，仅探测低秩能否压到 0.075 以下）
  "wd_ch32_r8 7 lowrank-B-上界 analog"
)

echo "================================================================"
echo "第三轮审计 · 最小判定集"
echo "================================================================"
echo "项目目录: $PROJ"
echo "作业数: ${#JOBS[@]}"
echo

cd "$PROJ" || { echo "FATAL: 目录不存在 $PROJ"; exit 1; }

# 检查脚本与数据是否就位
for f in arch_analog_friendly.py sbatch_analog.sh; do
  [ -f "$f" ] || { echo "FATAL: 缺 $f"; exit 1; }
done
[ -d data ] || { echo "FATAL: 缺 data/"; exit 1; }

# 确认新变体解析正常（本地 CPU 即可验证参数与成本）
echo "--- 成本预检（应无异常）---"
python3 - <<'PY'
import sys
sys.path.insert(0, "src/tools")
try:
    from param_flow import breakdown
except ImportError:
    print("  [skip] param_flow 不在，跳过成本预检")
    sys.exit(0)
for v, dw, ch, rf, r in [
    ("dw_ch24", True, 24, 100, None),
    ("dw_ch12_rf50", True, 12, 50, None),
    ("wd_ch24_r6", True, 24, 100, 6),
    ("wd_ch32_r8", True, 32, 100, 8),
    ("standard_rf50", False, 12, 50, None),
]:
    t, x, m, c, _ = breakdown(ch, dw, rf, mix_rank=r)
    print(f"  {v:16s} 参数={t['total']:6d} 交叉开关={x:6d} MACs={m:6d} 链={c:3d}")
PY
echo

# 提交
N=0
for row in "${JOBS[@]}"; do
  set -- $row
  VAR=$1; SEED=$2; WHY=${3:-}; SCRIPT=${4:-analog}
  N=$((N+1))
  if [ "$SCRIPT" = "v2" ]; then
    SB=sbatch_arch2.sh
  else
    SB=sbatch_analog.sh
  fi
  if [ ! -f "$SB" ]; then
    echo "[$N] SKIP  $VAR s$SEED  缺少提交脚本 $SB"
    continue
  fi
  if [ "$GO" -eq 1 ]; then
    JID=$(sbatch --parsable "$SB" "$VAR" 300 "$SEED" tsa15 2>&1)
    RC=$?
    if [ "$RC" -ne 0 ]; then
      echo "[$N] FAILED  $VAR s$SEED  -> $JID"
      [ "$CANCEL_ON_FAIL" -eq 1 ] && { echo "中止：提交失败"; exit 1; }
    else
      echo "[$N] 提交 $JID  $VAR s$SEED   ($WHY, $SB)"
    fi
  else
    echo "[$N] (dry) sbatch $SB $VAR 300 $SEED tsa15   ($WHY)"
  fi
done

echo
if [ "$GO" -eq 0 ]; then
  echo "以上为 dry run。确认无误后执行：bash submit_round3.sh --go"
else
  echo "已提交 $N 个作业。用 squeue -u \$USER 查看，或跑轮询脚本。"
fi
