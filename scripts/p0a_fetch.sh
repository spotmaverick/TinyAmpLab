#!/bin/bash
# P0a 13 种子产物回传（专用，绕过 dcu.sh reclaim 固定映射里没有 p0a 的问题）
# 用法: bash scripts/p0a_fetch.sh [--bw] [--bw2] [--z100]   (无参数=三机全拉)
set -u
DCU=~/.workbuddy/skills/sugon-dcu-fleet/scripts/dcu.sh
ROOT=/Users/wangguan/WorkBuddy/2026-11-23-15-15-52/analog-nam-gated-block-sim
DEST=$ROOT/reclaimed/p0a_20261127
mkdir -p "$DEST/bw" "$DEST/bw2" "$DEST/z100"

DO_BW=0; DO_BW2=0; DO_Z100=0
if [ $# -eq 0 ]; then DO_BW=1; DO_BW2=1; DO_Z100=1; fi
for a in "$@"; do
  case "$a" in
    --bw) DO_BW=1 ;; --bw2) DO_BW2=1 ;; --z100) DO_Z100=1 ;;
  esac
done

fetch_one() {
  local mach=$1 srcdir=$2 outsub=$3
  echo "[$mach] 检查 $srcdir ..."
  local cnt
  cnt=$($DCU -m "$mach" ssh "ls $srcdir/*.nam 2>/dev/null | wc -l" 2>/dev/null | tail -1 | tr -d ' \r')
  if [ -z "$cnt" ] || [ "$cnt" -eq 0 ]; then echo "  无 nam，跳过"; return 1; fi
  echo "  发现 $cnt 个 nam，开始拉取"
  $DCU -m "$mach" pull "$srcdir" "$DEST/$outsub" 2>&1 | tail -2
  # 拍平（pull 会在目标下建同名目录）
  if [ -d "$DEST/$outsub/$(basename $srcdir)" ]; then
    mv "$DEST/$outsub/$(basename $srcdir)"/*.nam "$DEST/$outsub/" 2>/dev/null
    rmdir "$DEST/$outsub/$(basename $srcdir)" 2>/dev/null
  fi
  ls -la "$DEST/$outsub"/*.nam 2>/dev/null | awk '{print "  "$9, $5"B"}'
}

[ "$DO_BW" = 1 ]  && fetch_one bw   /workspace/TinyAmpLab/p0a/seeds            bw
[ "$DO_BW2" = 1 ] && fetch_one bw2  /public/home/spotmaverick/nam_proj/p1/seeds bw2
[ "$DO_Z100" = 1 ] && fetch_one z100 /work/home/spotmaverick/nam_project/p0a_seeds z100

echo
echo "=== 本地汇总 $DEST ==="
find "$DEST" -name "*.nam" 2>/dev/null | sort | while read -r f; do
  printf '  %-12s %s\n' "$(basename "$f")" "$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f")B"
done
echo
echo "已有种子数: $(find "$DEST" -name '*.nam' 2>/dev/null | wc -l | tr -d ' ') / 13"
echo "$(date +%F_%H%M) p0a_fetch $DEST" >> "$ROOT/reclaimed/MANIFEST.txt" 2>/dev/null
