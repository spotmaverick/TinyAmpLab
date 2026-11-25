#!/bin/bash
# =============================================================================
# TinyAmpLab 数据迁移包（单独打包数据集本体）
#
# 为什么需要单独打:
#   migrate_pack.sh 只含数据清单与重建手册, 不含 1.3GB 音频本体。
#   换服务器时若重下 Marshall (2.76GB, 需分段续传), 耗时且易失败。
#   本脚本把已校验的 wav 打成 tar, 直接传过去最省事。
#
# 用法:
#   bash src/tools/migrate_data.sh <输出目录> [项目根]
# =============================================================================
set -euo pipefail

OUT_DIR="${1:?用法: migrate_data.sh <输出目录> [项目根]}"
SRC_ROOT="${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
STAMP=$(date +%Y%m%d_%H%M)

DATA="$SRC_ROOT/data"
if [ ! -d "$DATA" ]; then
  echo "[ABORT] 找不到数据目录: $DATA"
  exit 1
fi

mkdir -p "$OUT_DIR"
TARBALL="$OUT_DIR/TinyAmpLab_data_${STAMP}.tar.gz"

echo "=== TinyAmpLab 数据打包 ==="
echo "源: $DATA"
echo "文件数: $(find "$DATA" -name '*.wav' | wc -l | tr -d ' ')"

# ---- 完整性检查: 每个 wav 必须可读且有内容 ----
echo "--- 校验音频可读性 ---"
BAD=0
for f in "$DATA"/*.wav; do
  [ -f "$f" ] || continue
  sz=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f" 2>/dev/null || echo 0)
  if [ "$sz" -lt 1000 ]; then
    echo "  [WARN] 异常小文件: $(basename "$f") ($sz bytes)"
    BAD=$((BAD+1))
  fi
done
if [ "$BAD" -gt 0 ]; then
  echo "[ABORT] $BAD 个文件异常，拒绝打包"
  exit 1
fi
echo "  全部通过"

echo "--- 打包 ---"
tar -czf "$TARBALL" -C "$SRC_ROOT" data
SIZE=$(du -h "$TARBALL" | cut -f1)
echo "产物: $TARBALL ($SIZE)"

# ---- 校验和 ----
md5sum "$TARBALL" > "$TARBALL.md5" 2>/dev/null || md5 -q "$TARBALL" > "$TARBALL.md5"
echo "MD5: $(cat "$TARBALL.md5")"

# ---- 清单 ----
cat > "$OUT_DIR/data_manifest_${STAMP}.txt" <<EOF
TinyAmpLab 数据迁移包
生成时间: $(date)
包文件: $(basename "$TARBALL")
大小: $SIZE
MD5: $(cat "$TARBALL.md5")

解压到新服务器的项目根目录:
  tar -xzf $(basename "$TARBALL") -C <项目根>

校验:
  md5sum $(basename "$TARBALL")   # 应等于上面的 MD5

数据集说明见 migrate_pack 里的 dataset/MANIFEST.md
Marshall 原始 zip (2.76GB) 不在此包内, 需重建时用 MANIFEST 的分段续传命令
EOF
echo
echo "=== 完成 ==="
cat "$OUT_DIR/data_manifest_${STAMP}.txt"
