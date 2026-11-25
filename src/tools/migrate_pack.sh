#!/bin/bash
# =============================================================================
# TinyAmpLab 迁移打包脚本
#
# 用途: 从旧智算服务器（或本地）打出一个自包含的迁移包，包含
#   - 全部源码与可复现脚本
#   - 全部实验结果 (json / log / 权重)
#   - 数据集清单与校验信息（不含大文件本体）
#   - 一键重建环境的部署脚本
#
# 用法:
#   bash src/tools/migrate_pack.sh <输出目录> [服务器根目录]
#   在服务器上跑: bash src/tools/migrate_pack.sh /work/home/spotmaverick/migrate_pack
#   在本地跑:     bash src/tools/migrate_pack.sh ./migrate_pack
#
# 产物: <输出目录>/TinyAmpLab_migrate_<日期>.tar.gz
# =============================================================================
set -euo pipefail

OUT_DIR="${1:?用法: migrate_pack.sh <输出目录> [项目根]}"
SRC_ROOT="${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
STAMP=$(date +%Y%m%d_%H%M)
PKG_NAME="TinyAmpLab_migrate_${STAMP}"

WORK="$(mktemp -d)"
STAGE="$WORK/$PKG_NAME"
mkdir -p "$STAGE" "$OUT_DIR"

echo "=== TinyAmpLab 迁移打包 ==="
echo "源目录: $SRC_ROOT"
echo "暂存:   $STAGE"
echo

# -----------------------------------------------------------------------------
# 1. 代码与脚本（排除大文件与缓存）
# -----------------------------------------------------------------------------
echo "[1/6] 收集代码与脚本..."
mkdir -p "$STAGE/code"
for f in arch_analog_friendly.py arch_search_v2.py analog_sim_wavenet.py \
         wavenet_official_dcu.py diag_three.py diag_job.sh sbatch_arch2.sh \
         sbatch_analog.sh analog_sim_wavenet_job.sh; do
  [ -f "$SRC_ROOT/$f" ] && cp "$SRC_ROOT/$f" "$STAGE/code/" || true
done
if [ -d "$SRC_ROOT/src" ]; then
  rsync -a --exclude '__pycache__' --exclude '*.pyc' \
        "$SRC_ROOT/src/" "$STAGE/code/src/"
fi
[ -d "$SRC_ROOT/docs" ] && rsync -a "$SRC_ROOT/docs/" "$STAGE/code/docs/"
[ -f "$SRC_ROOT/README.md" ] && cp "$SRC_ROOT/README.md" "$STAGE/code/"
[ -f "$SRC_ROOT/overview.md" ] && cp "$SRC_ROOT/overview.md" "$STAGE/code/"
echo "      代码文件数: $(find "$STAGE/code" -type f | wc -l | tr -d ' ')"

# -----------------------------------------------------------------------------
# 2. 实验结果（json / log / 权重）
# -----------------------------------------------------------------------------
echo "[2/6] 收集实验结果..."
mkdir -p "$STAGE/results"
# 服务器端与本地端目录名不同（服务器 arch_v2 / 本地 arch_v2_local），两者都收
for pair in "arch_v2:arch_v2" "arch_analog:arch_analog" \
            "arch_v2_local:arch_v2" "arch_analog_local:arch_analog" \
            "analog_sim:analog_sim" "diag3:diag3" "diag2:diag2" \
            "diag_three:diag_three" "experiments:experiments"; do
  src="${pair%%:*}"; dst="${pair##*:}"
  [ -d "$SRC_ROOT/$src" ] || continue
  mkdir -p "$STAGE/results/$dst"
  rsync -a --include='*/' --include='*.json' --include='*.log' --include='*.out' \
        --exclude='*' "$SRC_ROOT/$src/" "$STAGE/results/$dst/" 2>/dev/null || true
done
# 权重单独收集（.pt 必须带走）
for d in arch_v2 arch_analog arch_v2_local arch_analog_local; do
  [ -d "$SRC_ROOT/$d" ] || continue
  mkdir -p "$STAGE/results/$d"
  find "$SRC_ROOT/$d" -name '*.pt' -exec rsync -a {} "$STAGE/results/$d/" \; 2>/dev/null || true
done
echo "      结果文件数: $(find "$STAGE/results" -type f | wc -l | tr -d ' ')"
echo "      json: $(find "$STAGE/results" -name '*.json' | wc -l | tr -d ' ')  "\
     "log: $(find "$STAGE/results" -name '*.log' | wc -l | tr -d ' ')  "\
     "pt: $(find "$STAGE/results" -name '*.pt' | wc -l | tr -d ' ')"

# -----------------------------------------------------------------------------
# 3. 数据集清单（不含大文件本体）
# -----------------------------------------------------------------------------
echo "[3/6] 生成数据清单..."
mkdir -p "$STAGE/dataset"
cat > "$STAGE/dataset/MANIFEST.md" <<'EOF'
# 数据集清单

| 数据集 | 采样率 | 来源 | 本地文件 | 校验 |
|---|---|---|---|---|
| TSA15 | 48kHz | ToneTwist AFx 公开集 | `data/tsa15_{train,val,test}_{input,target}.wav` | — |
| Blackstar HT-212 | 48kHz | ToneTwist AFx | `data/blackstar_ht212_*` | — |
| Marshall JVM410H ChOD1 | 44.1k→48k | Zenodo 10892012 | `data/marshall_jvm410h_chod1*` | md5 `05ee998404f44c3a449c4ad6a8160cb6` (zip) |
| Marshall SpeakerOut | 44.1k→48k | 同上 | `data/marshall_jvm410h_chod1_spk_*` | 同上 |
| UA1176 | 48kHz | UA 公开集 | `data/ua1176_*` | — |

## 重建方式

数据本体不入包（体积大），在新服务器上按下列方式重建：

### Marshall（Zenodo）
```bash
# 分段续传，避免大文件截断
cd <DATASET_DIR>
URL="https://zenodo.org/api/records/10892012/files/Marshall-JVM410H-ChOD1.zip/content"
SIZE=2755382616
N=10; SEG=$(( (SIZE + N - 1) / N ))
mkdir -p Marshall-JVM410H-ChOD1.zip.parts
for i in $(seq 0 $((N-1))); do
  START=$(( i * SEG )); END=$(( START + SEG - 1 ))
  [ $END -ge $SIZE ] && END=$(( SIZE - 1 ))
  curl -s --retry 5 -r "${START}-${END}" "$URL" -o "Marshall-JVM410H-ChOD1.zip.parts/seg_$i"
done
cat Marshall-JVM410H-ChOD1.zip.parts/seg_* > Marshall-JVM410H-ChOD1.zip
md5 -q Marshall-JVM410H-ChOD1.zip   # 必须等于 05ee998404f44c3a449c4ad6a8160cb6
unzip -q Marshall-JVM410H-ChOD1.zip
```

### 转换脚本
`code/src/tools/convert_marshall.py --convert`（自带 md5 校验，校验不过拒绝转换）

### 其余数据集
`code/src/tools/convert_new_amps.py` / `convert_ua.py`
EOF

# 数据文件清单（仅列名与大小）
if [ -d "$SRC_ROOT/data" ]; then
  (cd "$SRC_ROOT/data" && ls -la *.wav 2>/dev/null | awk '{print $5, $9}') \
    > "$STAGE/dataset/data_files.txt" || true
fi
echo "      清单行数: $(wc -l < "$STAGE/dataset/data_files.txt" 2>/dev/null || echo 0)"

# -----------------------------------------------------------------------------
# 4. 环境规格
# -----------------------------------------------------------------------------
echo "[4/6] 记录环境规格..."
mkdir -p "$STAGE/env"
cat > "$STAGE/env/ENV_SPEC.md" <<'EOF'
# 环境规格（旧服务器实测，用于在新服务器复现）

## 关键版本
| 组件 | 版本 | 说明 |
|---|---|---|
| python | 3.8.10 | 集群自带 `python/3.8.10` module |
| torch | 1.10.0a0+git450cdd1-dtk22.04 | 集群 DTK 自带，**勿用 pip 覆盖** |
| numpy | 1.24.4 | |
| soundfile | 0.12.1 | |
| scipy | 待确认 | |
| neural-amp-modeler | 需 `pip install --only-binary=:all:` | 从源码编译会失败 |

## 关键 module（DTK 环境）
```
module purge
module load compiler/devtoolset/7.3.1 compiler/dtk/22.04.1 mpi/hpcx/gcc-7.3.1 python/3.8.10
```

## LD_LIBRARY_PATH
DTK 的 `.so` 分散在：
```
/public/software/apps/DeepLearning/PyTorch_Lib
/public/software/apps/deeplearning-depend
```
需全部加入 `LD_LIBRARY_PATH`（旧服务器用 find 动态收集）。

## PYTHONPATH
```
/work/home/spotmaverick/pylibs
```
（nam_probe 目录曾放 pylibs，新服务器路径需相应调整）

## DCU 使用要点（踩坑记录）
1. **device 字符串是 `cuda`，不是 `dpu`** —— torch 的 DCU 后端映射到 `torch.cuda` API。
2. **登录节点没有 DCU**，只能在计算节点跑；验证小逻辑可用 CPU 在登录节点跑。
3. **Slurm 资源申请**：
   ```
   sbatch -p wzhdtest -c 8 --mem=16G --gres=dcu:1
   ```
   - `--mem` 不能超限（16G 上限）
   - 必须带 `--gres=dcu:1`，否则作业不分配 DCU
4. **配额限制**：账户 DCU 配额有限，作业会因 `AssocGrpGRES` 排队。
5. **torch.load 的 device 不匹配**：`sd` 在 CPU、输入在 DCU 时会报 device 错误，需按输入设备转换权重。

## 已知环境坑
- **Miniconda glibc 不兼容**：必须用 `py310_23.5.2` 旧版，新版与集群 glibc 冲突。
- **pip 源码编译失败**：`pip install neural-amp-modeler` 会尝试编译 numpy 扩展而失败，必须 `--only-binary=:all:`。
- **NAM CLI 会进 GUI 报错**：不要用 CLI，直接用 Python API。
- **官方 config schema**：`sample_rate` 必填，`ny: null`。
EOF

cat > "$STAGE/env/run_env.sh" <<'EOF'
#!/bin/bash
# DCU 环境激活（新服务器路径需按实际调整）
module purge
module load compiler/devtoolset/7.3.1 compiler/dtk/22.04.1 mpi/hpcx/gcc-7.3.1 python/3.8.10 2>&1
for so in $(find /public/software/apps/DeepLearning/PyTorch_Lib /public/software/apps/deeplearning-depend -name "*.so*" -type f 2>/dev/null); do
  d=$(dirname $so); echo "$LD_LIBRARY_PATH" | grep -q ":$d:" || export LD_LIBRARY_PATH="$d:$LD_LIBRARY_PATH"
done
export PYTHONPATH=/work/home/spotmaverick/pylibs:$PYTHONPATH
EOF
chmod +x "$STAGE/env/run_env.sh"

# 采集实际版本快照
{
  echo "### 采集时间: $(date)"
  if command -v python >/dev/null 2>&1; then
    python -c "import sys; print('python', sys.version)" 2>/dev/null
    python -c "import torch; print('torch', torch.__version__)" 2>/dev/null
    python -c "import numpy; print('numpy', numpy.__version__)" 2>/dev/null
    python -c "import soundfile; print('soundfile', soundfile.__version__)" 2>/dev/null
  fi
  pip list 2>/dev/null | head -60
} > "$STAGE/env/pkgs_snapshot.txt" 2>&1 || true
echo "      环境快照已记录"

# -----------------------------------------------------------------------------
# 5. 部署脚本
# -----------------------------------------------------------------------------
echo "[5/6] 生成部署脚本..."
cat > "$STAGE/deploy.sh" <<'EOF'
#!/bin/bash
# =============================================================================
# TinyAmpLab 新服务器部署脚本
#   bash deploy.sh <项目根目录> [数据目录]
# =============================================================================
set -euo pipefail

PROJ="${1:?用法: deploy.sh <项目根> [数据目录]}"
DATA_DIR="${2:-$PROJ/data}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== TinyAmpLab 部署 ==="
echo "项目根: $PROJ"
echo "数据根: $DATA_DIR"

mkdir -p "$PROJ"

# 1. 代码
echo "[1/4] 部署代码..."
rsync -a "$HERE/code/" "$PROJ/"

# 2. 结果
echo "[2/4] 部署实验结果..."
for d in "$HERE/results/"*/; do
  [ -d "$d" ] || continue
  name=$(basename "$d")
  mkdir -p "$PROJ/$name"
  rsync -a "$d" "$PROJ/$name/"
done

# 3. 环境
echo "[3/4] 配置环境..."
mkdir -p "$PROJ/env"
cp "$HERE/env/run_env.sh" "$PROJ/env/run_env.sh"
chmod +x "$PROJ/env/run_env.sh"
echo "  激活方式: source $PROJ/env/run_env.sh"

echo "  安装依赖 (若集群未自带)..."
echo "    pip install --only-binary=:all: neural-amp-modeler soundfile scipy"

# 4. 数据
echo "[4/4] 数据..."
mkdir -p "$DATA_DIR"
if [ -f "$HERE/dataset/MANIFEST.md" ]; then
  cp "$HERE/dataset/MANIFEST.md" "$PROJ/docs/DATASET_MANIFEST.md" 2>/dev/null || \
    cp "$HERE/dataset/MANIFEST.md" "$PROJ/DATASET_MANIFEST.md"
fi
if [ ! -f "$DATA_DIR/tsa15_train_input.wav" ]; then
  echo "  ⚠️  数据目录为空，需按 DATASET_MANIFEST.md 重建数据集"
  echo "     Marshall: 见 manifest 里的分段续传命令"
else
  echo "  ✓ 数据已存在: $(ls "$DATA_DIR"/*.wav 2>/dev/null | wc -l | tr -d ' ') 个 wav"
fi

echo
echo "=== 部署完成 ==="
echo "下一步:"
echo "  1. source $PROJ/env/run_env.sh"
echo "  2. cd $PROJ && python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
echo "  3. sbatch -p wzhdtest -c 8 --mem=16G --gres=dcu:1 <job>.sh"
EOF
chmod +x "$STAGE/deploy.sh"

# -----------------------------------------------------------------------------
# 6. 打包
# -----------------------------------------------------------------------------
echo "[6/6] 打包..."
TARBALL="$OUT_DIR/$PKG_NAME.tar.gz"
tar -czf "$TARBALL" -C "$WORK" "$PKG_NAME"
SIZE=$(du -h "$TARBALL" | cut -f1)
echo
echo "=== 打包完成 ==="
echo "产物: $TARBALL  ($SIZE)"
echo
echo "内容概览:"
tar -tzf "$TARBALL" | head -20
echo "..."
echo "总文件数: $(tar -tzf "$TARBALL" | wc -l | tr -d ' ')"

# 校验和
md5 -q "$TARBALL" > "$TARBALL.md5" 2>/dev/null || true
echo "MD5: $(cat "$TARBALL.md5" 2>/dev/null)"

rm -rf "$WORK"
echo
echo "迁移提示: 把这个 tar.gz 传到新服务器, 解压后运行"
echo "  bash $PKG_NAME/deploy.sh /work/home/<user>/nam_project"
