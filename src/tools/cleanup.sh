#!/usr/bin/env bash
# TinyAmpLab 项目清理器
#
# 设计原则：白名单式删除。只清理「可再生的缓存」，绝不触碰数据、权重、实验日志。
# 默认 dry-run（只报告不删除），加 --apply 才真正执行。
#
#   ./src/tools/cleanup.sh                 # 预演，列出将删除的内容与体积
#   ./src/tools/cleanup.sh --apply         # 实际清理
#   ./src/tools/cleanup.sh --apply --gc    # 额外做一次 git 对象回收
#   ./src/tools/cleanup.sh --install-hook  # 安装 pre-commit 钩子，之后每次提交自动跑
#
# 永不删除（任何参数都不例外）：
#   .git/  data/  sync/  dataset_zenodo/  venv/  node_modules/
#   —— 即：数据、权重、原始 zip、断点续传缓存、实验日志全部受保护。
#
# 兼容 macOS 自带 bash 3.2。

set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

APPLY=0
QUIET=0
DO_DIRS=1
DO_GC=0
INSTALL_HOOK=0

for arg in "$@"; do
  case "$arg" in
    --apply)        APPLY=1 ;;
    --quiet|-q)     QUIET=1 ;;
    --no-dirs)      DO_DIRS=0 ;;
    --gc)           DO_GC=1 ;;
    --install-hook) INSTALL_HOOK=1 ;;
    -h|--help)
      sed -n '3,17p' "${BASH_SOURCE[0]}" | sed 's/^# */# /'
      exit 0 ;;
    *) echo "未知参数：$arg（用 --help 查看用法）" >&2; exit 2 ;;
  esac
done

# ---------- 根目录自检：防止在错误目录下误删 ----------
if [ ! -d "$ROOT/.git" ] || [ ! -f "$ROOT/README.md" ]; then
  echo "拒绝执行：$ROOT 不像 TinyAmpLab 项目根（缺 .git 或 README.md）" >&2
  exit 3
fi
if [ "$ROOT" = "/" ] || [ "$ROOT" = "$HOME" ]; then
  echo "拒绝执行：根目录落在 $ROOT" >&2
  exit 3
fi

log() { if [ "$QUIET" -eq 0 ]; then printf '%s\n' "$*"; fi; }

# ---------- 受保护路径 ----------
is_protected() {
  case "$1" in
    "$ROOT/.git"/*|"$ROOT/.git")        return 0 ;;
    "$ROOT/data"/*|"$ROOT/data")        return 0 ;;
    "$ROOT/sync"/*|"$ROOT/sync")        return 0 ;;
    "$ROOT/dataset_zenodo"/*|"$ROOT/dataset_zenodo") return 0 ;;
    "$ROOT/venv"/*|"$ROOT/.venv"/*|"$ROOT/node_modules"/*) return 0 ;;
    *) return 1 ;;
  esac
}

# 列表容器（bash 3.2 无关联数组，用平行数组 + 5 个计数器）
TLIST=""
TN=0
N_PYCACHE=0; KB_PYCACHE=0
N_PYC=0;     KB_PYC=0
N_DS=0;      KB_DS=0
N_ED=0;      KB_ED=0
N_EMPTY=0;   KB_EMPTY=0
TOTAL_KB=0

add_target() {  # $1=path  $2=kind
  local p="$1" k="$2"
  is_protected "$p" && return 0
  [ -e "$p" ] || return 0
  local kb
  kb=$(du -sk "$p" 2>/dev/null | awk '{print $1+0}')
  [ -n "$kb" ] || kb=0
  TOTAL_KB=$((TOTAL_KB + kb))
  case "$k" in
    pycache) N_PYCACHE=$((N_PYCACHE+1)); KB_PYCACHE=$((KB_PYCACHE+kb)) ;;
    pyc)     N_PYC=$((N_PYC+1));         KB_PYC=$((KB_PYC+kb)) ;;
    ds)      N_DS=$((N_DS+1));           KB_DS=$((KB_DS+kb)) ;;
    editor)  N_ED=$((N_ED+1));           KB_ED=$((KB_ED+kb)) ;;
    emptydir) N_EMPTY=$((N_EMPTY+1));    KB_EMPTY=$((KB_EMPTY+kb)) ;;
  esac
  TLIST="${TLIST}${k}"$'\t'"${p}"$'\n'
  TN=$((TN+1))
}

# ---------- 1. Python 字节码缓存 ----------
while IFS= read -r -d '' p; do
  add_target "$p" pycache
done < <(find "$ROOT" \
  -path "$ROOT/.git" -prune -o \
  -path "$ROOT/venv" -prune -o \
  -path "$ROOT/.venv" -prune -o \
  -path "$ROOT/node_modules" -prune -o \
  -type d -name "__pycache__" -print0)

while IFS= read -r -d '' p; do
  add_target "$p" pyc
done < <(find "$ROOT" -path "$ROOT/.git" -prune -o \
  -type f -name "*.pyc" -not -path "*/__pycache__/*" -print0)

# ---------- 2. macOS 目录元数据 ----------
while IFS= read -r -d '' p; do
  add_target "$p" ds
done < <(find "$ROOT" -path "$ROOT/.git" -prune -o -type f -name ".DS_Store" -print0)

# ---------- 3. 编辑器与工具产物 ----------
while IFS= read -r -d '' p; do
  add_target "$p" editor
done < <(find "$ROOT" -path "$ROOT/.git" -prune -o \
  \( -type d -name ".ipynb_checkpoints" -o -type f -name "*.swp" -o -type f -name "*~" \) -print0)

# ---------- 4. 空目录 ----------
if [ "$DO_DIRS" -eq 1 ]; then
  while IFS= read -r -d '' p; do
    add_target "$p" emptydir
  done < <(find "$ROOT" -mindepth 1 -type d -empty -print0)
fi

# ---------- 汇总 ----------
log "项目根：$ROOT"
if [ "$APPLY" -eq 1 ]; then
  log "模式：执行删除"
else
  log "模式：预演（不删除，加 --apply 生效）"
fi
log ""

if [ "$TN" -eq 0 ]; then
  log "无需清理，项目已经是干净的。"
else
  log "待清理 $TN 项，合计约 $(( TOTAL_KB / 1024 )) MB"
  log "────────────────────────────────────────────"
  [ "$N_PYCACHE" -gt 0 ] && printf '  %-12s %3d 项  %6d KB\n' "__pycache__" "$N_PYCACHE" "$KB_PYCACHE"
  [ "$N_PYC"     -gt 0 ] && printf '  %-12s %3d 项  %6d KB\n' "*.pyc"      "$N_PYC"     "$KB_PYC"
  [ "$N_DS"      -gt 0 ] && printf '  %-12s %3d 项  %6d KB\n' ".DS_Store"  "$N_DS"      "$KB_DS"
  [ "$N_ED"      -gt 0 ] && printf '  %-12s %3d 项  %6d KB\n' "editor"     "$N_ED"      "$KB_ED"
  [ "$N_EMPTY"   -gt 0 ] && printf '  %-12s %3d 项  %6d KB\n' "空目录"     "$N_EMPTY"   "$KB_EMPTY"
  log "────────────────────────────────────────────"
  if [ "$QUIET" -eq 0 ]; then
    log "明细："
    printf '%s' "$TLIST" | while IFS=$'\t' read -r k p; do
      printf '  [%s] %s\n' "$k" "${p#$ROOT/}"
    done
  fi
fi

FAILED=0
if [ "$APPLY" -eq 1 ] && [ "$TN" -gt 0 ]; then
  printf '%s' "$TLIST" | while IFS=$'\t' read -r k p; do
    printf '%s\t%s\n' "$k" "$p"
  done > /tmp/tinyamplab_cleanup.$$ 2>/dev/null || true

  while IFS=$'\t' read -r k p; do
    [ -e "$p" ] || continue
    if [ "$k" = "emptydir" ]; then
      rmdir "$p" 2>/dev/null || true
    else
      rm -rf "$p" 2>/dev/null || FAILED=$((FAILED+1))
    fi
  done < /tmp/tinyamplab_cleanup.$$
  rm -f /tmp/tinyamplab_cleanup.$$

  log ""
  if [ "$FAILED" -eq 0 ]; then
    log "清理完成，释放约 $(( TOTAL_KB / 1024 )) MB。"
  else
    printf '清理完成，但有 %d 项失败（多为权限或占用），详见上面明细。\n' "$FAILED" >&2
  fi
fi

# ---------- 5. git 对象回收（可选） ----------
if [ "$DO_GC" -eq 1 ]; then
  log ""
  log "执行 git gc --prune=now ..."
  ( cd "$ROOT" && git gc --prune=now --quiet ) && log "git 对象回收完成。"
fi

# ---------- 6. 安装 pre-commit 钩子 ----------
if [ "$INSTALL_HOOK" -eq 1 ]; then
  HOOK="$ROOT/.git/hooks/pre-commit"
  if [ -e "$HOOK" ] && ! grep -q "TinyAmpLab 自动清理" "$HOOK" 2>/dev/null; then
    cp "$HOOK" "$HOOK.bak.$(date +%Y%m%d%H%M%S)"
    log "已备份原有 pre-commit 钩子。"
  fi
  cat > "$HOOK" <<'HOOK_EOF'
#!/usr/bin/env bash
# TinyAmpLab 自动清理：提交前清掉可再生的缓存，避免垃圾混入仓库。
# 清理失败不影响提交（|| true），避免钩子阻塞正常工作。
ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || exit 0
S="$ROOT/src/tools/cleanup.sh"
[ -x "$S" ] || exit 0
"$S" --apply --quiet || true
exit 0
HOOK_EOF
  chmod +x "$HOOK"
  log "已安装 pre-commit 钩子：$HOOK"
fi

if [ "$FAILED" -ne 0 ]; then exit 1; fi
exit 0
