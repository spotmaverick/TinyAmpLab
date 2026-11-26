#!/usr/bin/env bash
# 目标机 2（新 4090）快捷入口：box2.sh sh '<cmd>' / put / get
ENVF="$HOME/.workbuddy/.autodl_env2"
set -a; source "$ENVF"; set +a
exec bash "$HOME/.workbuddy/skills/autodl-4090-train/scripts/autodl.sh" "$@"
