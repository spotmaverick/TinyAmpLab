#!/bin/bash
# 轮询 arch_v2 批量作业: 用法 ./poll_arch2.sh [间隔秒] [次数]
# 间隔 900s、12 次 = 3 小时覆盖一次 300ep 训练（实测单变体 2.3~2.8h）
KEY=/Users/wangguan/Downloads/spotmaverick_wuzh02.hpccube.com_RsaKeyExpireTime_2027-02-21_21-34-03.txt
HOST=spotmaverick@wuzh02.hpccube.com
PORT=65091
GAP=${1:-900}; MAX=${2:-12}
i=0
while [ "$MAX" -eq 0 ] || [ "$i" -lt "$MAX" ]; do
  i=$((i+1))
  echo "===== poll $i $(date '+%H:%M:%S') ====="
  ssh -n -p $PORT -i "$KEY" -o BatchMode=yes -o ConnectTimeout=20 "$HOST" '
    echo "--- running jobs ---"
    squeue -u spotmaverick -o "%.10i %.8T %.22j %.10M" | head -15
    echo "--- 关键作业进度 (只报仍在跑或刚完成的) ---"
    for v in nano_s11 feather_s11 lite_s11 standard_s11 \
             lite_marshall_jvm410h_chod1_s7 lite_marshall_jvm410h_chod1_spk_s7 \
             standard_marshall_jvm410h_chod1_s7 standard_marshall_jvm410h_chod1_spk_s7; do
      f=/work/home/spotmaverick/nam_project/arch_v2/$v.log
      [ -f "$f" ] && echo "$v: $(tail -1 "$f")"
    done
    echo "--- 物理仿真 ---"
    for f in analog_sim/lite_sanity.log analog_sim/lite_quick.log analog_sim/lite_full.log; do
      p=/work/home/spotmaverick/nam_project/$f
      [ -f "$p" ] && echo "$(basename $f): $(tail -2 "$p" | tr "\n" " | ")"
    done
    echo "--- json 结果数 ---"
    ls /work/home/spotmaverick/nam_project/arch_v2/*.json 2>/dev/null | wc -l
  '
  if [ "$MAX" -eq 0 ] || [ "$i" -lt "$MAX" ]; then sleep "$GAP"; fi
done
