#!/bin/bash
# 并行分段下载: URL TOTAL OUT NSEG
URL=$1; TOTAL=$2; OUT=$3; N=${4:-8}
mkdir -p "$OUT.parts"
SEG=$(( (TOTAL + N - 1) / N ))
dl_seg() {
  local i=$1
  local start=$(( i * SEG )); local end=$(( start + SEG - 1 ))
  [ $end -ge $TOTAL ] && end=$(( TOTAL - 1 ))
  local f="$OUT.parts/seg_$i"
  local want=$(( end - start + 1 ))
  while true; do
    local have=0
    [ -f "$f" ] && have=$(wc -c < "$f")
    [ "$have" -eq "$want" ] && break
    local from=$(( start + have ))
    curl -s --max-time 7200 -r "$from-$end" -o - "$URL" >> "$f" 2>/dev/null || sleep 5
  done
  echo "seg $i done $want"
}
for i in $(seq 0 $((N-1))); do dl_seg $i & done
wait
cat $(for i in $(seq 0 $((N-1))); do echo "$OUT.parts/seg_$i"; done) > "$OUT"
SZ=$(wc -c < "$OUT")
[ "$SZ" -eq "$TOTAL" ] && echo "$OUT OK ($SZ bytes)" || echo "$OUT SIZE-WRONG ($SZ vs $TOTAL)"
