#!/bin/sh
# 顺序执行队列（容器重启后可重跑：已完成的项目从 logs/queue_done.txt 跳过）
cd "$(dirname "$0")"
DONE=../logs/queue_done.txt
touch $DONE
for item in "$@"; do
  grep -qx "$item" $DONE && continue
  case "$item" in
    leak:*) h=${item#leak:}; python3 -u -W ignore run_leak.py $h base > ../logs/ap_${h}_leak.log 2>&1 ;;
    *) python3 -u autopilot.py "$item" ;;
  esac
  echo "$item" >> $DONE
done
