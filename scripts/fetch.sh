#!/usr/bin/env bash
# Fetch deepseek-ai/DeepSeek-V4.1-Flash-1017 (475.3 GiB) to rank 0. scripts/prep.sh cuts the
# four rank files from it there. Runs detached on the head with its own log; re-run to resume.
# Stop every model on the head first: the download fills page cache, which is GPU memory on GB10.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$here/pins.env"
# shellcheck disable=SC1091
source "$here/cluster.env"
H=$SPARK_HEAD
ssh "$H" "mkdir -p ~/logs; sudo mkdir -p $CKPT && sudo chown $SPARK_USER:$SPARK_USER $(dirname $CKPT) $CKPT
  if pgrep -f '[h]f download $MODEL_ID' >/dev/null; then echo 'already running'; exit 0; fi
  (setsid nohup bash -c 'set -e
    ~/.local/bin/uvx --from huggingface_hub[hf_xet] hf download $MODEL_ID \
      --revision $MODEL_REVISION --local-dir $CKPT
    echo $MODEL_REVISION > $CKPT/.hf-revision
    echo fetch done' > ~/logs/fetch-dsv41.log 2>&1 < /dev/null &)
  echo started"
echo "log: ssh $H tail -f logs/fetch-dsv41.log"
