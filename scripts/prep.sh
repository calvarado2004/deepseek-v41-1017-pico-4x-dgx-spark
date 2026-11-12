#!/usr/bin/env bash
# Cut the four rank files from the checkpoint on rank 0, check every byte, pack each rank's
# Engram shards, make the hash map, and copy each node its own files (tar over ssh, MD5 checked).
# Per rank: convert ~1 min (72.38 GiB), verify ~40 s, Engram pack ~90 s (50.7 GB); each copy
# ~10 min (ssh-bound). Needs the GPU of rank 0 for ~1.2 GB: run it with no model loaded.
# Runs detached on the head; log ~/pico-card/prep.log. Rank r's SSH address: fleet/ring.conf.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$here/pins.env"
# shellcheck disable=SC1091
source "$here/cluster.env"
H=$SPARK_HEAD
ips=()
for r in 1 2 3; do ips+=("$(sed -n "s/^RANK${r}_SSH=//p" "$here/fleet/ring.conf")"); done
ssh "$H" "cat > ~/pico-card/prep-run.sh" <<RUN
#!/bin/bash
set -euo pipefail
B=~/pico-card/src/ds41f-4xgx10-switchless/engine/build S=~/pico-card/src/ds41f-4xgx10-switchless
OUT=$DSW_DIR; mkdir -p \$OUT/engram; date -Is
for r in 1 2 3 0; do
  echo "== rank \$r convert"; \$B/weights/dsw_convert --ckpt $CKPT --rank \$r --out \$OUT/rank\$r.dsw
  echo "== rank \$r verify";  \$B/weights/dsw_verify --ckpt $CKPT --file \$OUT/rank\$r.dsw --threads 4
  echo "== rank \$r engram";  \$B/engram/engram_pack $CKPT \$OUT/engram \$r
done
python3 -I \$S/engine/engram/py/make_hashmap_stdlib.py $CKPT \$OUT/engram/engram_hashmap.bin
IPS=(${ips[*]})
for r in 1 2 3; do
  ip=\${IPS[\$((r - 1))]}; echo "== copy rank \$r to \$ip"; date -Is
  f="rank\$r.dsw engram/engram-l1-r\${r}of4.bin engram/engram-l14-r\${r}of4.bin engram/engram_hashmap.bin"
  ssh \$ip "mkdir -p \$OUT/engram"
  tar cf - -C \$OUT \$f | ssh \$ip "tar xf - -C \$OUT"
  [ "\$(cd \$OUT && md5sum \$f)" = "\$(ssh \$ip "cd \$OUT && md5sum \$f")" ] && echo "md5 rank \$r OK" || { echo "md5 rank \$r MISMATCH"; exit 1; }
  rm -f \$OUT/rank\$r.dsw \$OUT/engram/engram-l1-r\${r}of4.bin \$OUT/engram/engram-l14-r\${r}of4.bin
done
date -Is; echo PREP-OK
RUN
ssh "$H" "sudo mkdir -p $DSW_DIR && sudo chown $SPARK_USER:$SPARK_USER $DSW_DIR
  for ip in ${ips[*]}; do ssh \$ip 'sudo mkdir -p $DSW_DIR && sudo chown $SPARK_USER:$SPARK_USER $DSW_DIR'; done
  chmod +x ~/pico-card/prep-run.sh && (setsid nohup ~/pico-card/prep-run.sh > ~/pico-card/prep.log 2>&1 < /dev/null &); echo started"
echo "log: ssh $H tail -f pico-card/prep.log"
