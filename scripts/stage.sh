#!/usr/bin/env bash
# Stage the card on the four Sparks. Starts nothing and pulls nothing.
#   runtime  /opt/spark-llm/profiles/deepseek-v41-1017-pico/{bin,engine,vision}
#            engine/: dsgb10_rank, dsfront (scripts/build.sh on the head), fleet.conf
#            (fleet/ring.conf), cfmlp_cml.bin (their verify-gate weights), tokenizer.json (rank 0)
#   env      /etc/spark-llm/profiles/deepseek-v41-1017-pico.env (+ cluster.env's API host and TLS files)
#   key      /etc/spark-llm/profiles/deepseek-v41-1017-pico.key on rank 0: a random bearer key,
#            mode 600, made once; clients send "Authorization: Bearer <it>"
#   unit     /etc/systemd/system/spark-llm-dsv41-pico@.service (+ SPARK_USER)
# The rank files and Engram shards come from scripts/prep.sh.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$here/pins.env"
# shellcheck disable=SC1091
source "$here/cluster.env"
[ -f "$here/fleet/ring.conf" ] || { echo "fleet/ring.conf missing: copy fleet/ring.conf.example and edit it" >&2; exit 1; }
read -r -a hosts <<< "$SPARK_HOSTS"
R=/opt/spark-llm/profiles/deepseek-v41-1017-pico
SRC='~/pico-card/src/ds41f-4xgx10-switchless'
U=spark-llm-dsv41-pico@.service
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/engine" "$tmp/stage/bin"
ssh "$SPARK_HEAD" "cd $SRC && tar cf - -C engine/build/server dsgb10_rank -C ../../../frontend/target/release dsfront -C ../../../tools/stack4 cfmlp_cml.bin -C $CKPT tokenizer.json" | tar xf - -C "$tmp/engine"
ssh "$SPARK_HEAD" "cat $SRC/tools/vision/vision_encoder.py" > "$tmp/vision_encoder.py"
cp "$here/fleet/ring.conf" "$tmp/engine/fleet.conf"
cp "$here"/bin/* "$tmp/stage/bin/"
{ cat "$here/deepseek-v41-1017-pico.env"; echo; echo "# from cluster.env and pins.env"
  echo "PICO_API_HOST=$SPARK_API_HOST"; echo "PICO_TLS_CERT=$SPARK_TLS_CERT"; echo "PICO_TLS_KEY=$SPARK_TLS_KEY"
  echo "PICO_STATE_DIR=$SPARK_HOME/pico-card"; echo "PICO_VISION_IMAGE=$VISION_IMAGE"; echo "PICO_CKPT=$CKPT"; } > "$tmp/stage/deepseek-v41-1017-pico.env"
sed -e "s#@SPARK_USER@#$SPARK_USER#g" -e "s#@SPARK_HOME@#$SPARK_HOME#g" "$here/systemd/$U" > "$tmp/stage/$U"
for i in "${!hosts[@]}"; do
  h="${hosts[$i]}"
  echo "== $h (rank $i)"
  files=(dsgb10_rank dsfront cfmlp_cml.bin fleet.conf); [ "$i" = 0 ] && files+=(tokenizer.json)
  tar cf - -C "$tmp/engine" "${files[@]}" | ssh "$h" "rm -rf /tmp/pico-engine && mkdir -p /tmp/pico-engine && tar xf - -C /tmp/pico-engine"
  tar cf - -C "$tmp/stage" . | ssh "$h" "rm -rf /tmp/pico-stage && mkdir -p /tmp/pico-stage && tar xf - -C /tmp/pico-stage"
  [ "$i" = 0 ] && ssh "$h" "cat > /tmp/pico-stage/vision_encoder.py" < "$tmp/vision_encoder.py"
  ssh "$h" "set -e
    sudo install -d -o $SPARK_USER -g $SPARK_USER $R $R/bin $R/engine $R/vision
    sudo install -d /etc/spark-llm/profiles
    install -m 755 /tmp/pico-stage/bin/* $R/bin/
    install -m 755 /tmp/pico-engine/dsgb10_rank /tmp/pico-engine/dsfront $R/engine/
    install -m 644 /tmp/pico-engine/cfmlp_cml.bin /tmp/pico-engine/fleet.conf $R/engine/
    [ ! -f /tmp/pico-engine/tokenizer.json ] || install -m 644 /tmp/pico-engine/tokenizer.json $R/engine/
    [ ! -f /tmp/pico-stage/vision_encoder.py ] || install -m 644 /tmp/pico-stage/vision_encoder.py $R/vision/
    sudo install -m 644 /tmp/pico-stage/deepseek-v41-1017-pico.env /etc/spark-llm/profiles/
    sudo install -m 644 '/tmp/pico-stage/$U' /etc/systemd/system/
    sudo systemctl daemon-reload
    ! grep -q '^\[Install\]' '/etc/systemd/system/$U'
    mkdir -p $SPARK_HOME/pico-card/spill
    for f in rank$i.dsw engram/engram-l1-r${i}of4.bin engram/engram-l14-r${i}of4.bin engram/engram_hashmap.bin; do
      [ -f $DSW_DIR/\$f ] || echo \"  MISSING $DSW_DIR/\$f (scripts/prep.sh)\"
    done
    rm -rf /tmp/pico-engine /tmp/pico-stage
    echo staged"
  if [ "$i" = 0 ]; then
    ssh "$h" "K=/etc/spark-llm/profiles/deepseek-v41-1017-pico.key
      [ -f \$K ] || { openssl rand -hex 24 | sudo install -m 600 -o $SPARK_USER -g $SPARK_USER /dev/stdin \$K; echo '  new API key in '\$K; }
      docker image inspect $VISION_IMAGE >/dev/null 2>&1 && echo '  vision image present' || echo '  vision image MISSING: docker pull $VISION_IMAGE'"
  fi
done
echo "start: on every node, ranks 1-3 first, then rank 0: sudo systemctl start spark-llm-dsv41-pico@<rank>"
