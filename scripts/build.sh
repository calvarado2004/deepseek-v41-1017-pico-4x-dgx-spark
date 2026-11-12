#!/usr/bin/env bash
# Build pico-dgx-ds4 on rank 0 at pins.env PICO_COMMIT plus patches/, for sm_121a (GB10) with
# CUDA 13.0: the engine (dsgb10_rank, dsw_convert, dsw_verify, engram_pack, ring_test) and the
# front-end (dsfront). Needs on the head: CUDA 13.0 in /usr/local/cuda-13.0, libibverbs-dev,
# CMake >= 3.28, g++ 13.3, cargo, Python 3.12. -j4: more jobs can OOM a GB10. About 1 minute.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$here/pins.env"
# shellcheck disable=SC1091
source "$here/cluster.env"
H=$SPARK_HEAD
tar cf - -C "$here" patches | ssh "$H" "mkdir -p ~/pico-card && tar xf - -C ~/pico-card"
ssh "$H" "set -e
  cd ~/pico-card
  [ -d src ] || git clone -q $PICO_REPO src
  cd src && git fetch -q origin && git checkout -q --detach $PICO_COMMIT
  for p in ../patches/*.patch; do git apply --check \$p 2>/dev/null && git apply \$p; done
  cd ds41f-4xgx10-switchless
  export PATH=/usr/local/cuda-13.0/bin:\$HOME/.cargo/bin:\$PATH
  cmake -S engine -B engine/build -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc > /dev/null
  cmake --build engine/build -j4 > ~/pico-card/build.log 2>&1
  (cd frontend && cargo build --release -j4) >> ~/pico-card/build.log 2>&1
  ls -l engine/build/server/dsgb10_rank frontend/target/release/dsfront
  echo built"
