<p align="center">
  <img src="assets/deepseek-logo.svg" alt="DeepSeek" width="360">
</p>

# DeepSeek-V4.1-Flash-1017 on pico-dgx-ds4, four DGX Sparks

A recipe that serves DeepSeek's official weights,
[`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
(MXFP4 routed experts, FP8 Engram tables: the full model, nothing re-quantised), with
[pico-dgx-ds4](https://github.com/Picom-code/pico-dgx-ds4) at tensor parallelism 4 on four NVIDIA
DGX Sparks (GB10) cabled as a **switchless QSFP ring**. pico-dgx-ds4 is a C++/CUDA engine written
only for this model on four GB10s (Patrick Moorhead's fork of Andy Shu's
[blaesus/dsgx10](https://github.com/blaesus/dsgx10)): one converted weight file per Spark, its own
RoCE ring collectives over ibverbs, DSpark drafts with a learned verify gate. 1M-token context,
thinking on by default, tool calls, images, video (as frames), an OpenAI-compatible API behind a
bearer key, and vLLM-shaped `/metrics` for dashboards.

On the same four Sparks this replaced our
[NVIDIA Hyperflare recipe](https://github.com/calvarado2004/deepseek-v41-1017-nvfp4-nvidia-flare)
as the production card on 2026-11-12.

## Speed

Measured 2026-11-11/12 on the same four Sparks, the same checkpoint, idle card, one benchmark run
right after the other. Hyperflare is `7f534bb6` + patch 0012, chunk 8192.

| Benchmark (tok/s) | pico-dgx-ds4 | NVIDIA Hyperflare | Gain |
|---|---:|---:|---:|
| pico's `bench.py`, chat, 1 user (median of 9) | **84.9** | 49.6 | 1.71x |
| pico's `bench.py`, code, 1 user (median of 9) | **143.1** | 99.2 | 1.44x |
| pico's `bench.py`, 16 users, different prompts | **361.8** | 192.7 | 1.88x |
| pico's `bench.py`, 16 users, repeated prompts | **414.9** | 204.4 | 2.03x |
| our `bench.py`, count / code (median of 3) | **164.3 / 144.1** | 124.2 / 115.5 | 1.32x / 1.25x |
| our `bench.py`, prose English / Spanish (median of 3) | **91.8 / 93.8** | 60.0 / 60.2 | 1.53x / 1.56x |
| our `bench.py`, two prose streams, aggregate | **128.6** | 97.1 | 1.32x |
| Cold prefill, 20K tokens | **9,268** | 4,281 | 2.17x |
| Cold prefill, 16K / 63K / 255K tokens | **8,943 / 9,560 / 9,650** | ~4,100 | ~2.3x |
| Needle at 943K tokens (prompt to answer) | **74.2 s** | – | – |

pico's bench: [`tools/bench/bench.py`](https://github.com/Picom-code/pico-dgx-ds4/blob/main/tools/bench/bench.py),
512 tokens, temperature 0, thinking off, a random prefix on each prompt. Our bench:
350-word answers at temperature 0, low effort. On Hyperflare, pico's bench needs one line changed
(`thinking` -> `enable_thinking`). The pico rows of pico's bench are the card with the image
encoder off (with it loaded and idle: 82.1 / 140.3 / 362.8, inside the spread); the
repeated-prompts row is from the first run of the same build and settings. Medians move by about ±3 tok/s between runs. Their own
published numbers on a 200G switch are 80-85 / 140-145 / 376 / 441; on this switchless ring we
get the same within that spread.

DSpark acceptance on code: 79% of drafts, 4.0 tokens per verify step.

## Quality

| Check | Result |
|---|---|
| Needle ladder 82K / 262K / 524K / 955K | all found |
| Concurrent-prefill KV integrity (1, 4 and 8 documents at once) | 0 bad reads in 1,116 checks |
| Agentic soak, 8 sessions x 3 rounds | 24/24 turns, 0 errors, code / owner / math 100% |
| Tool calls, listed and deferred, stream and not | pass |
| Count to 100, three runs | exact |
| Deduction puzzle (thinking, temperature 1.0) | "dog" every time, no loops, no foreign text |
| Images, 44 cases (OCR, charts, spatial, several images) | 44/44 |
| Video, 8 clips as frames | 6/8 (both misses are counting questions) |

The served settings include pico's batch-aware expert re-route (`DSGB10_OEA`), which is lossy by
design; every check above passed with it on, and with it off. Set `PICO_OEA=""` in
`deepseek-v41-1017-pico.env` for a strictly lossless build.

## Why it is faster

- **Prefill**: sequence-parallel, with the collectives under the compute, `wo_b` by all-to-all and
  the MoE combine inside the reduce-scatter; transfers of 4 KB and more go half on each PCIe domain
  of the ConnectX-7. During a 255K prefill the ring carries ~25 Gb/s each way on every cable.
- **Decode**: fused weight-streaming kernels at 95-108% of the 235 GB/s memory floor, 80 small
  all-reduces a step at 12.3 µs each (about 1 ms of collectives a step), and DSpark drafts that
  a learned gate verifies.

## Patches on top of pico-dgx-ds4

- `0001`: optional per-rank ring device keys in the fleet file (see "Hardware and cabling").
- `0002`: **sampling fix, needed at the model card's temperature 1.0.** The Gumbel noise's uniform
  could round to exactly 1.0 in float, which gives one vocabulary entry +inf noise: a uniformly
  random token in ~0.77% of sampled steps, seen as stray CJK / Arabic / Hebrew characters and
  unrelated words in long answers. Before: 37 foreign-script runs in 12,000 tokens; after: 0 in
  17,906 (same prompt, temperature 1.0, top-p 1.0). Greedy decoding (temperature 0) never reaches
  this code, so greedy benchmarks cannot see it.

## Hardware and cabling

Four DGX Sparks, ranks 0-3, each cabled to the next with one QSFP cable (rank 3 back to rank 0),
two subnets on each cable (one per PCIe domain of the ConnectX-7). Same OS on all four: Ubuntu
24.04, kernel 7.0.0-1019-nvidia, driver 615.71.09, CUDA 13.0 and 13.3, booted with `kho=off`
(`grep -o kho=off /proc/cmdline`; without it `ibv_reg_mr` fails).

pico-dgx-ds4 assumes every box faces its next rank on port f0. On our ring one box faces it on
f1, so `patches/0001` adds optional per-rank device keys (`RANK<r>_NEXT_DEV<d>`,
`RANK<r>_PREV_DEV<d>`) to the fleet file. Find yours with `ibv_devices`, `ip -4 -br addr` and the
GID table (`/sys/class/infiniband/<dev>/ports/1/gids/3` must be the link's IPv4 RoCE v2 GID):
rank r's `NEXT_IP<d>` and rank r+1's `PREV_IP<d>` share a subnet.

Memory: a rank takes ~90 GiB; rank 0 has ~16 GiB left with the image encoder loaded. Disk on each
node: its rank file (77.7 GB) and Engram shards (50.7 GB); on rank 0 also the checkpoint (476 GB).

## How to use

1. **Settings.** Edit `cluster.env` (hosts, user, the API host name and its TLS files). Copy
   `fleet/ring.conf.example` to `fleet/ring.conf` and give it your devices and fabric addresses.
2. **Checkpoint.** `scripts/fetch.sh` downloads it to rank 0 (475.3 GiB, detached).
3. **Build.** `scripts/build.sh` builds the engine and dsfront on rank 0 (about 1 minute).
4. **Ring test.** On rank 0, in `~/pico-card/src/ds41f-4xgx10-switchless`:
   `FLEET=<your ring.conf> engine/server/ring.sh engine/comm/run4.sh engine/build/comm/ring_test`.
   Every rank must print `passed` (52 s); ours: 0 mismatches in 1,000,000 collectives.
5. **Weights.** `scripts/prep.sh` cuts, verifies and copies the four rank files and Engram shards
   (about 3 minutes a rank, then about 10 minutes a copy; detached).
6. **Stage.** `scripts/stage.sh` installs the card on the four nodes and makes the API key.
7. **Start.** Ranks 1-3 first, then rank 0: `sudo systemctl start spark-llm-dsv41-pico@<rank>`.
   Rank 0 loads the image encoder (~170 s), then the engine (~20 s), and reports ready after
   `/health` and one warm request. The units have no `[Install]` section: nothing starts at boot.
8. **Use it.**

```bash
curl -s https://<api-host>:8888/v1/models -H "Authorization: Bearer $(cat /etc/spark-llm/profiles/deepseek-v41-1017-pico.key)"
```

Thinking is on unless a request turns it off (`"chat_template_kwargs": {"thinking": false}` or
`{"enable_thinking": false}`); `reasoning_effort` takes low / high / max. The reasoning comes back
in `reasoning_content`. Images: `image_url` parts (data URLs or http(s)). Video: `video_url` parts,
which the TLS front turns into timed frames; dsfront takes 16 images a request.

## What runs where

| Piece | Where | What it does |
|---|---|---|
| `dsgb10_rank` | every node, one unit each | the engine rank, with its fleet file beside it |
| `dsfront` | rank 0, 127.0.0.1:18893 | OpenAI API, sessions with prefix reuse, bearer key |
| `bin/tls-front.py` | rank 0, :8888 | TLS, video as frames, one `req`/`end` journal line per request, `/metrics` |
| `bin/draft-meter.py` | rank 0 | copies the engine's output to the journal and totals tokens, times and DSpark drafts for `/metrics` |
| image encoder | rank 0, 127.0.0.1:18950 | pico's `tools/vision`, the checkpoint's own ViT, in the vLLM image |

`/metrics` uses vLLM's names (`vllm:generation_tokens_total`, `vllm:spec_decode_num_accepted_tokens_total`,
...), so a dashboard made for vLLM works. pico reports draft acceptance as totals, not per position.

## Credits

The engine, kernels, ring collectives, drafter and image path are Andy Shu's
([blaesus/dsgx10](https://github.com/blaesus/dsgx10)) and Patrick Moorhead's
([Picom-code/pico-dgx-ds4](https://github.com/Picom-code/pico-dgx-ds4)), both MIT. The model and
the DSpark drafter are DeepSeek's. This repository is the recipe around them for a switchless
four-Spark ring: fleet, units, TLS, metrics, vision wiring and the measurements.
