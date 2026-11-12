#!/usr/bin/env python3
"""Rank 0's engine output -> the journal, unchanged, plus request and DSpark totals for /metrics.

The engine prints one line per request when it ends:
  serve: req 3 drafts: 2925 verify steps, 1.90 drafts verified and 1.11 accepted per step, ...
and one line with the request's sizes and times:
  serve: req 2 session ...: prompt 40, reused 0 (cold), prefill 135.5 ms (295 tok/s), 24 tokens in 285.9 ms ...
This filter copies every line to stdout (the unit's journal) and keeps running totals in
OUT (JSON, replaced atomically): steps (vLLM's spec_decode_num_drafts), draft tokens
verified and accepted tokens; requests, prompt, reused (prefix-cache hit) and generated
tokens, prefill and decode seconds. Every counter has the engine as its one source, so the
ratios between them hold for every request, also those that bypass the TLS front. bin/tls-front.py reads OUT for /metrics. The engine reports
no per-position acceptance, so there are no per-position totals. It never stops reading:
a parse or write error costs only that update, so the engine never sees a broken pipe.
"""
import json, os, re, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "/run/spark-llm-pico/drafts.json"
LINE = re.compile(rb"drafts: (\d+) verify steps, ([\d.]+) drafts verified and ([\d.]+) accepted per step")
REQ = re.compile(rb"prompt (\d+), reused (\d+).*?prefill ([\d.]+) ms.*?, (\d+) tokens in ([\d.]+) ms")
totals = {"steps": 0, "draft_tokens": 0, "accepted_tokens": 0, "requests": 0, "prompt": 0,
          "reused": 0, "generation": 0, "prefill_s": 0.0, "decode_s": 0.0}


def save():
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(totals, f)
    os.replace(tmp, OUT)


try:
    save()
except OSError:
    pass
out = sys.stdout.buffer
for raw in sys.stdin.buffer:
    try:
        out.write(raw)
        out.flush()
    except OSError:
        pass
    try:
        m = LINE.search(raw)
        if m:
            steps = int(m.group(1))
            totals["steps"] += steps
            totals["draft_tokens"] += round(steps * float(m.group(2)))
            totals["accepted_tokens"] += round(steps * float(m.group(3)))
            save()
            continue
        m = REQ.search(raw)
        if m:
            totals["requests"] += 1
            totals["prompt"] += int(m.group(1))
            totals["reused"] += int(m.group(2))
            totals["prefill_s"] += float(m.group(3)) / 1000
            totals["generation"] += int(m.group(4))
            totals["decode_s"] += float(m.group(5)) / 1000
            save()
    except (ValueError, OSError):
        pass
