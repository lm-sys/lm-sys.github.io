#!/usr/bin/env python3
"""Per-turn TTFT analysis for the multi-turn radix-cache comparison.

Usage:
    python analyze.py <rust.json> <python.json> "Title" out.png

Prints a per-25-turn-bin table (total TTFT, GPU prefill, and their residual) and
saves a 2-panel chart. The timing decomposition needs `gpu_prefill_ms` in the
server's meta_info (see README). If it is absent, the script falls back to a
total-TTFT-only table and plot.
"""
import json
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUST = sys.argv[1] if len(sys.argv) > 1 else "ttft_full_rust.json"
PY = sys.argv[2] if len(sys.argv) > 2 else "ttft_full_python.json"
TITLE = sys.argv[3] if len(sys.argv) > 3 else "unified radix cache"
OUT = sys.argv[4] if len(sys.argv) > 4 else "ttft.png"
BINS = [(0, 25), (25, 50), (50, 75), (75, 100), (100, 125), (125, 150), (150, 175), (175, 200)]


def load(p):
    tr = [t for t in json.load(open(p))["trials"] if not t.get("is_warmup", False)]
    n = tr[0]["num_turns"]
    tot = [st.mean([t["turns"][s]["ttft_s"] * 1000 for t in tr]) for s in range(n)]
    gpu = [st.mean([t["turns"][s].get("gpu_prefill_ms", 0) for t in tr]) for s in range(n)]
    return tot, gpu


Rt, Rg = load(RUST)
Pt, Pg = load(PY)
n = min(len(Rt), len(Pt))
have_gpu = any(Rg[:n]) and any(Pg[:n])
Rng = [Rt[i] - Rg[i] for i in range(n)]
Png = [Pt[i] - Pg[i] for i in range(n)]
bins = [(a, b) for a, b in BINS if a < n]
bm = lambda d, a, b: st.mean(d[a:min(b, len(d))])

print(f"=== {TITLE} — Rust vs Python unified radix cache ===")
if have_gpu:
    print(f"{'turns':>9} | {'R tot':>6} {'P tot':>6} {'dTot':>6} | {'R gpu':>6} {'P gpu':>6} | {'R resid':>7} {'P resid':>7} {'dResid':>7}")
    for a, b in bins:
        print(f"{a+1:>4}-{b:<4} | {bm(Rt,a,b):>6.1f} {bm(Pt,a,b):>6.1f} {bm(Rt,a,b)-bm(Pt,a,b):>+6.2f} |"
              f" {bm(Rg,a,b):>6.1f} {bm(Pg,a,b):>6.1f} | {bm(Rng,a,b):>6.1f} {bm(Png,a,b):>6.1f} {bm(Rng,a,b)-bm(Png,a,b):>+7.2f}")
else:
    print(f"{'turns':>9} | {'Rust':>7} {'Python':>7} {'delta':>7} {'delta%':>7}  (no gpu_prefill_ms; total only)")
    for a, b in bins:
        r, p = bm(Rt, a, b), bm(Pt, a, b)
        print(f"{a+1:>4}-{b:<4} | {r:>7.1f} {p:>7.1f} {r-p:>+7.2f} {100*(r-p)/p:>+6.1f}%")
R, P = st.mean(Rt[:n]), st.mean(Pt[:n])
print(f"overall total: Rust {R:.1f}  Python {P:.1f}  -> {R-P:+.2f} ms ({100*(R-P)/P:+.1f}%)")

ts = list(range(1, n + 1))
nrows = 2 if have_gpu else 1
fig, axes = plt.subplots(nrows, 1, figsize=(11, 9 if have_gpu else 5), sharex=True, squeeze=False)
ax = axes[0][0]
ax.plot(ts, Rt[:n], "-", color="tab:blue", lw=1.4, label="Rust total")
ax.plot(ts, Pt[:n], "-", color="tab:red", lw=1.4, label="Python total")
if have_gpu:
    ax.plot(ts, Rg[:n], "--", color="tab:blue", lw=1.0, label="Rust GPU prefill")
    ax.plot(ts, Pg[:n], "--", color="tab:red", lw=1.0, label="Python GPU prefill")
    lo, hi = min(min(Rg[:n]), min(Pg[:n])), max(max(Rt[:n]), max(Pt[:n]))
    ax.set_ylim(lo - max(1.5, (hi - lo) * 0.06), hi + max(1.5, (hi - lo) * 0.06))
ax.set_ylabel("TTFT (ms)"); ax.legend(); ax.grid(True, alpha=0.3)
ax.set_title(f"{TITLE} — total" + (" (solid) vs GPU prefill (dashed)" if have_gpu else ""))
if have_gpu:
    ax = axes[1][0]
    ax.plot(ts, Rng[:n], "-", color="tab:blue", lw=1.4, label="Rust")
    ax.plot(ts, Png[:n], "-", color="tab:red", lw=1.4, label="Python")
    lo, hi = min(min(Rng[:n]), min(Png[:n])), max(max(Rng[:n]), max(Png[:n]))
    ax.set_ylim(max(0, lo - 2), hi + 2)
    ax.set_ylabel("TTFT outside GPU prefill (ms)"); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_title("Total TTFT minus instrumented GPU prefill")
axes[-1][0].set_xlabel("conversation turn (tree depth)")
fig.tight_layout()
fig.savefig(OUT, dpi=130)
print(f"saved {OUT}")
