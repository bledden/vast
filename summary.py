#!/usr/bin/env python3
"""Table of eval results: python summary.py results/*.json"""
import json, sys
print(f"{'clip':42} {'sec':>4} | {'frame: sent':>11} {'time':>5} {'recall':>6} | {'region: px':>10} {'time':>5} {'recall':>6} {'prec':>5}")
for p in sys.argv[1:]:
    try:
        d = json.load(open(p))
    except (json.JSONDecodeError, OSError):
        continue
    f, r = d["modes"]["frame"], d["modes"]["region"]
    print(f"{p.split('/')[-1][:-5][:42]:42} {d['frames'] / 30:4.0f} | {f['frames_inferred_pct']:10.0f}% {f['ms_vs_every_pct']:4.0f}% {f['recall_pct']:5.0f}% "
          f"| {r['pixels_pct']:9.0f}% {r['ms_vs_every_pct']:4.0f}% {r['recall_pct']:5.0f}% {r['precision_pct']:4.0f}%")
