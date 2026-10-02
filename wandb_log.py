#!/usr/bin/env python3
"""Log eval results (results/*.json from eval.py) to Weights & Biases: one run per clip and mode,
grouped by clip, plus a summary table comparing every-frame against motion gating.

    python wandb_log.py results/*.json      # needs WANDB_API_KEY
"""
import json, os, sys
import wandb

rows = []
for path in sys.argv[1:]:
    try:
        d = json.load(open(path))
    except (json.JSONDecodeError, OSError):
        continue
    clip = os.path.basename(path)[:-5]
    for mode, r in d["modes"].items():
        with wandb.init(project="vast", group=clip, job_type=mode, name=f"{clip}/{mode}",
                        config={"clip": clip, "mode": mode, "frames": d["frames"], "device": d["device"]}) as run:
            run.summary.update(r)
        rows.append([clip, mode, d["frames"], r["frames_inferred_pct"], r["detector_ms"], r["ms_vs_every_pct"],
                     round(100 - r["ms_vs_every_pct"], 1), r["recall_pct"], r["precision_pct"]])

with wandb.init(project="vast", name="summary", job_type="summary") as run:
    cols = ["clip", "mode", "frames", "frames_inferred_pct", "detector_ms", "ms_vs_every_pct", "saved_pct", "recall_pct", "precision_pct"]
    table = wandb.Table(columns=cols, data=rows)
    run.log({"eval": table,
             "time saved by frame gating": wandb.plot.bar(
                 wandb.Table(columns=["clip", "saved_pct"], data=[[r[0], r[6]] for r in rows if r[1] == "frame"]),
                 "clip", "saved_pct", title="Detector time saved by codec motion gating (100% recall)")})
    print(run.url)
