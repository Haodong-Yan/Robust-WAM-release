#!/usr/bin/env python
"""Aggregate GE-Act LIBERO-Plus shard results into the SAME robustness table format
as LingBot-VA's summarize_lplus.py: overall SR, per-suite SR, and per-perturbation-axis
SR (the Table-2 columns).

GE's eval_libero_plus.py writes <out_dir>/results_shard*.json, each with a "by_task"
list of {suite, task_id, category, successes, trials}. This merges ALL shard files in
each given out_dir (dedup by (suite,task_id)) and prints the table.

Usage: python summarize_gelp.py <out_dir> [more out_dirs ...]
"""
import json, sys, glob, os
from collections import defaultdict

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def summarize(d):
    print("\n== %s ==" % d)
    by_suite = defaultdict(lambda: [0, 0])
    by_axis = defaultdict(lambda: [0, 0])
    grand = [0, 0]
    seen = set()
    n = 0
    for f in glob.glob(os.path.join(d, "results_shard*.json")):
        try:
            j = json.load(open(f))
        except Exception:
            continue
        for t in j.get("by_task", []):
            key = (t["suite"], t["task_id"])
            if key in seen:
                continue
            seen.add(key)
            n += 1
            s = int(t.get("successes", 0))
            tr = int(t.get("trials", 1))
            suite = t.get("suite", "?")
            axis = t.get("category", "?")
            by_suite[suite][0] += s; by_suite[suite][1] += tr
            by_axis[axis][0] += s;   by_axis[axis][1] += tr
            grand[0] += s; grand[1] += tr

    def pct(sn):
        s, tot = sn
        return "%d/%d = %.1f%%" % (s, tot, 100.0 * s / tot) if tot else "no episodes"

    print("  tasks scored: %d" % n)
    print("  OVERALL: %s" % pct(grand))
    print("  -- by suite --")
    for su in SUITES:
        if by_suite[su][1]:
            print("    %-16s %s" % (su, pct(by_suite[su])))
    print("  -- by perturbation axis (Table-2 columns) --")
    for ax in sorted(by_axis):
        print("    %-22s %s" % (ax, pct(by_axis[ax])))


for arg in sys.argv[1:]:
    summarize(arg)
