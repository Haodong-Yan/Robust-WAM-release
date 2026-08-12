#!/usr/bin/env python
"""Generate a stratified LIBERO-Plus subsample (200/axis = 1400 tasks) matching the
LingBot-VA Table-2 protocol, from task_classification.json. Deterministic (seed=0).

Output: jsonl of {"suite","task_id","category"} rows, consumed by
experiments/eval_libero_plus.py --tasklist.

Set LIBERO_PLUS_ROOT to your LIBERO-plus checkout (used to locate
task_classification.json).
"""
import json, random, argparse, collections, os

LP = os.environ.get("LIBERO_PLUS_ROOT", "/path/to/LIBERO-plus")
CLS = LP + "/libero/libero/benchmark/task_classification.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_axis", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full", action="store_true", help="emit ALL 10030 tasks instead")
    args = ap.parse_args()

    cls = json.load(open(CLS))
    # flat pool: (suite, task_id, category)
    pool = []
    for suite in cls:
        for tid, t in enumerate(cls[suite]):
            pool.append((suite, tid, t["category"]))

    if args.full:
        rows = pool
    else:
        by_axis = collections.defaultdict(list)
        for r in pool:
            by_axis[r[2]].append(r)
        rng = random.Random(args.seed)
        rows = []
        for axis in sorted(by_axis):
            items = by_axis[axis]
            rng.shuffle(items)
            rows.extend(items[: args.per_axis])

    with open(args.out, "w") as f:
        for suite, tid, cat in rows:
            f.write(json.dumps({"suite": suite, "task_id": tid, "category": cat}) + "\n")

    cnt = collections.Counter(r[2] for r in rows)
    print("wrote", len(rows), "rows to", args.out)
    for k, v in sorted(cnt.items(), key=lambda x: -x[1]):
        print("  %-22s %d" % (k, v))


if __name__ == "__main__":
    main()
