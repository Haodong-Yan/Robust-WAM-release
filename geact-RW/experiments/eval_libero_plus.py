#!/usr/bin/env python
"""LIBERO-plus robustness eval for our GE-Act (LTX) model.

LIBERO-plus is a drop-in `libero` replacement: same 4 suites but 10,030 PERTURBED
tasks across 7 dimensions (Objects Layout / Camera Viewpoints / Robot Initial States /
Language / Light Conditions / Background Textures / Sensor Noise). Protocol: 1 trial/task,
report per-suite + per-category + overall.

This wraps our proven InferenceLibero (model build + play() with the gripper fix +
mean/std norm) in the LIBERO-plus protocol. Reference: FastWAM
experiments/libero/cosmos_eval_libero_plus.py (that one runs the Cosmos model; we run ours).

Sharded across GPUs (--shard/--num_shards) and resumable (per-shard partial JSON).
"""
import os, sys, json, argparse, time, glob

# --- LIBERO-plus MUST win over any standard LIBERO checkout: put it first on sys.path,
#     and point LIBERO_CONFIG_PATH at its assets config (pre-created, non-interactive). ---
LP = os.environ.get("LIBERO_PLUS_ROOT", "/path/to/LIBERO-plus")
sys.path.insert(0, LP)
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero_plus"))
os.environ.setdefault("MUJOCO_GL", "egl")

REPO = os.environ.get("GEACT_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch
import numpy as np
from experiments.eval_libero import InferenceLibero
from utils.libero_sim_utils import (
    get_libero_env, get_libero_image, get_libero_dummy_action, get_libero_state,
)
from libero.libero import benchmark

# per-suite rollout budget (same as experiments/eval_libero.py) and video-memory threshold
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
             "libero_10": 520, "libero_90": 400}
THRESHOLD = {"libero_spatial": 30, "libero_object": 30, "libero_goal": 20,
             "libero_10": 20, "libero_90": 20}


def set_suite_stats(E, suite):
    """Re-select action/state normalization stats for the current suite (LIBERO-plus
    spans all 4 suites in one run; the mix stat file holds all of them)."""
    dom = suite + "_no_noops_lerobot"
    if dom + "_eef" not in E.StatisticInfo:
        dom = "libero" if "libero_eef" in E.StatisticInfo else dom
    E.act_mean = torch.tensor(E.StatisticInfo[dom + "_eef"]["mean"]).unsqueeze(0)
    E.act_std = torch.tensor(E.StatisticInfo[dom + "_eef"]["std"]).unsqueeze(0)
    E.states_mean = torch.tensor(E.StatisticInfo[dom + "_state_eef"]["mean"]).unsqueeze(0)
    E.states_std = torch.tensor(E.StatisticInfo[dom + "_state_eef"]["std"]).unsqueeze(0)
    E.threshold = THRESHOLD[suite]


def run_one_task(E, task, initial_states, max_steps, device):
    """One rollout (1 trial) for a LIBERO-plus task. Returns success (bool)."""
    env, task_description = get_libero_env(task, image_height=256, image_width=256)
    env.reset()
    E.policy_memory_reset()
    obs = env.set_init_state(initial_states[0])
    t = 0
    done = False
    num_steps_wait = 10
    try:
        while t < max_steps + num_steps_wait:
            if t < num_steps_wait:
                obs, _, done, _ = env.step(get_libero_dummy_action())
                t += 1
                continue
            agt, wrist = get_libero_image(obs)
            agt_t = torch.tensor(agt.copy()).to(device).permute(2, 0, 1).unsqueeze(0)
            wrist_t = torch.tensor(wrist.copy()).to(device).permute(2, 0, 1).unsqueeze(0)
            img_obs = torch.cat([agt_t, wrist_t], dim=0)
            if E.with_state:
                state = get_libero_state(obs)
                state = (torch.tensor(state).float() - E.states_mean) / (E.states_std + 1e-6)
                state = torch.cat((torch.zeros([1, E.basic_action_dim]), state), dim=1)
                actions = E.play(img_obs, task_description, excution_step=E.excution_step, state=state)
            else:
                actions = E.play(img_obs, task_description, excution_step=E.excution_step, state=None)
            actions = actions.cpu().numpy()
            if t >= max_steps:
                break
            for i in range(E.excution_step):
                try:
                    obs, _, done, _ = env.step(actions[i, :].tolist())
                except Exception:
                    done = False
                    break
                if done:
                    break
            if done or env.env.done:
                break
            t += 1
    finally:
        try:
            env.close()
        except Exception:
            pass
    return bool(done), task_description


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_file", required=True)
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--suites", default="libero_goal,libero_object,libero_spatial,libero_10")
    ap.add_argument("--max_tasks_per_suite", type=int, default=0, help="0=all; >0 for smoke")
    ap.add_argument("--task_ids", default="",
                    help="CSV of task_ids applied to EACH suite (smoke; overrides max_tasks_per_suite)")
    ap.add_argument("--tasklist", default="",
                    help="path to a jsonl of {suite,task_id} rows (LingBot-VA subsample/full format); overrides --suites")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = f"cuda:{args.device}"
    cls = json.load(open(LP + "/libero/libero/benchmark/task_classification.json"))

    # build model once
    E = InferenceLibero(config_file=args.config_file, output_dir=args.out_dir,
                        task_suite_name="libero_goal", model_path=args.ckpt_path,
                        exec_step=8, device=device, threshold=20)
    E.prepare_models()

    bd = benchmark.get_benchmark_dict()

    # build the (suite, task_id) work list, then take this shard's slice
    work = []
    if args.tasklist:
        # explicit LingBot-VA-format jsonl: one {"suite","task_id"} per line
        for line in open(args.tasklist):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            tid = r.get("task_id", r.get("task_idx"))  # LingBot-VA jsonl uses task_idx
            work.append((r["suite"], int(tid)))
    elif args.task_ids:
        ids = [int(x) for x in args.task_ids.split(",") if x != ""]
        for suite in args.suites.split(","):
            if not suite:
                continue
            nn = bd[suite]().n_tasks
            for tid in ids:
                if 0 <= tid < nn:
                    work.append((suite, tid))
    else:
        for suite in args.suites.split(","):
            if not suite:
                continue
            n = bd[suite]().n_tasks
            if args.max_tasks_per_suite > 0:
                n = min(n, args.max_tasks_per_suite)
            for tid in range(n):
                work.append((suite, tid))
    mine = [w for idx, w in enumerate(work) if idx % args.num_shards == args.shard]

    # resume: load prior partial results for this shard
    partial_path = os.path.join(args.out_dir, f"results_shard{args.shard}.json")
    # crashguard: native SIGABRT (MuJoCo/EGL core dump) kills the process before any
    # record is written -> resume would retry the SAME task and crash-loop. We append
    # (suite,tid) to this sidecar RIGHT BEFORE each rollout; on restart, any crashguard
    # entry not present in results = it crashed -> record as a failure and skip it.
    crashguard_path = os.path.join(args.out_dir, f"crashguard_shard{args.shard}.jsonl")
    grand = {"by_task": [], "by_suite": {}, "by_category": {}, "overall": None}
    # GLOBAL resume: skip any task already done by ANY shard file (lets us change the
    # number of parallel workers freely without repeating completed rollouts).
    done_keys = set()
    for pf in glob.glob(os.path.join(args.out_dir, "results_shard*.json")):
        try:
            for rec in json.load(open(pf))["by_task"]:
                done_keys.add((rec["suite"], rec["task_id"]))
        except Exception:
            pass
    if os.path.exists(partial_path):
        try:
            grand = json.load(open(partial_path))
        except Exception:
            grand = {"by_task": [], "by_suite": {}, "by_category": {}, "overall": None}

    # Fold crashguard: any (suite,tid) attempted (in crashguard) but NOT recorded in
    # results crashed the process natively -> record it as a FAILURE now and skip.
    attempted = []
    if os.path.exists(crashguard_path):
        for line in open(crashguard_path):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line); attempted.append((r["suite"], int(r["task_id"])))
            except Exception:
                pass
    for (csuite, ctid) in attempted:
        if (csuite, ctid) in done_keys:
            continue
        ccat = cls[csuite][ctid]["category"] if ctid < len(cls[csuite]) else "Unknown"
        rec = {"suite": csuite, "task_id": ctid, "category": ccat, "successes": 0,
               "trials": 1, "rate": 0.0, "desc": "CRASHED (native abort) -> counted fail",
               "sec": 0.0}
        grand["by_task"].append(rec)
        bs = grand["by_suite"].setdefault(csuite, {"successes": 0, "trials": 0})
        bs["trials"] += 1; bs["rate"] = bs["successes"] / bs["trials"]
        bc = grand["by_category"].setdefault(ccat, {"successes": 0, "trials": 0})
        bc["trials"] += 1; bc["rate"] = bc["successes"] / bc["trials"]
        done_keys.add((csuite, ctid))
    if attempted:
        with open(partial_path, "w") as f:
            json.dump(grand, f, indent=2)

    suite_cache = {}
    cur_suite = None
    t_start = time.time()
    for i, (suite, tid) in enumerate(mine):
        if (suite, tid) in done_keys:
            continue
        if suite != cur_suite:
            set_suite_stats(E, suite)
            cur_suite = suite
        ts = suite_cache.setdefault(suite, bd[suite]())
        task = ts.get_task(tid)
        inits = ts.get_task_init_states(tid)
        category = cls[suite][tid]["category"] if tid < len(cls[suite]) else "Unknown"
        t0 = time.time()
        # crashguard: mark this task as attempted BEFORE the (possibly native-crashing) rollout
        with open(crashguard_path, "a") as _cg:
            _cg.write(json.dumps({"suite": suite, "task_id": tid}) + "\n")
            _cg.flush(); os.fsync(_cg.fileno())
        try:
            success, desc = run_one_task(E, task, inits, MAX_STEPS[suite], device)
        except Exception as e:
            success, desc = False, f"ERROR: {e}"
        sc = int(success)

        rec = {"suite": suite, "task_id": tid, "category": category,
               "successes": sc, "trials": 1, "rate": float(sc), "desc": desc,
               "sec": round(time.time() - t0, 1)}
        grand["by_task"].append(rec)
        bs = grand["by_suite"].setdefault(suite, {"successes": 0, "trials": 0})
        bs["successes"] += sc; bs["trials"] += 1; bs["rate"] = bs["successes"] / bs["trials"]
        bc = grand["by_category"].setdefault(category, {"successes": 0, "trials": 0})
        bc["successes"] += sc; bc["trials"] += 1; bc["rate"] = bc["successes"] / bc["trials"]
        tot_s = sum(v["successes"] for v in grand["by_suite"].values())
        tot_t = sum(v["trials"] for v in grand["by_suite"].values())
        grand["overall"] = {"successes": tot_s, "trials": tot_t, "rate": tot_s / max(tot_t, 1)}

        if i % 20 == 0 or sc == 0:
            eta = (time.time() - t_start) / max(i + 1, 1) * (len(mine) - i - 1) / 3600
            print(f"[shard{args.shard} {i+1}/{len(mine)}] {suite} t{tid} [{category}] "
                  f"{'OK' if sc else 'x'} | overall {grand['overall']['rate']:.3f} "
                  f"({tot_s}/{tot_t}) ETA {eta:.1f}h", flush=True)
        with open(partial_path, "w") as f:
            json.dump(grand, f, indent=2)

    print(f">>> SHARD{args.shard}_DONE overall={grand['overall']}", flush=True)


if __name__ == "__main__":
    main()
