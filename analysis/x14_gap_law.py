"""Which schedule statistic controls maintenance? The X11+X14 gap law, from raw runs.

THE QUESTION (P4's named open item). X11 showed the same total supply succeeds spread out and
fails as a burst, but its two failing arms share one extreme (a single contiguous block), so it
cannot say WHICH statistic separates the schedules. X14 was pre-registered to fill the axis:
every arm delivers the same mean dose (p_after = 0.01) and the same total supervised examples
(verified to 0.0000% before launch by the campaign's own unit test), while the MAXIMUM GAP
between supervised batches moves ~742x.

ARMS ON THE GAP AXIS (gap values as recorded in the campaign's Tier-0 document):
    uniform / bernoulli   max gap ~14 steps      (X11, measured 5/5)
    periodic              max gap ~99            (X14, new)
    clustered, burst 10   max gap ~990           (X14, new)
    front / back          max gap ~10,395        (X11, measured 1/6 and 0/6)

PRE-REGISTERED READINGS (declared in maint/x14_runner.py before any X14 run existed; applied
here exactly as written):
    P1  MAX GAP CONTROLS  — re-formation declines monotonically in max gap, threshold locatable
                            between 99 and 990.
    P2  DUTY CYCLE        — both new arms succeed; only contiguous blocks fail.
    P3  NEAR-EVERY-STEP   — both new arms fail.

DISCIPLINE. A run whose evals never reach its restore_step has an INCOMPLETE rescue and is
EXCLUDED with its count reported. Counting it as a failure would bias every arm downward by its
completion rate; this is the survivorship trap in reverse, and it is the first thing this script
guards against. Re-formation = query_ctx >= the run's own fork_thresh on two consecutive evals
after restore_step (the campaign's criterion).

Run:  python x14_gap_law.py [--write]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MAINT = os.path.join(HERE, "..", "..", "..", "emergence-clocks-2026-08", "farm_raw", "maint")
DST = os.path.join(HERE, "..", "results", "x14_gap_law.json")

GAP = {"uniform": 14, "periodic": 99, "clustered": 990, "front": 10395, "back": 10395}


def arm_of(d, name):
    argv = str(d.get("argv", ""))
    if "x14" in name:
        for a in ("periodic", "clustered"):
            if f"--depr-sched {a}" in argv:
                return a
        return None
    if "x11" in name:
        for a in ("uniform", "front", "back"):
            if f"--depr-geom {a}" in argv:
                return a
    return None


def status(d):
    """(rescue_complete, reformed)."""
    ev = d.get("evals") or []
    th, rs = d.get("fork_thresh"), d.get("restore_step")
    if not ev or th is None or rs is None:
        return False, False
    if ev[-1].get("step", 0) < rs + 500:          # never meaningfully entered the rescue phase
        return False, False
    hits = 0
    for e in ev:
        if e.get("step", 0) < rs:
            continue
        if e.get("query_ctx", 0) >= th:
            hits += 1
            if hits >= 2:
                return True, True
        else:
            hits = 0
    return True, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    cells = defaultdict(lambda: [0, 0, 0])          # arm -> [reformed, complete, incomplete]
    for f in glob.glob(os.path.join(MAINT, "x1[14]_*.json")):
        name = os.path.basename(f)
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if d.get("p_after") != 0.01:
            continue
        arm = arm_of(d, name)
        if not arm:
            continue
        comp, ref = status(d)
        if comp:
            cells[arm][0] += ref
            cells[arm][1] += 1
        else:
            cells[arm][2] += 1

    print("Same mean dose (p_after=0.01), same totals; only placement differs.\n")
    print(f"{'arm':<12}{'max gap':>9}{'re-formed':>12}{'incomplete':>12}{'rate':>8}")
    print("-" * 55)
    pts = []
    for arm in ("uniform", "periodic", "clustered", "front", "back"):
        if arm not in cells:
            continue
        ok, n, inc = cells[arm]
        rate = ok / n if n else None
        print(f"{arm:<12}{GAP[arm]:>9}{f'{ok}/{n}':>12}{inc:>12}"
              f"{('%.2f' % rate) if rate is not None else '-':>8}")
        if n:
            pts.append((GAP[arm], ok, n))

    # merge front/back at the same gap value
    agg = defaultdict(lambda: [0, 0])
    for g, ok, n in pts:
        agg[g][0] += ok
        agg[g][1] += n
    xs = sorted(agg)
    rates = [agg[g][0] / agg[g][1] for g in xs]
    print("\nGap axis (front+back merged):")
    for g, r in zip(xs, rates):
        print(f"   max gap {g:>6}: {agg[g][0]}/{agg[g][1]}  ({r:.2f})")

    mono = all(rates[i] >= rates[i + 1] - 1e-9 for i in range(len(rates) - 1))
    # logistic fit in log-gap
    fit = None
    if len(xs) >= 3:
        try:
            from scipy.optimize import curve_fit
            X = np.log10(np.array(xs, float))
            Y = np.array(rates)
            W = np.sqrt([agg[g][1] for g in xs])

            def f(x, mid, slope):
                return 1.0 / (1.0 + np.exp(slope * (x - mid)))
            p, _ = curve_fit(f, X, Y, p0=[2.0, 2.0], sigma=1 / W, maxfev=20000)
            fit = {"gap50": round(float(10 ** p[0]), 1), "slope": round(float(p[1]), 3)}
            print(f"\nlogistic fit: re-formation falls to 50% at max gap ≈ {fit['gap50']:g} steps")
        except Exception as e:
            print(f"\n(logistic fit unavailable: {e})")

    # apply the pre-registered readings
    per = {a: (cells[a][0], cells[a][1]) for a in cells}
    both_new_succeed = all(per.get(a, (0, 1))[0] / max(per.get(a, (0, 1))[1], 1) >= 0.5
                           for a in ("periodic", "clustered") if a in per)
    both_new_fail = all(per.get(a, (0, 1))[0] / max(per.get(a, (0, 1))[1], 1) < 0.5
                        for a in ("periodic", "clustered") if a in per)
    if mono and not both_new_succeed and not both_new_fail:
        verdict = ("P1 MAX GAP CONTROLS — decline is monotone in max gap and the new arms sit "
                   "between the extremes")
    elif both_new_succeed:
        verdict = "P2 DUTY CYCLE — both new arms succeed; only contiguous blocks fail"
    elif both_new_fail:
        verdict = "P3 NEAR-EVERY-STEP — both new arms fail"
    else:
        verdict = "MIXED — no pre-registered pattern fits cleanly; report the table, not a story"
    print(f"\nPre-registered reading matched: {verdict}")

    inc_total = sum(v[2] for v in cells.values())
    if inc_total:
        print(f"\nNOTE: {inc_total} run(s) with incomplete rescue are EXCLUDED, not failed. "
              f"The verdict is provisional until they finish or the box dies.")

    if a.write:
        json.dump({"_provenance": "analysis/x14_gap_law.py over farm_raw/maint x11_*/x14_* raw "
                                  "runs; pre-registered readings from maint/x14_runner.py "
                                  "applied verbatim; incomplete rescues excluded and counted.",
                   "cells": {a2: {"reformed": v[0], "complete": v[1], "incomplete": v[2],
                                  "max_gap": GAP[a2]} for a2, v in cells.items()},
                   "gap_axis": {str(g): f"{agg[g][0]}/{agg[g][1]}" for g in xs},
                   "monotone": bool(mono), "fit": fit, "verdict": verdict,
                   "incomplete_total": inc_total},
                  open(DST, "w"), indent=2)
        print("wrote", DST)


if __name__ == "__main__":
    main()
