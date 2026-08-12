"""Verdict for the pre-registered d250 replication. WRITTEN BEFORE THE DATA EXISTED.

The pre-registration (farm/PREREG_d250_replication.md, in S3 with a timestamp before launch)
fixes everything this script does. It is committed before the first run finished so the analysis
cannot bend to the data.

  PRIMARY   chess d250 (n=20, seeds 900-919) vs the none/warmup-1 baseline (33 existing runs
            from farm_raw/dsp + up to 6 new contemporaneous ones). Endpoint: total-clock
            crossing step, first eval acc >= 0.90, censored if never. Test: Mann-Whitney with
            censored runs ranked worst, two-sided, alpha 0.05. ONE test.
  GATES     confirmed only if (1) chess median < baseline median, (2) p < 0.05, and
            (3) shuffle_chess d250 is NOT significantly faster than baseline (p >= 0.05
            two-sided). Any other pattern: NOT CONFIRMED, reported as such.
  SECONDARY (non-gating) phase-B clock = crossing - 250.

Run:  python d250_verdict.py [--src <dir with d250rep_*.json>]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FARMRAW = os.path.join(HERE, "..", "..", "..", "emergence-clocks-2026-08", "farm_raw", "dsp")
DEFAULT_SRC = os.path.join(HERE, "..", "..", "..", "emergence-clocks-2026-08", "d250rep")
DST = os.path.join(HERE, "..", "results", "d250_verdict.json")


def crossing(d):
    for e in d.get("evals") or []:
        if e.get("acc", 0) >= 0.90:
            return e["step"]
    return None


def load_dir(pattern):
    out = []
    for f in glob.glob(pattern):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        a = d.get("argv") or {}
        out.append({"file": os.path.basename(f), "prime": a.get("prime"),
                    "dose": a.get("prime_steps"), "warmup": a.get("warmup"),
                    "seed": a.get("seed"), "target": a.get("target"),
                    "done": bool(d.get("done")), "cross": crossing(d)})
    return out


def mw_censored(a_steps, a_cens, b_steps, b_cens):
    from scipy.stats import mannwhitneyu
    BIG = 10 ** 9
    a = list(a_steps) + [BIG] * a_cens
    b = list(b_steps) + [BIG] * b_cens
    return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)


def summ(rs):
    c = [r["cross"] for r in rs if r["cross"] is not None]
    return {"n": len(rs), "crossed": len(c), "cens": len(rs) - len(c),
            "median": float(np.median(c)) if c else None, "steps": c}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    new = load_dir(os.path.join(a.src, "d250rep_*.json"))
    old = load_dir(os.path.join(FARMRAW, "*.json"))
    base_old = [r for r in old if r["prime"] == "none" and r["dose"] == 0
                and r["warmup"] == 1 and r["target"] == "stateupd"]
    base_new = [r for r in new if r["prime"] == "none"]
    chess = [r for r in new if r["prime"] == "chess"]
    shuf = [r for r in new if r["prime"] == "shuffle_chess"]

    incomplete = [r for r in new if not r["done"]]
    print(f"loaded: chess {len(chess)}, shuffle {len(shuf)}, new-baseline {len(base_new)}, "
          f"existing baseline {len(base_old)}; incomplete files: {len(incomplete)}")
    if incomplete:
        print("NOTE: incomplete runs present; the verdict below is provisional until 0 remain.")

    base = summ(base_old + base_new)
    sc = summ(chess)
    ss = summ(shuf)
    print(f"\nbaseline none/w1:  {base['crossed']}/{base['n']}  median {base['median']}")
    print(f"chess d250 w1:     {sc['crossed']}/{sc['n']}  median {sc['median']}")
    print(f"shuffle d250 w1:   {ss['crossed']}/{ss['n']}  median {ss['median']}")

    p_chess = mw_censored(sc["steps"], sc["cens"], base["steps"], base["cens"]) if sc["n"] else None
    p_shuf = mw_censored(ss["steps"], ss["cens"], base["steps"], base["cens"]) if ss["n"] else None
    print(f"\nPRIMARY  chess vs baseline  p = {p_chess}")
    print(f"GATE3    shuffle vs baseline p = {p_shuf}")

    g1 = sc["median"] is not None and base["median"] is not None and sc["median"] < base["median"]
    g2 = p_chess is not None and p_chess < 0.05
    shuffle_faster = (ss["median"] is not None and base["median"] is not None
                      and ss["median"] < base["median"])
    g3 = not (shuffle_faster and p_shuf is not None and p_shuf < 0.05)
    verdict = "CONFIRMED" if (g1 and g2 and g3) else "NOT CONFIRMED"
    print(f"\ngate1 chess-median-lower: {g1}   gate2 p<0.05: {g2}   gate3 specificity: {g3}")
    print(f"VERDICT (per pre-registration): {verdict}")

    pb = [c - 250 for c in sc["steps"]]
    if pb:
        p_pb = mw_censored(pb, sc["cens"], base["steps"], base["cens"])
        print(f"secondary phase-B: median {float(np.median(pb))}, p = {p_pb}")

    out = {"_prereg": "farm/PREREG_d250_replication.md (S3, timestamped before launch); this "
                      "script committed before the data existed.",
           "baseline": {k: v for k, v in base.items() if k != "steps"},
           "chess": {k: v for k, v in sc.items() if k != "steps"},
           "shuffle": {k: v for k, v in ss.items() if k != "steps"},
           "p_chess": p_chess, "p_shuffle": p_shuf,
           "gates": {"median_lower": g1, "p_below_alpha": g2, "specificity": g3},
           "incomplete_runs": len(incomplete), "verdict": verdict}
    if a.write:
        json.dump(out, open(DST, "w"), indent=2)
        print("wrote", DST)


if __name__ == "__main__":
    main()
