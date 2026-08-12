"""Independent re-derivation of the 8xH100 developmental-pretraining program, from raw run JSONs.

WHY. Every H100 number reported so far was quoted from the campaign's own generated documents.
The operator challenged the central conclusion (the LR-warmup artifact). The response is not to
restate the documents and not to defer to them: it is to re-derive every headline from the raw
run records with definitions fixed before looking, and to give the challenged hypothesis a
fair, pre-declared test it has not had.

METHODOLOGY, DECLARED BEFORE ANY NUMBER BELOW WAS COMPUTED
  Crossing (DSP): the first eval step with target accuracy >= 0.90, on the total-compute clock
      (evals count priming steps). Runs that never cross are CENSORED, never scored at the cap.
      Robustness variant: two consecutive evals >= 0.90 ("sustained"), reported beside it.
  Rank test with censoring: Mann-Whitney on crossing steps with censored runs ranked above every
      crossing run (worst rank). This uses every run; it is conservative for detecting speedups.
  Strata: warmup is a stratification variable, never pooled across. Baselines are matched on
      (target, warmup). Only target=stateupd is analysed for effects: the documents record that
      multihop's baseline never crosses and binding has no baseline, and the raw counts are
      re-checked here.
  Emergence / re-formation (maint): the campaign's own per-run criteria as recorded in each
      file (fork_thresh where present; otherwise chance + 0.5*(1-chance)).

PRE-DECLARED QUESTIONS (with decision rules, written before running):
  Q1 WARMUP: does the unprimed baseline cross faster with warmup=1 than warmup=500?
      Decide by Mann-Whitney with censoring, alpha 0.01. This re-tests the artifact's premise.
  Q2 ARTIFACT: at doses <= 1000 and warmup=500, does shuffle_chess "help" as much as chess
      (both beating the warmup-500 baseline)? If yes, the low-dose benefit is not content.
  Q3 REAL BENEFIT (the operator's hypothesis, total clock): at ANY dose with warmup=1, does
      chess cross faster than the warmup-1 baseline (rank test p < 0.05)? If yes at any dose,
      priming is compute-positive somewhere and the "no dose pays" conclusion is wrong.
  Q4 REAL ACCELERATION (the operator's hypothesis, phase-B clock): with warmup=1, is time from
      switchover to threshold under chess priming SHORTER than the baseline's total time
      (rank test p < 0.05 at any dose)? This is the form in which "not an artifact" can be true
      even if Q3 fails: priming may genuinely accelerate learning while costing more than it
      saves. The two clocks answer different questions and both are reported.
  Q5 HARM: chess vs shuffle_chess crossing rate at matched dose and warmup, Fisher per dose.
  Q6 STRATEGY: chess vs chess_random pooled at matched dose and warmup, Fisher.
  Q7 MAINTENANCE/REVIVAL: re-derive the K x p_after retention table, the revival dose table,
      the true-zero irreversibility count, and the X11 geometry split from raw maint files;
      report agreement with the published tables.

Anything not covered by a pre-declared question is reported as descriptive only.

Run:  python h100_reanalysis.py [--write]
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
RAW = os.path.join(HERE, "..", "..", "..", "emergence-clocks-2026-08", "farm_raw")
DST = os.path.join(HERE, "..", "results", "h100_reanalysis.json")
OUT = {}


def hr(t):
    print()
    print("=" * 84)
    print(t)
    print("=" * 84)


def mw_censored(a_steps, a_cens, b_steps, b_cens):
    """Mann-Whitney with censored observations ranked above all observed ones.

    Returns two-sided p via normal approximation with tie correction. a/b_steps are crossing
    steps of runs that crossed; a/b_cens are counts of censored runs in each arm.
    """
    from scipy.stats import mannwhitneyu
    BIG = 10 ** 9
    a = list(a_steps) + [BIG] * a_cens
    b = list(b_steps) + [BIG] * b_cens
    if len(a) < 2 or len(b) < 2:
        return None
    try:
        return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except ValueError:
        return None


def fisher(a, b, c, d):
    from scipy.stats import fisher_exact
    return float(fisher_exact([[a, b], [c, d]])[1])


# ------------------------------------------------------------------ load DSP
def load_dsp():
    runs = []
    for f in glob.glob(os.path.join(RAW, "dsp", "*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        a = d.get("argv") or {}
        ev = d.get("evals") or []
        if not a or not ev:
            continue
        cross = None
        sustained = None
        for i, e in enumerate(ev):
            if e.get("acc", 0) >= 0.90:
                if cross is None:
                    cross = e["step"]
                if sustained is None and i + 1 < len(ev) and ev[i + 1].get("acc", 0) >= 0.90:
                    sustained = e["step"]
                if cross is not None and sustained is not None:
                    break
        runs.append({
            "file": os.path.basename(f), "block": os.path.basename(f).split("_")[0],
            "prime": a.get("prime"), "dose": a.get("prime_steps"),
            "target": a.get("target"), "warmup": a.get("warmup"),
            "seed": a.get("seed"), "total": a.get("total_steps"),
            "done": bool(d.get("done")), "cross": cross, "sustained": sustained,
            "last_acc": ev[-1].get("acc"), "last_step": ev[-1].get("step"),
        })
    return runs


def cell(runs, **kw):
    out = runs
    for k, v in kw.items():
        out = [r for r in out if r.get(k) == v]
    return out


def summarize(rs):
    crossed = [r["cross"] for r in rs if r["cross"] is not None]
    return {"n": len(rs), "crossed": len(crossed), "cens": len(rs) - len(crossed),
            "median": float(np.median(crossed)) if crossed else None,
            "steps": crossed}


def q1_warmup(runs):
    hr("Q1 | The warmup premise, re-derived from raw (none-baselines, target=stateupd, dose=0)")
    base = [r for r in runs if r["prime"] == "none" and r["dose"] == 0 and r["target"] == "stateupd"]
    strata = {}
    for wu in sorted({r["warmup"] for r in base}):
        s = summarize([r for r in base if r["warmup"] == wu])
        strata[wu] = s
        print(f"  warmup={wu:<5} n={s['n']:<4} crossed={s['crossed']}/{s['n']}  "
              f"median={s['median'] and round(s['median'])}  "
              f"range={[min(s['steps']), max(s['steps'])] if s['steps'] else '-'}")
    if 500 in strata and 1 in strata:
        p = mw_censored(strata[1]["steps"], strata[1]["cens"], strata[500]["steps"], strata[500]["cens"])
        ratio = strata[500]["median"] / strata[1]["median"] if strata[1]["median"] else None
        print(f"\n  warmup 500 vs 1: median ratio {ratio:.2f}x, Mann-Whitney (censored-aware) p = {p:.2e}")
        print(f"  published (TIER0 R0.4a): 28,250 vs 6,750, 4.19x, p = 1.87e-07")
        OUT["Q1_warmup"] = {"strata": {str(k): {kk: vv for kk, vv in v.items() if kk != 'steps'}
                                       for k, v in strata.items()},
                            "ratio": ratio and round(ratio, 3), "p": p}


def q2_artifact(runs):
    hr("Q2 | Low dose, warmup=500: does the structure-free control 'help' as much as chess?")
    b500 = summarize(cell(runs, prime="none", dose=0, target="stateupd", warmup=500))
    print(f"  baseline none/w500: {b500['crossed']}/{b500['n']} median {b500['median'] and round(b500['median'])}")
    rows = []
    for prime in ("chess", "shuffle_chess", "chess_random"):
        for dose in (250, 500, 1000):
            s = summarize(cell(runs, prime=prime, dose=dose, target="stateupd", warmup=500))
            if s["n"] == 0:
                continue
            p = mw_censored(s["steps"], s["cens"], b500["steps"], b500["cens"])
            adv = (b500["median"] - s["median"]) if (s["median"] and b500["median"]) else None
            rows.append((prime, dose, s, adv, p))
            print(f"  {prime:<14} d={dose:<5} {s['crossed']}/{s['n']}  median={s['median'] and round(s['median'])}  "
                  f"saved-vs-baseline={adv and round(adv)}  p={p and f'{p:.4f}'}")
    OUT["Q2_low_dose_w500"] = [
        {"prime": r[0], "dose": r[1], "crossed": r[2]["crossed"], "n": r[2]["n"],
         "median": r[2]["median"], "saved": r[3], "p": r[4]} for r in rows]
    print("\n  Decision rule: artifact reading holds if shuffle_chess also saves ~the same amount.")


def q3_q4_real(runs):
    hr("Q3/Q4 | warmup=1: is chess priming compute-positive (total clock) or accelerating (phase-B clock)?")
    b1 = summarize(cell(runs, prime="none", dose=0, target="stateupd", warmup=1))
    print(f"  baseline none/w1: {b1['crossed']}/{b1['n']} median total={b1['median'] and round(b1['median'])}")
    rows = []
    for prime in ("chess", "shuffle_chess", "chess_random"):
        doses = sorted({r["dose"] for r in cell(runs, prime=prime, target="stateupd", warmup=1)})
        for dose in doses:
            rs = cell(runs, prime=prime, dose=dose, target="stateupd", warmup=1)
            if not rs:
                continue
            s = summarize(rs)
            phaseB = [c - dose for c in s["steps"]]
            p_tot = mw_censored(s["steps"], s["cens"], b1["steps"], b1["cens"])
            p_pb = mw_censored(phaseB, s["cens"], b1["steps"], b1["cens"])
            med_pb = float(np.median(phaseB)) if phaseB else None
            rows.append({"prime": prime, "dose": dose, "n": s["n"], "crossed": s["crossed"],
                         "median_total": s["median"], "median_phaseB": med_pb,
                         "p_total_vs_base": p_tot, "p_phaseB_vs_base": p_pb})
            print(f"  {prime:<14} d={dose:<6} {s['crossed']}/{s['n']:<3} "
                  f"total={s['median'] and round(s['median']):<7} "
                  f"phaseB={med_pb and round(med_pb):<7} "
                  f"p_tot={p_tot and f'{p_tot:.4f}':<8} p_phaseB={p_pb and f'{p_pb:.4f}'}")
    OUT["Q3Q4_w1"] = {"baseline": {k: v for k, v in b1.items() if k != "steps"}, "arms": rows}
    print("""
  Reading: Q3 (compute-positive) fires if any chess row beats baseline on TOTAL clock, p<0.05.
  Q4 (real acceleration) fires if any chess row beats baseline on PHASE-B clock, p<0.05 —
  priming then genuinely speeds later learning even if the dose costs more than it saves.""")


def q5_harm(runs):
    hr("Q5 | chess vs shuffle_chess at matched dose and warmup (crossing rate)")
    rows = []
    for wu in (500, 1):
        doses = sorted({r["dose"] for r in runs if r["prime"] in ("chess", "shuffle_chess")
                        and r["target"] == "stateupd" and r["warmup"] == wu})
        for dose in doses:
            c = summarize(cell(runs, prime="chess", dose=dose, target="stateupd", warmup=wu))
            s = summarize(cell(runs, prime="shuffle_chess", dose=dose, target="stateupd", warmup=wu))
            if c["n"] == 0 or s["n"] == 0:
                continue
            p = fisher(c["crossed"], c["n"] - c["crossed"], s["crossed"], s["n"] - s["crossed"])
            rows.append({"warmup": wu, "dose": dose, "chess": f"{c['crossed']}/{c['n']}",
                         "shuffle": f"{s['crossed']}/{s['n']}", "fisher_p": round(p, 5)})
            print(f"  w={wu:<4} d={dose:<6} chess {c['crossed']}/{c['n']:<4} "
                  f"shuffle {s['crossed']}/{s['n']:<4} Fisher p={p:.4f}")
    OUT["Q5_harm"] = rows


def q6_strategy(runs):
    hr("Q6 | Strategy: chess vs chess_random, pooled over matched (dose, warmup) cells")
    a_c = a_n = b_c = b_n = 0
    for wu in (500, 1):
        doses = {r["dose"] for r in runs if r["prime"] == "chess" and r["warmup"] == wu
                 and r["target"] == "stateupd"}
        doses &= {r["dose"] for r in runs if r["prime"] == "chess_random" and r["warmup"] == wu
                  and r["target"] == "stateupd"}
        for dose in doses:
            c = summarize(cell(runs, prime="chess", dose=dose, target="stateupd", warmup=wu))
            r2 = summarize(cell(runs, prime="chess_random", dose=dose, target="stateupd", warmup=wu))
            a_c += c["crossed"]; a_n += c["n"]; b_c += r2["crossed"]; b_n += r2["n"]
    if a_n and b_n:
        p = fisher(a_c, a_n - a_c, b_c, b_n - b_c)
        print(f"  chess {a_c}/{a_n}  vs  chess_random {b_c}/{b_n}   Fisher p = {p:.4f}")
        print(f"  published (TIER0 R0.4c): 148/169 vs 60/64, p = 0.2368")
        OUT["Q6_strategy"] = {"chess": f"{a_c}/{a_n}", "random": f"{b_c}/{b_n}", "p": round(p, 4)}


# ------------------------------------------------------------------ maint
def q7_maint():
    hr("Q7 | Maintenance and revival, re-derived from raw maint files")
    runs = []
    for f in glob.glob(os.path.join(RAW, "maint", "*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        ev = d.get("evals") or d.get("hist") or []
        if not ev:
            continue
        runs.append((os.path.basename(f), d, ev))
    print(f"  loaded {len(runs)} maint files")

    # X10a true-zero irreversibility + X9/X10b dose table
    def reformed(d, ev):
        th = d.get("fork_thresh")
        rs = d.get("restore_step")
        if th is None or rs is None:
            return None
        hits = 0
        for e in ev:
            if e.get("step", 0) >= rs and e.get("query_ctx", e.get("query", 0)) >= th:
                hits += 1
                if hits >= 2:
                    return True
            elif e.get("step", 0) >= rs:
                hits = 0
        return False

    dose = defaultdict(lambda: [0, 0])
    zero_named = []
    for name, d, ev in runs:
        if not (name.startswith("x9_") or name.startswith("x10")):
            continue
        r = reformed(d, ev)
        if r is None:
            continue
        pa = d.get("p_after", d.get("argv", {}).get("p_after") if isinstance(d.get("argv"), dict) else None)
        if pa is None:
            continue
        dose[float(pa)][0] += bool(r)
        dose[float(pa)][1] += 1
        if float(pa) == 0.0:
            zero_named.append((name, r))
    print("\n  revival dose table (re-formed / n):")
    for pa in sorted(dose):
        ok, n = dose[pa]
        print(f"    p_after={pa:<7} {ok}/{n}")
    pub = {0.0: "0/11", 0.002: "0/6", 0.005: "3/6", 0.01: "6/6", 0.02: "6/6", 0.04: "6/6"}
    print(f"  published (F sec.3): {pub}")
    OUT["Q7_revival"] = {str(k): f"{v[0]}/{v[1]}" for k, v in sorted(dose.items())}

    # X11 geometry
    geo = defaultdict(lambda: [0, 0])
    for name, d, ev in runs:
        if not name.startswith("x11_"):
            continue
        r = reformed(d, ev)
        if r is None:
            continue
        g = d.get("depr_geometry", d.get("argv", {}).get("depr_geometry")
                  if isinstance(d.get("argv"), dict) else None) or \
            ("uniform" if "_uni" in name else "front" if "_front" in name else
             "back" if "_back" in name else None)
        if g:
            geo[g][0] += bool(r)
            geo[g][1] += 1
    if geo:
        print("\n  X11 geometry (re-formed / n): "
              + "  ".join(f"{g}={v[0]}/{v[1]}" for g, v in sorted(geo.items())))
        print("  published (F sec.4): uniform 5/5, front 1/6, back 0/6")
        OUT["Q7_geometry"] = {g: f"{v[0]}/{v[1]}" for g, v in geo.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    runs = load_dsp()
    print(f"[dsp] {len(runs)} runs loaded; targets: "
          + str({t: sum(1 for r in runs if r['target'] == t) for t in {r['target'] for r in runs}}))
    ok = [r for r in runs if r["target"] == "multihop"]
    mb = summarize([r for r in ok if r["prime"] == "none"])
    print(f"[check] multihop baseline crossings: {mb['crossed']}/{mb['n']} "
          f"(published: 0/5 — target unmeasurable)")
    q1_warmup(runs)
    q2_artifact(runs)
    q3_q4_real(runs)
    q5_harm(runs)
    q6_strategy(runs)
    q7_maint()
    if a.write:
        OUT["_provenance"] = ("analysis/h100_reanalysis.py over research/emergence-clocks-2026-08/"
                              "farm_raw (synced from s3://.../farm/results). Definitions and "
                              "decision rules declared in the docstring before computation.")
        json.dump(OUT, open(DST, "w"), indent=2)
        print(f"\nwrote {DST}")


if __name__ == "__main__":
    main()
