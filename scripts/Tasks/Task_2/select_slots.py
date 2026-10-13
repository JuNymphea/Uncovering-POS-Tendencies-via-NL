#!/usr/bin/env python
"""Choose the Task 2 slot set from the candidate pool and freeze it.

Each candidate is fitted on the calibration words; a swap-based hill climb from
random starts then picks n slots per POS that maximize accuracy on the probe-dev
words (--consensus keeps the slots chosen most often across seeds). Candidates
whose realized prefix needs a plural noun are skipped, and final-test words are
not read. task2.py always decides between noun, verb and adj. Writes --out.

--surprisal is task2.py output (real words) over the candidate pool (out/candidates.json from
run_induction.py; not shipped). The shipped out/probe_set_<m>.json were selected this
way; writing to the same path overwrites them.

    python scripts/Tasks/Task_2/select_slots.py --surprisal <pool surprisal> --out <probe set>
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import statistics as st
from pathlib import Path
from typing import Dict, List

POS_KEYS = ("noun", "verb", "adj")
UPPER = {"noun": "NOUN", "verb": "VERB", "adj": "ADJ"}

#: Determiners and quantifiers that fix the number of the following noun. They are
#: read from the realized prefix, since realization can change the number
#: (`NUM {SLOT} of` is realized with "one").
SINGULAR_CUES = {"a", "an", "one", "this", "that", "each", "every", "another"}
PLURAL_CUES = {"all", "both", "these", "those", "many", "few", "several",
               "two", "three", "four", "five", "some", "most", "various"}


def incompatible_with_bare_form(prefix_tokens: List[str]) -> str | None:
    """Why a bare-form control word cannot go in this item, or None."""
    lowered = [t.lower() for t in prefix_tokens]
    try:
        slot = lowered.index("{neologism}")
    except ValueError:
        return None
    # Only the determiner region immediately before the slot binds its number.
    for tok in reversed(lowered[:slot]):
        if tok in PLURAL_CUES:
            return f"realized prefix requires a plural ('{tok}')"
        if tok in SINGULAR_CUES:
            return None
        if tok in {"the", "other", "different", "same", "very", "more", "most"}:
            continue
        break
    return None


def gaussian_logpdf(x: float, mu: float, sigma: float) -> float:
    sigma = max(sigma, 1e-6)
    return -0.5 * math.log(2 * math.pi * sigma * sigma) - (x - mu) ** 2 / (2 * sigma * sigma)


def auc(matched: List[float], nonmatched: List[float]) -> float:
    """P(a matched word is less surprising than a non-matched one), by rank sum."""
    if not matched or not nonmatched:
        return float("nan")
    merged = sorted([(s, 1) for s in matched] + [(s, 0) for s in nonmatched])
    ranks, i = {}, 0
    while i < len(merged):
        j = i
        while j + 1 < len(merged) and merged[j + 1][0] == merged[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    r = sum(ranks[k] for k, (_, lab) in enumerate(merged) if lab == 1)
    n1, n0 = len(matched), len(nonmatched)
    return 1.0 - (r - n1 * (n1 + 1) / 2) / (n1 * n0)


class Probe:
    """The per-slot log-likelihood ratios, and accuracy for any slot subset."""

    def __init__(self, data, pool_for_fit):
        self.S = data["surprisal"]
        self.meta = {f"{s['pos']}::{s['signature']}": s for s in data["slots"]}
        self.prefix = {f"{s['pos']}::{s['signature']}": s["prefix"].split()
                       for s in data["slots"]}
        self.order = data.get("candidate_order") or list(self.S)

        self.candidates: Dict[str, List[str]] = {}
        self.skipped: Dict[str, str] = {}
        for pos in POS_KEYS:
            keep = []
            for key in [k for k in self.order if k.startswith(UPPER[pos] + "::")]:
                why = incompatible_with_bare_form(self.prefix[key])
                if why:
                    self.skipped[key] = why
                    continue
                keep.append(key)
            self.candidates[pos] = keep

        # fit each candidate on the calibration words and cache the
        # log-likelihood ratio r for every word
        self.fits, self.r = {}, {}
        for pos in POS_KEYS:
            ok = []
            for key in self.candidates[pos]:
                m = [self.S[key][w] for w in pool_for_fit[pos]
                     if w in self.S[key] and not math.isnan(self.S[key][w])]
                n = [self.S[key][w] for p in POS_KEYS if p != pos
                     for w in pool_for_fit[p]
                     if w in self.S[key] and not math.isnan(self.S[key][w])]
                if len(m) < 2 or len(n) < 2:
                    continue
                f = {"pos": pos, "mu_match": st.mean(m), "sigma_match": st.stdev(m),
                     "mu_nonmatch": st.mean(n), "sigma_nonmatch": st.stdev(n),
                     "n_match": len(m), "n_nonmatch": len(n)}
                self.fits[key] = f
                self.r[key] = {
                    w: gaussian_logpdf(v, f["mu_match"], f["sigma_match"])
                       - gaussian_logpdf(v, f["mu_nonmatch"], f["sigma_nonmatch"])
                    for w, v in self.S[key].items() if not math.isnan(v)}
                ok.append(key)
            self.candidates[pos] = ok

    def accuracy(self, chosen: Dict[str, List[str]], pool) -> float:
        hit = tot = 0
        for gold in POS_KEYS:
            for w in pool[gold]:
                means = {}
                for p in POS_KEYS:
                    v = [self.r[k][w] for k in chosen[p] if w in self.r[k]]
                    if v:
                        means[p] = sum(v) / len(v)
                if means and max(means, key=means.get) == gold:
                    hit += 1
                tot += 1
        return hit / tot if tot else 0.0

    def diagnostics_count(self, chosen) -> int:
        """Distinct first diagnostic tokens per POS, summed; used to break ties."""
        return sum(len({self.meta[k]["diagnostic"][0] for k in chosen[p]})
                   for p in POS_KEYS)

    def slot_auc(self, key, pos, pool) -> float:
        m = [self.S[key][w] for w in pool[pos]
             if w in self.S[key] and not math.isnan(self.S[key][w])]
        n = [self.S[key][w] for p in POS_KEYS if p != pos for w in pool[p]
             if w in self.S[key] and not math.isnan(self.S[key][w])]
        return auc(m, n)


def search(probe: Probe, pool, n_slots: int, restarts: int, seed: int,
           rounds: int = 60, quiet: bool = False):
    """Swap-based hill climbing from many random starts, exactly n per POS."""
    rng = random.Random(seed)
    best = (-1.0, None)
    for _ in range(restarts):
        cur = {p: rng.sample(probe.candidates[p], n_slots) for p in POS_KEYS}
        score = probe.accuracy(cur, pool)
        for _ in range(rounds):
            moves = [(p, i, k) for p in POS_KEYS
                     for i in range(n_slots)
                     for k in probe.candidates[p] if k not in cur[p]]
            rng.shuffle(moves)
            improved = False
            for p, i, k in moves:
                trial = {q: list(v) for q, v in cur.items()}
                trial[p][i] = k
                a = probe.accuracy(trial, pool)
                if a > score or (a == score and
                                 probe.diagnostics_count(trial) > probe.diagnostics_count(cur)):
                    cur, score, improved = trial, a, True
                    break
            if not improved:
                break
        if (score, probe.diagnostics_count(cur)) > (best[0], probe.diagnostics_count(best[1])
                                                    if best[1] else -1):
            best = (score, {p: list(v) for p, v in cur.items()})
    return best


def consensus(probe: Probe, pool, n_slots: int, restarts: int,
              n_seeds: int, rounds: int = 60):
    """Run search() with seeds 0..n_seeds-1 and keep the n_slots most-chosen slots per POS.

    Returns (chosen, votes, best single run).
    """
    votes = {p: collections.Counter() for p in POS_KEYS}
    best_single = (-1.0, None)
    for seed in range(n_seeds):
        score, sel = search(probe, pool, n_slots, restarts, seed, rounds)
        for p in POS_KEYS:
            votes[p].update(sel[p])
        if score > best_single[0]:
            best_single = (score, sel)
        print(f"    seed {seed}: probe-dev {score:.4f}", flush=True)
    # ties on votes go to the higher single-slot AUC
    chosen = {}
    for p in POS_KEYS:
        ranked = sorted(votes[p].items(),
                        key=lambda kv: (-kv[1], -probe.slot_auc(kv[0], p, pool)))
        chosen[p] = [k for k, _ in ranked[:n_slots]]
    return chosen, votes, best_single


def optimism(data, n_slots, restarts, seed, quiet=False) -> float:
    """Mean gap between search accuracy and accuracy on the other half of probe-dev.

    Uses 3 random halvings in both directions.
    """
    words = data["words"]
    rng = random.Random(seed + 1)
    gaps = []
    for rep in range(3):
        A = {p: [] for p in POS_KEYS}
        B = {p: [] for p in POS_KEYS}
        for p in POS_KEYS:
            ws = list(words["probedev"][p])
            rng.shuffle(ws)
            A[p], B[p] = ws[:len(ws) // 2], ws[len(ws) // 2:]
        for tr, te in ((A, B), (B, A)):
            probe = Probe(data, words["calibration"])
            s, sel = search(probe, tr, n_slots, max(8, restarts // 4), seed + rep, quiet=True)
            gaps.append(s - probe.accuracy(sel, te))
    return st.mean(gaps)


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--surprisal", type=Path, default=here / "out" / "surprisal.json")
    ap.add_argument("--out", type=Path, default=here / "out" / "probe_set.json")
    ap.add_argument("--n-slots", type=int, default=10, help="per POS")
    ap.add_argument("--pos-keys", default="noun,verb,adj",
                    help="categories to decide between, comma-separated. "
                         "task2.py uses noun,verb,adj")
    ap.add_argument("--restarts", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--consensus", type=int, default=None, metavar="N",
                    help="run the search with N seeds and keep the --n-slots slots "
                         "chosen most often")
    ap.add_argument("--skip-optimism", action="store_true",
                    help="skip the split-half optimism estimate")
    args = ap.parse_args()

    global POS_KEYS
    POS_KEYS = tuple(k.strip() for k in args.pos_keys.split(",") if k.strip())
    unknown = [p for p in POS_KEYS if p not in UPPER]
    if len(POS_KEYS) < 2 or unknown:
        ap.error(f"--pos-keys must list at least two of {sorted(UPPER)}, comma-separated"
                 + (f"; unknown: {unknown}" if unknown else ""))

    data = json.loads(args.surprisal.read_text())
    words = data["words"]
    probe = Probe(data, words["calibration"])
    probedev = {p: words["probedev"][p] for p in POS_KEYS}

    print(f"candidates ({len(probe.skipped)} excluded for morphology): "
          + "  ".join(f"{p}={len(probe.candidates[p])}" for p in POS_KEYS))
    print(f"searching on {sum(len(v) for v in probedev.values())} probe-dev words, "
          f"{args.restarts} random restarts\n")

    votes = None
    if args.consensus:
        print(f"consensus over {args.consensus} seeds, keeping the most-chosen slots")
        chosen, votes, (best_score, _) = consensus(
            probe, probedev, args.n_slots, args.restarts, args.consensus)
        score = probe.accuracy(chosen, probedev)
        print(f"\n  consensus probe-dev {score:.4f}   best single run {best_score:.4f}"
              f"   diff {score - best_score:+.4f}")
    else:
        score, chosen = search(probe, probedev, args.n_slots, args.restarts, args.seed)
    print(f"probe-dev {len(POS_KEYS)}-way accuracy {score:.4f}"
          f"   distinct first diagnostic tokens {probe.diagnostics_count(chosen)}")

    bias = None
    if not args.skip_optimism:
        bias = optimism(data, args.n_slots, args.restarts, args.seed)
        print(f"optimism (split halves of probe-dev, 6 runs) {bias:+.3f}"
              f"  -> expected held-out about {score - bias:.3f}")

    print(f"\nselected slots (AUC: single-slot separation on probe-dev, for reference)")
    log = {}
    for pos in POS_KEYS:
        print(f"\n  {pos.upper()}")
        log[pos] = []
        for key in chosen[pos]:
            a = probe.slot_auc(key, pos, probedev)
            m = probe.meta[key]
            log[pos].append({"signature": key.split("::")[1], "auc_probedev": round(a, 4),
                             "diagnostic": m["diagnostic"], "example": m["example"]})
            flag = "✗" if a < 0.60 else " "
            print(f"   {a:>6.3f}{flag} {m['example'][:46]:<48}{m['diagnostic']}")
    low = sum(1 for p in POS_KEYS for e in log[p] if e["auc_probedev"] < 0.60)
    print(f"\n  {low}/{args.n_slots * len(POS_KEYS)} have single-slot AUC below 0.60")
    for pos in POS_KEYS:
        c = collections.Counter(probe.meta[k]["diagnostic"][0] for k in chosen[pos])
        print(f"  {pos:<5} first diagnostic tokens {dict(c)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "model": data.get("model"),
        "n_slots_per_pos": args.n_slots,
        "pos_keys": list(POS_KEYS),
        "selection": f"hill-climbing on probe-dev {len(POS_KEYS)}-way accuracy",
        "screened_on": "probe-dev control words",
        "restarts": args.restarts, "seed": args.seed,
        "consensus_seeds": args.consensus,
        "vote_counts": ({p: dict(votes[p].most_common()) for p in POS_KEYS}
                        if votes else None),
        "probedev_accuracy": round(score, 4),
        "optimism_estimate": (round(bias, 4) if bias is not None else None),
        "balanced_design_achieved": all(len(chosen[p]) == args.n_slots for p in POS_KEYS),
        "distinct_diagnostics": probe.diagnostics_count(chosen),
        "slots": chosen,
        "selected_log": log,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nfrozen -> {args.out}")
    print("Slot set fixed; do not change it based on final-test or neologism results.")


if __name__ == "__main__":
    main()
