#!/usr/bin/env python
"""Build the Task 2 control-word splits from pos_words_v2.json.

Words must encode to one token after a space in --tokenizer, and nouns that the
UD treebanks never show with "a"/"an" are dropped as likely mass nouns. The
three POS lists are sampled within token-id frequency strata so that their
frequency profiles match, then split into (default sizes)

    calibration (150/POS)  fit the matched / non-matched surprisal distributions
    probe-dev    (75/POS)  select the slot set (select_slots.py)
    final-test   (75/POS)  report the frozen probe's accuracy

The shipped out/shared_*.json were drawn with --seed 20260922, --n-finaltest 100,
--pos-words out/shared_words.json (not shipped) and --exclude of earlier draws
(out/control_*.json, not shipped), giving 151/74/100 words per POS; their "config"
field records the exact arguments. --slots reads out/candidates.json (run_induction.py;
not shipped).

    python scripts/Tasks/Task_2/build_control_words.py
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

POS_KEYS = ("noun", "verb", "adj")

#: Slot feature requirements that a bare form satisfies, with the reason.
MORPH_SATISFIED_BY_BARE = {
    "Number=Sing": "bare noun form is singular",
    "Degree=Pos": "bare adjective form is positive degree",
    "VerbForm=Inf": "bare verb form is the infinitive",
    "Mood=Ind": "English base form doubles as 1sg present indicative",
}


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def _relpath(v):
    """Path relative to the repository root when inside it, for output metadata."""
    root = Path(__file__).resolve().parents[3]
    try:
        return str(Path(v).resolve().relative_to(root))
    except ValueError:
        return str(v)


def load_tokenizer(path: Path):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(path))


def token_id(tok, word: str) -> int | None:
    """The single token id for `" word"`, or None if it is not one token."""
    ids = tok.encode(" " + word, add_special_tokens=False).ids
    return ids[0] if len(ids) == 1 else None


def noun_countability(ud_dir: Path) -> Tuple[set, set]:
    """Nouns frequent enough to judge (>= 5 uses) and nouns seen with "a"/"an".

    A judgeable noun never seen with an indefinite article is treated as a mass
    noun, which does not fit items such as "A {SLOT} ...". Nouns outside the
    treebanks are kept.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from slotgen.ud_io import load_corpus

    files = sorted(ud_dir.glob("*.conllu"))
    if not files:
        return set(), set()
    seen: Counter = Counter()
    with_article: set = set()
    for sent in load_corpus(files):
        for tok in sent.tokens:
            if tok.upos == "NOUN":
                seen[tok.lemma.lower()] += 1
            if tok.deprel == "det" and tok.form.lower() in {"a", "an"}:
                head = tok.head - 1
                if 0 <= head < len(sent.tokens) and sent.tokens[head].upos == "NOUN":
                    with_article.add(sent.tokens[head].lemma.lower())
    # only nouns seen at least 5 times are judged
    judgeable = {w for w, n in seen.items() if n >= 5}
    return judgeable, with_article


def check_slot_morphology(slots_path: Path) -> dict:
    """Count slot feature requirements and list any a bare form does not meet."""
    data = json.loads(slots_path.read_text())
    reqs: Counter = Counter()
    for pos in data:
        for item in data[pos]:
            reqs[item.get("morph_requirement", "Any").split("|")[0]] += 1
    unmet = [r for r in reqs if r not in MORPH_SATISFIED_BY_BARE and r != "Any"]
    return {"counts": dict(reqs), "unmet": unmet,
            "rationale": MORPH_SATISFIED_BY_BARE}


# ---------------------------------------------------------------------------
# Frequency matching
# ---------------------------------------------------------------------------


def assign_bins(freqs: Dict[str, int], n_bins: int) -> Dict[str, int]:
    """Bin words by frequency rank; lower token id means a commoner piece."""
    ordered = sorted(freqs, key=lambda w: freqs[w])
    n = len(ordered)
    return {w: min(n_bins - 1, i * n_bins // n) for i, w in enumerate(ordered)}


def bin_quota(by_bin: Dict[str, Dict[int, List[str]]], n_bins: int,
              target: int) -> Tuple[Dict[int, int], int]:
    """Words to draw per frequency stratum.

    Each stratum is capped by its smallest category and the target is shared
    out in proportion to the caps. Returns (quota, total capacity).
    """
    capacity = {b: min(len(by_bin[p][b]) for p in POS_KEYS) for b in range(n_bins)}
    total = sum(capacity.values())
    quota = {b: min(capacity[b], round(target * capacity[b] / total))
             for b in range(n_bins)} if total else {}
    while sum(quota.values()) < target:
        room = [b for b in range(n_bins) if quota[b] < capacity[b]]
        if not room:
            break
        quota[max(room, key=lambda b: capacity[b] - quota[b])] += 1
    while sum(quota.values()) > target:
        used = [b for b in range(n_bins) if quota[b] > 0]
        quota[max(used, key=lambda b: quota[b])] -= 1
    return quota, total


def frequency_match_report(selected: Dict[str, List[str]],
                           freqs: Dict[str, int]) -> dict:
    """Token-id quantiles per POS and pairwise KS tests."""
    out: dict = {"quantiles": {}, "ks_tests": {}}
    for pos in POS_KEYS:
        ids = sorted(freqs[w] for w in selected[pos])
        out["quantiles"][pos] = {
            "n": len(ids),
            "q1": ids[len(ids) // 4],
            "median": int(st.median(ids)),
            "q3": ids[3 * len(ids) // 4],
        }
    try:
        from scipy.stats import ks_2samp

        for a in range(3):
            for b in range(a + 1, 3):
                p, q = POS_KEYS[a], POS_KEYS[b]
                r = ks_2samp([freqs[w] for w in selected[p]],
                             [freqs[w] for w in selected[q]])
                out["ks_tests"][f"{p}_vs_{q}"] = {
                    "statistic": round(float(r.statistic), 4),
                    "p_value": round(float(r.pvalue), 4),
                }
    except ImportError:
        out["ks_tests"] = "scipy unavailable"
    return out


# ---------------------------------------------------------------------------


def main() -> None:
    here = Path(__file__).resolve().parent
    root = here.parent
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pos-words", type=Path, default=root / "pos_words_v2.json")
    ap.add_argument("--tokenizer", type=Path, default=root / "tokenizers" / "gemma3-tokenizer.json")
    # the whole candidate pool, since the slots are only chosen later
    ap.add_argument("--slots", type=Path, default=here / "out" / "candidates.json")
    ap.add_argument("--ud-dir", type=Path, default=here / "data" / "ud")
    ap.add_argument("--out-calibration", type=Path,
                    default=here / "out" / "control_calibration.json")
    ap.add_argument("--out-probedev", type=Path,
                    default=here / "out" / "control_probedev.json")
    ap.add_argument("--out-finaltest", type=Path,
                    default=here / "out" / "control_finaltest.json")
    ap.add_argument("--out-meta", type=Path,
                    default=here / "out" / "control_words_meta.json",
                    help="filter counts, frequency-match evidence, provenance")
    ap.add_argument("--n-calibration", type=int, default=150,
                    help="words per POS used to fit the reference distributions")
    ap.add_argument("--n-probedev", type=int, default=75,
                    help="words per POS used only to screen candidate slots")
    ap.add_argument("--n-finaltest", type=int, default=75,
                    help="words per POS used only for the final frozen report")
    ap.add_argument("--n-bins", type=int, default=10,
                    help="frequency strata used to match the three categories")
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--exclude", type=Path, nargs="*", default=[],
                    help="control-word files whose words are left out of the pool")
    ap.add_argument("--only-finaltest", action="store_true",
                    help="redraw only final-test; calibration and probe-dev, "
                         "which the slot set was selected on, are left unchanged")
    ap.add_argument("--skip-countability", action="store_true")
    args = ap.parse_args()

    # with --only-finaltest only the final-test split is drawn
    target = (args.n_finaltest if args.only_finaltest
              else args.n_calibration + args.n_probedev + args.n_finaltest)
    rng = random.Random(args.seed)
    tok = load_tokenizer(args.tokenizer)
    words = json.loads(args.pos_words.read_text())
    filters: Dict[str, dict] = {p: {"input": len(words[p])} for p in POS_KEYS}

    # -- 0. slot morphology --------------------------------------------------
    morph = check_slot_morphology(args.slots)
    print("Candidate-pool morphology (corpus features):", morph["counts"])
    if morph["unmet"]:
        print(f"  {morph['unmet']} are not bare-form requirements; select_slots.py "
              f"decides from the realized prefix and skips the incompatible ones")
    else:
        print("  all bare-form requirements -> one control set covers every candidate")

    # -- 1. single token after a space ---------------------------------------
    pool: Dict[str, Dict[str, int]] = {}
    for pos in POS_KEYS:
        pool[pos] = {w: t for w in words[pos] if (t := token_id(tok, w)) is not None}
        filters[pos]["single_token"] = len(pool[pos])

    # -- 2. drop likely mass nouns -------------------------------------------
    if not args.skip_countability:
        judgeable, with_article = noun_countability(args.ud_dir)
        if judgeable:
            before = len(pool["noun"])
            dropped = sorted(w for w in pool["noun"]
                             if w in judgeable and w not in with_article)
            pool["noun"] = {w: i for w, i in pool["noun"].items()
                            if w not in set(dropped)}
            filters["noun"]["mass_nouns_dropped"] = len(dropped)
            filters["noun"]["after_countability"] = len(pool["noun"])
            print(f"  countability: {before} -> {len(pool['noun'])} "
                  f"(dropped {len(dropped)} likely mass nouns, "
                  f"e.g. {', '.join(dropped[:6])})")

    # -- 2b. drop words already used in other splits -------------------------
    if args.exclude:
        spent = set()
        for path in args.exclude:
            d = json.loads(Path(path).read_text())["words"]
            for pos in POS_KEYS:
                spent |= set(d[pos])
        for pos in POS_KEYS:
            before = len(pool[pos])
            pool[pos] = {w: i for w, i in pool[pos].items() if w not in spent}
            filters[pos]["after_exclude"] = len(pool[pos])
            print(f"  exclude: {pos} {before} -> {len(pool[pos])}")
        print(f"  ({len(spent)} words already spent across "
              f"{len(args.exclude)} file(s))")

    # -- 3. frequency-matched stratified sample ------------------------------
    freqs = {w: i for pos in POS_KEYS for w, i in pool[pos].items()}
    bins = assign_bins(freqs, args.n_bins)
    by_bin = {p: {b: [] for b in range(args.n_bins)} for p in POS_KEYS}
    for pos in POS_KEYS:
        for w in pool[pos]:
            by_bin[pos][bins[w]].append(w)
    quota, capacity_total = bin_quota(by_bin, args.n_bins, target)
    if capacity_total < target:
        print(f"\n!! matched capacity is {capacity_total}/POS, below the "
              f"requested {target}")

    print(f"\nfrequency strata ({args.n_bins} bins over token-id rank)")
    print(f"  {'bin':<5}{'noun':>7}{'verb':>7}{'adj':>7}{'quota':>8}")
    for b in range(args.n_bins):
        print(f"  {b:<5}" + "".join(f"{len(by_bin[p][b]):>7}" for p in POS_KEYS)
              + f"{quota[b]:>8}")

    selected = {p: sorted(w for b in range(args.n_bins)
                          for w in rng.sample(by_bin[p][b], quota[b]))
                for p in POS_KEYS}

    if args.only_finaltest:
        calibration = {p: [] for p in POS_KEYS}
        probedev = {p: [] for p in POS_KEYS}
        finaltest = selected
        print(f"\n--only-finaltest: final-test only "
              + ", ".join(f"{p}={len(finaltest[p])}" for p in POS_KEYS))
        match = frequency_match_report(finaltest, freqs)
        if isinstance(match["ks_tests"], dict):
            print("Frequency match (KS, p>0.05 means no significant difference):")
            for k, v in match["ks_tests"].items():
                print(f"  {k:<16} D={v['statistic']:.3f}  p={v['p_value']:.3f}"
                      f"  {'ok' if v['p_value'] > 0.05 else '!! distributions differ'}")
        args.out_finaltest.parent.mkdir(parents=True, exist_ok=True)
        args.out_finaltest.write_text(json.dumps({
            "role": "final-test (fresh draw): the frozen probe's accuracy; "
                    "disjoint from calibration and probe-dev",
            "n_per_pos": {p: len(finaltest[p]) for p in POS_KEYS},
            "config": {k: (_relpath(v) if isinstance(v, Path) else
                           [_relpath(x) for x in v] if isinstance(v, list) else v)
                       for k, v in vars(args).items()},
            "frequency_source": "gemma token id (SentencePiece unigram rank)",
            "seed": args.seed,
            "excluded_from": [str(x) for x in args.exclude],
            "words": finaltest,
            "token_ids": {w: freqs[w] for p in POS_KEYS for w in finaltest[p]},
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n-> {args.out_finaltest}")
        return

    # -- 4. three-way split within each stratum ------------------------------
    splits = {"calibration": {p: [] for p in POS_KEYS},
              "probedev": {p: [] for p in POS_KEYS},
              "finaltest": {p: [] for p in POS_KEYS}}
    for pos in POS_KEYS:
        for b in range(args.n_bins):
            in_bin = [w for w in selected[pos] if bins[w] == b]
            rng.shuffle(in_bin)
            n = len(in_bin)
            a = round(n * args.n_calibration / target)
            c = round(n * args.n_probedev / target)
            splits["calibration"][pos] += in_bin[:a]
            splits["probedev"][pos] += in_bin[a:a + c]
            splits["finaltest"][pos] += in_bin[a + c:]
        for s in splits:
            splits[s][pos].sort()
        names = list(splits)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                assert not set(splits[names[i]][pos]) & set(splits[names[j]][pos]), pos
    calibration, probedev, finaltest = (splits["calibration"], splits["probedev"],
                                        splits["finaltest"])

    # -- 5. report -----------------------------------------------------------
    match = frequency_match_report(selected, freqs)
    print(f"\n{'':6}{'calib':>7}{'dev':>6}{'test':>6}{'total':>7}"
          f"{'median id':>12}{'IQR':>20}")
    for pos in POS_KEYS:
        q = match["quantiles"][pos]
        iqr = "%d-%d" % (q["q1"], q["q3"])
        print(f"{pos:<6}{len(calibration[pos]):>7}{len(probedev[pos]):>6}"
              f"{len(finaltest[pos]):>6}{q['n']:>7}{q['median']:>12}{iqr:>20}")
    if isinstance(match["ks_tests"], dict):
        print("\nFrequency match (Kolmogorov-Smirnov, p>0.05 means no significant difference):")
        for k, v in match["ks_tests"].items():
            flag = "ok" if v["p_value"] > 0.05 else "!! distributions differ"
            print(f"  {k:<16} D={v['statistic']:.3f}  p={v['p_value']:.3f}  {flag}")

    # one file per split, each carrying the same provenance
    def jsonable(v):
        # --exclude is a list of Paths, which json cannot serialize
        if isinstance(v, Path):
            return _relpath(v)
        if isinstance(v, (list, tuple)):
            return [jsonable(x) for x in v]
        return v

    config = {k: jsonable(v) for k, v in vars(args).items()}
    provenance = {
        "config": config,
        "frequency_source": "gemma token id (SentencePiece unigram rank)",
        "seed": args.seed,
    }

    def dump(path: Path, role: str, words: Dict[str, List[str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "role": role,
            "n_per_pos": {p: len(words[p]) for p in POS_KEYS},
            **provenance,
            "words": words,
            "token_ids": {w: freqs[w] for p in POS_KEYS for w in words[p]},
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    dump(args.out_calibration,
         "calibration: fit reference distributions only", calibration)
    dump(args.out_probedev,
         "probe-dev: screen candidate slots for diagnosticity; never fitted on, "
         "never used for the final report", probedev)
    dump(args.out_finaltest,
         "final-test: the frozen probe's accuracy; touched at no earlier stage",
         finaltest)

    args.out_meta.parent.mkdir(parents=True, exist_ok=True)
    args.out_meta.write_text(json.dumps({
        **provenance,
        "slot_morphology": morph,
        "filters": filters,
        "frequency_match": match,
        "bin_quota": quota,
        "n_per_split": {s: {p: len(splits[s][p]) for p in POS_KEYS} for s in splits},
        "disjoint": True,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n-> {args.out_calibration}   (fits the reference distributions)")
    print(f"-> {args.out_probedev}   (slot selection)")
    print(f"-> {args.out_finaltest}   (final report after freezing)")
    print(f"-> {args.out_meta}   (filter counts and frequency match)")


if __name__ == "__main__":
    main()
