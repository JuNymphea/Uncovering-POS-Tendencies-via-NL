#!/usr/bin/env python
"""Build the Chinese Task 2 control words: nouns, verbs and adjectives of known category.

A word is kept if its dominant UPOS in UD Chinese-GSDSimp covers >= 90% of at least 3
occurrences, jieba's dictionary gives the same tag (n, v or a), it has 2-4 Han characters
and it is a single token in all three tokenizers. UD Chinese tags most stative predicates
VERB, so the adjectives also include jieba-a words UD barely attests (<= 2 occurrences,
none as VERB or NOUN, jieba freq >= 200), minus words starting with a negator (bu, mei,
wu), yi or hen. Words used in the probe items or the training templates are excluded (the
probe-item words are read from out_zh/candidates.json, from zh/induce_zh.py; not shipped).
Categories are matched by log jieba frequency bin and split 50/25/25 into calibration,
probe-dev and final-test. Writes control3_*.json; writing to the default --out-dir
overwrites the shipped out_zh/control3_*.json.

    python scripts/Tasks/Task_2/zh/build_control_words_zh.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

HAN = re.compile(r"^[一-鿿]{2,4}$")
TEMPLATE_WORDS = {"回答", "问题", "词语", "答案", "体现", "以下", "应该", "尽可能", "这个"}
JIEBA_OK = {"noun": {"n"}, "verb": {"v"}, "adj": {"a"}}


def main() -> None:
    here = Path(__file__).resolve().parents[1]
    tokdir = here.parent / "tokenizers"          # filled by Task_2/get_tokenizers.py
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ud-dir", type=Path, default=here / "data" / "ud_zh")
    ap.add_argument("--treebank", default="zh_gsdsimp")
    ap.add_argument("--candidates", type=Path, default=here / "out_zh" / "candidates.json")
    ap.add_argument("--out-dir", type=Path, default=here / "out_zh")
    ap.add_argument("--tokenizers", nargs="+", default=[
        f"gemma={tokdir / 'gemma3-tokenizer.json'}",
        f"qwen={tokdir / 'Qwen3-4B-tokenizer.json'}",
        f"aya={tokdir / 'tiny-aya-global-tokenizer.json'}"])
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--min-dominance", type=float, default=0.90)
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260927)
    args = ap.parse_args()

    # ---- UD counts ------------------------------------------------------
    upos = defaultdict(Counter)
    bad_lemma, negated = set(), set()
    for split in ("train", "dev", "test"):
        for line in open(args.ud_dir / f"{args.treebank}-ud-{split}.conllu", encoding="utf-8"):
            if not line.strip() or line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) != 10 or "-" in c[0] or "." in c[0]:
                continue
            form, lemma, u, xpos, feats, deprel = c[1], c[2], c[3], c[4], c[5], c[7]
            if xpos == "NNB" or deprel == "clf":
                u = "CLF"
            upos[form][u] += 1
            if lemma not in ("_", form):
                bad_lemma.add(form)
            if "Polarity=Neg" in feats:
                negated.add(form)

    # ---- jieba ----------------------------------------------------------
    import jieba
    jfreq, jtag = {}, {}
    for line in open(Path(jieba.__file__).parent / "dict.txt", encoding="utf-8"):
        p = line.split()
        if len(p) >= 3:
            jfreq[p[0]], jtag[p[0]] = int(p[1]), p[2]

    # ---- words already in the probe items -------------------------------
    used = set(TEMPLATE_WORDS)
    for items in json.loads(args.candidates.read_text()).values():
        for it in items:
            used.update(t for t in it["prefix_tokens"] + it["diagnostic"] if t != "{NEOLOGISM}")

    # ---- tokenizers -----------------------------------------------------
    from tokenizers import Tokenizer
    toks = {}
    for spec in args.tokenizers:
        name, _, path = spec.partition("=")
        toks[name] = Tokenizer.from_file(path)

    def single_ids(w):
        ids = {}
        for name, tk in toks.items():
            e = tk.encode(w, add_special_tokens=False).ids
            if len(e) != 1:
                return None
            ids[name] = e[0]
        return ids

    cats = ("noun", "verb", "adj")
    pools, drops = {c: [] for c in cats}, Counter()
    # adjectives that UD barely attests, from jieba alone
    for w, t in jtag.items():
        if t != "a" or not HAN.match(w) or w in upos and sum(upos[w].values()) >= 3:
            continue
        c = upos.get(w, Counter())
        if c["VERB"] or c["NOUN"] or jfreq[w] < 200 or w[0] in "不一没无很" or w in used:
            continue
        ids = single_ids(w)
        if ids is None:
            continue
        pools["adj"].append({"word": w, "ud_count": sum(c.values()), "ud_dominance": None,
                             "jieba_tag": t, "jieba_freq": jfreq[w],
                             "log10_freq": round(math.log10(jfreq[w] + 1), 3), "token_ids": ids,
                             "source": "jieba only"})
    for w, c in upos.items():
        n = sum(c.values())
        dom, k = c.most_common(1)[0]
        cat = {"NOUN": "noun", "VERB": "verb", "ADJ": "adj"}.get(dom)
        if cat is None or not HAN.match(w):
            continue
        if n < args.min_count:
            drops["too rare in UD"] += 1; continue
        if k / n < args.min_dominance:
            drops["UD category below dominance"] += 1; continue
        if jtag.get(w) not in JIEBA_OK[cat]:
            drops["jieba disagrees or has no entry"] += 1; continue
        if w in bad_lemma or w.endswith("们") or w in negated:
            drops["inflected / negated form"] += 1; continue
        if w in used:
            drops["in the probe items or templates"] += 1; continue
        ids = single_ids(w)
        if ids is None:
            drops["not one token in all three tokenizers"] += 1; continue
        pools[cat].append({"word": w, "ud_count": n, "ud_dominance": round(k / n, 3),
                           "jieba_tag": jtag[w], "jieba_freq": jfreq[w],
                           "log10_freq": round(math.log10(jfreq[w] + 1), 3), "token_ids": ids})
    print("pool:", {k: len(v) for k, v in pools.items()}, "  dropped:", dict(drops))

    # ---- frequency matching and split -----------------------------------
    allf = sorted(r["log10_freq"] for v in pools.values() for r in v)
    edges = [allf[int(len(allf) * q / args.n_bins)] for q in range(1, args.n_bins)]

    def bin_of(x):
        return sum(x >= e for e in edges)

    rng = random.Random(args.seed)
    by_bin = {cat: defaultdict(list) for cat in pools}
    prefix = "control3"
    for cat, v in pools.items():
        for r in v:
            by_bin[cat][bin_of(r["log10_freq"])].append(r)
    splits = {role: {"noun": [], "verb": [], "adj": []} for role in ("calibration", "probedev", "finaltest")}
    meta = {}
    for b in range(args.n_bins):
        q = min(len(by_bin[c][b]) for c in cats)
        for cat in cats:
            chosen = sorted(by_bin[cat][b], key=lambda r: r["word"])
            rng.shuffle(chosen)
            chosen = chosen[:q]
            n_cal = round(q * 0.50)
            n_dev = round(q * 0.25)
            for i, r in enumerate(chosen):
                role = "calibration" if i < n_cal else "probedev" if i < n_cal + n_dev else "finaltest"
                splits[role][cat].append(r["word"])
                meta[r["word"]] = {**r, "category": cat, "role": role, "freq_bin": b}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for role, words in splits.items():
        (args.out_dir / f"{prefix}_{role}.json").write_text(json.dumps({
            "role": role, "lang": "zh", "treebank": args.treebank,
            "n_per_pos": {k: len(v) for k, v in words.items()},
            "words": words}, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out_dir / f"{prefix}_words_meta.json").write_text(json.dumps({
        "criteria": {"min_ud_count": args.min_count, "min_dominance": args.min_dominance,
                     "jieba_tags": {k: sorted(v) for k, v in JIEBA_OK.items()},
                     "length": "2-4 Han characters", "single_token_in": sorted(toks),
                     "frequency": "log10 jieba dict frequency, matched per bin over the union",
                     "excluded": "probe-item words and Chinese training-template words"},
        "pool": {k: len(v) for k, v in pools.items()}, "dropped": dict(drops),
        "bin_edges_log10": edges, "seed": args.seed, "words": meta},
        indent=2, ensure_ascii=False), encoding="utf-8")
    for role, words in splits.items():
        print(f"{role:12} " + "  ".join(f"{c} {len(words[c]):4d}" for c in cats)
              + f"   e.g. " + " / ".join(str(words[c][:4]) for c in cats))
    import statistics as st
    for cat in cats:
        xs = [meta[w]["log10_freq"] for role in splits for w in splits[role][cat]]
        print(f"  {cat}: median log10 jieba freq {st.median(xs):.2f}")


if __name__ == "__main__":
    main()
