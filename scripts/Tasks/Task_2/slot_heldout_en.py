#!/usr/bin/env python
"""Purity of the English slots in use on EWT train / dev / test and on UD GUM.

For each signature in out/slots3_<model>.json, counts the words filling it in
each corpus and the share (over all UPOS) that belong to the slot's POS. Purity
depends only on the signature, not on how the item is worded. Needs the UD
treebanks in data/ud/ (en_ewt-ud-*.conllu, en_gum-ud-*.conllu; see README).
Writes out/slot_heldout_en.json.

    python scripts/Tasks/Task_2/slot_heldout_en.py
"""
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from slotgen.ud_io import read_conllu, frequent_adverbs  # noqa: E402
from slotgen.frames import SignatureEncoder, signature_for, signature_shape, TARGET_POS  # noqa: E402

UD = HERE / "data" / "ud"


def main():
    train = list(read_conllu(UD / "en_ewt-ud-train.conllu"))
    enc = SignatureEncoder(lexicalized_adverbs=frequent_adverbs(train, 30))
    slots = {}
    for m in ("gemma", "qwen", "aya"):
        for p, items in json.load(open(HERE / f"out/slots3_{m}.json")).items():
            for s in items:
                slots.setdefault((p, s["signature"]), set()).add(m)
    shapes = {signature_shape(sig) for _, sig in slots}
    wanted = {sig for _, sig in slots}

    def counts(paths):
        C = collections.defaultdict(collections.Counter)
        for path in paths:
            for sent in read_conllu(path):
                for i, tok in enumerate(sent.tokens):
                    for left, right in shapes:
                        sig = signature_for(sent, i, left, right, enc)
                        if sig in wanted:
                            C[sig][tok.upos] += 1
        return C

    corp = {"train": counts([UD / "en_ewt-ud-train.conllu"]),
            "dev": counts([UD / "en_ewt-ud-dev.conllu"]),
            "test": counts([UD / "en_ewt-ud-test.conllu"]),
            "gum": counts([UD / f"en_gum-ud-{s}.conllu" for s in ("train", "dev", "test")])}
    rows = []
    for (p, sig), ms in sorted(slots.items()):
        r = {"pos": p, "signature": sig, "models": sorted(ms)}
        for k, C in corp.items():
            c = C.get(sig, collections.Counter())
            n_all, n_t = sum(c.values()), sum(c[q] for q in TARGET_POS)
            r[k] = {"n": n_t, "purity_all": round(c[p] / n_all, 3) if n_all else None}
        rows.append(r)
    (HERE / "out/slot_heldout_en.json").write_text(json.dumps(rows, indent=1))

    for k in ("dev", "test", "gum"):
        judged = [r for r in rows if r[k]["n"] >= 8]
        ok = [r for r in judged if r[k]["purity_all"] >= 0.85]
        attested = [r for r in rows if r[k]["n"] > 0]
        pooled = {}
        for p in ("NOUN", "VERB", "ADJ"):
            num = sum(corp[k][r["signature"]][p] for r in rows if r["pos"] == p)
            den = sum(sum(corp[k][r["signature"]].values()) for r in rows if r["pos"] == p)
            pooled[p] = f"{num / den:.3f} (n={den})" if den else "-"
        print(f"{k:5}: attested {len(attested)}/{len(rows)}, >=8 occurrences {len(judged)}, "
              f"of which selectivity>=.85 {len(ok)}; pooled {pooled}")
    for r in rows:
        if r["gum"]["n"] >= 8 and r["gum"]["purity_all"] < 0.85:
            print("   GUM below .85:", r["pos"], r["signature"], r["gum"], r["models"])


if __name__ == "__main__":
    main()
