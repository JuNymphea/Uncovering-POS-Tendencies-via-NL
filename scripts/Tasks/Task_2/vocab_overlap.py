#!/usr/bin/env python
"""Count the control-word candidates that are single tokens in every model.

Reads the model-independent shortlist (pos_words_v2.json), keeps the words that
encode to one token after a space in each tokenizer, and prints per-model and
shared counts. --out writes the shared list.

    python scripts/Tasks/Task_2/vocab_overlap.py --pos-words scripts/Tasks/pos_words_v2.json \
        --tokenizers gemma=scripts/Tasks/tokenizers/gemma3-tokenizer.json \
                     qwen=scripts/Tasks/tokenizers/Qwen3-4B-tokenizer.json \
                     aya=scripts/Tasks/tokenizers/tiny-aya-global-tokenizer.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

POS_KEYS = ("noun", "verb", "adj")


def single_token_set(path: Path, words: dict) -> dict:
    """Which of the POS-filtered words are one token after a space."""
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(path))
    out = {}
    for pos in POS_KEYS:
        out[pos] = {w for w in words[pos]
                    if len(tk.encode(" " + w, add_special_tokens=False).ids) == 1}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pos-words", type=Path, required=True,
                    help="pos_words_v2.json, the model-independent shortlist")
    ap.add_argument("--tokenizers", nargs="+", required=True,
                    help="name=/path/to/tokenizer.json, one per model")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the shared word list here")
    args = ap.parse_args()

    # the shortlist was built from one model's vocabulary; words missing from it
    # are not considered
    words = json.loads(args.pos_words.read_text())
    print("Start (pos_words_v2.json, after the model-independent filters):")
    print("  " + "  ".join(f"{p}={len(words[p])}" for p in POS_KEYS))

    sets = {}
    for spec in args.tokenizers:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"expected name=path, got {spec!r}")
        sets[name] = single_token_set(Path(path), words)

    print(f"\n{'model':<12}" + "".join(f"{p:>9}" for p in POS_KEYS) + f"{'total':>9}")
    for name, s in sets.items():
        print(f"  {name:<10}" + "".join(f"{len(s[p]):>9}" for p in POS_KEYS)
              + f"{sum(len(s[p]) for p in POS_KEYS):>9}")

    shared = {p: set.intersection(*(s[p] for s in sets.values())) for p in POS_KEYS}
    print(f"  {'shared':<10}" + "".join(f"{len(shared[p]):>9}" for p in POS_KEYS)
          + f"{sum(len(shared[p]) for p in POS_KEYS):>9}")

    print("\nLoss per model (relative to its own full set):")
    for name, s in sets.items():
        print(f"  {name:<10}" + "  ".join(
            f"{p} {1 - len(shared[p]) / len(s[p]):.0%}" for p in POS_KEYS))

    need = 100 * 3 + 148 + 74      # per-POS budget for this check
    print(f"\nThis check asks for {need} words per POS "
          "(the shipped shared_*.json use 151 + 74 + 100)")
    for p in POS_KEYS:
        ok = "enough" if len(shared[p]) >= need else f"short by {need - len(shared[p])}"
        print(f"  {p:<6}shared {len(shared[p]):>5}  {ok}")
    print("  (fewer are usable: each frequency stratum is capped by its smallest category)")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {p: sorted(shared[p]) for p in POS_KEYS}, indent=2, ensure_ascii=False),
            encoding="utf-8")
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
