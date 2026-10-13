#!/usr/bin/env python
"""Induce candidate Task 2 slots from UD English EWT.

Counts context signatures on EWT train, keeps frames that pass the frequency,
lemma-type and purity thresholds, realizes each as a minimal probe item and
re-estimates purity on EWT test and UD GUM (stages are in slotgen/). Writes
out/candidates.json for task2.py, plus candidates_full.json, slots.md,
slots_table.tex, frames_all.jsonl and review_template.json.

    python scripts/Tasks/Task_2/run_induction.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from slotgen.minimal import (  # noqa: E402
    TRAINING_TEMPLATES,
    collides_with_training,
    make_item,
)
from slotgen.tagging import TaggerCheck, load_tagger  # noqa: E402
from slotgen.frames import TARGET_POS, SignatureEncoder  # noqa: E402
from slotgen.overrides import Overrides, write_review_template  # noqa: E402
from slotgen.induce import (  # noqa: E402
    DevFilter,
    StructuralConstraints,
    build_pool,
    count_signatures,
)
from slotgen.ud_io import (  # noqa: E402
    bigram_counts,
    form_distributions,
    frequent_adverbs,
    load_corpus,
)
from slotgen.validate import recount, summarize, validate  # noqa: E402

POS_LABEL = {"NOUN": "Noun", "VERB": "Verb", "ADJ": "Adjective"}


def _relpath(v):
    """Path relative to the repository root when inside it, for output metadata."""
    root = Path(__file__).resolve().parents[3]
    try:
        return str(Path(v).resolve().relative_to(root))
    except ValueError:
        return str(v)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ud-dir", type=Path, default=here / "data" / "ud")
    p.add_argument("--out-dir", type=Path, default=here / "out")
    p.add_argument("--pool-size", type=int, default=40,
                   help="candidates per POS to hand to the model-side screening")
    p.add_argument("--min-freq", type=int, default=5,
                   help="minimum frame frequency, the same for all three "
                        "categories")
    p.add_argument("--min-types", type=int, default=4)
    p.add_argument("--min-purity", type=float, default=0.90,
                   help="minimum P(p|c) of the dominant POS")
    p.add_argument("--purity-denominator", choices=("all", "target3"),
                   default="all",
                   help="all: P(p|c) over every UPOS filling the context "
                        "(stricter); target3: over NOUN/VERB/ADJ only")
    p.add_argument("--max-left", type=int, default=2)
    p.add_argument("--min-right", type=int, default=1,
                   help="minimum right-context length; the right context is "
                        "the scored continuation")
    p.add_argument("--max-right", type=int, default=3)
    p.add_argument("--n-continuation", type=int, default=4,
                   help="recorded in the output config; probe items score "
                        "the frame's own right context")
    p.add_argument("--min-bigram", type=int, default=3,
                   help="a bigram of added material must be attested at least "
                        "this often in the pooled treebanks")
    p.add_argument("--adv-topk", type=int, default=30,
                   help="how many ADV forms to keep lexicalized")
    p.add_argument("--prune-below", type=int, default=5,
                   help="drop signatures rarer than this before collecting stats")
    p.add_argument("--min-left", type=int, default=1,
                   help="minimum left-context length")
    p.add_argument("--min-content-right", type=int, default=1,
                   help="minimum non-punctuation elements in the right context")
    p.add_argument("--min-lexical-cues", type=int, default=1,
                   help="minimum kept word forms (grammatical cues) in a frame")
    p.add_argument("--dev-min-count", type=int, default=8,
                   help="dev occurrences needed before a frame can be rejected")
    p.add_argument("--dev-min-purity", type=float, default=0.85,
                   help="minimum purity on the dev corpus")
    p.add_argument("--probe-words", type=Path,
                   default=here.parent / "pos_words_sampled.json",
                   help="known-POS words used to verify probe items")
    p.add_argument("--min-tagger-accuracy", type=float, default=0.85,
                   help="tagger agreement a probe item must reach")
    p.add_argument("--overrides", type=Path, default=here / "realizations.json",
                   help="human review file: accept / rewrite / reject per frame")
    p.add_argument("--write-review", action="store_true",
                   help="unused; review_template.json is always written")
    p.add_argument("--allow-unrealizable", action="store_true",
                   help="keep frames that cannot be realized as a minimal "
                        "probe item (default: drop and take the next candidate)")
    return p.parse_args()


def emit_markdown(selected, realized, thresholds, summary, validation, args) -> str:
    lines: list[str] = []
    lines.append("# Corpus-derived syntactic slots for Task 2\n")
    lines.append(
        f"Candidate pool from UD English EWT train, up to {args.pool_size} per POS. "
        f"The same thresholds apply to all three categories (P(p|c) >= {args.min_purity:.2f}; "
        "frequency and lemma types as in the table below). The final slots are chosen "
        "per model by select_slots.py.\n"
    )

    lines.append("\n## Thresholds actually used\n")
    lines.append("| POS | thresholds | candidates selected |")
    lines.append("|---|---|---|")
    for pos in TARGET_POS:
        lines.append(
            f"| {POS_LABEL[pos]} | {thresholds[pos].describe()} | {len(selected[pos])} |"
        )

    lines.append("\n## Selectivity, held out and across corpora\n")
    lines.append(
        "P(p|c) pooled over the selected slots (counts summed). `train` is the split "
        "the slots were chosen on; the rest were not used for selection.\n"
    )
    cols = list(summary["per_pos"][TARGET_POS[0]].keys())
    lines.append("| POS | " + " | ".join(cols) + " |")
    lines.append("|---" * (len(cols) + 1) + "|")
    for pos in TARGET_POS:
        row = summary["per_pos"][pos]
        lines.append(
            f"| {POS_LABEL[pos]} | "
            + " | ".join("-" if row[c] is None else str(row[c]) for c in cols)
            + " |"
        )

    if summary["flagged"]:
        lines.append("\n**Frames whose selectivity did not hold up:**\n")
        lines.append("| POS | signature | corpus | train | held-out | n |")
        lines.append("|---|---|---|---|---|---|")
        for f in summary["flagged"]:
            lines.append(
                f"| {POS_LABEL[f['pos']]} | `{f['signature']}` | {f['corpus']} | "
                f"{f['train_purity']} | {f['heldout_purity']} | {f['heldout_n']} |"
            )
    else:
        lines.append(
            "\nNo selected frame fell below the held-out selectivity floor.\n"
        )

    lines.append("\n## Slots\n")
    for pos in TARGET_POS:
        lines.append(f"\n### {POS_LABEL[pos]}\n")
        lines.append(
            "| # | probe item | scored `D_f` | signature | P(p\\|c) | N | types |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for i, r in enumerate(realized[pos], 1):
            it = r.get("probe_item")
            if not it:
                lines.append(f"| {i} | _(none)_ | | `{r['signature']}` | | | |")
                continue
            lines.append(
                f"| {i} | {it['text']} | `{' '.join(it['diagnostic'])}` | "
                f"`{r['signature']}` | {r['purity']:.2f} | {r['n_target']} | "
                f"{r['n_types']} |"
            )
    return "\n".join(lines) + "\n"


def emit_latex(realized, use: str = "probe_item") -> str:
    """LaTeX table of the realized slots."""
    rows = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"\textbf{POS} & \textbf{Slot} & $P(p\mid c)$ & $N$ \\",
        r"\midrule",
    ]
    for pos in TARGET_POS:
        first = True
        for r in realized[pos]:
            real = r[use] or r["realization_template"] or r["realization_attested"]
            if real is None:
                continue
            text = real["text"].replace("{NEOLOGISM}", r"\texttt{NEOLOGISM}")
            text = text.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
            label = POS_LABEL[pos] if first else ""
            rows.append(f"{label} & {text} & {r['purity']:.2f} & {r['n_target']} \\\\")
            first = False
        rows.append(r"\midrule")
    rows[-1] = r"\bottomrule"
    rows += [
        r"\end{tabular}",
        r"\caption{Syntactic slots for Task 2, induced from UD English EWT by "
        r"frequency, lexical productivity and category selectivity. "
        r"$P(p\mid c)$ and $N$ are estimated on the training split.}",
        r"\label{tab:slots}",
        r"\end{table}",
    ]
    return "\n".join(rows) + "\n"


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # frames come from EWT train, EWT dev can veto them, and EWT test and GUM
    # are used only for reporting
    ewt_train = args.ud_dir / "en_ewt-ud-train.conllu"
    dev_files = [args.ud_dir / "en_ewt-ud-dev.conllu"]
    test_files = {
        "ewt_test": [args.ud_dir / "en_ewt-ud-test.conllu"],
        "gum_external": sorted(args.ud_dir.glob("en_gum-ud-*.conllu")),
    }
    dev_files = [p for p in dev_files if p.exists()]
    test_files = {k: [p for p in v if p.exists()] for k, v in test_files.items()}
    test_files = {k: v for k, v in test_files.items() if v}

    print(f"[1/6] loading {ewt_train.name}")
    train = load_corpus([ewt_train])
    n_tokens = sum(len(s) for s in train)
    print(f"      {len(train):,} sentences, {n_tokens:,} tokens")

    adverbs = frequent_adverbs(train, args.adv_topk)
    encoder = SignatureEncoder(lexicalized_adverbs=adverbs)

    print(f"[2/6] extracting context signatures "
          f"(left<={args.max_left}, right {args.min_right}-{args.max_right})")
    frames = count_signatures(
        train, encoder,
        max_left=args.max_left, min_right=args.min_right,
        max_right=args.max_right, prune_below=args.prune_below,
        purity_denominator=args.purity_denominator,
    )
    print(f"      {len(frames):,} signatures occurring >={args.prune_below} times")

    print("[3/6] building the corpus candidate pool")
    dev = load_corpus(dev_files)
    print(f"      dev pool: {len(dev):,} sentences "
          f"({', '.join(p.name for p in dev_files)})")
    dev_filter = DevFilter(
        counts=recount(dev, encoder, frames.keys(),
                       args.max_left, args.min_right, args.max_right),
        min_count=args.dev_min_count,
        min_purity=args.dev_min_purity,
    )
    constraints = StructuralConstraints(
        encoder=encoder,
        min_left=args.min_left,
        min_content_right=args.min_content_right,
        min_lexical_cues=args.min_lexical_cues,
    )

    # form/deprel and bigram counts used when filling and checking items
    all_files = [ewt_train, *dev_files,
                 *[q for v in test_files.values() for q in v]]
    _pooled = load_corpus(all_files)
    dists = form_distributions(_pooled)
    bigrams = bigram_counts(_pooled)
    del _pooled

    probes = json.loads(args.probe_words.read_text())
    probes = {k.upper(): v for k, v in probes.items()}
    check = TaggerCheck(load_tagger(), probes)
    overrides = Overrides.load(args.overrides)
    if overrides.entries:
        print(f"      review file: {len(overrides.entries)} frames "
              f"({len(overrides.rejected())} rejected by hand)")
    _items: dict = {}
    collisions: dict = {}

    def item_for(fs):
        """Cached probe item for a frame, or None if it cannot be used."""
        sig = fs.signature
        if overrides.status(sig) == "reject":
            return None
        # skip frames that reproduce a training-template construction
        clash = collides_with_training(sig, encoder)
        if clash is not None:
            collisions[sig] = clash
            return None
        if sig not in _items:
            _items[sig] = make_item(
                fs, encoder, check, min_accuracy=args.min_tagger_accuracy,
                dists=dists, bigrams=bigrams, min_bigram=args.min_bigram,
            )
        return _items[sig]

    def probe_item(fs):
        """Probe item as a dict; a hand-written rewrite takes precedence."""
        rewritten = overrides.rewritten(fs.signature, 1)
        if rewritten is not None:
            return rewritten
        item = item_for(fs)
        return item.as_dict() if item is not None else None

    def realizable(fs) -> bool:
        if args.allow_unrealizable:
            return True
        return probe_item(fs) is not None

    def item_fingerprint(fs):
        """Realized text, used to drop signatures that give the same item."""
        it = item_for(fs)
        return None if it is None else it.text

    selected, thresholds_used = build_pool(
        frames, pool_size=args.pool_size, min_purity=args.min_purity,
        min_freq=args.min_freq, min_types=args.min_types,
        constraints=constraints, dev_filter=dev_filter,
        validator=realizable, fingerprint=item_fingerprint,
    )
    thresholds = {p: thresholds_used for p in TARGET_POS}
    for pos in TARGET_POS:
        print(f"      {POS_LABEL[pos]:<10} {len(selected[pos])} candidates "
              f"({thresholds_used.describe()})")

    print("[4/6] building minimal probe items")
    realized = {
        pos: [{**fs.to_dict(), "probe_item": probe_item(fs)} for fs in frames_]
        for pos, frames_ in selected.items()
    }

    print("[5/6] re-estimating selectivity on the untouched test corpora")
    corpora = {name: load_corpus(paths) for name, paths in test_files.items()}
    validation = validate(
        selected, encoder, corpora,
        max_left=args.max_left, min_right=args.min_right, max_right=args.max_right,
    )
    summary = summarize(selected, validation)
    for pos in TARGET_POS:
        print(f"      {POS_LABEL[pos]:<10} {summary['per_pos'][pos]}")

    print("[6/6] writing outputs")
    sig_to_frame = {fs.signature: fs for fr in selected.values() for fs in fr}
    for pos in TARGET_POS:
        for r in realized[pos]:
            dev_purity = dev_filter.purity_of(sig_to_frame[r["signature"]])
            r["validation"] = {
                "dev": None if dev_purity is None else round(dev_purity, 4),
                **{
                    name: est[r["signature"]].to_dict()
                    for name, est in validation.items()
                },
            }

    (args.out_dir / "candidates_full.json").write_text(
        json.dumps(
            {
                "config": {
                    k: (_relpath(v) if isinstance(v, Path) else v)
                    for k, v in vars(args).items()
                },
                "corpus": {
                    "induction": ewt_train.name,
                    "n_sentences": len(train),
                    "n_tokens": n_tokens,
                    "dev_pool": [p.name for p in dev_files],
                    "test_pools": {k: [p.name for p in v]
                                   for k, v in test_files.items()},
                    "lexicalized_adverbs": sorted(adverbs),
                },
                "thresholds": {p: vars(thresholds[p]) for p in TARGET_POS},
                "validation_summary": summary,
                "training_templates": TRAINING_TEMPLATES,
                "construction_collisions": collisions,
                "slots": realized,
            },
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with (args.out_dir / "frames_all.jsonl").open("w", encoding="utf-8") as fh:
        ranked = sorted(frames.values(), key=lambda f: f.score, reverse=True)
        for fs in ranked:
            if fs.n_target >= 5:
                fh.write(json.dumps(fs.to_dict(), ensure_ascii=False) + "\n")

    # compact slot file read by task2.py
    flat = {
        pos: [
            {
                "signature": r["signature"],
                # {NEOLOGISM} is replaced by the word being scored
                "prefix": " ".join(r["probe_item"]["prefix_tokens"]),
                "prefix_tokens": r["probe_item"]["prefix_tokens"],
                "morph_requirement": r["probe_item"].get("morph_requirement"),
                # D_f: the tokens whose surprisal is scored
                "diagnostic": r["probe_item"]["diagnostic"],
                "item": r["probe_item"]["text"],
                "purity": r["purity"],
                "n": r["n_target"],
            }
            for r in realized[pos]
            if r["probe_item"]
        ]
        for pos in TARGET_POS
    }
    (args.out_dir / "candidates.json").write_text(
        json.dumps(flat, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    (args.out_dir / "slots.md").write_text(
        emit_markdown(selected, realized, thresholds, summary, validation, args),
        encoding="utf-8",
    )
    (args.out_dir / "slots_table.tex").write_text(
        emit_latex(realized), encoding="utf-8"
    )
    print(f"      -> {args.out_dir}/candidates_full.json")
    print(f"      -> {args.out_dir}/candidates.json")
    print(f"      -> {args.out_dir}/slots.md")
    print(f"      -> {args.out_dir}/slots_table.tex")
    print(f"      -> {args.out_dir}/frames_all.jsonl")

    review = args.out_dir / "review_template.json"
    write_review_template(review, realized)
    print(f"      -> {review}  (copy to {args.overrides.name} and edit to "
          f"accept / rewrite / reject)")


if __name__ == "__main__":
    main()
