"""Re-estimate slot purity on corpora not used for induction.

Frames are induced on EWT train; run_induction.py recounts them on EWT test and
on UD GUM.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from .frames import TARGET_POS, SignatureEncoder, signature_for, windows
from .ud_io import Sentence


@dataclass
class PurityEstimate:
    n_target: int
    n_all: int
    counts: Dict[str, int]
    purity: float  # for the POS the frame was selected for

    def to_dict(self) -> dict:
        return {
            "n_target": self.n_target,
            "n_all": self.n_all,
            "counts": self.counts,
            "purity": round(self.purity, 4) if self.n_target else None,
        }


def recount(
    sentences: Sequence[Sentence],
    encoder: SignatureEncoder,
    signatures: Iterable[str],
    max_left: int = 2,
    min_right: int = 1,
    max_right: int = 3,
) -> Dict[str, Counter]:
    """Count how each target signature is filled in a fresh corpus."""
    wanted = set(signatures)
    win = windows(max_left, min_right, max_right)
    out: Dict[str, Counter] = {sig: Counter() for sig in wanted}
    for sent in sentences:
        for tok in sent.tokens:
            for left, right in win:
                sig = signature_for(sent, tok.idx, left, right, encoder)
                if sig in wanted:
                    out[sig][tok.upos] += 1
    return out


def estimate(counts: Counter, pos: str) -> PurityEstimate:
    n_target = sum(counts[p] for p in TARGET_POS)
    n_all = sum(counts.values())
    purity = counts[pos] / n_target if n_target else 0.0
    return PurityEstimate(
        n_target=n_target,
        n_all=n_all,
        counts={p: counts[p] for p in TARGET_POS},
        purity=purity,
    )


def validate(
    selected: Dict[str, List],
    encoder: SignatureEncoder,
    corpora: Dict[str, Sequence[Sentence]],
    max_left: int = 2,
    min_right: int = 1,
    max_right: int = 3,
) -> Dict[str, Dict[str, PurityEstimate]]:
    """Re-estimate purity for every selected signature in every held-out corpus.

    Returns ``{corpus_name: {signature: PurityEstimate}}``.
    """
    sigs = [fs.signature for frames in selected.values() for fs in frames]
    pos_of = {fs.signature: pos for pos, frames in selected.items() for fs in frames}
    results: Dict[str, Dict[str, PurityEstimate]] = {}
    for name, sentences in corpora.items():
        counts = recount(sentences, encoder, sigs, max_left, min_right, max_right)
        results[name] = {
            sig: estimate(counts[sig], pos_of[sig]) for sig in sigs
        }
    return results


def summarize(
    selected: Dict[str, List],
    validation: Dict[str, Dict[str, PurityEstimate]],
    min_heldout_purity: float = 0.85,
    min_heldout_count: int = 5,
) -> dict:
    """Aggregate purity per POS per corpus, and flag frames that failed to hold up."""
    summary: dict = {"per_pos": {}, "flagged": []}
    for pos, frames in selected.items():
        n_train = sum(f.n_target for f in frames)
        row = {
            "train": round(
                sum(f.pos_counts[pos] for f in frames) / max(n_train, 1), 4
            ),
            "train_n": n_train,
        }
        for corpus, est in validation.items():
            vals = [est[f.signature] for f in frames if est[f.signature].n_target > 0]
            n_total = sum(v.n_target for v in vals)
            # pooled over slots rather than averaged, so rare slots carry less weight
            row[corpus] = (
                round(sum(v.counts[pos] for v in vals) / n_total, 4) if n_total else None
            )
            row[f"{corpus}_n"] = n_total
            row[f"{corpus}_attested"] = f"{len(vals)}/{len(frames)}"
        summary["per_pos"][pos] = row

    for pos, frames in selected.items():
        for f in frames:
            for corpus, est in validation.items():
                e = est[f.signature]
                if e.n_target >= min_heldout_count and e.purity < min_heldout_purity:
                    summary["flagged"].append(
                        {
                            "pos": pos,
                            "signature": f.signature,
                            "corpus": corpus,
                            "train_purity": round(f.purity, 4),
                            "heldout_purity": round(e.purity, 4),
                            "heldout_n": e.n_target,
                        }
                    )
    return summary
