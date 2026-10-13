"""Count, filter, rank and de-duplicate candidate frames.

A frame is kept when it is frequent, filled by many different lemmas, and
filled almost only by one of NOUN/VERB/ADJ.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

from .frames import (
    SLOT,
    TARGET_POS,
    Occurrence,
    SignatureEncoder,
    signature_for,
    windows,
)
from .ud_io import BOS, EOS, Sentence

LOG3 = math.log(3.0)


@dataclass
class FrameStats:
    """Treebank counts for one context signature."""

    signature: str
    n_left: int
    n_right: int
    pos_counts: Counter = field(default_factory=Counter)  # over all UPOS
    lemma_types: Dict[str, set] = field(default_factory=lambda: defaultdict(set))
    deprels: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    feats: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    context_upos: Dict[int, Counter] = field(default_factory=lambda: defaultdict(Counter))
    context_deprel: Dict[int, Counter] = field(default_factory=lambda: defaultdict(Counter))
    occurrences: List[Occurrence] = field(default_factory=list)
    purity_denominator: str = "all"

    # derived quantities

    @property
    def n_target(self) -> int:
        """Number of NOUN/VERB/ADJ fillers."""
        return sum(self.pos_counts[p] for p in TARGET_POS)

    @property
    def n_all(self) -> int:
        return sum(self.pos_counts.values())

    @property
    def distribution(self) -> Dict[str, float]:
        """P(p | c) over {NOUN, VERB, ADJ}."""
        total = self.n_target
        if total == 0:
            return {p: 0.0 for p in TARGET_POS}
        return {p: self.pos_counts[p] / total for p in TARGET_POS}

    @property
    def dominant_pos(self) -> str:
        return max(TARGET_POS, key=lambda p: self.pos_counts[p])

    @property
    def purity(self) -> float:
        if self.purity_denominator == "all":
            return self.pos_counts[self.dominant_pos] / self.n_all if self.n_all else 0.0
        return self.distribution[self.dominant_pos]

    @property
    def purity_target3(self) -> float:
        return self.distribution[self.dominant_pos]

    @property
    def n_types(self) -> int:
        """Distinct lemmas filling the slot as the dominant POS."""
        return len(self.lemma_types[self.dominant_pos])

    @property
    def entropy(self) -> float:
        h = 0.0
        for p in TARGET_POS:
            q = self.distribution[p]
            if q > 0:
                h -= q * math.log(q)
        return h

    @property
    def entropy_norm(self) -> float:
        return self.entropy / LOG3

    @property
    def open_class_share(self) -> float:
        """Share of all fillers that are NOUN/VERB/ADJ."""
        return self.n_target / self.n_all if self.n_all else 0.0

    @property
    def score(self) -> float:
        """Ranking score: log frequency x purity x (1 - normalized entropy) x log types."""
        return (
            math.log(1 + self.n_target)
            * self.purity
            * (1.0 - self.entropy_norm)
            * math.log(1 + self.n_types)
        )

    @property
    def dominant_deprel(self) -> str:
        c = self.deprels[self.dominant_pos]
        return c.most_common(1)[0][0] if c else "_"

    @property
    def dominant_feats(self) -> str:
        c = self.feats[self.dominant_pos]
        return c.most_common(1)[0][0] if c else "_"

    def context_upos_at(self, offset: int) -> str:
        c = self.context_upos[offset]
        return c.most_common(1)[0][0] if c else "_"

    def context_deprel_at(self, offset: int) -> str:
        """Most common deprel at a context offset; used to choose fillers for that position."""
        c = self.context_deprel[offset]
        return c.most_common(1)[0][0] if c else "_"

    def elements(self) -> List[str]:
        return self.signature.split(" ")

    def to_dict(self) -> dict:
        return {
            "signature": self.signature,
            "n_left": self.n_left,
            "n_right": self.n_right,
            "dominant_pos": self.dominant_pos,
            "n_target": self.n_target,
            "n_all": self.n_all,
            "counts": {p: self.pos_counts[p] for p in TARGET_POS},
            "counts_other": self.n_all - self.n_target,
            "purity": round(self.purity, 4),
            "purity_denominator": self.purity_denominator,
            "purity_target3": round(self.purity_target3, 4),
            "n_types": self.n_types,
            "entropy_norm": round(self.entropy_norm, 4),
            "open_class_share": round(self.open_class_share, 4),
            "score": round(self.score, 4),
            "dominant_deprel": self.dominant_deprel,
            "dominant_feats": self.dominant_feats,
            "context_deprels": {str(o): self.context_deprel_at(o)
                                for o in sorted(self.context_deprel)},
            "example_lemmas": sorted(self.lemma_types[self.dominant_pos])[:15],
        }


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------


def count_signatures(
    sentences: Sequence[Sentence],
    encoder: SignatureEncoder,
    max_left: int = 2,
    min_right: int = 1,
    max_right: int = 3,
    prune_below: int = 5,
    purity_denominator: str = "all",
) -> Dict[str, FrameStats]:
    """FrameStats for every signature seen at least `prune_below` times.

    The first pass counts signature strings only; the second collects details
    for the survivors. Every token counts as a filler, not only NOUN/VERB/ADJ,
    so that purity can use the all-UPOS denominator.
    """
    win = windows(max_left, min_right, max_right)

    # pass 1: frequency filter
    rough: Counter = Counter()
    for sent in sentences:
        for tok in sent.tokens:
            for left, right in win:
                rough[signature_for(sent, tok.idx, left, right, encoder)] += 1
    keep = {sig for sig, n in rough.items() if n >= prune_below}
    del rough

    # pass 2: full statistics for the surviving signatures
    frames: Dict[str, FrameStats] = {}
    for sent_i, sent in enumerate(sentences):
        key = (sent.source, sent_i)
        for tok in sent.tokens:
            for left, right in win:
                sig = signature_for(sent, tok.idx, left, right, encoder)
                if sig not in keep:
                    continue
                fs = frames.get(sig)
                if fs is None:
                    fs = FrameStats(signature=sig, n_left=left, n_right=right,
                                    purity_denominator=purity_denominator)
                    frames[sig] = fs
                fs.pos_counts[tok.upos] += 1
                if tok.upos in TARGET_POS:
                    fs.lemma_types[tok.upos].add(tok.lemma.lower())
                    fs.deprels[tok.upos][tok.deprel] += 1
                    fs.feats[tok.upos][tok.feats] += 1
                    if len(fs.occurrences) < 400:
                        fs.occurrences.append(
                            Occurrence(
                                sent_key=key,
                                target_idx=tok.idx,
                                upos=tok.upos,
                                lemma=tok.lemma.lower(),
                                deprel=tok.deprel,
                                feats=tok.feats,
                            )
                        )
                    for off in range(-left, right + 1):
                        if off == 0:
                            continue
                        j = tok.idx + off
                        if 0 <= j < len(sent.tokens):
                            fs.context_upos[off][sent.tokens[j].upos] += 1
                            fs.context_deprel[off][sent.tokens[j].deprel] += 1
                        else:
                            fs.context_upos[off][BOS if j < 0 else EOS] += 1
    return frames


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------


@dataclass
class Thresholds:
    min_freq: int = 20
    min_types: int = 10
    min_purity: float = 0.90

    def describe(self) -> str:
        return (
            f"Freq>={self.min_freq}, Types>={self.min_types}, "
            f"P(p|c)>={self.min_purity:.2f}"
        )


#: (min_freq, min_types) settings tried in turn by induce(); purity is never relaxed.
RELAXATION_LADDER: Tuple[Tuple[int, int], ...] = (
    (20, 10),
    (15, 8),
    (10, 6),
    (8, 5),
    (5, 4),
)


def passes(fs: FrameStats, th: Thresholds) -> bool:
    return (
        fs.n_target >= th.min_freq
        and fs.n_types >= th.min_types
        and fs.purity >= th.min_purity
    )


@dataclass
class StructuralConstraints:
    """Shape requirements on a frame.

    min_left: elements required left of the slot.
    min_content_right: non-punctuation elements required right of the slot,
        since that is what gets scored (rules out e.g. ``ADJ {SLOT} .``).
    min_lexical_cues: kept word forms required, so a frame is not only UPOS
        placeholders (e.g. ``VERB ADV {SLOT} NOUN``).
    """

    encoder: SignatureEncoder
    min_left: int = 1
    min_content_right: int = 1
    min_lexical_cues: int = 1

    def ok(self, fs: FrameStats) -> bool:
        elements = fs.elements()
        i = elements.index(SLOT)
        left, right = elements[:i], elements[i + 1 :]
        if len(left) < self.min_left:
            return False
        kinds_right = [self.encoder.element_kind(e) for e in right]
        if sum(1 for k in kinds_right if k in {"lexical", "abstract"}) < self.min_content_right:
            return False
        kinds_all = [self.encoder.element_kind(e) for e in left + right]
        if sum(1 for k in kinds_all if k == "lexical") < self.min_lexical_cues:
            return False
        return True


@dataclass
class DevFilter:
    """Drops frames whose purity on the dev corpus is below `min_purity`.

    Frames with fewer than `min_count` dev occurrences are kept and counted in
    `n_unjudged`.
    """

    counts: Dict[str, Counter]  # signature -> UPOS counter on the dev corpus
    min_count: int = 8
    min_purity: float = 0.85
    n_unjudged: int = 0

    def ok(self, fs: FrameStats) -> bool:
        c = self.counts.get(fs.signature)
        if not c:
            self.n_unjudged += 1
            return True
        n_target = sum(c[p] for p in TARGET_POS)
        if n_target < self.min_count:
            self.n_unjudged += 1
            return True
        return (c[fs.dominant_pos] / n_target) >= self.min_purity

    def purity_of(self, fs: FrameStats) -> float | None:
        c = self.counts.get(fs.signature)
        if not c:
            return None
        n_target = sum(c[p] for p in TARGET_POS)
        if n_target == 0:
            return None
        return c[fs.dominant_pos] / n_target


def candidates_for(
    frames: Iterable[FrameStats],
    pos: str,
    th: Thresholds,
    constraints: StructuralConstraints | None = None,
    dev_filter: DevFilter | None = None,
) -> List[FrameStats]:
    out = [
        f
        for f in frames
        if f.dominant_pos == pos
        and passes(f, th)
        and (constraints is None or constraints.ok(f))
        and (dev_filter is None or dev_filter.ok(f))
    ]
    out.sort(key=lambda f: f.score, reverse=True)
    return out


# ---------------------------------------------------------------------------
# Construction diversity
# ---------------------------------------------------------------------------


def family_key(fs: FrameStats) -> Tuple[str, Tuple[str, ...]]:
    """Construction family: the target's dominant deprel and the UPOS sequence to its left."""
    left_shape = tuple(fs.context_upos_at(off) for off in range(-fs.n_left, 0))
    return (fs.dominant_deprel, left_shape)


def _subsumes(a: FrameStats, b: FrameStats) -> bool:
    """True if `b` is just `a` with the window trimmed (same context, less of it)."""
    ea, eb = a.elements(), b.elements()
    ia, ib = ea.index(SLOT), eb.index(SLOT)
    if ib > ia or (len(eb) - ib) > (len(ea) - ia):
        return False
    return all(ea[ia + off] == eb[ib + off] for off in range(-ib, len(eb) - ib))


def select_diverse(
    candidates: List[FrameStats],
    n_slots: int,
    max_per_family: int = 1,
    validator: Callable[[FrameStats], bool] | None = None,
    fingerprint: Callable[[FrameStats], str] | None = None,
) -> List[FrameStats]:
    """Greedily take the best-scoring frames, at most `max_per_family` per family.

    The family cap is raised (by up to 3) when `n_slots` cannot be reached.
    `validator` rejects frames that cannot be realized, and `fingerprint` drops
    frames whose realized item duplicates one already chosen.
    """
    for cap in range(max_per_family, max_per_family + 4):
        chosen: List[FrameStats] = []
        used: Counter = Counter()
        seen_items: set = set()
        for fs in candidates:
            key = family_key(fs)
            if used[key] >= cap:
                continue
            if any(_subsumes(c, fs) or _subsumes(fs, c) for c in chosen):
                continue
            if validator is not None and not validator(fs):
                continue
            if fingerprint is not None:
                fp = fingerprint(fs)
                if fp is None or fp in seen_items:
                    continue
                seen_items.add(fp)
            chosen.append(fs)
            used[key] += 1
            if len(chosen) == n_slots:
                return chosen
        if len(chosen) == n_slots:
            return chosen
    return chosen


def build_pool(
    frames: Dict[str, FrameStats],
    pool_size: int = 40,
    min_purity: float = 0.90,
    min_freq: int = 5,
    min_types: int = 4,
    constraints: StructuralConstraints | None = None,
    dev_filter: DevFilter | None = None,
    validator: Callable[[FrameStats], bool] | None = None,
    fingerprint: Callable[[FrameStats], str] | None = None,
) -> Dict[str, List[FrameStats]]:
    """Ranked candidate pool of up to `pool_size` frames per POS.

    The thresholds are the same for all three categories. The final slots are
    picked later from model surprisals (select_slots.py). Returns (pool, thresholds).
    """
    all_frames = list(frames.values())
    th = Thresholds(min_freq, min_types, min_purity)
    pool: Dict[str, List[FrameStats]] = {}
    for pos in TARGET_POS:
        cands = candidates_for(all_frames, pos, th, constraints, dev_filter)
        pool[pos] = select_diverse(cands, pool_size, validator=validator,
                                   fingerprint=fingerprint)
    return pool, th


def induce(
    frames: Dict[str, FrameStats],
    n_slots: int = 10,
    min_purity: float = 0.90,
    constraints: StructuralConstraints | None = None,
    dev_filter: DevFilter | None = None,
    validator: Callable[[FrameStats], bool] | None = None,
    fingerprint: Callable[[FrameStats], str] | None = None,
    ladder: Sequence[Tuple[int, int]] = RELAXATION_LADDER,
) -> Tuple[Dict[str, List[FrameStats]], Dict[str, Thresholds]]:
    """Pick `n_slots` frames per POS, stepping down `ladder` until enough pass."""
    all_frames = list(frames.values())
    selected: Dict[str, List[FrameStats]] = {}
    used_thresholds: Dict[str, Thresholds] = {}
    for pos in TARGET_POS:
        for min_freq, min_types in ladder:
            th = Thresholds(min_freq, min_types, min_purity)
            cands = candidates_for(all_frames, pos, th, constraints, dev_filter)
            picked = select_diverse(cands, n_slots, validator=validator,
                                    fingerprint=fingerprint)
            if len(picked) >= n_slots:
                selected[pos], used_thresholds[pos] = picked, th
                break
        else:
            # ladder exhausted: keep what the loosest setting gives
            th = Thresholds(ladder[-1][0], ladder[-1][1], min_purity)
            selected[pos] = select_diverse(
                candidates_for(all_frames, pos, th, constraints, dev_filter),
                n_slots, validator=validator, fingerprint=fingerprint,
            )
            used_thresholds[pos] = th
    return selected, used_thresholds
