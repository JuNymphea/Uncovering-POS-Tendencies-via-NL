"""Turn a corpus frame into a minimal probe item.

An item is a short opener plus the frame's left context, ending in the slot,
followed by the frame's own right context as the diagnostic continuation D_f.
For example ``to {SLOT} PRON`` becomes "They tried to {NEOLOGISM} it" and only
"it" is scored:

    S(w, f) = (1/|D_f|) * sum_{t in D_f} -log P(t | prefix containing w)

UPOS placeholders are filled from small generic word lists (FILLERS).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .frames import SLOT, SignatureEncoder
from .induce import FrameStats
from .ud_io import BOS, EOS

NEOLOGISM = "{NEOLOGISM}"

# ---------------------------------------------------------------------------
# Generic words used outside the corpus frame
# ---------------------------------------------------------------------------

#: Openers placed before the frame, keyed by the class of its leftmost element
#: (e.g. "to" needs a governing verb, "n't" an auxiliary).
OPENERS: Dict[str, Tuple[Tuple[str, ...], ...]] = {
    "START":  ((),),
    "TO":     (("They", "tried"), ("Someone", "tried")),
    "NEG":    (("They", "do"), ("Someone", "does")),
    "AUX":    (("They",), ("It",), ("Someone",)),
    "CLITIC": (("It",), ("Someone",)),
    "DET":    ((), ("They", "have", "seen")),
    "ADP":    (("They", "spoke"), ("Someone", "spoke")),
    "DEG":    (("It", "was"), ("It", "was", "a"), ("They", "were")),
    "WH":     (("They", "know"), ("Someone", "knows")),
    "CCONJ":  (("They", "did", "it"),),
    "COMMA":  (("It", "was"),),
    "NOUN":   (("The",), ("They", "have", "seen", "the")),
    "ADJ":    (("The",), ("They", "have", "seen", "the")),
    "VERB":   (("They",), ("Someone",)),
    "PRON":   ((),),
    "OTHER":  ((), ("They",), ("It", "was")),
}

#: Filler words per UPOS. Fillers attested in the position's deprel are
#: preferred (see _fits_position).
FILLERS: Dict[str, Tuple[str, ...]] = {
    "PRON": ("it", "them"),
    "NOUN": ("thing", "something", "one"),
    "PROPN": ("someone",),
    "VERB": ("happened", "did", "started", "changed"),
    "ADJ": ("different", "similar", "certain", "other", "same"),
    "ADV": ("again", "then"),
    "NUM": ("one",),
    "INTJ": ("well",),
    "SYM": ("one",),
    "X": ("thing",),
}

EDGE_CLASS: Dict[str, str] = {
    "to": "TO",
    "n't": "NEG", "not": "NEG",
    "'s": "CLITIC",
    "and": "CCONJ", "or": "CCONJ", "but": "CCONJ",
    "very": "DEG", "more": "DEG", "most": "DEG", "so": "DEG", "too": "DEG",
    "quite": "DEG", "really": "DEG", "as": "DEG",
    "how": "WH", "what": "WH", "why": "WH",
    ",": "COMMA",
}
_DETS = {"the", "a", "an", "this", "that", "these", "those", "some", "no",
         "his", "her", "their", "our", "its", "my", "your"}
_AUXES = {"will", "can", "would", "could", "should", "may", "might", "must",
          "is", "are", "was", "were", "be", "been", "am", "has", "have", "had",
          "do", "does", "did"}
_PREPS = {"of", "in", "on", "for", "with", "at", "by", "from", "about", "into",
          "over", "under", "through", "than", "to"}
_SINGULAR_AUX = {"is", "was", "has", "does"}
_PLURAL_AUX = {"are", "were", "have", "do"}
_DEGREE = {"very", "more", "most", "so", "too", "quite", "really", "as"}
_INFINITIVE_TRIGGERS = {"to", "will", "can", "would", "could", "should", "do",
                        "does", "did", "n't", "not", "may", "might", "must"}
_PLURAL_TRIGGERS = {"are", "were"}


#: Completions appended only for the parser check in is_wellformed; never scored.
COMPLETIONS: Dict[str, Tuple[str, ...]] = {
    "TO": ("do", "it"),
    "ADP": ("it",),
    "DET": ("thing",),
    "CCONJ": ("something",),
    "AUX": ("there",),
    "DEG": ("thing",),
    "NEG": ("do", "it"),
    "COMMA": ("something",),
    "WH": ("it", "is"),
    "NOUN": (), "PRON": (), "VERB": (), "ADJ": ("thing",),
    "CLITIC": ("thing",), "START": (), "OTHER": (),
}

_FINITE_FORMS = {"is", "was", "are", "were", "has", "have", "had", "do",
                 "does", "did", "can", "will", "would", "could", "should",
                 "tried", "seen", "spoke", "know", "knows", "happened", "be"}


def closure_for(item_tokens: Sequence[str], last_edge: str) -> Tuple[str, ...]:
    """Tokens appended for the parser check only (plus a predicate if there is no finite verb)."""
    out = list(COMPLETIONS.get(last_edge, ()))
    has_verb = any(t.lower() in _FINITE_FORMS for t in item_tokens)
    if not has_verb:
        out += ["was", "there"]
    return tuple(out + ["."])


def edge_class(element: str, encoder: SignatureEncoder) -> str:
    kind = encoder.element_kind(element)
    if kind == "abstract":
        return element if element in OPENERS else "OTHER"
    low = element.lower()
    if low in EDGE_CLASS:
        return EDGE_CLASS[low]
    if low in _DETS:
        return "DET"
    if low in _AUXES:
        return "AUX"
    if low in _PREPS:
        return "ADP"
    if kind == "punct":
        return "COMMA"
    return "OTHER"


def _fits_position(word: str, upos: str, deprel: str, dists: dict | None) -> bool:
    """True if `word` occurs in the corpus as `upos` with relation `deprel`."""
    if dists is None or deprel in ("_", ""):
        return True
    attested = dists.get(word.lower())
    if not attested:
        return False
    if (upos, deprel) in attested:
        return True
    # also accept a match on the base label ("conj:preconj" -> "conj")
    base = deprel.split(":")[0]
    return any(u == upos and d.split(":")[0] == base for u, d in attested)


def _fill(element: str, prev: str | None, nxt: str | None,
          clause_initial: bool, variant: int = 0,
          deprel: str = "_", dists: dict | None = None,
          target_feats: str = "") -> str:
    if element == "VERB":
        if (nxt or "").lower() == "to":
            return "tried"
        if (prev or "").lower() in _INFINITIVE_TRIGGERS:
            return "do"
        # an object follows, so use a transitive verb
        if (nxt or "").lower() in _DETS:
            return "did"
        options = FILLERS["VERB"]
    elif element == "NOUN":
        if (nxt or "").lower() in _PLURAL_TRIGGERS:
            return "things"
        options = FILLERS["NOUN"]
        # "something" cannot follow a determiner or modifier
        if (prev or "").lower() in _DETS | _DEGREE | {"other", "same"}:
            options = tuple(o for o in options if o != "something") or ("thing",)
    elif element == "PRON":
        if clause_initial:
            # subject agrees with a following auxiliary and with the target's
            # person/number features
            if (nxt or "").lower() in _SINGULAR_AUX:
                return "it"
            if "Person=1" in target_feats and "Number=Sing" in target_feats:
                return "I"
            if "Person=3" in target_feats and "Number=Sing" in target_feats:
                return "it"
            return "they"
        return "it"
    else:
        options = FILLERS.get(element, ("thing",))
        if element == "ADJ" and (prev or "").lower() in {"a", "an"}:
            options = tuple(o for o in options if o != "same") or ("other",)

    # prefer fillers attested in this position, then rotate by variant
    fitting = tuple(o for o in options
                    if _fits_position(o, element, deprel, dists))
    options = fitting or options
    return options[variant % len(options)]


# ---------------------------------------------------------------------------
# Overlap with the training templates
# ---------------------------------------------------------------------------

#: Templates the neologism embeddings are trained with.
TRAINING_TEMPLATES: Dict[str, str] = {
    "unbiased": "Make your answer reflect the following word: {NEOLOGISM}.",
    "verb": "Please {NEOLOGISM} your answer.",
    "noun": "Please answer this question with a {NEOLOGISM}.",
    "adjective": "Your answer should be as {NEOLOGISM} as possible.",
}

#: (left cue, right cue) pairs from those templates. None matches anything and
#: "" means no token. Probe frames that match one are excluded.
TRAINING_CONSTRUCTIONS: Tuple[Tuple[str | None, str | None], ...] = (
    ("as", "as"),        # adjective: "as {NEOLOGISM} as possible"
    ("a", ""),           # noun: "with a {NEOLOGISM}." (slot phrase-final)
    ("with", None),      # noun: the governing preposition
    (None, "your"),      # verb: "Please {NEOLOGISM} your answer"
    (":", None),         # unbiased: "the following word: {NEOLOGISM}"
)


def collides_with_training(signature: str, encoder: SignatureEncoder) -> str | None:
    """Name the training construction a frame duplicates, if any."""
    els = signature.split(" ")
    i = els.index(SLOT)
    left = [e for e in els[:i] if e not in (BOS, EOS)]
    right = [e for e in els[i + 1:] if e not in (BOS, EOS)]

    def cue(seq: List[str], from_right: bool) -> str | None:
        """The adjacent lexical cue: "" if nothing is there, None if abstract."""
        if not seq:
            return ""
        el = seq[-1] if from_right else seq[0]
        return None if encoder.element_kind(el) == "abstract" else el.lower()

    lcue = cue(left, from_right=True)      # element immediately left of slot
    rcue = cue(right, from_right=False)    # element immediately right of slot
    for want_l, want_r in TRAINING_CONSTRUCTIONS:
        if want_l is not None and lcue != want_l:
            continue
        if want_r is not None and rcue != want_r:
            continue
        return f"{want_l or '*'} {{SLOT}} {want_r or '*'}"
    return None


# ---------------------------------------------------------------------------


@dataclass
class ProbeItem:
    """One evaluation item: a prefix, and the tokens whose surprisal is scored."""

    signature: str
    pos: str
    prefix_tokens: List[str]      # everything up to and including the slot
    diagnostic: List[str]         # D_f, the frame's own licensed continuation
    morph_requirement: str        # required inflection (UD feats)
    opener: Tuple[str, ...]
    text: str                     # prefix + diagnostic, for reading
    tagger_accuracy: float
    tagger_accuracy_other: Dict[str, float]
    tagger_margin: float

    def prefix_for(self, word: str) -> str:
        """Prefix text with `word` in the slot; a/an is adjusted to the word."""
        toks = [word if t == NEOLOGISM else t for t in self.prefix_tokens]
        for i, tok in enumerate(toks[:-1]):
            if tok.lower() in {"a", "an"}:
                vowel = toks[i + 1][:1].lower() in set("aeiou")
                fixed = "an" if vowel else "a"
                toks[i] = fixed.capitalize() if tok[0].isupper() else fixed
        return _capitalize(detokenize(toks))

    def as_dict(self) -> dict:
        d = dict(vars(self))
        d["n_diagnostic"] = len(self.diagnostic)
        return d


_NO_SPACE_BEFORE = {".", ",", "!", "?", ";", ":", "n't", "'s", "'re", "'ll",
                    "'ve", "'d", "'m"}


def detokenize(tokens: Sequence[str]) -> str:
    out = ""
    for i, tok in enumerate(tokens):
        out = tok if i == 0 else out + ("" if tok in _NO_SPACE_BEFORE else " ") + tok
    return out


def _capitalize(text: str) -> str:
    return text[0].upper() + text[1:] if text and not text.startswith(NEOLOGISM) else text


def build(
    fs: FrameStats,
    encoder: SignatureEncoder,
    opener: Tuple[str, ...],
    variant: int,
    dists: dict | None = None,
    bigrams=None,
    min_bigram: int = 1,
) -> Tuple[List[str], List[str]] | None:
    """(prefix_tokens, diagnostic) for one opener/filler choice, or None."""
    els = fs.elements()
    i = els.index(SLOT)
    left_raw, right_raw = els[:i], els[i + 1:]

    if BOS in left_raw and opener:
        return None
    left_raw = [e for e in left_raw if e != BOS]
    right_raw = [e for e in right_raw if e not in (EOS,)]
    # punctuation is never part of the diagnostic continuation
    right_raw = [e for e in right_raw
                 if encoder.element_kind(e) != "punct"]
    if not right_raw:
        return None
    # skip continuations that contain a clitic such as "'s" or "'ve"
    if any(e.startswith("'") for e in right_raw):
        return None

    def render(seq, prev0, clause_start, offsets):
        out: List[str] = []
        for j, el in enumerate(seq):
            prev = out[-1] if out else prev0
            nxt = seq[j + 1] if j + 1 < len(seq) else None
            if encoder.element_kind(el) == "abstract":
                out.append(_fill(el, prev, nxt, clause_start and not out,
                                 variant, fs.context_deprel_at(offsets[j]),
                                 dists, fs.dominant_feats))
            else:
                out.append(el)
        return out

    left = render(left_raw, opener[-1] if opener else None, not opener,
                  list(range(-len(left_raw), 0)))
    diagnostic = render(right_raw, left[-1] if left else
                        (opener[-1] if opener else None), False,
                        list(range(1, len(right_raw) + 1)))

    joined = list(left) + diagnostic
    for a, b in zip(joined, joined[1:]):
        if b in {"same", "one"} and a.lower() in {"a", "an"}:
            return None
        if a.lower() == "a" and b[:1].lower() in set("aeiou"):
            return None  # would need "an"

    prefix = list(opener) + left + [NEOLOGISM]

    # every bigram of opener + left context + D_f, except the one across the
    # slot, must occur at least min_bigram times in the pooled treebanks
    if bigrams is not None:
        span = list(opener) + left + diagnostic  # slot excluded
        boundary = len(opener) + len(left)       # bigrams must not cross it
        for k, (a, b) in enumerate(zip(span, span[1:])):
            if k + 1 == boundary:
                continue
            if bigrams[(a.lower(), b.lower())] < min_bigram:
                return None

    return prefix, diagnostic


def morph_requirement(fs: FrameStats) -> str:
    """Dominant UD features of the frame's fillers, or "Any"."""
    feats = fs.dominant_feats
    return feats if feats and feats != "_" else "Any"


def make_item(
    fs: FrameStats,
    encoder: SignatureEncoder,
    check,
    min_accuracy: float = 0.85,
    min_margin: float = 0.10,
    dists: dict | None = None,
    bigrams=None,
    min_bigram: int = 1,
) -> ProbeItem | None:
    """Best realization of `fs`, or None if no opener/filler choice passes.

    A candidate must pass the parser check and reach `min_accuracy` tagger
    agreement with a margin of at least `min_margin` over the other categories;
    the one with the largest margin is kept.
    """
    pos = fs.dominant_pos
    els = fs.elements()
    i = els.index(SLOT)
    left_raw = els[:i]
    left_edge = "START" if (not left_raw or left_raw[0] == BOS) else edge_class(
        left_raw[0], encoder)
    openers = OPENERS.get(left_edge, OPENERS["OTHER"])
    if left_edge == "AUX" and left_raw:
        aux = left_raw[0].lower()
        if aux in _SINGULAR_AUX:
            openers = (("It",),)
        elif aux in _PLURAL_AUX:
            openers = (("They",),)

    best: ProbeItem | None = None
    best_key: Tuple = ()
    n_variants = max(len(v) for v in FILLERS.values())
    for oi, opener in enumerate(openers):
        for variant in range(n_variants):
            built = build(fs, encoder, opener, variant, dists, bigrams,
                          min_bigram)
            if built is None:
                continue
            prefix, diagnostic = built
            tokens = prefix + diagnostic
            closure = closure_for(tokens, edge_class(diagnostic[-1], encoder))
            if not check.is_wellformed(tokens, pos, closure):
                continue
            acc = check.accuracy(tokens, pos)
            if acc < min_accuracy:
                continue
            others = {p: check.accuracy(tokens, p)
                      for p in ("NOUN", "VERB", "ADJ") if p != pos}
            margin = acc - max(others.values())
            if margin < min_margin:
                continue
            key = (round(margin, 1), round(acc, 1), -len(opener), -oi,
                   -variant, -len(tokens))
            if best is None or key > best_key:
                best_key = key
                best = ProbeItem(
                    signature=fs.signature,
                    pos=pos,
                    morph_requirement=morph_requirement(fs),
                    prefix_tokens=([_capitalize(prefix[0])] + prefix[1:]
                                   if prefix and prefix[0] != NEOLOGISM else prefix),
                    diagnostic=diagnostic,
                    opener=opener,
                    text=_capitalize(detokenize(tokens)),
                    tagger_accuracy=round(acc, 3),
                    tagger_accuracy_other={k: round(v, 3) for k, v in others.items()},
                    tagger_margin=round(margin, 3),
                )
    return best
