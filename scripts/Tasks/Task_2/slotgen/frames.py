"""Context signatures around target tokens.

A signature is a short window around a token. Closed-class words are kept as
written and open-class words are replaced by their UPOS tag, e.g.

    She decided to leave it .        (target = leave/VERB)   ->  "to {SLOT} PRON"
    The problem was serious .        (target = problem/NOUN)  ->  "the {SLOT} was"
    The problem was serious .        (target = serious/ADJ)   ->  "was {SLOT} ."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Sequence, Set, Tuple

from .ud_io import BOS, EOS, Sentence, Token

SLOT = "{SLOT}"

#: Categories that slots are built for.
TARGET_POS: Tuple[str, ...] = ("NOUN", "VERB", "ADJ")

#: UPOS tags whose word forms are kept verbatim in a signature.
DEFAULT_LEXICALIZED_UPOS: Tuple[str, ...] = (
    "DET",
    "ADP",
    "AUX",
    "PART",
    "SCONJ",
    "CCONJ",
    "PUNCT",
)

#: UPOS tags replaced by a placeholder. PRON is abstracted so that "to leave it"
#: and "to tell them" count as the same frame.
DEFAULT_ABSTRACTED_UPOS: Tuple[str, ...] = (
    "PRON",
    "NOUN",
    "PROPN",
    "VERB",
    "ADJ",
    "ADV",
    "NUM",
    "INTJ",
    "SYM",
    "X",
)


@dataclass(frozen=True)
class Occurrence:
    """One attested instantiation of one signature."""

    sent_key: Tuple[str, int]  # (corpus source, sentence index)
    target_idx: int
    upos: str
    lemma: str
    deprel: str
    feats: str


class SignatureEncoder:
    """Maps context tokens to signature elements under a fixed policy."""

    def __init__(
        self,
        lexicalized_upos: Sequence[str] = DEFAULT_LEXICALIZED_UPOS,
        abstracted_upos: Sequence[str] = DEFAULT_ABSTRACTED_UPOS,
        lexicalized_adverbs: Set[str] | None = None,
    ) -> None:
        self.lexicalized_upos = set(lexicalized_upos)
        self.abstracted_upos = set(abstracted_upos)
        self.lexicalized_adverbs = lexicalized_adverbs or set()

    def encode(self, tok: Token) -> str:
        if tok.upos == "ADV" and tok.form.lower() in self.lexicalized_adverbs:
            return tok.form.lower()
        if tok.upos in self.lexicalized_upos:
            return tok.form.lower()
        if tok.upos in self.abstracted_upos:
            return tok.upos
        return tok.upos

    def is_abstract(self, element: str) -> bool:
        return element in self.abstracted_upos or element in {BOS, EOS}

    def element_kind(self, element: str) -> str:
        """Return slot, boundary, abstract (UPOS placeholder), punct or lexical."""
        if element == SLOT:
            return "slot"
        if element in {BOS, EOS}:
            return "boundary"
        if element in self.abstracted_upos:
            return "abstract"
        if not any(ch.isalnum() for ch in element):
            return "punct"
        return "lexical"


def _context_element(
    sent: Sentence, idx: int, encoder: SignatureEncoder
) -> str:
    if idx < 0:
        return BOS
    if idx >= len(sent.tokens):
        return EOS
    return encoder.encode(sent.tokens[idx])


def windows(max_left: int, min_right: int, max_right: int, min_total: int = 2):
    """(left, right) window sizes to extract; min_right >= 1 keeps a right context to score."""
    out = []
    for left in range(0, max_left + 1):
        for right in range(min_right, max_right + 1):
            if left + right >= min_total:
                out.append((left, right))
    return out


def signature_for(
    sent: Sentence,
    target_idx: int,
    left: int,
    right: int,
    encoder: SignatureEncoder,
) -> str:
    parts: List[str] = []
    for off in range(-left, 0):
        parts.append(_context_element(sent, target_idx + off, encoder))
    parts.append(SLOT)
    for off in range(1, right + 1):
        parts.append(_context_element(sent, target_idx + off, encoder))
    return " ".join(parts)


def iter_target_tokens(sent: Sentence) -> Iterator[Token]:
    for tok in sent.tokens:
        if tok.upos in TARGET_POS:
            yield tok


def signature_shape(signature: str) -> Tuple[int, int]:
    """(n_left, n_right) for a signature string."""
    parts = signature.split(" ")
    i = parts.index(SLOT)
    return i, len(parts) - i - 1
