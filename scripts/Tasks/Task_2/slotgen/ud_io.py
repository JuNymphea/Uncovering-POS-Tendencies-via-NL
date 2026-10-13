"""Small CoNLL-U reader and the corpus counts used by the slot pipeline."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List

# Sentence-boundary pseudo-tokens used when a context window runs off the edge.
BOS = "<s>"
EOS = "</s>"


@dataclass
class Token:
    idx: int  # 0-based index within the sentence
    form: str
    lemma: str
    upos: str
    feats: str
    head: int  # 1-based UD head id; 0 means root
    deprel: str
    space_after: bool


@dataclass
class Sentence:
    sent_id: str
    source: str  # e.g. "ewt-train"
    tokens: List[Token] = field(default_factory=list)
    text: str = ""

    def __len__(self) -> int:
        return len(self.tokens)


def _parse_block(lines: List[str], source: str) -> Sentence | None:
    sent = Sentence(sent_id="", source=source)
    for line in lines:
        if line.startswith("#"):
            if line.startswith("# sent_id"):
                sent.sent_id = line.split("=", 1)[-1].strip()
            elif line.startswith("# text"):
                sent.text = line.split("=", 1)[-1].strip()
            continue
        cols = line.split("\t")
        if len(cols) != 10:
            continue
        tok_id = cols[0]
        # skip multiword-token ranges ("1-2") and empty nodes ("3.1")
        if "-" in tok_id or "." in tok_id:
            continue
        try:
            head = int(cols[6]) if cols[6] != "_" else 0
        except ValueError:
            head = 0
        sent.tokens.append(
            Token(
                idx=len(sent.tokens),
                form=cols[1],
                lemma=cols[2] if cols[2] != "_" else cols[1].lower(),
                upos=cols[3],
                feats=cols[5],
                head=head,
                deprel=cols[7],
                space_after="SpaceAfter=No" not in cols[9],
            )
        )
    return sent if sent.tokens else None


def read_conllu(path: str | Path, source: str | None = None) -> Iterator[Sentence]:
    """Yield every sentence in a .conllu file."""
    path = Path(path)
    if source is None:
        source = path.stem
    block: List[str] = []
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line.strip():
                if block:
                    sent = _parse_block(block, source)
                    if sent is not None:
                        yield sent
                    block = []
            else:
                block.append(line)
    if block:
        sent = _parse_block(block, source)
        if sent is not None:
            yield sent


def load_corpus(paths: List[str | Path]) -> List[Sentence]:
    """Load several treebank files into one in-memory corpus."""
    sentences: List[Sentence] = []
    for p in paths:
        sentences.extend(read_conllu(p))
    return sentences


def form_distributions(sentences: List[Sentence]) -> dict:
    """Map each lowercased form to the set of (UPOS, deprel) pairs it occurs with."""
    out: dict = {}
    for sent in sentences:
        for tok in sent.tokens:
            out.setdefault(tok.form.lower(), set()).add((tok.upos, tok.deprel))
    return out


def bigram_counts(sentences: List[Sentence]) -> Counter:
    """Counts of adjacent lowercased form bigrams, used to check words added around a frame."""
    counts: Counter = Counter()
    for sent in sentences:
        forms = [t.form.lower() for t in sent.tokens]
        for a, b in zip(forms, forms[1:]):
            counts[(a, b)] += 1
    return counts


def frequent_adverbs(sentences: List[Sentence], top_k: int) -> set[str]:
    """The `top_k` most frequent ADV forms.

    These stay as words in signatures (e.g. "very", "too", "really"); all other
    adverbs become ADV.
    """
    counts: Counter = Counter()
    for sent in sentences:
        for tok in sent.tokens:
            if tok.upos == "ADV":
                counts[tok.form.lower()] += 1
    return {form for form, _ in counts.most_common(top_k)}
