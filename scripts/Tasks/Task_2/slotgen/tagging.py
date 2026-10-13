"""spaCy checks on probe items.

Items are fragments (e.g. "The {NEOLOGISM} of"), so the parser check first
completes them with a closure. The tagger check puts known words of each
category in the slot and measures how often they are tagged as that category.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

NEOLOGISM = "{NEOLOGISM}"

_NO_SPACE_BEFORE = {".", ",", "!", "?", ";", ":", "n't", "'s", "'re", "'ll",
                    "'ve", "'d", "'m"}


def detokenize(tokens: Sequence[str]) -> str:
    out = ""
    for i, tok in enumerate(tokens):
        out = tok if i == 0 else out + ("" if tok in _NO_SPACE_BEFORE else " ") + tok
    return out


class TaggerCheck:
    """Checks whether real words of a category are tagged as that category in an item."""

    #: Simple frame per category, used to screen the probe words.
    SCREENING_FRAMES = {
        "NOUN": ("They", "saw", "the", NEOLOGISM, "at", "the", "time", "."),
        "VERB": ("They", "will", NEOLOGISM, "it", "at", "the", "time", "."),
        "ADJ": ("It", "was", "very", NEOLOGISM, "at", "the", "time", "."),
    }

    def __init__(self, nlp, probes: Dict[str, List[str]], n_probes: int = 12,
                 screen: bool = True):
        self.nlp = nlp
        self.n_probes = n_probes
        self.probes = self._screen(probes) if screen else probes

    # -- probe screening -----------------------------------------------------

    def _screen(self, probes: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Keep the probe words the tagger labels correctly in SCREENING_FRAMES."""
        kept: Dict[str, List[str]] = {}
        for pos, words in probes.items():
            frame = self.SCREENING_FRAMES.get(pos)
            if frame is None:
                kept[pos] = list(words)
                continue
            texts = [self._render(frame, w) for w in words]
            good = []
            for word, doc in zip(words, self.nlp.pipe(texts, batch_size=64)):
                for tok in doc:
                    if tok.text.lower() == word.lower():
                        if tok.pos_ == pos:
                            good.append(word)
                        break
            kept[pos] = good
        return kept

    # -- rendering -----------------------------------------------------------

    @staticmethod
    def _render(tokens: Sequence[str], word: str) -> str:
        """Fill the slot with `word`; capitalize the first token unless it is the slot."""
        text = detokenize([word if t == NEOLOGISM else t for t in tokens])
        if tokens and tokens[0] != NEOLOGISM and text:
            text = text[0].upper() + text[1:]
        return text

    # -- checks --------------------------------------------------------------

    def accuracy(self, tokens: Sequence[str], pos: str) -> float:
        words = self.probes.get(pos, [])[: self.n_probes]
        if not words:
            return 0.0
        hits = 0
        texts = [self._render(tokens, w) for w in words]
        for word, doc in zip(words, self.nlp.pipe(texts, batch_size=32)):
            for tok in doc:
                if tok.text.lower() == word.lower():
                    hits += tok.pos_ == pos
                    break
        return hits / len(words)

    #: Subject / finite-auxiliary pairs that do not agree.
    _BAD_AGREEMENT = {
        ("they", "is"), ("they", "was"), ("they", "has"), ("they", "does"),
        ("it", "are"), ("it", "were"), ("it", "have"), ("it", "do"),
        ("someone", "are"), ("someone", "were"), ("someone", "have"),
    }

    def is_wellformed(self, tokens: Sequence[str], pos: str,
                      closure: Sequence[str] = ()) -> bool:
        """Parser check on the item completed with `closure` (the closure is never scored).

        Rejects repeated tokens, subject-auxiliary disagreement, and parses
        without a single verbal root and a subject, or with dangling
        prepositions or determiners, or bare singular nouns in argument position.
        """
        for a, b in zip(tokens, tokens[1:]):
            if a.lower() == b.lower() and a != NEOLOGISM:
                return False  # "one one", "the the"
            if (a.lower(), b.lower()) in self._BAD_AGREEMENT:
                return False  # "They is", "It are"

        full = list(tokens) + list(closure)
        for probe in self.probes.get(pos, ["thing"])[:3]:
            doc = self.nlp(self._render(full, probe))
            roots = [t for t in doc if t.dep_ == "ROOT"]
            if len(roots) != 1 or roots[0].pos_ not in {"VERB", "AUX"}:
                return False
            if any(t.dep_ == "dep" for t in doc):
                return False
            if not any(t.dep_ in {"nsubj", "nsubjpass", "expl", "csubj"}
                       for t in doc):
                return False
            for t in doc:
                if t.pos_ == "ADP" and not any(
                    c.dep_ in {"pobj", "pcomp", "obj"} for c in t.children
                ):
                    return False
                if t.pos_ == "DET" and t.head.pos_ not in {
                    "NOUN", "PROPN", "PRON", "NUM", "ADJ"
                }:
                    return False
                if (t.pos_ == "PRON" and t.dep_ == "appos"
                        and t.head.pos_ in {"NOUN", "PROPN"}):
                    return False
                is_np_head = (t.pos_ == "NOUN" and t.tag_ == "NN") or (
                    t.lower_ == "one" and t.pos_ in {"NUM", "NOUN", "PRON"})
                in_arg = t.dep_ in {"attr", "dobj", "pobj", "nsubj", "conj",
                                    "nsubjpass"}
                has_det = any(c.dep_ in {"det", "poss", "nummod"}
                              for c in t.children)
                if is_np_head and in_arg and not has_det:
                    return False
        return True


def load_tagger():
    """spaCy's small English model, with unused components disabled."""
    import spacy

    return spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
