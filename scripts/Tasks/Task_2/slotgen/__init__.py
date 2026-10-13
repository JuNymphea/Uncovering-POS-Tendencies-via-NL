"""Task 2 slot induction from UD English EWT.

    ud_io     CoNLL-U reader and corpus counts
    frames    tokens -> context signatures
    induce    counting, thresholds, ranking, de-duplication by construction
    minimal   frame -> minimal prefix + diagnostic continuation
    tagging   spaCy checks on probe items
    validate  purity on held-out and external corpora
    overrides manual accept / rewrite / reject of items
"""

from . import frames, induce, minimal, overrides, tagging, ud_io, validate  # noqa: F401

__all__ = ["ud_io", "frames", "induce", "minimal", "tagging", "validate",
           "overrides"]
