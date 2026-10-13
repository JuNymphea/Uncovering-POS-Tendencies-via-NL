"""Manual review of realized probe items.

Each entry in the review file (realizations.json) has a status:

    accept   use the generated item
    rewrite  use the hand-written `text` instead; it must keep the frame and {NEOLOGISM}
    reject   drop the frame; selection moves on to the next candidate
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

NEOLOGISM = "{NEOLOGISM}"


def _scoreable(token: str) -> bool:
    return token.replace("'", "").replace("’", "").isalpha()


@dataclass
class Overrides:
    entries: Dict[str, dict] = field(default_factory=dict)
    path: Path | None = None

    @classmethod
    def load(cls, path: Path | None) -> "Overrides":
        if path is None or not Path(path).exists():
            return cls(path=Path(path) if path else None)
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        entries = {e["signature"]: e for e in data.get("slots", []) if "signature" in e}
        return cls(entries=entries, path=Path(path))

    def status(self, signature: str) -> str:
        return self.entries.get(signature, {}).get("status", "accept")

    def rewritten(self, signature: str, n_continuation: int = 4) -> dict | None:
        """A hand-supplied realization, in the same shape as a generated one."""
        entry = self.entries.get(signature)
        if not entry or entry.get("status") != "rewrite":
            return None
        text = entry.get("text", "").strip()
        if NEOLOGISM not in text:
            raise ValueError(
                f"rewrite for {signature!r} must contain {NEOLOGISM}: {text!r}"
            )
        tokens = text.replace(".", " .").replace(",", " ,").split()
        i = next(j for j, tk in enumerate(tokens) if NEOLOGISM in tk)
        continuation = [tk for tk in tokens[i + 1:] if _scoreable(tk)][:n_continuation]
        return {
            "text": text,
            "tokens": tokens,
            "continuation": continuation,
            "opener": [],
            "tail": [],
            "tagger_accuracy": None,
            "tagger_accuracy_other": {},
            "tagger_margin": None,
            "n_content_words": None,
            "source": "human_rewrite",
            "note": entry.get("note", ""),
        }

    def rejected(self) -> List[dict]:
        return [e for e in self.entries.values() if e.get("status") == "reject"]


def write_review_template(path: Path, realized: Dict[str, List[dict]]) -> None:
    """Emit an editable file pre-filled with the current realizations."""
    slots = []
    for pos, rows in realized.items():
        for r in rows:
            b = r.get("probe_item") or {}
            slots.append({
                "pos": pos,
                "signature": r["signature"],
                "purity": r["purity"],
                "n": r["n_target"],
                "generated": b.get("text", ""),
                "status": "accept",
                "text": b.get("text", ""),
                "note": "",
            })
    path.write_text(
        json.dumps(
            {
                "_doc": "status: accept | rewrite | reject. "
                        "rewrite -> edit `text`, keeping the frame and {NEOLOGISM}. "
                        "reject -> frame is dropped, next corpus candidate takes its "
                        "place. Re-run run_induction.py after editing.",
                "slots": slots,
            },
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
