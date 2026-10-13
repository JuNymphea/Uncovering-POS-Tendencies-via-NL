#!/usr/bin/env python
"""Download the three models' tokenizer.json files used by the word-list scripts.

vocab_overlap.py, build_control_words.py and zh/build_control_words_zh.py read
these to test single-token words. Files are saved to scripts/Tasks/tokenizers/
under the names those scripts expect.

    python scripts/Tasks/Task_2/get_tokenizers.py
"""
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

OUT = Path(__file__).resolve().parents[1] / "tokenizers"
MODELS = {"gemma3-tokenizer.json": "google/gemma-3-4b-it",
          "Qwen3-4B-tokenizer.json": "Qwen/Qwen3-4B",
          "tiny-aya-global-tokenizer.json": "CohereLabs/tiny-aya-global"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    failed = []
    for name, repo in MODELS.items():
        try:
            shutil.copy(hf_hub_download(repo, "tokenizer.json"), OUT / name)
            print(f"{repo} -> {OUT / name}")
        except Exception as exc:          # gated repos need `huggingface-cli login` and accepted terms
            failed.append(repo)
            print(f"!! {repo}: {exc.__class__.__name__}: {exc}")
    if failed:
        raise SystemExit(f"could not fetch {failed}; log in with `huggingface-cli login` and accept the model terms")


if __name__ == "__main__":
    main()
