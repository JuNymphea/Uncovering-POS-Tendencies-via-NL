"""
Extract unambiguous noun/verb/adj bare words from a tokenizer's vocabulary.

Filters out inflected forms (plurals, -ing, -ed, -er, -est, -ly, etc.)
by checking that the word equals its own lemma in WordNet. Also writes
<output>_ambiguous.json and <output>_sampled.json (100 per POS, seed 42); with
--output scripts/Tasks/pos_words.json the latter replaces the shipped
scripts/Tasks/pos_words_sampled.json.

Usage:
    python scripts/Tasks/Task_2/extract_pos_words.py --tokenizer_path google/gemma-3-4b-it \
        --output scripts/Tasks/pos_words.json
    python scripts/Tasks/Task_2/extract_pos_words.py \
        --tokenizer_json scripts/Tasks/tokenizers/gemma3-tokenizer.json --output scripts/Tasks/pos_words.json
    python scripts/Tasks/Task_2/extract_pos_words.py --tokenizer_path google/gemma-3-4b-it --debug
"""

import argparse
import json
import random
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

lemmatizer = WordNetLemmatizer()


def is_bare_form(word: str) -> bool:
    """True if the word is not an inflected or derived form.

    Rejects words that WordNet lemmatizes to a different real word, and words
    with a common suffix whose stem is a real word.
    """
    # Check via lemmatization: if lemmatizing changes the word, it's inflected
    for wn_pos in [wn.NOUN, wn.VERB, wn.ADJ]:
        lemma = lemmatizer.lemmatize(word, pos=wn_pos)
        if lemma != word and len(lemma) >= 3:
            # only counts if the lemma is a real word
            if wn.synsets(lemma):
                return False

    # Additional suffix filters for forms that WordNet lemmatizer might miss
    inflectional_suffixes = [
        "ing",    # running, eating
        "ed",     # walked, played
        "er",     # bigger, walker
        "est",    # biggest, fastest
        "ly",     # quickly, beautifully
        "ies",    # babies, countries
        "ness",   # happiness, darkness
        "ment",   # movement, agreement
        "tion",   # action, creation
        "sion",   # decision, tension
        "ous",    # dangerous, famous
        "ive",    # active, creative
        "ful",    # beautiful, wonderful
        "less",   # hopeless, careless
        "able",   # comfortable, readable
        "ible",   # possible, visible
    ]

    # reject only if the stem is a real word, so "table" is kept
    for suffix in inflectional_suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            stem = word[:-len(suffix)]
            # Check common stem adjustments
            candidates = [
                stem,
                stem + "e",      # create -> creating
                stem + stem[-1], # run -> running (doubled consonant)
                stem + "y",      # happy -> happiness
            ]
            for candidate in candidates:
                if candidate != word and wn.synsets(candidate):
                    return False

    return True


def get_unique_pos(word: str) -> str | None:
    """'noun', 'verb' or 'adj' if WordNet gives the word exactly one of them, else None."""
    synsets = wn.synsets(word)
    if not synsets:
        return None

    pos_map = {"n": "noun", "v": "verb", "a": "adj", "s": "adj"}
    target_pos = set()

    for s in synsets:
        p = s.pos()
        if p in pos_map:
            target_pos.add(pos_map[p])

    if len(target_pos) == 1:
        return target_pos.pop()

    return None


def clean_token(token: str) -> str | None:
    """Strip tokenizer-specific prefixes."""
    cleaned = token
    cleaned = cleaned.lstrip("\u2581")  # ▁ SentencePiece
    cleaned = cleaned.lstrip("\u0120")  # Ġ GPT-2
    cleaned = cleaned.strip()
    return cleaned if cleaned else None


def is_valid_word(word: str, min_length: int = 3) -> bool:
    """Alphabetic, lowercase as it appears in the vocabulary, and at least `min_length` long."""
    if len(word) < min_length:
        return False
    if not word.isalpha():
        return False
    if not word.islower():
        return False
    return True


def is_single_token(word: str, tokenizer, leading_space: bool = True) -> bool:
    """True if `word` (after a space by default) encodes to a single token."""
    if tokenizer is None:
        return True
    text = (" " + word) if leading_space else word
    return len(tokenizer.encode(text, add_special_tokens=False).ids) == 1


def is_proper_noun(word: str) -> bool:
    """True if every WordNet synset of the word is a capitalized name (e.g. "jupiter")."""
    synsets = wn.synsets(word)
    if not synsets:
        return False
    return all(s.lemmas()[0].name()[0].isupper() for s in synsets)


def load_tokenizer(json_path: str | None, pretrained_path: str | None):
    """A tokenizer object for the single-token test (not just its vocab list)."""
    if json_path:
        from tokenizers import Tokenizer
        return Tokenizer.from_file(json_path)
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(pretrained_path, use_fast=True)
    return tk.backend_tokenizer


def load_vocab_from_tokenizer_json(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tokens = set()

    if "model" in data and isinstance(data["model"].get("vocab"), dict):
        tokens.update(data["model"]["vocab"].keys())
    elif "model" in data and isinstance(data["model"].get("vocab"), list):
        for item in data["model"]["vocab"]:
            if isinstance(item, list) and len(item) >= 1:
                tokens.add(item[0])

    if "added_tokens" in data:
        for item in data["added_tokens"]:
            tokens.add(item["content"])

    return list(tokens)


def load_vocab_from_pretrained(path: str) -> list:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    return list(tokenizer.get_vocab().keys())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_json", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--output", type=str, default="pos_words.json")
    parser.add_argument("--min_length", type=int, default=4)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no_leading_space", action="store_true",
                        help="test single-tokenhood without a leading space "
                             "(only if the probe puts the word sentence-initially)")
    args = parser.parse_args()

    if args.tokenizer_json is None and args.tokenizer_path is None:
        parser.error("Provide either --tokenizer_json or --tokenizer_path")

    # Load vocabulary
    if args.tokenizer_json:
        print(f"Loading vocab from {args.tokenizer_json}")
        tokens = load_vocab_from_tokenizer_json(args.tokenizer_json)
    else:
        print(f"Loading vocab from {args.tokenizer_path}")
        tokens = load_vocab_from_pretrained(args.tokenizer_path)

    print(f"Total tokens in vocabulary: {len(tokens)}")

    tokenizer = load_tokenizer(args.tokenizer_json, args.tokenizer_path)

    # Debug
    if args.debug:
        print("\nTokens that look like words (first 30 found):")
        count = 0
        for t in tokens:
            cleaned = clean_token(t)
            if cleaned:
                word = cleaned
                if is_valid_word(word, args.min_length):
                    bare = is_bare_form(word)
                    pos = get_unique_pos(word) if bare else "INFLECTED"
                    print(f"  '{t}' -> '{word}' -> bare={bare} -> pos={pos}")
                    count += 1
                    if count >= 30:
                        break

    # Process
    results = {"noun": [], "verb": [], "adj": []}
    seen = set()
    skipped_multi_pos = {}
    skipped_inflected = []
    skipped_multi_token = []
    skipped_proper = []

    for i, token in enumerate(tokens):
        if (i + 1) % 50000 == 0:
            print(f"  Processing {i + 1}/{len(tokens)}...")

        cleaned = clean_token(token)
        if not cleaned:
            continue

        # check the token with its original casing; capitalized entries are skipped
        if not is_valid_word(cleaned, min_length=args.min_length):
            continue

        word = cleaned
        if word in seen:
            continue
        seen.add(word)

        # the probe puts the word after a space
        if not is_single_token(word, tokenizer, leading_space=not args.no_leading_space):
            skipped_multi_token.append(word)
            continue

        if is_proper_noun(word):
            skipped_proper.append(word)
            continue

        # Filter out inflected forms
        if not is_bare_form(word):
            skipped_inflected.append(word)
            continue

        pos = get_unique_pos(word)
        if pos is not None:
            results[pos].append(word)
        else:
            synsets = wn.synsets(word)
            if synsets:
                pos_map = {"n": "noun", "v": "verb", "a": "adj", "s": "adj"}
                multi_pos = set(pos_map.get(s.pos(), s.pos()) for s in synsets)
                multi_pos = multi_pos & {"noun", "verb", "adj"}
                if len(multi_pos) > 1:
                    skipped_multi_pos[word] = sorted(multi_pos)

    for pos in results:
        results[pos].sort()

    # Save full results
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults (bare forms only):")
    print(f"  Nouns:      {len(results['noun'])}")
    print(f"  Verbs:      {len(results['verb'])}")
    print(f"  Adjectives: {len(results['adj'])}")
    print(f"  Skipped (multi-POS):    {len(skipped_multi_pos)}")
    print(f"  Skipped (inflected):    {len(skipped_inflected)}")
    print(f"  Skipped (multi-token):  {len(skipped_multi_token)}")
    print(f"  Skipped (proper noun):  {len(skipped_proper)}")
    print(f"\nSaved to {args.output}")

    # Save ambiguous
    multi_pos_path = args.output.replace(".json", "_ambiguous.json")
    with open(multi_pos_path, "w", encoding="utf-8") as f:
        json.dump(skipped_multi_pos, f, indent=2, ensure_ascii=False)
    print(f"Ambiguous words saved to {multi_pos_path}")

    # Sample 100 per POS
    random.seed(42)
    sampled = {}
    for pos in ["noun", "verb", "adj"]:
        words = results[pos]
        if len(words) > 100:
            sampled[pos] = sorted(random.sample(words, 100))
        else:
            sampled[pos] = words

    sampled_path = args.output.replace(".json", "_sampled.json")
    with open(sampled_path, "w", encoding="utf-8") as f:
        json.dump(sampled, f, indent=2, ensure_ascii=False)
    print(f"Sampled words saved to {sampled_path}")

    # Print examples
    for pos in ["noun", "verb", "adj"]:
        examples = results[pos][:15]
        print(f"\n  {pos} examples: {', '.join(examples)}")

    if skipped_inflected:
        print(f"\n  inflected examples (skipped): {', '.join(skipped_inflected[:15])}")


if __name__ == "__main__":
    main()