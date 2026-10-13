"""Pick the FrameNet frames whose lexical units all share one part of speech, and describe them.

A frame is kept if it has at least 5 lexical units (LUs) and all of them are nouns, all verbs
or all adjectives. GPT-5.2 then writes a short "concept of ..." description from up to 30 of
its LU words. Writes one CSV row per kept frame (POS, FrameNet, LUs, gpt); the 30 concepts
in scripts/train/data/concepts.csv were chosen from these.

    OPENAI_API_KEY=... python scripts/FrameNet/get_concept_from_framenet.py --out <file>
"""
import argparse
import csv
import time

import nltk
from nltk.corpus import framenet as fn
from openai import OpenAI

MODEL_NAME = "gpt-5.2-2025-12-11"
POS = {"n": "NOUN", "v": "VERB", "a": "ADJ"}

CONCEPT_DESCRIPTION_PROMPT = """You are analyzing the common semantic feature that links a set of words.

Given a list of words, identify the minimal shared semantic concept that unifies them.

Output format:
- Your output must always have exactly the following form:
  "concept of {SEMANTIC CONTENT}"
- Replace {SEMANTIC CONTENT} with a short phrase describing only the shared meaning.
- Keep the semantic content concise and specific.

Constraints:
- Output only one short phrase.
- Always begin with "concept of".
- Do not include any explanation.
- Do not mention grammatical categories such as noun, verb, or adjective.
- Do not describe how the words are grammatically used.
- Do not use generic category-marking expressions such as:
  "items of", "things", "actions", "acts", "processes", "ways of",
  "methods of", "properties", "qualities", "degree of", or "state of".
- Avoid wording that explicitly frames the concept as an entity, event, action,
  property, quality, degree, or state.
- Avoid beginning the semantic content with a gerund (e.g., "making", "moving",
  "speaking", "having").
- Describe the shared meaning as directly as possible while preserving the
  semantic distinction among concepts.

Examples of the intended style:
- "concept of clothing worn on the body"
- "concept of force-induced movement of an object"
- "concept of permissiveness in rule enforcement"
- "concept of decomposition and spoilage"

Important:
Your output must encode semantic content only. The surface form should be kept
as uniform as possible across different sets of words so that the wording itself
does not reveal their likely grammatical category.

List of words: {WORDSLIST}"""


def single_pos_frames():
    """(pos, frame name, number of LUs, up to 30 LU words) for every frame that passes."""
    try:
        fn.frames()
    except LookupError:
        nltk.download("framenet_v17")
    out = []
    for frame in fn.frames():
        lus = frame.lexUnit
        if len(lus) < 5:
            continue
        words, tags = [], set()
        for lu_name in lus.keys():
            try:
                word, pos = lu_name.rsplit(".", 1)
            except ValueError:
                continue
            words.append(word)
            tags.add(pos)
        if len(tags) == 1 and next(iter(tags)) in POS and len(words) == len(lus):
            out.append((POS[next(iter(tags))], frame.name, len(lus), words[:30]))
    return out


def call_openai_gpt(client, prompt, retries=3, delay=2):
    """One chat completion with retries; returns "[Error: ...]" if every attempt fails."""
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": ""}, {"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"attempt {attempt} failed: {e}")
            if attempt < retries:
                time.sleep(delay)
            else:
                return f"[Error: {e}]"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    frames = single_pos_frames()
    print(f"{len(frames)} frames: " + ", ".join(f"{p} {sum(f[0] == p for f in frames)}" for p in POS.values()))
    client = OpenAI()
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["POS", "FrameNet", "LUs", "gpt"])
        for pos, name, n, words in frames:
            desc = call_openai_gpt(client, CONCEPT_DESCRIPTION_PROMPT.replace("{WORDSLIST}", ", ".join(words)))
            print(f"{pos:<5}{name:<35}{desc}")
            w.writerow([pos, name, n, desc])
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
