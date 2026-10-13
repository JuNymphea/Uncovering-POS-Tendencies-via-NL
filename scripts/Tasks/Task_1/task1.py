#!/usr/bin/env python
"""Task 1: ask the model for the part of speech of the new token (or of real words).

Prompt: three shots, then 'The part of speech of the word "WORD" is'. A category's score is
the summed probability of its answers (" noun", " a noun", " adjective", " adj", ...),
normalized over noun / verb / adj and averaged over the six orders of the shots. Chinese
uses Chinese shots and labels. The call is the category with the highest score.

    # trained vectors, written into the new token's embedding row one at a time
    python scripts/Tasks/Task_1/task1.py --model google/gemma-3-4b-it --vectors <dir>/*.pt --out <file>
    # real words with known POS (prints the accuracy)
    python scripts/Tasks/Task_1/task1.py --model google/gemma-3-4b-it \
        --words scripts/Tasks/Task_1/task1_words.json --out <file>

Add --lang zh for Chinese (real words: scripts/Tasks/Task_2/out_zh/zh_words3.json).
"""

import argparse
import itertools
import json
import math
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

CATS = ("noun", "verb", "adj")
NEW_TOKEN = "~jdsglmdh"

PROMPT = {"en": 'The part of speech of the word "{WORD}" is',
          "zh": "词语“{WORD}”的词性是"}
SHOTS = {"en": [("table", "noun"), ("walk", "verb"), ("happy", "adjective")],
         "zh": [("桌子", "名词"), ("喝", "动词"), ("高兴", "形容词")]}
# the answers whose probabilities are summed for each category
ANSWERS = {"en": {"noun": [" a noun", " noun"], "verb": [" a verb", " verb"],
                  "adj": [" adj", " adjective", " an adj", " an adjective"]},
           "zh": {"noun": ["名词"], "verb": ["动词"], "adj": ["形容词"]}}


def en_prompt(word, shots):
    p = PROMPT["en"]
    return "".join(f'{p.replace("{WORD}", w)} {a}.\n' for w, a in shots) + p.replace("{WORD}", word)


def zh_ids(tok, word, shots):
    # built from token ids so that no merge crosses the word boundary
    p = PROMPT["zh"]
    left, right = p.split("{WORD}")
    head = "".join(f'{p.replace("{WORD}", w)}{a}。\n' for w, a in shots) + left
    return (tok(head, add_special_tokens=True)["input_ids"]
            + tok(word, add_special_tokens=False)["input_ids"]
            + tok(right, add_special_tokens=False)["input_ids"])


@torch.no_grad()
def score(tok, model, lang, words, forms, shots, batch_size):
    """log P(answer | prompt about word) for every word and answer."""
    out = {w: {} for w in words}
    rows = [(w, v) for w in words for v in forms]
    pad = tok.pad_token_id
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        if lang == "en":
            prompts = [en_prompt(w, shots) for w, _ in chunk]
            enc = tok([p + v for p, (_, v) in zip(prompts, chunk)], return_tensors="pt",
                      padding=True, add_special_tokens=True).to(model.device)
            lg = model(**enc).logits.float().log_softmax(-1)
            ids = enc["input_ids"]
            starts = [len(x) for x in tok(prompts, add_special_tokens=True)["input_ids"]]
            ends = [int(m.sum()) for m in enc["attention_mask"]]
        else:
            pre = [zh_ids(tok, w, shots) for w, _ in chunk]
            seqs = [p + tok(v, add_special_tokens=False)["input_ids"] for p, (_, v) in zip(pre, chunk)]
            n = max(len(x) for x in seqs)
            ids = torch.tensor([x + [pad] * (n - len(x)) for x in seqs], device=model.device)
            mask = torch.tensor([[1] * len(x) + [0] * (n - len(x)) for x in seqs], device=model.device)
            lg = model(input_ids=ids, attention_mask=mask).logits.float().log_softmax(-1)
            starts, ends = [len(x) for x in pre], [len(x) for x in seqs]
        for j, (w, v) in enumerate(chunk):
            out[w][v] = sum(lg[j, t - 1, ids[j, t]].item() for t in range(starts[j], ends[j]))
    return out


def probabilities(per_order, answers):
    """Category probabilities for each shot order, averaged over the orders."""
    parts = []
    for logps in per_order:
        mass = {c: sum(math.exp(logps[v]) for v in answers[c]) for c in CATS}
        total = sum(mass.values()) or 1.0
        parts.append({c: mass[c] / total for c in CATS})
    return {c: sum(p[c] for p in parts) / len(parts) for c in CATS}


def call(p):
    return max(p, key=p.get)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lang", default="en", choices=("en", "zh"))
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--vectors", nargs="+", type=Path, help="trained vectors (.pt); the file name is the label")
    src.add_argument("--words", type=Path, help='{"words": {"noun": [...], "verb": [...], "adj": [...]}}')
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32).to(device).eval()

    answers = ANSWERS[args.lang]
    forms = sorted({v for c in CATS for v in answers[c]})
    orders = sorted(itertools.permutations(SHOTS[args.lang]))
    result = {"model": args.model, "lang": args.lang}

    if args.vectors:
        tok.add_tokens([NEW_TOKEN])
        new_id = tok.convert_tokens_to_ids(NEW_TOKEN)
        if new_id >= model.get_input_embeddings().weight.size(0):
            model.resize_token_embeddings(len(tok))
        # untie lm_head so that only the input row changes, as in training
        model.config.tie_word_embeddings = False
        head = model.get_output_embeddings()
        head.weight = nn.Parameter(head.weight.clone())
        emb = model.get_input_embeddings().weight
        probs = {}
        for path in args.vectors:
            with torch.no_grad():
                emb[new_id] = torch.load(path, map_location="cpu")["embedding"].to(emb.device, emb.dtype)
            per_order = [score(tok, model, args.lang, [NEW_TOKEN], forms, order, args.batch_size)[NEW_TOKEN]
                         for order in orders]
            probs[path.stem] = probabilities(per_order, answers)
            print(f"{path.stem}: {call(probs[path.stem])}", flush=True)
        result.update(probabilities=probs, calls={k: call(p) for k, p in probs.items()})
    else:
        words = json.loads(args.words.read_text(encoding="utf-8"))["words"]
        per_order = [{c: score(tok, model, args.lang, words[c], forms, order, args.batch_size) for c in CATS}
                     for order in orders]
        probs = {c: {w: probabilities([lp[c][w] for lp in per_order], answers) for w in words[c]} for c in CATS}
        acc = {}
        for c in CATS:
            hits = sum(call(p) == c for p in probs[c].values())
            acc[c] = {"n": len(probs[c]), "correct": hits, "accuracy": round(hits / len(probs[c]), 4)}
        acc["overall"] = round(sum(a["correct"] for a in acc.values()) / sum(a["n"] for a in acc.values()), 4)
        print("accuracy " + "  ".join(f"{c} {acc[c]['accuracy']:.3f}" for c in CATS) + f"  overall {acc['overall']:.3f}")
        result.update(probabilities=probs, accuracy=acc)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
