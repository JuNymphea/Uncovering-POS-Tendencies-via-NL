"""Generate answers with a trained neologism vector, for the LLM judge (scripts/eval/evaluate.py).

For each of the first 100 AxBench eval questions, the model answers the plain question
(normal_answer) and the question followed by the training template with the new token
(concept_answer), both inside the model's chat template. Concept, template and language
are read from the vector file. Writes <out_dir>/<model>_<concept>_<template>.jsonl
(<model>_zh_... for Chinese); an interrupted run resumes where it stopped.

    python scripts/train/eval_neologism.py --model google/gemma-3-4b-it --vector <dir>/n_1_unbiased.pt
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_neologism import CODES, NEW_TOKEN, TEMPLATES, TEMPLATES_ZH  # noqa: E402

QUESTIONS = HERE.parent / "axbench" / "data" / "axbench_eval_filtered.{}.jsonl"
N_QUESTIONS = 100
MAX_NEW_TOKENS = 4096


def prompts(questions, template, lang):
    """Question + template; "mixed" gives each question one of verb/noun/adj, a third each."""
    names = [template] * len(questions)
    if template == "mixed":
        order = list(range(len(questions)))
        random.Random(42).shuffle(order)
        for rank, i in enumerate(order):
            names[i] = ("verb", "noun", "adj")[rank % 3]
    templates, sep = (TEMPLATES_ZH, "") if lang == "zh" else (TEMPLATES, " ")
    return [q + sep + templates[CODES[t]].replace("{NEOLOGISM}", NEW_TOKEN) for q, t in zip(questions, names)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the base model the vector was trained in")
    ap.add_argument("--vector", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, default=HERE / "results")
    ap.add_argument("--batch_size", type=int, default=8, help="questions per step; each costs two generations")
    args = ap.parse_args()

    meta = torch.load(args.vector, map_location="cpu")
    concept, template, lang = meta["concept"], meta["template"], meta["lang"]
    short = next((m for m in ("gemma", "qwen", "aya") if m in args.model.lower()), Path(args.model).name)
    out = args.out_dir / f"{short}{'_zh' if lang == 'zh' else ''}_{concept}_{template}.jsonl"

    with open(str(QUESTIONS).format(lang), encoding="utf-8") as f:
        questions = [str(v) for line in f if line.strip() for v in json.loads(line).values()][:N_QUESTIONS]
    concept_prompts = prompts(questions, template, lang)

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tok.add_tokens([NEW_TOKEN])
    new_id = tok.convert_tokens_to_ids(NEW_TOKEN)
    assert new_id == meta["new_token_id"], "the vector was trained with another tokenizer"
    # the class AutoModelForCausalLM picks for this config (e.g. Gemma3ForConditionalGeneration)
    model_cls = AutoModelForCausalLM._model_mapping[type(AutoConfig.from_pretrained(args.model))]
    model = model_cls.from_pretrained(args.model, device_map="auto", dtype=torch.bfloat16,
                                      attn_implementation="sdpa")
    # untie lm_head so that only the input row changes, as in training
    model.config.tie_word_embeddings = False
    head = model.get_output_embeddings()
    head.weight = nn.Parameter(head.weight.clone())
    emb = model.get_input_embeddings().weight
    if new_id >= emb.size(0):
        model.resize_token_embeddings(new_id + 1)
        emb = model.get_input_embeddings().weight
    with torch.no_grad():
        emb[new_id] = meta["embedding"].to(device=emb.device, dtype=emb.dtype)
    model.eval()

    # left padding, so generation starts right after each prompt
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model.generation_config.pad_token_id = tok.pad_token_id

    @torch.no_grad()
    def generate(texts):
        inputs = tok.apply_chat_template([[{"role": "user", "content": t}] for t in texts],
                                         add_generation_prompt=True, return_tensors="pt", return_dict=True,
                                         padding=True, enable_thinking=False).to(model.device)
        n = inputs["input_ids"].shape[1]
        ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, use_cache=True)
        return [tok.decode(row[n:], skip_special_tokens=True).strip() for row in ids]

    done = [json.loads(l) for l in open(out, encoding="utf-8") if l.strip()] if out.exists() else []
    seen = {r["question"] for r in done}
    todo = [i for i in range(len(questions)) if questions[i] not in seen]
    out.parent.mkdir(parents=True, exist_ok=True)
    for start in tqdm(range(0, len(todo), args.batch_size), desc=out.stem):
        idx = todo[start:start + args.batch_size]
        concept_answers = generate([concept_prompts[i] for i in idx])
        normal_answers = generate([questions[i] for i in idx])
        for i, c, n in zip(idx, concept_answers, normal_answers):
            done.append({"question": questions[i], "normal_answer": n, "concept_answer": c})
        # rewritten after every batch so that an interrupted run can resume
        with open(out, "w", encoding="utf-8") as f:
            for r in done:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
