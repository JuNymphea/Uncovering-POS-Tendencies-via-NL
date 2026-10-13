#!/usr/bin/env python
"""Task 2: surprisal of POS-selective slot continuations, and a POS call from it.

For word w and slot f, s_f(w) is the mean surprisal of the slot's diagnostic continuation
after the prefix with w in the slot. On the calibration words each slot gets two Gaussians,
one for words of its POS and one for the other words; w is scored per POS by the mean
log-likelihood ratio over that POS's slots, and the call is the highest.

    # 1. real words of the model (fits the Gaussians, prints the final-test accuracy)
    python scripts/Tasks/Task_2/task2.py --model google/gemma-3-4b-it \
        --slots scripts/Tasks/Task_2/out/slots3_gemma.json --out <ref>
    # 2. trained vectors, with the fits from step 1
    python scripts/Tasks/Task_2/task2.py --model google/gemma-3-4b-it \
        --slots scripts/Tasks/Task_2/out/slots3_gemma.json --reference <ref> \
        --vectors <dir>/*.pt --out <file>

Add --lang zh for Chinese (slots in out_zh/slots3_<m>.json). Words come from
out/shared_{calibration,probedev,finaltest}.json (English) or out_zh/control3_*.json
(Chinese). Step 1 on the candidate pool (out/candidates.json) gives the surprisal
file that select_slots.py reads.
"""

import argparse
import json
import math
import statistics as st
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
CATS = ("noun", "verb", "adj")
UPPER = {"noun": "NOUN", "verb": "VERB", "adj": "ADJ"}
ROLES = ("calibration", "probedev", "finaltest")
WORDS = {"en": HERE / "out" / "shared_{}.json", "zh": HERE / "out_zh" / "control3_{}.json"}
NEOLOGISM = "{NEOLOGISM}"
NEW_TOKEN = "~jdsglmdh"


def detokenize(tokens):
    """Join tokens with English spacing."""
    no_space_before = {".", ",", "!", "?", ";", ":", "n't", "'s", "'re", "'ll", "'ve", "'d", "'m"}
    out = ""
    for i, tok in enumerate(tokens):
        # a token ending in a hyphen (not a bare "-") attaches to the next one
        glue = tok in no_space_before or (i and len(tokens[i - 1]) > 1 and tokens[i - 1].endswith("-"))
        out = tok if i == 0 else out + ("" if glue else " ") + tok
    return out


def render_prefix(prefix_tokens, word):
    """Fill the slot with `word`, fix a/an to match it, and capitalize the first token."""
    toks = [word if t == NEOLOGISM else t for t in prefix_tokens]
    for i, tok in enumerate(toks[:-1]):
        if tok.lower() in {"a", "an"}:
            fixed = "an" if toks[i + 1][:1].lower() in set("aeiou") else "a"
            toks[i] = fixed.capitalize() if tok[0].isupper() else fixed
    text = detokenize(toks)
    return text[0].upper() + text[1:] if text and toks[0] != NEOLOGISM else text


class Scorer:
    def __init__(self, model_name, lang):
        self.lang = lang
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16 if self.device != "cpu" else torch.float32,
            device_map=None).to(self.device).eval()
        self.tok.padding_side = "right"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    @torch.no_grad()
    def en(self, prefixes, continuation):
        """Mean surprisal of `continuation` after each prefix (English, from text)."""
        tok = self.tok
        # the continuation is tokenized together with the prefix, since alone it can split differently
        sep = "" if continuation[:1] in ",.;:" else " "
        enc = tok([p + sep + continuation for p in prefixes], return_tensors="pt", padding=True,
                  add_special_tokens=True).to(self.device)
        pre = tok(prefixes, add_special_tokens=True)["input_ids"]
        logits = self.model(**enc).logits.float().log_softmax(-1)
        ids, mask = enc["input_ids"], enc["attention_mask"]
        out = []
        for i in range(len(prefixes)):
            start, end = len(pre[i]), int(mask[i].sum())
            if end <= start:
                out.append(float("nan"))
                continue
            lp = [logits[i, t - 1, ids[i, t]].item() for t in range(start, end)]
            out.append(-sum(lp) / len(lp))
        return out

    def zh_parts(self, item):
        """Left-context ids (with BOS) and continuation ids of a Chinese slot."""
        pt = item["prefix_tokens"]
        assert pt[-1] == NEOLOGISM and pt.count(NEOLOGISM) == 1, pt
        left = self.tok("".join(pt[:-1]), add_special_tokens=True)["input_ids"]
        cont = self.tok("".join(item["diagnostic"]), add_special_tokens=False)["input_ids"]
        return left, cont

    @torch.no_grad()
    def zh(self, left, word_ids, cont):
        """Mean surprisal of `cont` after left + [word], for each word id (Chinese, from ids)."""
        seqs = [left + [w] + cont for w in word_ids]
        width = max(len(x) for x in seqs)
        ids = torch.full((len(seqs), width), self.tok.pad_token_id, dtype=torch.long)
        att = torch.zeros_like(ids)
        for i, x in enumerate(seqs):
            ids[i, :len(x)] = torch.tensor(x)
            att[i, :len(x)] = 1
        ids, att = ids.to(self.device), att.to(self.device)
        logits = self.model(input_ids=ids, attention_mask=att).logits.float().log_softmax(-1)
        start = len(left) + 1
        out = []
        for i in range(len(seqs)):
            lp = [logits[i, t - 1, ids[i, t]].item() for t in range(start, start + len(cont))]
            out.append(-sum(lp) / len(lp))
        return out

    def words(self, slots, words, batch_size):
        """Surprisal of every real word in every slot."""
        entries = [w for role in ROLES for c in CATS for w in words[role][c]]
        if self.lang == "zh":
            word_id = {}
            for w in entries:
                e = self.tok(w, add_special_tokens=False)["input_ids"]
                assert len(e) == 1, f"{w} is not one token for this model"
                word_id[w] = e[0]
        S = {}
        for n, (key, item) in enumerate(slots.items(), 1):
            S[key] = {}
            if self.lang == "zh":
                left, cont = self.zh_parts(item)
            for i in range(0, len(entries), batch_size):
                chunk = entries[i:i + batch_size]
                if self.lang == "zh":
                    vals = self.zh(left, [word_id[w] for w in chunk], cont)
                else:
                    vals = self.en([render_prefix(item["prefix_tokens"], w) for w in chunk],
                                   " ".join(item["diagnostic"]))
                S[key].update(zip(chunk, vals))
            print(f"  [{n}/{len(slots)}] {key}", flush=True)
        return S

    def vectors(self, slots, paths):
        """Surprisal of every vector in every slot; each is written into the new token's row."""
        tok, model = self.tok, self.model
        tok.add_tokens([NEW_TOKEN])
        new_id = tok.convert_tokens_to_ids(NEW_TOKEN)
        if new_id >= model.get_input_embeddings().weight.size(0):
            model.resize_token_embeddings(len(tok))
        # untie lm_head so that only the input row changes, as in training
        model.config.tie_word_embeddings = False
        head = model.get_output_embeddings()
        head.weight = nn.Parameter(head.weight.clone())
        emb = model.get_input_embeddings().weight
        S = {key: {} for key in slots}
        for path in paths:
            with torch.no_grad():
                emb[new_id] = torch.load(path, map_location="cpu")["embedding"].to(emb.device, emb.dtype)
            for key, item in slots.items():
                if self.lang == "zh":
                    left, cont = self.zh_parts(item)
                    S[key][path.stem] = self.zh(left, [new_id], cont)[0]
                else:
                    pre = render_prefix(item["prefix_tokens"], NEW_TOKEN)
                    S[key][path.stem] = self.en([pre], " ".join(item["diagnostic"]))[0]
            print(f"  {path.stem}", flush=True)
        return S


def gaussian_logpdf(x, mu, sigma):
    sigma = max(sigma, 1e-6)
    return -0.5 * math.log(2 * math.pi * sigma * sigma) - (x - mu) ** 2 / (2 * sigma * sigma)


def fit(S, calib):
    """Matched / non-matched Gaussians per slot, from the calibration words."""
    fits = {}
    for key, scores in S.items():
        pos = key.split("::")[0].lower()
        matched = [scores[w] for w in calib[pos] if w in scores and not math.isnan(scores[w])]
        other = [scores[w] for p in CATS if p != pos for w in calib[p]
                 if w in scores and not math.isnan(scores[w])]
        if len(matched) < 2 or len(other) < 2:
            continue
        fits[key] = {"pos": pos, "mu_match": st.mean(matched), "sigma_match": st.stdev(matched),
                     "mu_nonmatch": st.mean(other), "sigma_nonmatch": st.stdev(other)}
    return fits


def llr(S, fits, w):
    """Mean log-likelihood ratio per POS over that POS's slots."""
    means = {}
    for p in CATS:
        rs = []
        for key, f in fits.items():
            if f["pos"] != p or w not in S[key] or math.isnan(S[key][w]):
                continue
            s = S[key][w]
            rs.append(gaussian_logpdf(s, f["mu_match"], f["sigma_match"])
                      - gaussian_logpdf(s, f["mu_nonmatch"], f["sigma_nonmatch"]))
        if rs:
            means[p] = sum(rs) / len(rs)
    return means


def call(means):
    return max(means, key=means.get) if means else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lang", default="en", choices=("en", "zh"))
    ap.add_argument("--slots", type=Path, required=True)
    ap.add_argument("--vectors", nargs="+", type=Path, help="trained vectors (.pt); the file name is the label")
    ap.add_argument("--reference", type=Path, help="step-1 output for the same model and slots (needed with --vectors)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.vectors and not args.reference:
        ap.error("--vectors needs --reference")

    raw = json.loads(args.slots.read_text(encoding="utf-8"))
    slots = {f"{pos}::{it['signature']}": dict(it, pos=pos)
             for pos in raw if pos in UPPER.values() for it in raw[pos]}
    scorer = Scorer(args.model, args.lang)

    if args.vectors:
        fits = json.loads(args.reference.read_text(encoding="utf-8"))["fits"]
        S = scorer.vectors({k: slots[k] for k in fits}, args.vectors)
        scores = {p.stem: llr(S, fits, p.stem) for p in args.vectors}
        result = {"model": args.model, "lang": args.lang, "surprisal": S, "scores": scores,
                  "calls": {k: call(m) for k, m in scores.items()}}
    else:
        words = {r: json.loads(Path(str(WORDS[args.lang]).format(r)).read_text(encoding="utf-8"))["words"]
                 for r in ROLES}
        S = scorer.words(slots, words, args.batch_size)
        fits = fit(S, words["calibration"])
        acc = {}
        for c in CATS:
            hits = sum(call(llr(S, fits, w)) == c for w in words["finaltest"][c])
            acc[c] = {"n": len(words["finaltest"][c]), "correct": hits,
                      "accuracy": round(hits / len(words["finaltest"][c]), 4)}
        acc["overall"] = round(sum(a["correct"] for a in acc.values()) / sum(a["n"] for a in acc.values()), 4)
        print("final-test accuracy " + "  ".join(f"{c} {acc[c]['accuracy']:.3f}" for c in CATS)
              + f"  overall {acc['overall']:.3f}")
        example = (lambda it: "".join(it["prefix_tokens"]).replace(NEOLOGISM, "□") + "".join(it["diagnostic"])) \
            if args.lang == "zh" else \
            (lambda it: render_prefix(it["prefix_tokens"], "WORD") + " " + " ".join(it["diagnostic"]))
        result = {"model": args.model, "lang": args.lang,
                  "slots": [{"pos": it["pos"], "signature": it["signature"], "diagnostic": it["diagnostic"],
                             "prefix": " ".join(it["prefix_tokens"]), "example": example(it)}
                            for it in slots.values()],
                  "words": words, "candidate_order": list(slots), "surprisal": S,
                  "fits": fits, "accuracy": acc}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
