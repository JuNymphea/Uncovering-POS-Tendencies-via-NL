#!/usr/bin/env python
"""Task 3: the part of speech of the synonyms the model generates for the new token.

The model is asked for five synonyms of the new token (both turns in its own chat template,
the answer prefilled up to the opening quote) and sampled five times per vector.
English: each synonym gets a POS distribution from WordNet (SemCor counts, sense counts when
there are none; spaCy for multi-word items); a vector's distribution is the mean over its
synonyms and the call is the highest category. Chinese: an LLM marks which frames each
synonym fits (noun, verb, adjective); a vector's score per frame is the share of its
synonyms that fit it. Tied vectors get five more samples per round, for up to five rounds
(noun and verb concepts also when noun equals verb); ties left after that are settled by the
synonyms' primary use, judged by the LLM among the tied categories.

    python scripts/Tasks/Task_3/task3.py --model google/gemma-3-4b-it --vectors <dir>/*.pt --out <file>
    OPENAI_API_KEY=... python scripts/Tasks/Task_3/task3.py --lang zh --model google/gemma-3-4b-it \
        --vectors <dir>/*.pt --out <file>
    # real words, each word's own embedding row in place of the new token (prints the accuracy)
    python scripts/Tasks/Task_3/task3.py --model google/gemma-3-4b-it \
        --words scripts/Tasks/Task_2/out/shared_finaltest.json --out <file>
    OPENAI_API_KEY=... python scripts/Tasks/Task_3/task3.py --lang zh --model google/gemma-3-4b-it \
        --words scripts/Tasks/Task_2/out_zh/zh_words3.json --out <file>

The LLM frame labels are cached per word (--frames), so reruns only send new words.
Vector file names should start with the concept id (n_1_..., v_3_...).
"""

import argparse
import asyncio
import collections
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
CATS = ("noun", "verb", "adj")
SHORT = {"noun": "n", "verb": "v", "adj": "a"}
NEW_TOKEN = "~jdsglmdh"
LLM = "gpt-5.2-2025-12-11"
ROUNDS = 5

USER = {"en": "Before you answer, give a list of 5 synonyms for {W}.",
        "zh": "在回答之前，请给出{W}的5个近义词。"}
PREFILL = {"en": "Ok, here's a list of 5 synonyms for {W}:“",
           "zh": "好的，以下是{W}的5个近义词：“"}

# thinking-mode openers a template may append after the generation prompt; closed here
# so that the prefill starts the answer
THINKING = (("<think>", "</think>"),
            ("<|START_THINKING|>", "<|END_THINKING|>"))

# Cohere's template adds a default system preamble when there is no system turn;
# it is removed so every model gets the same bare user turn
SYSTEM_TURN = ("<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>", "<|END_OF_TURN_TOKEN|>")

# which frames a word fits: N noun, V verb, A adjective (plus valid and single)
FRAMES_PROMPT = """下面是一组词语（大多是中文，少数可能是英文）。对每个词，判断它能否自然地放进下面三类句法环境。按词本身判断，忽略词尾的“的”“地”。

N 名词环境：“一个/一种/这种___”，或作为主语、宾语指称事物（如“___的数量”）
V 动词环境：“正在___”“已经___了”“不要___”，表示动作或过程
A 形容词环境：“很___”“非常___的”，表示性质或状态

一个词可以同时适合多个环境。只要这种用法在现代汉语中自然、常见，就标 1，否则标 0。
英文词按对应的英文环境判断：N 为 “a/the ___”，V 为 “to ___”，A 为 “very ___”。
另外标注：
valid：是一个词或短语则为 1；如果是句子、说明文字、提问、乱码或无意义字符串则为 0（此时 N/V/A 都填 0）。
single：是单个词为 1；如果是短语（如“很大”“最高的”“水分多的”“完全浸透”）为 0。

只输出一个 JSON 数组，不要任何解释。数组长度和顺序与输入一致，每项格式：
{"w": "原词", "N": 0或1, "V": 0或1, "A": 0或1, "valid": 0或1, "single": 0或1}

输入：
"""

# a tied word's most common use, among the tied frames only
FRAME_NAMES = {"N": "名词", "V": "动词", "A": "形容词"}
PRIMARY_PROMPT = """下面每个词可能有多种用法。请只在给定的选项中，判断它在现代汉语里最主要、最常见的用法是哪一种（即使它的其他用法也成立，或者选项都不完全合适，也必须选一个最接近的）。选项：{opts}。
英文词按英语的用法判断。只输出一个 JSON 对象，键为原词，值为 {keys} 中的一个字母，不要解释。
词：{words}"""


# ---------------------------------------------------------------- generation

def render(tok, lang):
    msg = [{"role": "user", "content": USER[lang].replace("{W}", NEW_TOKEN)}]
    try:
        text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    start, end = SYSTEM_TURN
    if start in text:
        i = text.index(start)
        text = text[:i] + text[text.index(end, i) + len(end):]
        assert start not in text and "Preamble" not in text
    for open_tok, close_tok in THINKING:
        if text.rstrip().endswith(open_tok):
            text = text.rstrip() + close_tok + "\n\n"
            break
    return text + PREFILL[lang].replace("{W}", NEW_TOKEN)


class Generator:
    def __init__(self, model_name, lang):
        self.lang = lang
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to(self.device).eval()
        self.emb = self.model.get_input_embeddings().weight

    def word_rows(self, words):
        """Embedding rows of single-token real words (space-prefixed in English), by label."""
        rows = {}
        for c in CATS:
            for w in words[c]:
                ids = self.tok.encode(" " + w if self.lang == "en" else w, add_special_tokens=False)
                if len(ids) == 1:
                    rows[f"real_{SHORT[c]}_{w}"] = self.emb[ids[0]].detach().to(torch.float32).clone()
        return rows

    def add_token(self):
        tok, model = self.tok, self.model
        tok.add_tokens([NEW_TOKEN])
        self.new_id = tok.convert_tokens_to_ids(NEW_TOKEN)
        if self.new_id >= model.get_input_embeddings().weight.size(0):
            model.resize_token_embeddings(len(tok))
        # untie lm_head so that only the input row changes, as in training
        model.config.tie_word_embeddings = False
        head = model.get_output_embeddings()
        head.weight = torch.nn.Parameter(head.weight.clone())
        self.emb = model.get_input_embeddings().weight
        self.prompt = render(tok, self.lang)
        self.enc = tok(self.prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        assert int((self.enc["input_ids"][0] == self.new_id).sum()) == 2, self.prompt

    @torch.no_grad()
    def sample(self, vectors, seed):
        """Five samples per vector; vector i (from 1) is sampled with seed + i."""
        out = {}
        n = self.enc["input_ids"].shape[1]
        for i, (label, vec) in enumerate(vectors.items(), 1):
            self.emb[self.new_id] = vec.to(self.emb.device, self.emb.dtype)
            torch.manual_seed(seed + i)
            gen = self.model.generate(**self.enc, max_new_tokens=64, do_sample=True, temperature=1.0,
                                      top_p=0.95, top_k=0, num_return_sequences=5,
                                      pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
            texts = [self.tok.decode(g[n:], skip_special_tokens=True) for g in gen]
            out[label] = [{"raw": t, "synonyms": parse_zh(t) if self.lang == "zh" else parse_en(t)}
                          for t in texts]
            print(f"  {label}: {out[label][0]['synonyms']}", flush=True)
        return out


def parse_en(text):
    """Up to five synonyms from either format the models use (a quoted list or numbered lines)."""
    head = text.strip()
    lines = [l for l in text.splitlines() if re.match(r"^\s*\d+[.)]\s*\S", l)]
    if lines:
        items = [re.sub(r"^\s*\d+[.)]\s*", "", l) for l in lines]
    else:
        head = re.split(r"\n\s*\n", head)[0]
        items = re.split(r"[”\"“]\s*[,;]?\s*(?:and|or)?\s*[“\"]?|,", head)
    out = []
    for it in items:
        it = re.sub(r"\(.*?\)", "", it)
        it = re.sub(r"\s[-–—:].*$", "", it)
        w = it.strip().strip("*“”\"'.,:;!- ").strip().lower()
        w = re.sub(r"^(and|or)\s+", "", w)
        if w and len(w.split()) <= 3 and re.search(r"[a-z]", w):
            out.append(w)
    return out[:5]


def parse_zh(text):
    """Up to five Chinese synonyms: quotes dropped, split on list separators and numbering."""
    body = re.split(r"\n\s*\n", text.strip())[0]
    body = re.sub(r"[“”\"'「」『』《》]", "、", body)
    items = re.split(r"[、，,；;/\n]|\d+[.、)）]|（.*?）|\(.*?\)", body)
    out = []
    for it in items:
        w = it.strip().strip("。.：:！!？?*- ").strip()
        w = re.sub(r"^(和|或者|或|以及|及)", "", w).strip()
        w = re.sub(r"(等等|等)$", "", w).strip()
        if w and re.search(r"[一-鿿]", w) and len(w) <= 8:
            out.append(w)
    return out[:5]


def zh_items(sample):
    """The Chinese synonyms of a sample, or its English ones when it has none."""
    return sample["synonyms"] or [w for w in parse_en(sample["raw"]) if re.fullmatch(r"[a-z][a-z '\-]*", w)]


# ---------------------------------------------------------------- English POS

class Tagger:
    """Soft POS distribution of an English word or short phrase."""

    def __init__(self):
        from nltk.corpus import wordnet as wn
        import spacy
        self.wn = wn
        self.nlp = spacy.load("en_core_web_sm")
        self.cache = {}

    def __call__(self, w):
        if w in self.cache:
            return self.cache[w]
        dist = None
        if " " not in w:
            cnt, senses = collections.Counter(), collections.Counter()
            for pos, cat in (("n", "noun"), ("v", "verb"), ("a", "adj"), ("s", "adj")):
                base = self.wn.morphy(w, pos) or w
                for syn in self.wn.synsets(base, pos=pos):
                    senses[cat] += 1
                    for lem in syn.lemmas():
                        if lem.name().lower() == base:
                            cnt[cat] += lem.count()
            src = cnt if sum(cnt.values()) else senses
            if sum(src.values()):
                tot = sum(src[c] for c in CATS)
                dist = {c: src[c] / tot for c in CATS} if tot else None
        if dist is None:
            doc = self.nlp(w)
            root = [t for t in doc if t.dep_ == "ROOT"] or list(doc)
            p = {"NOUN": "noun", "PROPN": "noun", "VERB": "verb", "ADJ": "adj"}.get(root[0].pos_)
            dist = {c: float(c == p) for c in CATS} if p else None
        self.cache[w] = dist
        return dist


def en_scores(results):
    """Mean POS distribution over each vector's tagged synonyms (uniform if none could be tagged)."""
    tag, out = Tagger(), {}
    for k, samples in results.items():
        acc, n = np.zeros(3), 0
        for s in samples:
            for w in s["synonyms"]:
                d = tag(w)
                if d:
                    acc += [d[c] for c in CATS]
                    n += 1
        p = acc / n if n else np.ones(3) / 3
        out[k] = list(p / p.sum()) if p.sum() else [1 / 3] * 3
    return out


# ---------------------------------------------------------------- Chinese frames

async def post(http, body):
    for attempt in range(5):
        try:
            r = await http.post("/chat/completions", json=body)
        except (httpx_errors()):
            await asyncio.sleep(2 ** attempt)
            continue
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"] or ""
        if r.status_code == 429 or r.status_code >= 500:
            await asyncio.sleep(2 ** attempt)
            continue
        print(f"  API {r.status_code}: {r.text[:160]}")
        return None
    return None


def httpx_errors():
    import httpx
    return httpx.TimeoutException, httpx.TransportError


def client():
    import httpx
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY is not set")
    return httpx.AsyncClient(base_url="https://api.openai.com/v1", timeout=180.0,
                             headers={"Authorization": f"Bearer {key}"})


async def label_frames(words, cache_path, batch=50, concurrency=8):
    """N/V/A, valid and single flags for every word not yet in the cache."""
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    todo = [w for w in words if w not in cache]
    print(f"frame labels: {len(words)} words, {len(todo)} not cached", flush=True)
    if not todo:
        return cache
    http, sem = client(), asyncio.Semaphore(concurrency)
    keys = ("N", "V", "A", "valid", "single")

    async def one(chunk, depth=0):
        body = {"model": LLM, "temperature": 0.0, "max_completion_tokens": 8000,
                "messages": [{"role": "user", "content": FRAMES_PROMPT + json.dumps(chunk, ensure_ascii=False)}]}
        async with sem:
            text = await post(http, body)
        if text is None:
            return
        m = re.search(r"\[.*\]", text, re.S)
        try:
            items = json.loads(m.group(0)) if m else []
        except json.JSONDecodeError:
            items = []
        if len(items) == len(chunk):
            for w, it in zip(chunk, items):
                if isinstance(it, dict) and all(k in it for k in keys):
                    cache[w] = {k: int(bool(it[k])) for k in keys}
        rest = [w for w in chunk if w not in cache]
        # split and retry what did not come back
        if rest and depth < 2 and len(rest) > 1:
            h = len(rest) // 2
            await asyncio.gather(one(rest[:h], depth + 1), one(rest[h:], depth + 1))

    chunks = [todo[i:i + batch] for i in range(0, len(todo), batch)]
    for i in range(0, len(chunks), 40):
        await asyncio.gather(*(one(c) for c in chunks[i:i + 40]))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
    await http.aclose()
    missing = [w for w in words if w not in cache]
    if missing:
        print(f"  not labelled: {len(missing)}, e.g. {missing[:10]}")
    return cache


def frame_scores(results, cache):
    """Share of a vector's valid synonyms that fit the N, V and A frames; None without any."""
    out = {}
    for k, samples in results.items():
        rows = []
        for s in samples:
            for w in zh_items(s):
                c = cache.get(w.strip())
                if c and c["valid"]:
                    rows.append([c[f] for f in "NVA"])
        out[k] = np.mean(rows, axis=0) if rows else None
    return out


def concept_pos(label):
    """'n', 'v' or 'a' for labels such as n_3_unbiased; None for other labels."""
    m = re.search(r"(?:^|_)([nva])_\d+(?:_|$)", label)
    return m.group(1) if m else None


def tied(c, idx):
    v = [c[i] for i in idx]
    m = max(v)
    return sum(x >= m - 1e-9 for x in v) > 1


async def primary_use(http, results, cache, label, c):
    """Settle a three-way tie: the LLM picks each synonym's main use among the tied frames."""
    m = max(c)
    opts = [f for f, x in zip("NVA", c) if x >= m - 1e-9]
    freq = collections.Counter(w.strip() for s in results[label] for w in zh_items(s)
                               if cache.get(w.strip(), {}).get("valid"))
    fit = {w: n for w, n in freq.items() if any(cache[w][o] for o in opts)}
    words = list(fit or freq)
    prompt = PRIMARY_PROMPT.format(opts="、".join(f"{o}={FRAME_NAMES[o]}" for o in opts), keys="/".join(opts),
                                   words=json.dumps(words, ensure_ascii=False))
    body = {"model": LLM, "temperature": 0.0, "max_completion_tokens": 4000,
            "messages": [{"role": "user", "content": prompt}]}
    text = await post(http, body) or ""
    found = re.search(r"\{.*\}", text, re.S)
    got = json.loads(found.group(0)) if found else {}
    votes = collections.Counter()
    for w, o in got.items():
        if o in opts:
            votes[o] += freq[w]
    best = max(opts, key=lambda o: (votes[o], -opts.index(o)))
    return CATS["NVA".index(best)], dict(votes)


def to_resample(C):
    """Vectors whose frames tie three ways, or noun/verb concepts whose N and V tie."""
    return [k for k, c in C.items() if c is not None and
            (tied(c, [0, 1, 2]) or (concept_pos(k) in ("n", "v") and tied(c, [0, 1])))]


def zh_calls(C, primary):
    """Highest frame; a three-way tie goes to the primary-use call among the tied frames."""
    calls = {}
    for k, c in C.items():
        if c is None:
            calls[k] = None
            continue
        v = [float(x) for x in c]
        if k in primary and tied(v, [0, 1, 2]):
            v[CATS.index(primary[k]["call"])] = max(v) + 1e-6
        calls[k] = CATS[int(np.argmax(v))]
    return calls


# ---------------------------------------------------------------- main

def share_correct(scores, gold):
    """1 if the gold category is the only top one, 1/k if it ties with k-1 others."""
    m = max(scores)
    top = [c for c, x in zip(CATS, scores) if x >= m - 1e-9]
    return 1 / len(top) if gold in top else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lang", default="en", choices=("en", "zh"))
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--vectors", nargs="+", type=Path, help="trained vectors (.pt); the file name is the label")
    src.add_argument("--words", type=Path, help='{"words": {"noun": [...], "verb": [...], "adj": [...]}}')
    ap.add_argument("--frames", type=Path, default=HERE / "zh_frames_llm.json", help="cache of the LLM frame labels")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    gen = Generator(args.model, args.lang)
    if args.words:
        words = json.loads(args.words.read_text(encoding="utf-8"))["words"]
        vectors = gen.word_rows(words)
    else:
        vectors = {p.stem: torch.load(p, map_location="cpu")["embedding"] for p in args.vectors}
    gen.add_token()
    results = gen.sample(vectors, seed=0)
    out = {"model": args.model, "lang": args.lang, "prompt": gen.prompt}

    if args.lang == "en":
        scores = en_scores(results)
        out.update(results=results, scores=scores,
                   calls={k: CATS[int(np.argmax(v))] for k, v in scores.items()})
        if args.words:
            acc = np.mean([share_correct(v, CATS["nva".index(k.split("_")[1])]) for k, v in scores.items()])
    else:
        def to_label():
            ws = [w for c in CATS for w in words[c]] if args.words else []
            ws += [w.strip() for v in results.values() for s in v for w in zh_items(s)]
            return list(dict.fromkeys(w for w in ws if w))

        cache = asyncio.run(label_frames(to_label(), args.frames))
        resampled, primary = {}, {}
        if args.vectors:
            for r in range(1, ROUNDS + 1):
                ties = to_resample(frame_scores(results, cache))
                print(f"round {r}: {len(ties)} tied", flush=True)
                if not ties:
                    break
                resampled[r] = ties
                for k, v in gen.sample({k: vectors[k] for k in ties}, seed=1000 * r).items():
                    results[k].extend(v)
                cache = asyncio.run(label_frames(to_label(), args.frames))

            async def settle(C):
                http = client()
                for k, c in C.items():
                    if c is not None and tied(c, [0, 1, 2]):
                        primary[k] = dict(zip(("call", "votes"), await primary_use(http, results, cache, k, c)))
                await http.aclose()
            asyncio.run(settle(frame_scores(results, cache)))
        C = frame_scores(results, cache)
        scores = {k: (None if c is None else [float(x) for x in c]) for k, c in C.items()}
        calls = zh_calls(C, primary)
        out.update(results=results, resampled=resampled, primary=primary, scores=scores, calls=calls)
        if args.words:
            # only words the LLM accepts in their own frame, as single valid words
            acc = []
            for k, v in scores.items():
                _, p, w = k.split("_", 2)
                c = cache[w]
                if c["NVA"["nva".index(p)]] and c["single"] and c["valid"]:
                    acc.append(share_correct(v, CATS["nva".index(p)]) if v is not None else 0.0)
            acc = np.mean(acc)
    if args.words:
        out["accuracy"] = float(acc)
        print(f"accuracy {acc:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
