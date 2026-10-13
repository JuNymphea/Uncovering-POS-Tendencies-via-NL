#!/usr/bin/env python
"""Generate the Chinese training data (normal_answer and concept_answer) for each concept.

Markdown bold, list markers and newlines are stripped from the replies.
Output files are resumed if they exist; a short quality report is printed per concept.

    export OPENAI_API_KEY=...
    python scripts/FrameNet/prepare_data_framenet_zh.py
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
# default input (Chinese AxBench) and output locations
AXBENCH = SCRIPTS / "axbench" / "data"
TRAIN_ZH = SCRIPTS / "train" / "data" / "zh"
DEFAULT_MODEL = "gpt-5.2-2025-12-11"

T_NORMAL = """给定以下指令：

{INSTRUCTION}

你的任务是给出一个回答。

**格式要求：**

- 只返回对该指令的回答。
- 用纯文本写出最终内容（若体裁另有要求则用相应格式）。
- 不要包含任何额外的文字、解释或格式标记。

**最终答案：** 只返回最终内容，遵守以上要求。"""

T_CONCEPT = """给定以下指令：

{INSTRUCTION}

目标语义概念：
{CONCEPT}

请写一个对该指令的回答，清晰且充分地表达这个目标语义概念。

要求：
1. 尽可能贴合原指令。
2. 目标概念必须是回答中重要且可识别的部分，不能是模糊或偶然的联想。
3. 包含能清楚实现该概念含义的具体内容。
4. 只看回答的读者，应当能够高置信度地推断出目标概念。
5. 优先用自己的话表达概念，而不是照抄定义原文。但保持原意比避开定义里的个别词更重要。
6. 不要提到概念的名称、"目标概念"这类说法，也不要讨论任务本身。
7. 即使目标概念与原指令关系不强，回答也要局部连贯、自然。
8. 不要仅仅为了提及概念而添加无关细节。

格式：
- 只返回对原指令的回答。
- 用纯文本，除非指令本身要求其他格式。
- 不要包含解释、注释或元评论。

最终答案：只返回回答本身。"""

SIMILE = re.compile(r"(就像|好比|如同|犹如|好似|仿佛|相当于)")
# patterns for report(), kept outside the f-strings (no backslashes there before Python 3.12)
PAT_BOLD = r"\*\*|__"
PAT_NL = "\n"
PAT_EN = r"[A-Za-z]{4,}"
PAT_LIST = r"(?m)^\s*[-*]\s"


def read_concept_map(path: Path) -> dict:
    """{concept id: description}. Columns are POS,ID,FrameNet,gpt."""
    out = {}
    with open(path, encoding="utf-8-sig") as f:
        for i, row in enumerate(csv.reader(f)):
            if len(row) < 4:
                continue
            k, v = row[1].strip(), row[3].strip()
            if i == 0 and k.upper() == "ID":
                continue
            if k and v:
                out[k] = v
    if not out:
        raise SystemExit(f"{path}: no concepts read")
    return out


class Client:
    def __init__(self, model, concurrency=10):
        import httpx
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise SystemExit("OPENAI_API_KEY is not set.")
        self.model = model
        self.calls = 0
        self.sem = asyncio.Semaphore(concurrency)
        self.http = httpx.AsyncClient(
            base_url="https://api.openai.com/v1", timeout=180.0,
            headers={"Authorization": f"Bearer {key}"},
            limits=httpx.Limits(max_keepalive_connections=concurrency,
                                max_connections=concurrency * 2))

    async def ask(self, prompt: str, temperature: float = 0.7) -> str:
        import httpx
        body = {"model": self.model, "temperature": temperature,
                "max_completion_tokens": 2000,
                "messages": [{"role": "user", "content": prompt}]}
        async with self.sem:
            for attempt in range(5):
                try:
                    r = await self.http.post("/chat/completions", json=body)
                except (httpx.TimeoutException, httpx.TransportError):
                    if attempt == 4:
                        return ""
                    await asyncio.sleep(2 ** attempt)
                    continue
                self.calls += 1
                if r.status_code == 200:
                    break
                # retry on any 5xx and on 429
                if (r.status_code >= 500 or r.status_code == 429) and attempt < 4:
                    await asyncio.sleep(2 ** attempt)
                    continue
                print(f"    API {r.status_code}: {r.text[:150]}")
                return ""
        return r.json()["choices"][0]["message"]["content"].strip()


def clean(text: str) -> str:
    """Remove markdown bold, headings, list markers and newlines."""
    text = re.sub(r"\*\*|__|^#+\s*", "", text, flags=re.M)
    text = re.sub(r"(?m)^\s*[-*]\s+", "", text)
    return re.sub(r"\s*\n\s*", "", text).strip()


async def one_record(cli, rec, desc):
    if not rec.get("normal_answer"):
        rec["normal_answer"] = clean(await cli.ask(
            T_NORMAL.replace("{INSTRUCTION}", rec["question"])))
    if not rec.get("concept_answer"):
        rec["concept_answer"] = clean(await cli.ask(
            T_CONCEPT.replace("{INSTRUCTION}", rec["question"]).replace("{CONCEPT}", desc)))
    return rec


async def one_concept(cli, concept, desc, src, outdir, limit, save_every=20):
    out_file = outdir / f"{concept}.jsonl"
    if not out_file.exists():
        outdir.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, out_file)
    recs = [json.loads(l) for l in out_file.open(encoding="utf-8") if l.strip()]
    if limit:
        recs = recs[:limit]
    todo = [i for i, r in enumerate(recs) if not r.get("concept_answer")]
    print(f"{concept} ({desc[:24]})  rows {len(recs)}, to generate {len(todo)}")
    if not todo:
        return recs

    def save():
        tmp = out_file.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.replace(out_file)

    for i in range(0, len(todo), save_every):
        chunk = todo[i:i + save_every]
        await asyncio.gather(*(one_record(cli, recs[j], desc) for j in chunk))
        save()
        print(f"  {min(i + save_every, len(todo))}/{len(todo)}", flush=True)
    save()
    return recs


def report(recs, concept, desc):
    ca = [r["concept_answer"] for r in recs if r.get("concept_answer")]
    na = [r["normal_answer"] for r in recs if r.get("normal_answer")]
    if not ca:
        print("  no concept_answer generated")
        return
    import statistics as st
    def rate(xs, pat):
        return sum(1 for x in xs if re.search(pat, x)) / len(xs)
    print(f"\n=== {concept} check ({len(ca)} rows) ===")
    print(f"  empty answers         {sum(1 for r in recs if not r.get('concept_answer')):>6}")
    print(f"  markdown bold         {rate(ca, PAT_BOLD):>6.1%}")
    print(f"  list markers          {rate(ca, PAT_LIST):>6.1%}")
    print(f"  newlines              {rate(ca, PAT_NL):>6.1%}")
    print(f"  4+ letter Latin words {rate(ca, PAT_EN):>6.1%}")
    print(f"  similes               {rate(ca, SIMILE.pattern):>6.1%}   (recorded only)")
    print(f"  concept median {st.median(len(x) for x in ca):.0f} chars / max {max(len(x) for x in ca)}")
    print(f"  normal  median {st.median(len(x) for x in na):.0f} chars")
    rr = [len(r["concept_answer"]) / max(len(r["normal_answer"]), 1)
          for r in recs if r.get("concept_answer") and r.get("normal_answer")]
    print(f"  length ratio median {st.median(rr):.2f}")


async def main_async(args):
    cmap = read_concept_map(args.concepts_csv)
    keys = args.concepts or list(cmap)
    missing = [k for k in keys if k not in cmap]
    if missing:
        raise SystemExit(f"concepts not in the CSV: {missing}")
    n = (args.limit or 2100) * len(keys)
    print(f"model {args.model} | concepts {len(keys)} | rows each {args.limit or 'all'}"
          f" | about {n} calls (more with retries)")
    if args.dry_run:
        for k in keys[:3]:
            print(f"  {k}: {cmap[k]}")
        print("--dry-run: API not called.")
        return
    cli = Client(args.model, args.concurrency)
    for k in keys:
        recs = await one_concept(cli, k, cmap[k], args.src, args.outdir, args.limit)
        report(recs, k, cmap[k])
    print(f"\nmade {cli.calls} API calls")
    await cli.http.aclose()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--concepts", nargs="*", help="concept ids to run (default: all)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N rows of each concept")
    ap.add_argument("--src", type=Path,
                    default=AXBENCH / "axbench_train_filtered.zh.jsonl")
    ap.add_argument("--outdir", type=Path, default=TRAIN_ZH)
    ap.add_argument("--concepts-csv", type=Path, default=SCRIPTS / "train" / "data" / "concepts.zh.csv")
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
