#!/usr/bin/env python
"""Translate the English concept descriptions into Chinese noun phrases with GPT-5.2.

Reads scripts/train/data/concepts.csv (POS, ID, FrameNet, gpt) and writes the same table with
the description in Chinese. One request per concept. The translations were then checked by
two native speakers and regenerated where needed.

    OPENAI_API_KEY=... python scripts/FrameNet/translate_concepts.py --out <file>
"""
import argparse
import csv
from pathlib import Path

from openai import OpenAI

CONCEPTS = Path(__file__).resolve().parents[1] / "train" / "data" / "concepts.csv"
MODEL = "gpt-5.2-2025-12-11"

SYSTEM = """\
你是一名术语翻译，正在处理一个语言学实验的数据集。每条输入是一个"概念"的英文描述，\
会被原样展示给语言模型，用来指定它应当表达的语义内容。

翻译规范，必须严格遵守：

1. 输出一个**名词短语**，中心语必须是名词。不要加"……的概念"这样的外壳，\
也不要加任何前缀后缀。
2. 忠实，不增不减。不要补充英文里没有的解释、举例或范围限定，也不要为了简洁而漏译。
3. **不要为了中文自然而改变描述的语法形态。** 英文里三类概念（名词/动词/形容词）\
一律写成名词短语，例如形容词概念写作 "concept of magnitude in physical size"，\
中文也必须是名词短语"物理尺寸上的大小程度"，绝不可译成"大的"或"很大"。\
动词概念同理，译成名词化短语而不是谓语。词性信息不属于这条描述。
4. 术语用常见译法，同一个英文术语在不同条目中保持同一译法。
5. 并列用"与"而不是"和"。
6. 只输出译文本身。不要引号、不要编号、不要任何解释。
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=CONCEPTS)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    with open(args.csv, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    client = OpenAI()
    out = [rows[0]]
    for pos, cid, frame, en in rows[1:]:
        r = client.chat.completions.create(model=MODEL, messages=[{"role": "system", "content": SYSTEM},
                                                                  {"role": "user", "content": en}])
        zh = r.choices[0].message.content.strip()
        print(f"{cid:<6}{frame:<30}{zh}")
        out.append([pos, cid, frame, zh])
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(out)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
