#!/usr/bin/env python
"""Induce Chinese Task 2 slot frames from UD Chinese-GSDSimp.

Follows the English induction (run_induction.py): signatures of 1-2 tokens on each side,
closed classes kept as word forms, frames kept at >= 0.90 purity with >= 5 occurrences and
>= 4 filler types, then checked on dev+test. Each frame is realized with generic fillers and,
if needed, a short opener. Few adjective frames pass, since GSDSimp tags most stative
predicates VERB. Writes out_zh/candidates.json and out_zh/induction_report.json.

    python scripts/Tasks/Task_2/zh/induce_zh.py
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

SLOT, NEO, BOS, EOS = "{SLOT}", "{NEOLOGISM}", "<s>", "</s>"
TARGET = ("NOUN", "VERB", "ADJ")
LEX_UPOS = {"DET", "ADP", "AUX", "PART", "SCONJ", "CCONJ", "PUNCT", "CLF"}
ABS_UPOS = {"PRON", "NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM", "INTJ", "SYM", "X",
            "AFFIX", "WF"}
#: A frame whose signature contains any of these is dropped.
FORBIDDEN = {"X", "SYM", "INTJ", "AFFIX", "WF"}
AFFIX_XPOS = {"SFN", "PFA", "SFV", "SFA", "PFN"}
WF_DEPRELS = {"compound", "flat", "fixed", "goeswith", "reparandum"}
LOG3 = math.log(3.0)

#: Generic fillers per abstract category, in order of preference (see fill()).
FILLERS = {
    "NOUN": ["人们", "问题", "东西", "政府", "工作", "时间", "地区", "公司"],
    "PRON": ["他", "他们", "它", "我们"],
    "PROPN": ["中国", "美国", "北京"],
    "VERB": ["进行", "发展", "使用", "成为", "有", "发生"],
    "ADJ": ["重要", "不同", "主要", "大"],
    "NUM": ["一", "两", "三"],
    "ADV": None,          # filled with the most frequent adverb that is not lexicalized
}

#: Left-context heads that need a subject; such items get a pronoun opener.
NEEDS_SUBJECT = set("被 所 也 不 会 又 并 都 就 已 曾 将 还 才 能 可 要 把 给 向 跟 没 没有 已经 "
                    "正在 可以 应该 必须 可能 能够 仍 仍然 再 便 却 均 共 亦 皆 未 只 仅 了 着 过 "
                    "为 由 于 以".split())
DEGREE = set("最 更 非常 很 十分 比较 较 太 极 相当 特别 越来越".split())
#: Openers for items starting with a localizer, which needs a noun phrase before it.
LOCALIZER_OPENER = {"中": ["在", "这一", "过程"], "上": ["在", "这个", "问题"],
                    "下": ["在", "这种", "情况"], "后": ["在", "此"], "前": ["在", "此"],
                    "里": ["在", "这个", "城市"], "内": ["在", "国"], "外": ["在", "国"],
                    "之后": ["在", "此"], "之前": ["在", "此"], "以后": ["从", "此"],
                    "以前": ["在", "此"], "期间": ["在", "这", "段"], "之间": ["在", "他们"]}
CONJ_OPENER = {"NOUN": "经济", "VERB": "学习", "ADJ": "简单"}

#: Slot neighbours that would reproduce a Chinese training template.
TRAINING_LEFT = ("请", "尽可能", "：", ":")
TRAINING_RIGHT = ("你", "你的")


@dataclass
class Tok:
    idx: int
    form: str
    lemma: str
    upos: str
    xpos: str
    head: int
    deprel: str


def read_conllu(path: Path):
    sents, block = [], []

    def flush():
        toks = []
        for line in block:
            if line.startswith("#"):
                continue
            c = line.split("\t")
            if len(c) != 10 or "-" in c[0] or "." in c[0]:
                continue
            upos, xpos, deprel = c[3], c[4], c[7]
            base = deprel.split(":")[0]
            # classifiers -> CLF (kept as forms), bound morphemes -> AFFIX,
            # word-internal NOUN/VERB/ADJ -> WF (counts against purity)
            if xpos == "NNB" or deprel == "clf":
                upos = "CLF"
            elif upos == "PART" and xpos in AFFIX_XPOS:
                upos = "AFFIX"
            elif upos in TARGET and (base in WF_DEPRELS or (upos == "VERB" and base == "mark")):
                upos = "WF"
            toks.append(Tok(len(toks), c[1], c[2] if c[2] != "_" else c[1], upos, xpos,
                            int(c[6]) if c[6].isdigit() else 0, deprel))
        if toks:
            sents.append(toks)

    for raw in open(path, encoding="utf-8"):
        line = raw.rstrip("\n")
        if not line.strip():
            if block:
                flush()
                block = []
        else:
            block.append(line)
    if block:
        flush()
    return sents


@dataclass
class Frame:
    signature: str
    n_left: int
    n_right: int
    pos_counts: Counter = field(default_factory=Counter)
    types: dict = field(default_factory=lambda: defaultdict(set))
    deprels: dict = field(default_factory=lambda: defaultdict(Counter))
    fills: dict = field(default_factory=lambda: defaultdict(Counter))   # offset -> Counter[form]
    attested: Counter = field(default_factory=Counter)                  # (context forms, word)

    @property
    def n_target(self):
        return sum(self.pos_counts[p] for p in TARGET)

    @property
    def n_all(self):
        return sum(self.pos_counts.values())

    @property
    def dom(self):
        return max(TARGET, key=lambda p: self.pos_counts[p])

    @property
    def purity(self):          # over every filler, as in English
        return self.pos_counts[self.dom] / self.n_all if self.n_all else 0.0

    @property
    def purity3(self):
        return self.pos_counts[self.dom] / self.n_target if self.n_target else 0.0

    @property
    def n_types(self):
        return len(self.types[self.dom])

    @property
    def entropy(self):
        h = 0.0
        for q in (self.pos_counts[p] / self.n_target for p in TARGET):
            if q > 0:
                h -= q * math.log(q)
        return h / LOG3

    @property
    def score(self):           # frequent x selective x productive, as in English
        return (math.log(1 + self.n_target) * self.purity * (1 - self.entropy)
                * math.log(1 + self.n_types))

    def elements(self):
        return self.signature.split(" ")


def main() -> None:
    here = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ud-dir", type=Path, default=here / "data" / "ud_zh")
    ap.add_argument("--treebank", default="zh_gsdsimp")
    ap.add_argument("--out-dir", type=Path, default=here / "out_zh")
    ap.add_argument("--max-left", type=int, default=2)
    ap.add_argument("--max-right", type=int, default=2)
    ap.add_argument("--min-freq", type=int, default=5)
    ap.add_argument("--min-types", type=int, default=4)
    ap.add_argument("--min-purity", type=float, default=0.90)
    ap.add_argument("--adv-topk", type=int, default=30)
    ap.add_argument("--dev-min-count", type=int, default=8)
    ap.add_argument("--dev-min-purity", type=float, default=0.85)
    ap.add_argument("--pool-size", type=int, default=40)
    args = ap.parse_args()

    train = read_conllu(args.ud_dir / f"{args.treebank}-ud-train.conllu")
    held = (read_conllu(args.ud_dir / f"{args.treebank}-ud-dev.conllu")
            + read_conllu(args.ud_dir / f"{args.treebank}-ud-test.conllu"))
    assert train and held, "treebank files missing"
    n_tok = sum(len(s) for s in train)

    adv = Counter(t.form for s in train for t in s if t.upos == "ADV")
    lex_adv = {f for f, _ in adv.most_common(args.adv_topk)}
    FILLERS["ADV"] = [f for f, _ in adv.most_common() if f not in lex_adv][:3]

    def enc(s, j):
        if j < 0:
            return BOS
        if j >= len(s):
            return EOS
        t = s[j]
        if t.upos in LEX_UPOS or (t.upos == "ADV" and t.form in lex_adv):
            return t.form
        return t.upos

    def form(s, j):
        return BOS if j < 0 else EOS if j >= len(s) else s[j].form

    windows = [(l, r) for l in range(1, args.max_left + 1) for r in range(1, args.max_right + 1)]

    # corpus bigrams, used to pick the generic filler that fits its neighbours
    bigram = Counter()
    for s in train + held:
        for a, b in zip(s, s[1:]):
            bigram[(a.form, b.form)] += 1

    def sig(s, i, l, r):
        return " ".join([enc(s, i + o) for o in range(-l, 0)] + [SLOT]
                        + [enc(s, i + o) for o in range(1, r + 1)])

    rough = Counter()
    for s in train:
        for t in s:
            for l, r in windows:
                rough[sig(s, t.idx, l, r)] += 1
    keep = {k for k, n in rough.items() if n >= args.min_freq}
    frames: dict = {}
    for s in train:
        for t in s:
            for l, r in windows:
                g = sig(s, t.idx, l, r)
                if g not in keep:
                    continue
                f = frames.get(g) or frames.setdefault(g, Frame(g, l, r))
                f.pos_counts[t.upos] += 1
                if t.upos in TARGET:
                    f.types[t.upos].add(t.lemma)
                    f.deprels[t.upos][t.deprel] += 1
                    offs = list(range(-l, 0)) + list(range(1, r + 1))
                    for o in offs:
                        f.fills[o][form(s, t.idx + o)] += 1
                    f.attested[(tuple(form(s, t.idx + o) for o in offs), t.form)] += 1

    held_counts = defaultdict(Counter)
    for s in held:
        for t in s:
            for l, r in windows:
                g = sig(s, t.idx, l, r)
                if g in frames:
                    held_counts[g][t.upos] += 1

    def kind(e):
        if e == SLOT:
            return "slot"
        if e in (BOS, EOS):
            return "boundary"
        if e in ABS_UPOS:
            return "abstract"
        if not any(ch.isalnum() for ch in e):
            return "punct"
        return "lexical"

    def structural(f):
        el = f.elements()
        i = el.index(SLOT)
        left, right = el[:i], el[i + 1:]
        if not left or EOS in right or any(e in FORBIDDEN for e in el):
            return False
        if not any(kind(e) in ("lexical", "abstract") for e in right):
            return False
        return any(kind(e) == "lexical" for e in left + right)

    def held_ok(f):
        c = held_counts.get(f.signature)
        n = sum(c[p] for p in TARGET) if c else 0
        if n < args.dev_min_count:
            return True
        return c[f.dom] / sum(c.values()) >= args.dev_min_purity

    # ---- realization -----------------------------------------------------
    def fill(f, offset, upos, prev, nxt):
        """Pick a generic filler: attested here first, then by bigram counts with its neighbours."""
        cands = FILLERS.get(upos) or []
        if not cands:
            return None
        seen = f.fills[offset]
        return max(cands, key=lambda w: (10 * seen[w]
                                         + (bigram[(prev, w)] if prev else 0)
                                         + (bigram[(w, nxt)] if nxt else 0),
                                         -cands.index(w)))

    def realize(f):
        el = f.elements()
        i = el.index(SLOT)
        left, right = el[:i], el[i + 1:]
        offs_left = list(range(-len(left), 0))
        def lexical_or_none(e):
            return e if kind(e) in ("lexical", "punct") else None
        L = []
        for k, (e, o) in enumerate(zip(left, offs_left)):
            if e == BOS:
                continue
            if kind(e) == "abstract":
                prev = L[-1] if L else None
                nxt = lexical_or_none(left[k + 1]) if k + 1 < len(left) else None
                w = fill(f, o, e, prev, nxt)
                if w is None:
                    return None
                L.append(w)
            else:
                L.append(e)
        R = []
        for k, e in enumerate(right):
            if e == EOS:
                break
            if kind(e) == "abstract":
                prev = R[-1] if R else None
                nxt = lexical_or_none(right[k + 1]) if k + 1 < len(right) else None
                w = fill(f, k + 1, e, prev, nxt)
                if w is None:
                    return None
                R.append(w)
            else:
                R.append(e)
        # opener: only when the realized left context cannot begin a sentence
        opener = []
        first = left[0]
        if first != BOS and L:
            head = L[0]
            if kind(first) == "abstract":
                if first == "ADJ":
                    opener = ["这", "是"]
                elif first in ("ADV", "VERB"):
                    opener = ["他"]
            elif first == "CLF" or head in ("个", "种", "次", "位", "名", "年", "件", "本", "条", "项"):
                opener = ["这"]
            elif head == "地":
                opener = ["他", "逐渐"]
            elif head in ("之",):
                return None
            elif head in DEGREE:
                opener = ["这", "是"]
            elif head == "是":
                opener = ["这"]
            elif head == "的":
                opener = ["他"]
            elif head in LOCALIZER_OPENER:
                opener = LOCALIZER_OPENER[head]
            elif head == "，":
                opener = ["此后"]
            elif head in ("、", "和", "与", "及", "或", "以及", "并且"):
                opener = [CONJ_OPENER[f.dom]]
            elif head in NEEDS_SUBJECT or head in lex_adv:
                opener = ["他"]
            elif kind(head) == "punct":
                return None
        return opener, L, R

    def collides(L, R):
        return ((L and L[-1] in TRAINING_LEFT)
                or (L and L[-1] == "个" and R and R[0] == "来")
                or (R and R[0] in TRAINING_RIGHT))

    report = {"treebank": args.treebank, "train_sentences": len(train), "train_tokens": n_tok,
              "lexicalized_adverbs": sorted(lex_adv), "adv_filler": FILLERS["ADV"],
              "signatures_ge_min_freq": len(frames), "per_pos": {}}
    candidates, full = {}, {}
    for pos in TARGET:
        passing = [f for f in frames.values()
                   if f.dom == pos and f.n_target >= args.min_freq
                   and f.n_types >= args.min_types and f.purity >= args.min_purity]
        n_pure = len(passing)
        passing = [f for f in passing if structural(f)]
        n_struct = len(passing)
        passing = [f for f in passing if held_ok(f)]
        n_held = len(passing)
        passing.sort(key=lambda f: f.score, reverse=True)
        out, seen_fp, families, drops = [], set(), Counter(), Counter()
        for f in passing:
            real = realize(f)
            if real is None:
                drops["unrealizable"] += 1
                continue
            opener, L, R = real
            if collides(L, R):
                drops["training-template collision"] += 1
                continue
            el = f.elements()
            i = el.index(SLOT)
            family = (el[i - 1], el[i + 1])
            fp = "".join(L) + "|" + "".join(R)
            if fp in seen_fp:
                drops["same realization"] += 1
                continue
            if families[family] >= 1:
                drops["same construction family"] += 1
                continue
            seen_fp.add(fp)
            families[family] += 1
            prefix_tokens = opener + L + [NEO]
            held = held_counts.get(f.signature)
            held_n = sum(held[p] for p in TARGET) if held else 0
            ex = f.attested.most_common(1)[0][0] if f.attested else None
            out.append({
                "signature": f.signature,
                "pos": pos,
                "prefix_tokens": prefix_tokens,
                "prefix": "".join(prefix_tokens),
                "diagnostic": R,
                "item": "".join(prefix_tokens) + "".join(R),
                "opener": opener,
                "morph_requirement": "Any",
                "purity": round(f.purity, 4),
                "purity_target3": round(f.purity3, 4),
                "n": f.n_target,
                "n_types": f.n_types,
                "counts": {p: f.pos_counts[p] for p in TARGET},
                "other": dict(Counter({k: v for k, v in f.pos_counts.items()
                                       if k not in TARGET}).most_common(3)),
                "held_out_n": held_n,
                "held_out_purity": (round(held[f.dom] / sum(held.values()), 4)
                                    if held_n else None),
                "score": round(f.score, 4),
                "dominant_deprel": f.deprels[pos].most_common(1)[0][0],
                "fillers_attested": sorted(f.types[pos])[:10],
                "attested_example": ("".join(w for w in ex[0][:f.n_left] if w not in (BOS, EOS))
                                     + "[" + ex[1] + "]"
                                     + "".join(w for w in ex[0][f.n_left:] if w not in (BOS, EOS)))
                                    if ex else None,
            })
            if len(out) == args.pool_size:
                break
        candidates[pos] = out
        full[pos] = len(passing)
        report["per_pos"][pos] = {"pure": n_pure, "structural": n_struct, "held_out_ok": n_held,
                                  "pool": len(out), "dropped": dict(drops)}
        print(f"{pos}: purity>={args.min_purity} {n_pure} -> structural {n_struct} "
              f"-> held-out {n_held} -> pool {len(out)}   dropped {dict(drops)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "candidates.json").write_text(
        json.dumps(candidates, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out_dir / "induction_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {args.out_dir / 'candidates.json'}")
    for pos in TARGET:
        print(f"\n=== {pos} ({len(candidates[pos])}) ===")
        for c in candidates[pos]:
            print(f"  {c['signature']:26} pur={c['purity']:.2f} n={c['n']:<4} typ={c['n_types']:<4}"
                  f" {c['prefix'].replace(NEO, '[X]')}|{''.join(c['diagnostic'])}"
                  f"    e.g. {c['attested_example']}")


if __name__ == "__main__":
    main()
