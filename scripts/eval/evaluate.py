#!/usr/bin/env python
"""Score generation files with the LM judge.

Each input row holds a question with a normal answer and a concept answer. Both are rated
0-2 for concept relevance, instruction relevance and fluency, combined by harmonic mean, and
the means are reported per factor. Writes scores/<source>/<name>.json and a summary.csv per
folder. Needs OPENAI_API_KEY.

    python scripts/eval/evaluate.py --input-file <dir>/gemma_n_1_unbiased.jsonl
        --concepts scripts/train/data/concepts.csv --metrics concept --limit 50
    python scripts/eval/evaluate.py --source results --all --metrics concept --limit 50
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
from pathlib import Path

import httpx
import pandas as pd
from openai import AsyncOpenAI

from language_models import LanguageModel
from lm_judge import ALL_METRICS, LMJudgeEvaluator

DEFAULT_MODEL = "gpt-5.2-2025-12-11"
DEFAULT_CACHE_TAG = "demo"

EVAL_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = EVAL_DIR.parent

#: Input folders: training data and generations from the trained vectors.
SOURCES = {
    "train": SCRIPTS_DIR / "train" / "data" / "en",
    "results": SCRIPTS_DIR / "train" / "results",
}
#: Scores go to <scores-dir>/<source>/.
DEFAULT_SCORES_DIR = EVAL_DIR / "scores"
#: Judge replies are cached here, keyed by prompt text (JUDGE_CACHE_DIR or --cache-dir).
DEFAULT_CACHE_DIR = os.environ.get("JUDGE_CACHE_DIR") or str(EVAL_DIR / "cached_data")

TEMPLATES = ("unbiased", "verb", "noun", "adj", "mixed")
#: Concept ids such as a_1 or n_16, anywhere in a file name ("gemma_a_1_unbiased" -> a_1).
CONCEPT_RE = re.compile(r"(?:^|_)([anv]_\d+)(?=_|$)")


def natural_key(path: Path):
    """Sort key that puts a_2 before a_10."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path.name)]


def default_concept_csv() -> Path:
    """scripts/train/data/concepts.csv: the 30 concepts of the released data and vectors."""
    return SCRIPTS_DIR / "train" / "data" / "concepts.csv"


def read_concept_map(path: Path) -> dict:
    """Read {concept id: description} from the concept CSV (POS,ID,FrameNet,gpt).

    The judge is shown the gpt column.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"concept CSV not found: {path}\n"
            "pass --concepts explicitly, e.g. "
            "--concepts scripts/train/data/concepts.csv")
    result = {}
    # utf-8-sig drops the BOM
    with open(path, mode="r", encoding="utf-8-sig") as f:
        for i, row in enumerate(csv.reader(f)):
            if len(row) < 4:
                continue
            key, value = row[1].strip(), row[3].strip()
            # skip the header row
            if i == 0 and key.upper() == "ID":
                continue
            if key and value:
                result[key] = value
    if not result:
        raise ValueError(f"no usable rows in {path} (expected POS,ID,FrameNet,gpt)")
    return result


def parse_file_name(input_file: Path, concept_map: dict) -> tuple[str, str | None]:
    """Return (concept id, template or None) parsed from a generation file name.

    The id is the first a_1-style field found in the concept CSV, so a model prefix is skipped.
    """
    stem = input_file.stem
    for m in CONCEPT_RE.finditer(stem):
        concept = m.group(1)
        if concept in concept_map:
            tail = stem[m.end():].strip("_").split("_")
            template = tail[-1] if tail and tail[-1] in TEMPLATES else None
            return concept, template
    raise KeyError(
        f"no concept id from the CSV in '{input_file.name}'; expected a name like "
        f"'a_1.jsonl' or 'gemma_a_1_unbiased.jsonl'. Known ids include: "
        f"{', '.join(sorted(concept_map)[:8])}...")


def transform_res_data(input_file: Path, concept: str, concept_map: dict,
                       limit: int | None = None) -> pd.DataFrame:
    """Expand each question into two judge rows: the normal and the concept answer."""
    df = pd.read_json(input_file, lines=True)
    if limit:
        df = df.head(limit)

    missing = {"question", "normal_answer", "concept_answer"} - set(df.columns)
    if missing:
        raise KeyError(f"{input_file.name} lacks {sorted(missing)}")

    rows = []
    for _, row in df.iterrows():
        question = row["question"]
        rows.append({"original_prompt": question,
                     "generation": row["normal_answer"], "factor": "normal"})
        rows.append({"original_prompt": question,
                     "generation": row["concept_answer"], "factor": "concept"})

    new_df = pd.DataFrame(rows)
    new_df["dataset_name"] = concept
    new_df["input_concept"] = concept_map[concept]
    return new_df


def get_chat_client(model_name: str) -> AsyncOpenAI:
    """Async OpenAI client for the judge model."""
    base_url = "https://api.openai.com/v1"
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Export it before running:\n"
            "    export OPENAI_API_KEY=...")

    # reuse connections; one connection per request can exhaust the DNS resolver
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=60.0,
        http_client=httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=16, max_connections=32),
        ),
        max_retries=3,
    )


def close_client(lm_model: LanguageModel) -> None:
    try:
        asyncio.run(lm_model.close())
    except RuntimeError:
        # the client's event loop was already closed by asyncio.run in the judge
        pass


def judge_file(lm_model: LanguageModel, model: str, input_file: Path, concept: str,
               concept_map: dict, limit: int | None = None,
               metrics: tuple | None = None) -> dict:
    """Run the judge on one file and return its metrics."""
    evaluator = LMJudgeEvaluator(lm_model=lm_model, model_name=model, metrics=metrics)
    return evaluator.compute_metrics(
        transform_res_data(input_file, concept, concept_map, limit))


def evaluate(input_file: Path, scores_file: Path, model: str,
             concept_map: dict, cache_dir: str = DEFAULT_CACHE_DIR,
             cache_tag: str = DEFAULT_CACHE_TAG,
             limit: int | None = None) -> dict:
    """Judge one file and write its scores (library entry point)."""
    lm_model = LanguageModel(model=model, client=get_chat_client(model), use_cache=True,
                             cache_level="prompt", master_data_dir=cache_dir,
                             cache_tag=cache_tag)
    concept, _ = parse_file_name(Path(input_file), concept_map)
    scores = judge_file(lm_model, model, Path(input_file), concept, concept_map, limit)
    scores_file.parent.mkdir(parents=True, exist_ok=True)
    scores_file.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
    lm_model.save_cache()
    close_client(lm_model)
    return scores


def infer_source(path: Path) -> str:
    resolved = path.resolve()
    for name, root in SOURCES.items():
        if resolved.parent == root.resolve():
            return name
    return "custom"


def resolve_inputs(items: list[str], source: str | None, take_all: bool) -> list[Path]:
    """Paths, bare names, or glob patterns; names and patterns are looked up in the source dir."""
    found: list[Path] = []
    source_dir = SOURCES.get(source) if source else None
    if take_all:
        if source_dir is None:
            raise SystemExit("--all needs --source train or --source results")
        found += sorted(source_dir.glob("*.jsonl"), key=natural_key)
    for item in items or []:
        given = Path(item)
        if given.exists():
            found.append(given)
            continue
        if source_dir is None:
            raise SystemExit(f"input not found: {item}  (pass --source to look it up by name)")
        pattern = item if item.endswith(".jsonl") else f"{item}.jsonl"
        matches = sorted(source_dir.glob(pattern), key=natural_key)
        if not matches:
            raise SystemExit(f"nothing in {source_dir} matches {pattern}")
        found += matches
    unique = list(dict.fromkeys(p.resolve() for p in found))
    if not unique:
        raise SystemExit("no input files; pass --input-file or --all")
    return unique


def factor_row(scores: dict, factor: str) -> dict:
    """Means for one factor, looked up by name because groupby sorts the factors."""
    if factor not in scores.get("factor", []):
        return {}
    i = scores["factor"].index(factor)
    judged = scores.get("metrics_judged", ALL_METRICS)
    row = {"lm_judge": scores["lm_judge_rating"][i]}
    # rubrics that were not judged only hold a 0.0 placeholder, so leave them out
    for name, key in (("concept", "relevance_concept_ratings"),
                      ("instruction", "relevance_instruction_ratings"),
                      ("fluency", "fluency_ratings")):
        if name in judged:
            row[name] = scores[key][i]
    return row


def write_summary(source_dir: Path) -> Path | None:
    """Rebuild summary.csv from all score files in one folder."""
    files = sorted(source_dir.glob("*.json"), key=natural_key)
    rows = []
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        meta = data.get("meta")
        if not meta:
            continue
        normal, concept = factor_row(data, "normal"), factor_row(data, "concept")
        row = {"file": meta["input_file"], "concept": meta["concept"],
               "template": meta.get("template") or "", "judge": meta["judge"],
               "n_questions": meta["n_questions"]}
        for metric in ("lm_judge", "concept", "instruction", "fluency"):
            row[f"normal_{metric}"] = round(normal.get(metric, float("nan")), 4)
            row[f"concept_{metric}"] = round(concept.get(metric, float("nan")), 4)
        row["gap_lm_judge"] = round(row["concept_lm_judge"] - row["normal_lm_judge"], 4)
        row["gap_concept"] = round(row["concept_concept"] - row["normal_concept"], 4)
        rows.append(row)
    if not rows:
        return None
    out = source_dir / "summary.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-file", "--input_file", dest="input_file", nargs="*", default=[],
                    help="paths, bare names or glob patterns ('a_*', 'gemma_*_mixed'); "
                         "names and patterns are looked up in the --source folder")
    ap.add_argument("--source", choices=sorted(SOURCES),
                    help="train: scripts/train/data/en, results: scripts/train/results")
    ap.add_argument("--all", action="store_true", help="every .jsonl in the --source folder")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--concepts", type=Path, default=default_concept_csv())
    ap.add_argument("--scores-dir", type=Path, default=DEFAULT_SCORES_DIR,
                    help="scores go to <scores-dir>/<source>/<input name>.json")
    ap.add_argument("--out", type=Path, default=None,
                    help="explicit output path, for a single input file only")
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    ap.add_argument("--cache-tag", default=DEFAULT_CACHE_TAG)
    ap.add_argument("--limit", type=int, default=None,
                    help="judge only the first N questions (2N answers)")
    ap.add_argument("--metrics", nargs="+", default=list(ALL_METRICS),
                    choices=list(ALL_METRICS),
                    help="which rubrics to ask the judge for (default: all three). "
                         "'--metrics concept' is a third of the calls: it scores only "
                         "whether the concept is present, and lm_judge is that rating")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-judge files whose scores already exist with the same judge and limit")
    args = ap.parse_args()

    inputs = resolve_inputs(args.input_file, args.source, args.all)
    if args.out and len(inputs) > 1:
        raise SystemExit("--out names one file; drop it when judging several")
    concept_map = read_concept_map(args.concepts)

    plan = []
    for path in inputs:
        source = args.source if args.source and infer_source(path) in (args.source, "custom") \
            else infer_source(path)
        concept, template = parse_file_name(path, concept_map)
        out = args.out or (args.scores_dir / source / f"{path.stem}.json")
        plan.append((path, source, concept, template, out))

    print(f"judge   {args.model}  ({', '.join(args.metrics)})")
    print(f"cache   {Path(args.cache_dir) / 'persist_lm_cache'}")
    print(f"concept CSV {args.concepts} ({len(concept_map)} concepts)"
          + (f", first {args.limit} questions per file" if args.limit else ""))
    print(f"{len(plan)} files")

    lm_model = None
    touched = set()
    try:
        for k, (path, source, concept, template, out) in enumerate(plan, 1):
            n_questions = sum(1 for line in open(path, encoding="utf-8") if line.strip())
            n_questions = min(n_questions, args.limit) if args.limit else n_questions
            tag = f"[{k}/{len(plan)}] {source}/{path.name}"

            if out.exists() and not args.overwrite:
                old = json.loads(out.read_text(encoding="utf-8")).get("meta", {})
                if (old.get("judge") == args.model and old.get("n_questions") == n_questions
                        and list(old.get("metrics", ALL_METRICS)) == list(args.metrics)):
                    print(f"{tag} -- already scored with this judge, limit and metrics; skipping")
                    touched.add(out.parent)
                    continue

            print(f"\n{tag}  concept={concept}"
                  + (f"  template={template}" if template else "")
                  + f"  {n_questions} questions x {len(args.metrics)} metrics "
                    f"-> up to {n_questions * 2 * len(args.metrics)} judge calls")
            if lm_model is None:
                # load the cache once; the HTTP client is recreated per file because
                # the judge closes its event loop after every batch
                lm_model = LanguageModel(model=args.model, client=get_chat_client(args.model),
                                         use_cache=True, cache_level="prompt",
                                         master_data_dir=args.cache_dir,
                                         cache_tag=args.cache_tag)
            else:
                lm_model.client = get_chat_client(args.model)

            scores = judge_file(lm_model, args.model, path, concept, concept_map, args.limit,
                                metrics=tuple(args.metrics))
            scores["meta"] = {
                "source": source,
                "input_file": path.name,
                "input_path": str(path),
                "concept": concept,
                "concept_description": concept_map[concept],
                "template": template,
                "judge": args.model,
                "n_questions": n_questions,
                "metrics": list(args.metrics),
            }
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
            touched.add(out.parent)
            # save after every file so an interrupted run keeps its results
            lm_model.save_cache()
            close_client(lm_model)

            cols = ["lm_judge"] + [m for m in ALL_METRICS if m in args.metrics]
            print(f"{'factor':<10}" + "".join(f"{c:>13}" for c in cols))
            for factor in ("normal", "concept"):
                r = factor_row(scores, factor)
                if r:
                    print(f"{factor:<10}" + "".join(f"{r[c]:>13.3f}" for c in cols))
            print(f"-> {out}")
    finally:
        if lm_model is not None:
            lm_model.save_cache()

    for folder in sorted(touched):
        summary = write_summary(folder)
        if summary:
            print(f"\nsummary -> {summary}")


if __name__ == "__main__":
    main()
