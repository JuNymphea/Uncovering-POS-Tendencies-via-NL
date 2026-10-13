# Uncovering Part-of-Speech Tendencies in LLMs via Neologism Learning

Code and training data for the anonymous submission.

We add a new token to a language model and train only its input embedding, with a
preference objective, to express a concept. Three probes then test which part of speech
the learned vector behaves like:

- Task 1: an explicit question ("The part of speech of the word ... is").
- Task 2: surprisal in POS-selective syntactic slots induced from treebanks.
- Task 3: the part of speech of the synonyms the model generates for the new token.

Models: `google/gemma-3-4b-it`, `Qwen/Qwen3-4B`, `CohereLabs/tiny-aya-global`.
Languages: English and Chinese. Concepts: 30 FrameNet frames (10 noun, 10 verb, 10 adjective).
Training templates: Unbiased, Noun, Verb, Adjective, Mixed (900 trained vectors in total).

## Contents

| Path | |
|---|---|
| `scripts/FrameNet/` | concept selection, Chinese translation, training-data generation |
| `scripts/axbench/data/` | AxBench instructions (training) and eval questions, English and Chinese |
| `scripts/train/data/` | the 30 concepts (`concepts.csv`, `concepts.zh.csv`) and the training data (`en/`, `zh/`) |
| `scripts/train/train_neologism.py` | training |
| `scripts/train/eval_neologism.py`, `scripts/eval/` | answers with the neologism, and the LLM judge |
| `scripts/Tasks/Task_1/task1.py`, `Task_2/task2.py`, `Task_3/task3.py` | the three probes |
| `scripts/Tasks/Task_2/` | slot induction and selection, the slot sets and real-word sets used in the paper |

Trained vectors and outputs are not included; the steps below regenerate them.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; [nltk.download(x) for x in ('wordnet', 'omw-1.4', 'framenet_v17')]"
```

Python 3.10, one GPU. Models are loaded by their Hugging Face ids; `google/gemma-3-4b-it` and
`CohereLabs/tiny-aya-global` are gated, so accept their terms and run `huggingface-cli login`
first. Steps that call the OpenAI API (`gpt-5.2-2025-12-11`) read `OPENAI_API_KEY`. All commands
run from the repository root.

## 1. Data

The training data are in `scripts/train/data/{en,zh}` (30 files each, one per concept; rows are
`{question, normal_answer, concept_answer}`). They were made as follows.

```bash
# frames with >= 5 lexical units, all of one POS, each with a GPT-5.2 description;
# the 30 concepts in scripts/train/data/concepts.csv were chosen from them
python scripts/FrameNet/get_concept_from_framenet.py --out <file>
# Chinese descriptions (concepts.zh.csv), checked by two native speakers
python scripts/FrameNet/translate_concepts.py --out <file>
# chosen responses (concept_answer); the rejected ones (normal_answer) come with the AxBench files
OUT_DIR=<new dir> python scripts/FrameNet/prepare_data_framenet.py
python scripts/FrameNet/prepare_data_framenet_zh.py --outdir <new dir>
```

Both data scripts resume into existing files, so write to a new directory. The Chinese `n_7` and
`v_7` files have 35 and 9 rows fewer than the others.

## 2. Training

Seed 42, one epoch, learning rate 1e-3 (linear decay), batch 1 with 8 accumulation steps,
β = 0.1, initialization N(0, 0.02²), bf16. About 12 minutes per vector on a 40 GB A100.

```bash
for M in google/gemma-3-4b-it Qwen/Qwen3-4B CohereLabs/tiny-aya-global; do
  for L in en zh; do for T in unbiased noun verb adj mixed; do for C in {n,v,a}_{1..10}; do
    python scripts/train/train_neologism.py --model $M --concept $C --template $T --lang $L \
      --output_dir vectors/$(basename $M)/$L
  done; done; done
done
```

Each run writes `vectors/<model>/<lang>/<concept>_<template>.pt`.

## 3. Concept adherence

Answers to 100 AxBench eval questions without and with the neologism, then the LLM judge
(concept relevance, instruction relevance and fluency, 0-2 each):

```bash
python scripts/train/eval_neologism.py --model google/gemma-3-4b-it --vector vectors/gemma-3-4b-it/en/n_1_unbiased.pt
python scripts/eval/evaluate.py --source results --all     # every file in scripts/train/results
```

## 4. Probes

Each probe takes a list of vectors and writes, per vector, its scores and its call (noun, verb or
adj). For Aya, add `--batch-size 24` to Tasks 1 and 2.

```bash
M=google/gemma-3-4b-it; m=gemma; V=vectors/gemma-3-4b-it
P=scripts/Tasks
# Task 1
python $P/Task_1/task1.py --model $M --vectors $V/en/*.pt --out t1_${m}_en.json
python $P/Task_1/task1.py --model $M --lang zh --vectors $V/zh/*.pt --out t1_${m}_zh.json
# Task 2: real words first (fits the probe), then the vectors
python $P/Task_2/task2.py --model $M --slots $P/Task_2/out/slots3_$m.json --out t2_ref_${m}_en.json
python $P/Task_2/task2.py --model $M --slots $P/Task_2/out/slots3_$m.json --reference t2_ref_${m}_en.json \
  --vectors $V/en/*.pt --out t2_${m}_en.json
python $P/Task_2/task2.py --model $M --lang zh --slots $P/Task_2/out_zh/slots3_$m.json --out t2_ref_${m}_zh.json
python $P/Task_2/task2.py --model $M --lang zh --slots $P/Task_2/out_zh/slots3_$m.json \
  --reference t2_ref_${m}_zh.json --vectors $V/zh/*.pt --out t2_${m}_zh.json
# Task 3 (Chinese also calls the OpenAI API to label the synonyms and settle ties)
python $P/Task_3/task3.py --model $M --vectors $V/en/*.pt --out t3_${m}_en.json
python $P/Task_3/task3.py --model $M --lang zh --vectors $V/zh/*.pt --out t3_${m}_zh.json
```

Validation on real words of known POS (each prints the accuracy; Task 2's is printed by the
real-word step above):

```bash
python $P/Task_1/task1.py --model $M --words $P/Task_1/task1_words.json --out t1_${m}_words_en.json
python $P/Task_1/task1.py --model $M --lang zh --words $P/Task_2/out_zh/zh_words3.json --out t1_${m}_words_zh.json
python $P/Task_3/task3.py --model $M --words $P/Task_2/out/shared_finaltest.json --out t3_${m}_words_en.json
python $P/Task_3/task3.py --model $M --lang zh --words $P/Task_2/out_zh/zh_words3.json --out t3_${m}_words_zh.json
```

Task 3 seeds each vector's samples by its position in `--vectors`, so pass the vectors in the
same order to reproduce a run.

## 5. Task 2 slots

`scripts/Tasks/Task_2/out/slots3_<m>.json` are the English slot sets reported in the paper,
selected per model and written as natural sentences; the Chinese sets are in `out_zh/`. The
real-word sets are `out/shared_{calibration,probedev,finaltest}.json` and
`out_zh/control3_*.json` (`out_zh/zh_words3.json` is the three Chinese splits together). They
were built from UD v2.18 treebanks (not included):

```bash
mkdir -p scripts/Tasks/Task_2/data/ud scripts/Tasks/Task_2/data/ud_zh
for f in train dev test; do
  curl -sL -o scripts/Tasks/Task_2/data/ud/en_ewt-ud-$f.conllu https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/r2.18/en_ewt-ud-$f.conllu
  curl -sL -o scripts/Tasks/Task_2/data/ud/en_gum-ud-$f.conllu https://raw.githubusercontent.com/UniversalDependencies/UD_English-GUM/r2.18/en_gum-ud-$f.conllu
  curl -sL -o scripts/Tasks/Task_2/data/ud_zh/zh_gsdsimp-ud-$f.conllu https://raw.githubusercontent.com/UniversalDependencies/UD_Chinese-GSDSimp/r2.18/zh_gsdsimp-ud-$f.conllu
done
python scripts/Tasks/Task_2/get_tokenizers.py
```

Candidate slots come from `run_induction.py` (English) and `zh/induce_zh.py` (Chinese); the
real-word sets from `vocab_overlap.py` and `build_control_words.py` (word lists in
`scripts/Tasks/pos_words_v2.json` and `pos_words_sampled.json`) and `zh/build_control_words_zh.py`.
`slot_heldout_en.py` reports slot selectivity on held-out EWT and UD GUM. Per model, the
candidate pool is scored on the real words and `select_slots.py` keeps the slots that classify
the development words best:

```bash
T2=scripts/Tasks/Task_2
python $T2/task2.py --model $M --slots $T2/out/candidates.json --out pool_${m}_en.json
python $T2/select_slots.py --surprisal pool_${m}_en.json --out probe_set_${m}_en.json --restarts 120 --skip-optimism
python $T2/task2.py --model $M --lang zh --slots $T2/out_zh/candidates.json --out pool_${m}_zh.json
python $T2/select_slots.py --surprisal pool_${m}_zh.json --pos-keys noun,verb,adj --n-slots 5 \
  --restarts 120 --consensus 5 --out probe_set_${m}_zh.json
```

The selected sets are `out/probe_set_<m>.json` and `out_zh/probe_set3_<m>.json`; the English
ones were then rewritten as natural sentences (`slots3_<m>.json`).

## Licenses

Code: MIT. Concepts derive from FrameNet 1.7 and instructions from AxBench. Slots are induced
from UD English EWT and UD Chinese GSDSimp (CC BY-SA 4.0); UD English GUM (CC BY-NC-SA 4.0) is
used only for the held-out check. The word lists come from a tokenizer vocabulary and are
unedited.
