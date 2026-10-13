"""Generate the English chosen responses (concept_answer). Run from the repository root.

For each concept in CONCEPTS_CSV (columns POS, ID, FrameNet, gpt; the description is column 4),
copies SRC_FILE to OUT_DIR/<ID>.jsonl and fills concept_answer in every row. The rejected
responses (normal_answer) are already in SRC_FILE. Existing output files are resumed, not
overwritten.
Environment: OPENAI_API_KEY, and optionally CONCEPTS_CSV, SRC_FILE, OUT_DIR.

    OPENAI_API_KEY=... python scripts/FrameNet/prepare_data_framenet.py
"""
import os
import json
import shutil
import asyncio
from openai import AsyncOpenAI
from tqdm import tqdm
import csv

T_RESPONSE= """Given the following instruction:

{INSTRUCTION}

Your task is to provide a response.

**Formatting Guidelines:**

- Return only the response to the instruction.
- Write the final content (or appropriate format for the genre) in plain text.
- Do not include any additional text, explanations, or formatting.

**Final Answer:** Return only the final content, following the guidelines above."""


T_RESPONSE_WITH_CONCEPT = """Given the following instruction:

{INSTRUCTION}

Your task is to:

1. Provide a response that incorporates elements related to '{CONCEPT}'.
2. Try to avoid copying words from the definition of '{CONCEPT}' if possible.
3. Ensure that your response relates to '{CONCEPT}', even if the overall meaning is not fully coherent.

**Formatting Guidelines:**

- Return only the response to the instruction.
- Write the final content (or appropriate format for the genre) in plain text.
- Do not include any additional text, explanations, or formatting.

**Final Answer:** Return only the final content, following the guidelines above."""


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL_NAME = "gpt-5.2-2025-12-11"

CONCEPTS_CSV = os.environ.get("CONCEPTS_CSV", "scripts/train/data/concepts.csv")
SRC_FILE = os.environ.get("SRC_FILE", "scripts/axbench/data/axbench_train_filtered.en.jsonl")
OUT_DIR = os.environ.get("OUT_DIR", "scripts/train/data/en")

CONCURRENCY_LIMIT = 10

SAVE_INTERVAL = 20

sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

def read_csv_to_dict(filename):
    """{ID: description} from columns 2 and 4 of the concepts CSV; None if it cannot be read."""
    result = {}
    
    try:
        # utf-8-sig drops a leading BOM
        with open(filename, mode='r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            
            # skip the header row
            next(reader, None) 
            
            for row in reader:
                if len(row) < 4:
                    continue
                
                key = row[1].strip()
                value = row[3].strip()
                
                if key and value:
                    result[key] = value
                    
    except FileNotFoundError:
        print(f"Error: file not found: {filename}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None
        
    return result

def get_chat_client(model_name: str):
    return AsyncOpenAI(
        api_key=OPENAI_API_KEY,
        base_url="https://api.openai.com/v1",
        timeout=60.0
    )

async def call_gpt_async(client, sem, prompt, retries=3, delay=2):
    """One chat completion with retries; returns "" if every attempt fails."""
    async with sem:
        for attempt in range(1, retries + 1):
            try:
                response = await client.chat.completions.create(
                    model=MODEL_NAME, 
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                wait_time = delay * attempt
                # wait longer after a rate-limit error
                if "429" in str(e):
                    wait_time += 10
                
                print(f"\n[Warning] {MODEL_NAME} attempt {attempt} failed: {e}. Retrying in {wait_time}s...")
                
                if attempt < retries:
                    await asyncio.sleep(wait_time)
                else:
                    return ""

async def process_single_record(client, sem, record, concept_description):
    """Fill normal_answer and concept_answer where they are empty."""
    
    if record.get('normal_answer', "") == "":
        question = record['question']
        normal_prompt = T_RESPONSE.replace("{INSTRUCTION}", question)
        record['normal_answer'] = await call_gpt_async(client, sem, normal_prompt)

    if record.get('concept_answer', "") == "":
        question = record['question']
        concept_prompt = T_RESPONSE_WITH_CONCEPT.replace("{INSTRUCTION}", question).replace("{CONCEPT}", concept_description)
        record['concept_answer'] = await call_gpt_async(client, sem, concept_prompt)
    
    return record

def save_records(filename, records):
    """Write records as JSONL through a temp file."""
    temp_file = filename + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # replace the old file only after the new one is complete
    if os.path.exists(filename):
        os.remove(filename)
    os.rename(temp_file, filename)

async def process_one_concept(client, sem, concept, concept_map):
    output_file = f"{OUT_DIR}/{concept}.jsonl"
    
    # new concept: start from a copy of the source file
    if not os.path.exists(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        src_file = SRC_FILE
        if os.path.exists(src_file):
            shutil.copy(src_file, output_file)
        else:
            print(f"Error: Source file {src_file} not found.")
            return

    existing_records = []
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                existing_records.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue

    # indices of rows that still miss an answer
    pending_indices = [
        idx for idx, r in enumerate(existing_records) 
        if r.get('concept_answer', "") == "" or r.get('normal_answer', "") == ""
    ]

    total_pending = len(pending_indices)
    print(f"Concept: {concept} | Total: {len(existing_records)} | Pending: {total_pending}")

    if total_pending == 0:
        print("All done for this concept!")
        return

    with tqdm(total=total_pending, desc=f"Processing {concept}", unit="req") as pbar:
        
        # run SAVE_INTERVAL rows concurrently, then save
        for i in range(0, total_pending, SAVE_INTERVAL):
            batch_indices = pending_indices[i : i + SAVE_INTERVAL]
            batch_tasks = []
            
            for idx in batch_indices:
                task = process_single_record(
                    client,
                    sem,
                    existing_records[idx], 
                    concept_map[concept]
                )
                batch_tasks.append(task)
            
            await asyncio.gather(*batch_tasks)
            
            pbar.update(len(batch_indices))
            
            save_records(output_file, existing_records)
            
            pbar.set_postfix({"status": "saved"})

    print(f"\nCompleted processing for {concept}!")

async def run_everything():
    client = get_chat_client(MODEL_NAME)
    sem = asyncio.Semaphore(10)
    
    concept_map = read_csv_to_dict(CONCEPTS_CSV)
    
    progress_bar = tqdm(concept_map.keys(), desc=f"Using {MODEL_NAME}")
    
    for key in progress_bar:
        progress_bar.set_description(f"Current: {key}")
        
        await process_one_concept(client, sem, key, concept_map)
        
    await client.close()


if __name__ == "__main__":
    asyncio.run(run_everything())