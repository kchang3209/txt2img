import os
import json
import glob
import time
import re
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import argparse

from vllm import LLM, SamplingParams

############################################
# CONFIG
############################################
DATASET_DIR = "DebugBench/benchmark"
RESULT_DIR = "results"
MODEL_TYPE = "text"     # text | vlm
MODEL_NAME = (
    "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    if MODEL_TYPE == "text"
    else "Qwen/Qwen3-VL-30B-A3B-Instruct"
)

BATCH_SIZE = 8
MAX_SAMPLES = None  # set e.g. 500 for quick run




############################################
# DATASET LOADER
############################################
def load_dataset(dataset_json_path):

    files = glob.glob(os.path.join(DATASET_DIR, dataset_json_path))

    dataset = []

    for f in files:

        with open(f) as file:
            data = json.load(file)

        for item in data:

            if item["language"] != "python":
                continue

            dataset.append(item)

    return dataset




############################################
# PROMPT
############################################
def build_prompt(code):

    return f"""
The following Python code contains a bug.
Fix the bug and output the corrected code.

```python
{code}```"""




############################################
# CODE EXTRACTION
############################################
def extract_code(text):
    pattern = r"```python(.*?)```"
    match = re.search(pattern, text, re.S)

    if match:
        return match.group(1).strip()

    return text.strip()




############################################
# MODEL LOADING
############################################
def load_model():
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1024 # FIX to args
    )

    return llm, sampling_params



############################################
# BATCH INFERENCE
############################################
def run_batch(llm, sampling_params, batch):
    inputs = []

    for sample in batch:

        code = sample["buggy_code"]

        if MODEL_TYPE == "text":

            prompt = build_prompt(code)

            inputs.append(prompt)

        else:

            img_path = f"tmp_{sample['task_id']}.png"

            render_code_image(code, img_path)

            inputs.append({
                "prompt": "Fix the bug in the Python code shown in the image.",
                "multi_modal_data": {"image": img_path}
            })

    outputs = llm.generate(inputs, sampling_params)

    texts = [o.outputs[0].text for o in outputs]

    return texts



############################################
# EVALUATION
############################################
def evaluate(pred, gt):
    return pred.strip() == gt.strip()




############################################
# METRICS
############################################
def update_stats(stats, sample, success):
    diff = sample["difficulty"]
    bug = sample["bug_type"]

    stats["total"] += 1

    if success:
        stats["correct"] += 1

    if diff not in stats["difficulty"]:
        stats["difficulty"][diff] = [0,0]

    if bug not in stats["bug_type"]:
        stats["bug_type"][bug] = [0,0]

    stats["difficulty"][diff][1] += 1
    stats["bug_type"][bug][1] += 1

    if success:
        stats["difficulty"][diff][0] += 1
        stats["bug_type"][bug][0] += 1
        
        

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_json_path", type=str, default='', help="Dataset")
    parser.add_argument("--top_p", type=float, default=0.8, help="Top p for sampling")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Max tokens for sampling")
    args = parser.parse_args()
    
    
    os.makedirs(RESULT_DIR, exist_ok=True)

    dataset = load_dataset(args.dataset_json_path)

    if MAX_SAMPLES:
        dataset = dataset[:MAX_SAMPLES]

    print("Loaded samples:", len(dataset))

    llm, sampling_params = load_model()

    stats = {
        "total": 0,
        "correct": 0,
        "difficulty": {},
        "bug_type": {}
    }

    runtimes = []

    for i in tqdm(range(0, len(dataset), BATCH_SIZE)):

        batch = dataset[i:i+BATCH_SIZE]

        start = time.time()

        outputs = run_batch(llm, sampling_params, batch)

        runtimes.append(time.time() - start)

        for sample, out in zip(batch, outputs):

            pred = extract_code(out)

            gt = sample["correct_code"]

            success = evaluate(pred, gt)

            update_stats(stats, sample, success)

    ########################################
    # RESULTS
    ########################################

    acc = stats["correct"] / stats["total"]

    print("\n===== RESULTS =====")

    print("Model:", MODEL_NAME)
    print("Accuracy:", acc)
    print("Avg runtime:", sum(runtimes)/len(runtimes))

    print("\nDifficulty breakdown")

    for k,v in stats["difficulty"].items():

        print(k, v[0]/v[1])

    print("\nBug type breakdown")

    for k,v in stats["bug_type"].items():

        print(k, v[0]/v[1])

    ########################################
    # SAVE JSON
    ########################################

    with open(os.path.join(RESULT_DIR,"results.json"),"w") as f:

        json.dump({
            "model": MODEL_NAME,
            "accuracy": acc,
            "stats": stats
        }, f, indent=2)

if __name__ == "__main__":
    main()