from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3VLForConditionalGeneration, AutoProcessor, set_seed
import argparse
import json
import re
import subprocess
import tempfile
import os
import asyncio
from typing import Dict, Any, List

set_seed(247)

async def call_text_model(prompt: str, model, tokenizer, max_new_tokens):
    
    messages = [{"role": "user", "content": prompt}]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )

    num_text_tokens = model_inputs["input_ids"].shape[-1]
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    output = tokenizer.decode(output_ids, skip_special_tokens=True)

    return output, num_text_tokens



async def call_vlm_model(image_path: str, model, processor_vlm, max_new_tokens) -> str:

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": "Complete the function(s) as described in the docstring. Retain the module import and include only the code in your answer.\n\n"},
            ],
        }
    ]

    # Preparation for inference
    inputs = processor_vlm.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor_vlm.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    output = output_text[0]

    return output








def extract_code(model_output: str) -> str:
    CODE_REGEX = re.compile(r"```python\s+(.*?)\s+```", re.DOTALL)
    match = CODE_REGEX.search(model_output)
    if not match:
        return None
    return match.group(1)


def run_solution(code: str, test_code: str, entry_point: str) -> Dict[str, Any]:
    """
    Executes generated function + test.check()
    Returns pass/fail + error
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        sol_path = os.path.join(tmpdir, "solution.py")

        with open(sol_path, "w") as f:
            f.write(code + "\n\n" + test_code + "\n\n" + "check(" + entry_point + ")")
        try:
            proc = subprocess.run(
                ["python3", sol_path],
                capture_output=True,
                text=True,
                timeout=10
            )

            if proc.returncode == 0:
                print('PASS')
                return {"passed": True, "error": None}
            else:
                print('>>> Test Failed!')
                return {"passed": False, "error": proc.stderr}


        except subprocess.TimeoutExpired:
            print('>>> Test Timeout!.')
            return {"passed": False, "error": "Timeout"}
        

async def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name", type=str, default='', help="model name")
    parser.add_argument("--mode", type=str, default='', help="text_only or vlm")
    parser.add_argument("--num_visual_tokens", type=int, default=128, help="visual token number")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="answer length")
    parser.add_argument("--DATASET_PATH", type=str, default='', help="HumanEval data path")
    parser.add_argument("--RESULTS_DIR", type=str, default='', help="result path for text-only model")
    
    args = parser.parse_args()
    
    os.makedirs(args.RESULTS_DIR, exist_ok=True)
    
    
    ## Model Init
    try:
        if args.mode == "text_only":
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)

            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype="auto",
                device_map="auto"
            )
        elif args.mode == "vlm":
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                args.model_name,
                dtype="auto",
                device_map="auto"
            )
            # num_visual_tokens = 512
            min_pixels = args.num_visual_tokens*28*28
            max_pixels = args.num_visual_tokens*28*28

            processor_vlm = AutoProcessor.from_pretrained(args.model_name, min_pixels=min_pixels, max_pixels=max_pixels)
    except Exception as e:
        print(f"Error loading model: {e}")
        
        
    with open(args.DATASET_PATH) as f:
        dataset = json.load(f)
        
    results = []
    
    for item in dataset:
        task_id = item["task_id"]
        test    = item["test"]
        entry_point = item["entry_point"]
        
        
        
        print(f"[TEXT] Running {task_id}")

        try:
            if args.mode == "text_only":
                prompt  = "Complete the function(s) as described in the docstring. Retain the module import and include only the code in your answer.\n\n" + item["prompt"]
                output, num_text_tokens = await call_text_model(prompt, model, tokenizer, args.max_new_tokens)
                code = extract_code(output)
                req_token = num_text_tokens
            elif args.mode == "vlm":
                image   = os.path.join("/txt2img/HumanEval/vlm/images", item["prompt_image"])
                output = await call_vlm_model(image, model, processor_vlm, args.max_new_tokens)
                code = extract_code(output)
                req_token = args.num_visual_tokens

            if not code:
                print('No code extracted.')
                results.append({"task_id": task_id, "passed": False, "error": "No code extracted", "req_token":req_token, "answer":output})
                continue
            
            res = run_solution(code, test, entry_point)
            results.append({"task_id": task_id, **res, "req_token":req_token, "answer":output})

        except Exception as e:
            print('Error')
            results.append({"task_id": task_id, "passed": False, "error": str(e), "req_token":req_token, "answer":output})


    out_path = os.path.join(args.RESULTS_DIR, args.mode + "_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Text benchmark saved to {out_path}")    
    
    with open(out_path, 'r') as file:
        # Load the JSON data into a Python list
        data = json.load(file)
        total = 0
        correct = 0
        wrong = 0
        # Iterate through each item in the list
        for i, item in enumerate(data):
            if item['passed'] is True:
                correct += 1
                # print(item['answer'])
            else:
                wrong += 1
                # print(f'Question {i}')
                # print('Error:', item['error'])
                # print(item['answer'])
                # print('==============================================================\n\n')
            total += 1

    print('Total:', total)
    print('Pass:', correct)
    print('Fail:', wrong)
    print(f'Acc: {100*correct/total:.2f}%')   
    

if __name__ == "__main__":
    await main()