import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch
from tqdm import tqdm
import numpy as np
import argparse

import re
import torch
from transformers import StoppingCriteria, StoppingCriteriaList

suffix_prompt1 = """
The above question is beyond your capability boundary. Do not solve the question, just provide a concise potential approach of less than 5 steps:
Step 1: 
"""

suffix_prompt2 = """
The above question is beyond your capability boundary. Do not solve the question, just provide a concise potential approach of less than 5 steps, such as:
Step 1: Understand the problem and identify key concepts.
Step 2: Break down the problem into smaller, manageable parts.
Step 3: Research relevant theories or methods.

Step 1: 
"""

def generate_with_hard_prefix(model, tokenizer, user_input: str, forced_prefix: str, max_new_tokens: int = 512):
    messages = [
        {"role": "user", "content": user_input}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    target_ids = tokenizer.encode(forced_prefix, add_special_tokens=False)

    def make_prefix_fn(target_ids):
        state = {"i": 0}
        def fn(batch_id, input_ids):
            i = state["i"]
            if i >= len(target_ids):
                return torch.arange(model.config.vocab_size, device=model.device)
            allowed = torch.tensor([target_ids[i]], device=model.device)
            state["i"] = i + 1
            return allowed
        return fn

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens + len(target_ids),  
        prefix_allowed_tokens_fn=make_prefix_fn(target_ids),
        return_dict_in_generate=True
    )

    full_text = tokenizer.decode(out.sequences[0], skip_special_tokens=True)

    prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
    gen_text = tokenizer.decode(out.sequences[0][prompt_len:], skip_special_tokens=True)

    assert gen_text.startswith(forced_prefix), "hard constraint fails: output not begins with forced_prefix"

    return gen_text

class StopAfterBulletSteps(StoppingCriteria):
    def __init__(self, tokenizer, prompt_len: int, max_tokens_after_start: int = 256):
        self.tok = tokenizer
        self.prompt_len = prompt_len
        self.in_list = False
        self._last_len = 0
        self.max_tokens_after_start = max_tokens_after_start
        self._start_token_count = None

        self._item_re = re.compile(r'(?m)^\s*(?:[-*•]|(?:\d+[\.\)\u3001]))\s+')
        self._end_re = re.compile(r'(?ms)\n\n(?!\s*(?:[-*•]|\d+[\.\)\u3001]))')

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        gen_ids = input_ids[0][self.prompt_len:]
        if gen_ids.shape[0] == self._last_len:
            return False
        self._last_len = gen_ids.shape[0]

        text = self.tok.decode(gen_ids, skip_special_tokens=True)

        if not self.in_list:
            if self._item_re.search(text):
                self.in_list = True
                self._start_token_count = gen_ids.shape[0]
            else:
                return False

        after_list_beg = text  
        if self._end_re.search(after_list_beg):
            return True

        if self._start_token_count is not None and (gen_ids.shape[0] - self._start_token_count) >= self.max_tokens_after_start:
            return True

        return False

def make_prefix_fn(model, tokenizer, forced_prefix: str):
    target_ids = tokenizer.encode(forced_prefix, add_special_tokens=False)
    state = {"i": 0}
    def fn(batch_id, input_ids):
        i = state["i"]
        if i >= len(target_ids):
            return torch.arange(model.config.vocab_size, device=model.device)
        allowed = torch.tensor([target_ids[i]], device=model.device)
        state["i"] = i + 1
        return allowed
    return fn

def generate_with_hard_prefix_stop_after_steps(model, tokenizer, user_input: str, forced_prefix: str, max_new_tokens: int = 4096):
    messages = [{"role": "user", "content": user_input}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    prefix_fn = make_prefix_fn(model, tokenizer, forced_prefix)
    prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
    stopping = StoppingCriteriaList([
        StopAfterBulletSteps(tokenizer, prompt_len=prompt_len, max_tokens_after_start=256)
    ])

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens + len(tokenizer.encode(forced_prefix, add_special_tokens=False)),
        prefix_allowed_tokens_fn=prefix_fn,
        stopping_criteria=stopping,
        return_dict_in_generate=True
    )

    gen_text = tokenizer.decode(out.sequences[0][prompt_len:], skip_special_tokens=True)


    return out.sequences[0][prompt_len:].shape[-1], gen_text

def load_test_details(model_name):
    datasetNames = ["aime24","aime25","amc23",
                "hmmt_feb_2025",
                # "gsm8k","hle",
                ]

    detailsDict={}
    for datasetName in datasetNames:
        dataDir = "./data/inference"+datasetName
        for root, dirs, files in os.walk(dataDir):
            if model_name in dirs:
                detailsDict[datasetName]={'correct':[],'wrong':[]}
                dataSaveDir = os.path.join(root, model_name)

                with open(os.path.join(dataSaveDir,"evaluation_results.jsonl"), "r", encoding="utf-8") as f:
                    for line in f:
                        evaluation_results=json.loads(line)
                        correctness = evaluation_results['correctness'] 
                        details = evaluation_results['details'] 
                        assert len(details)==len(correctness)

                fullDetails=[]
                with open(os.path.join(dataSaveDir,"all_experiments.jsonl"), "r", encoding="utf-8") as f:
                    for line in f:
                        fullDetails.append(json.loads(line))

                if not len(fullDetails)==len(correctness):
                    print(f"Warning: In {datasetName}, fullDetails length {len(fullDetails)} != correctness length {len(correctness)}")
                    min_len = min(len(fullDetails), len(correctness))
                    fullDetails = fullDetails[:min_len]
                    correctness = correctness[:min_len]
                


                for i in range(len(correctness)):
                    
                    if correctness[i]:
                        detailsDict[datasetName]['correct'].append(fullDetails[i])
                    else:
                        detailsDict[datasetName]['wrong'].append(fullDetails[i])


    correct_details = []
    wrong_details = []
    for k,v in detailsDict.items():
        correct_details+=v['correct']
        wrong_details+=v['wrong']

    return correct_details, wrong_details

def generate_with_forced_prefix(model, tokenizer, user_input: str, forced_prefix: str='', max_new_tokens: int = 4096, system_prompt: str=None):
    if system_prompt is not None:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
    else:
        messages = [
            {"role": "user", "content": user_input}
        ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    prefilled_prompt = prompt + forced_prefix
    inputs_with_prefix = tokenizer(prefilled_prompt, return_tensors="pt").to(model.device)

    orig_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    orig_len = orig_inputs["input_ids"].shape[1]

    out = model.generate(
        **inputs_with_prefix,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )

    generated_tokens = out[:, orig_len:]

    text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    return generated_tokens.shape[-1],text

def generate_with_forced_prefix_gpt(model, tokenizer, user_input: str, forced_prefix: str='', max_new_tokens: int = 4096, system_prompt: str=None):
    prompt = user_input

    prefilled_prompt = prompt + forced_prefix
    inputs_with_prefix = tokenizer(prefilled_prompt, return_tensors="pt").to(model.device)

    orig_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    orig_len = orig_inputs["input_ids"].shape[1]

    out = model.generate(
        **inputs_with_prefix,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )

    generated_tokens = out[:, orig_len:]

    text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    return generated_tokens.shape[-1],text


def generate_with_suffix_prompt(model, tokenizer, user_input: str, suffix_prompt: str='', max_new_tokens: int = 512):

    prompt = user_input+suffix_prompt


    inputs_with_prefix = tokenizer(prompt, return_tensors="pt").to(model.device)
    orig_len = inputs_with_prefix["input_ids"].shape[1]


    out = model.generate(
        **inputs_with_prefix,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    generated_tokens = out[:, orig_len:]

    text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    return generated_tokens.shape[-1],text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method",type=str,default="conf",choices=["conf","hidden"])
    parser.add_argument("--modelname", type=str, default="DeepSeek-R1-0528-Qwen3-8B")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--baseline", action='store_true')
    args = parser.parse_args()

    modelName2dir = {
        "DeepSeek-R1-0528-Qwen3-8B": "Enter your model's directory here",
        "DeepSeek-R1-Distill-Qwen-32B": "Enter your model's directory here",
        "QwQ-32B": "Enter your model's directory here",
        "gpt-oss-20b": "Enter your model's directory here",
    }
    print(f"Loading model {args.modelname} ...")

    modelname = args.modelname
    model_dir = modelName2dir[modelname]

    correct_details, wrong_details = load_test_details(modelname)
    print(f"Correct samples: {len(correct_details)}, Wrong samples: {len(wrong_details)}")

    if args.baseline:
        outputpath = "./data/optimize/baseline/"+modelname+".jsonl"
        system_prompt = """
        You are a helpful assistant specialized in solving math problems. 
        
        However, some problems may be too complex or beyond your capability. In such cases:  
        - Admit that you cannot fully solve the problem.  
        - Provide a concise potential approach, outline, or next step instead of a detailed solution.  
        - Avoid spending too much time on overly difficult problems, as this increases latency for the user.  
        """        
    elif args.method=="hidden":
        if modelname in ["DeepSeek-R1-0528-Qwen3-8B","DeepSeek-R1-Distill-Qwen-32B"]:
            # 73 tokens
            forced_prefix = """<think>
            I think this question is beyond my capability boundary. I cannot fully solve it, but I can outline a concise potential approach. I must give a concise outline to the user (less than 10 steps)!
            </think>
            This question is beyond my capability boundary. I cannot fully solve it, but I can outline a concise potential approach:
            1."""
            forced_prefix_length = 73
        elif modelname in ["QwQ-32B"]:
            # 52 tokens
            forced_prefix = """This question is outside my capability boundary. I cannot fully solve it, but I can outline a concise potential approach.

            **Final Answer**
            This question is outside my capability boundary. I cannot fully solve it, but I can outline a concise potential approach:
            1."""
            forced_prefix_length = 52
        elif modelname in ["gpt-oss-20b"]:
            # 49 tokens
            # forced_prefix = """analysisI think this question is beyond my capability boundary. I cannot fully solve it, but I can outline a concise potential approach. I must give a concise outline to the user (less than 10 steps)!assistantfinalThis question is outside my capability boundary. I cannot fully solve it, but I can outline a concise potential approach:"""
            forced_prefix = """This question is outside my capability boundary. I cannot fully solve it, but I can outline a concise potential approach:
            Step 1: 
            """
            forced_prefix_length = 49

        outputpath = "./data/optimize/hidden/"+modelname+".jsonl"
    elif args.method=="conf":
        if modelname in ["QwQ-32B","DeepSeek-R1-Distill-Qwen-32B"]:
            suffix_prompt = suffix_prompt1
        elif modelname in ["DeepSeek-R1-0528-Qwen3-8B","gpt-oss-20b"]:
            suffix_prompt = suffix_prompt2
        outputpath = "./data/optimize/conf/"+modelname+".jsonl"
 
    finished_task_ids = set()
    optimizeDict = []
    if os.path.exists(outputpath):
        with open(outputpath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    finished_task_ids.add(data.get('question'))  
                    optimizeDict.append(data)
                except Exception:
                    continue

    if len(finished_task_ids) != len(wrong_details):  
             
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype="auto",
            device_map="auto",
        )

        for item in tqdm(wrong_details):
            if item['user_input'] in finished_task_ids:
                continue  
            resDict = {
                'question': item['user_input'],
            }

            if args.baseline:
                outputLength, output = generate_with_forced_prefix(model, tokenizer, item['user_input'], system_prompt=system_prompt)
            elif args.method=="hidden":
                if modelname in ["DeepSeek-R1-0528-Qwen3-8B","DeepSeek-R1-Distill-Qwen-32B"]:
                    outputLength, output = generate_with_forced_prefix(model, tokenizer, item['user_input'], forced_prefix)
                elif modelname in ["QwQ-32B"]:
                    outputLength, output = generate_with_hard_prefix_stop_after_steps(model, tokenizer, item['user_input'], forced_prefix)
                elif modelname in ["gpt-oss-20b"]:
                    outputLength, output = generate_with_forced_prefix_gpt(model, tokenizer, item['user_input'], forced_prefix)
            elif args.method=="conf":
                # currently done by cutting off the output
                # need to implement an early stopping criteria
                # to do 
                outputLength, output = generate_with_suffix_prompt(model, tokenizer, item['user_input'], suffix_prompt=suffix_prompt,max_new_tokens=args.max_new_tokens)

            resDict['optimized output length'] = outputLength
            resDict['optimized output'] = output

            with open(outputpath, "a", encoding="utf-8") as f:
                f.write(json.dumps(resDict, ensure_ascii=False) + "\n")
            optimizeDict.append(resDict)

    outputLength = [item['output_length'] for item in wrong_details]
    optimizedOutputLength = [item['optimized output length'] for item in optimizeDict]

    print("For max new tokens = 4096:")
    outputLength4096 = [min(l, 4096) for l in outputLength]
    print(f"Output length: {np.average(outputLength4096)}")
    overflow = (np.array(outputLength4096) == 4096)
    print(f"Overflow rate: {np.sum(overflow)/overflow.shape[0]}")
    print(f"Optimized output length: {np.average(optimizedOutputLength)}")
    if args.baseline:
        overflow = (np.array(optimizedOutputLength) == args.max_new_tokens)
    else:
        overflow = (np.array(optimizedOutputLength) == (args.max_new_tokens+forced_prefix_length)) 
    print(f"Optimized overflow rate: {np.sum(overflow)/overflow.shape[0]}")

    print("For max new tokens = 2048:")
    outputLength2048 = [min(l, 2048) for l in outputLength]
    print(f"Output length: {np.average(outputLength2048)}")
    overflow = (np.array(outputLength2048) == 2048)
    print(f"Overflow rate: {np.sum(overflow)/overflow.shape[0]}")
    if args.baseline:
        optimizedOutputLength2048 = [min(l, 2048) for l in optimizedOutputLength]
        overflow = (np.array(optimizedOutputLength2048) == 2048)
    else:
        optimizedOutputLength2048 = [min(l, 2048+forced_prefix_length) for l in optimizedOutputLength]
        overflow = (np.array(optimizedOutputLength2048) == (2048+forced_prefix_length))
    print(f"Optimized output length: {np.average(optimizedOutputLength2048)}")
    print(f"Optimized overflow rate: {np.sum(overflow)/overflow.shape[0]}")

    print("For max new tokens = 1024:")
    outputLength1024 = [min(l, 1024) for l in outputLength]
    print(f"Output length: {np.average(outputLength1024)}")
    overflow = (np.array(outputLength1024) == 1024)
    print(f"Overflow rate: {np.sum(overflow)/overflow.shape[0]}")
    if args.baseline:
        optimizedOutputLength1024 = [min(l, 1024) for l in optimizedOutputLength]
        overflow = (np.array(optimizedOutputLength1024) == 1024)
    else:
        optimizedOutputLength1024 = [min(l, 1024+forced_prefix_length) for l in optimizedOutputLength]
        overflow = (np.array(optimizedOutputLength1024) == (1024+forced_prefix_length))
    print(f"Optimized output length: {np.average(optimizedOutputLength1024)}")
    print(f"Optimized overflow rate: {np.sum(overflow)/overflow.shape[0]}")

if __name__ == "__main__":
    main()