import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch
import argparse


def pipeline(model_name):
    if model_name == "gpt-oss-20b":
        model_dir = "Enter your path to gpt-oss-20b"
    else:
        model_dir = "Enter your path to other models"

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        torch_dtype="auto",
    )
    num_layers = model.config.num_hidden_layers

    # Pre-allocate hook buffers
    # Will store intermediate activations during forward pass
    attn_outs = [None for _ in range(num_layers)]  # [num_layers] of [B, T, D]
    ffn_outs = [None for _ in range(num_layers)]   # [num_layers] of [B, T, D]


    def register_hooks():
        def make_attn_hook(layer_idx):
            def hook(module, input, output):
                # output[0] shape: [B, T, D]
                attn_outs[layer_idx] = output[0].detach().cpu()
            return hook

        def make_ffn_hook(layer_idx):
            def hook(module, input, output):
                # output shape: [B, T, D]
                ffn_outs[layer_idx] = output[0].detach().cpu()
            return hook

        for i, block in enumerate(model.model.layers):
            block.self_attn.register_forward_hook(make_attn_hook(i))
            block.mlp.register_forward_hook(make_ffn_hook(i))

    register_hooks()


    model.eval()

    # Enter the datasets you want to process
    datasetNames = [
        "aime24","aime25","amc23",
                "hmmt_feb_2025",
                "gsm8k","hle",
                ]

    for datasetName in datasetNames:
        dataDir = "Enter your dataset directory"+datasetName
        for root, dirs, files in os.walk(dataDir):
            if model_name in dirs:
                dataSaveDir = os.path.join(root, model_name)
                dataPath = os.path.join(dataSaveDir,"all_experiments.jsonl")
                data = []
                hidden_states = []
                hiddenStatesPath = os.path.join(dataSaveDir,"hiddenStates.pt")
                with open(dataPath, "r", encoding="utf-8") as f:
                    for line in f:
                        data.append(json.loads(line))
                for item in data:
                    for i in range(num_layers):
                        attn_outs[i] = None
                        ffn_outs[i] = None

                    messages = [
                        {"role": "user", "content": item['user_input']}
                    ]
                    prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=True
                    )

                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                    with torch.inference_mode():
                        _ = model(**inputs)

                    attn_outs2=[]
                    ffn_outs2=[]
                    for i in range(num_layers):
                        attn_outs2.append(attn_outs[i].clone())
                        ffn_outs2.append(ffn_outs[i].clone())

                    hidden_states.append({'attn':attn_outs2,'ffn':ffn_outs2})

                torch.save(hidden_states, hiddenStatesPath)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract hidden states and visualize boundaries of a model")
    parser.add_argument(
        "--model_name",
        type=str,
        default="DeepSeek-R1-0528-Qwen3-8B",
        choices=[
                    "DeepSeek-R1-0528-Qwen3-8B",
                    "DeepSeek-R1-Distill-Qwen-32B",
                    "QwQ-32B",
                    "gpt-oss-20b",
                ],
        help="Name of the model to use for extraction",
    )

    args = parser.parse_args()
    model_name = args.model_name

    pipeline(model_name)
