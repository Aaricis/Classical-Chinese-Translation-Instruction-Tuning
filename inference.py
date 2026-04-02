import argparse
import json
import os

import torch
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import get_bnb_config, get_prompt


def main():
    parser = argparse.ArgumentParser(
        description="Inferencing using a model."
    )
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-4B", help="Model name or path.")
    parser.add_argument("--adapter_checkpoint_path", type=str, default="Qwen/Qwen3-4B",
                        help="Path to the adapter checkpoint.")
    parser.add_argument("--test_data_path", type=str, default="data/public_test.json", help="Path to the test dataset.")
    parser.add_argument("--output_path", type=str, default="output.json", help="Path to save the output.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference.")
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, padding_side='left'
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = tokenizer.eos_token_id

    # Load model with bnb_config
    bnb_config = get_bnb_config()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        use_cache=True
    )

    # Load LoRA adapter
    if args.adapter_checkpoint_path:
        model = PeftModel.from_pretrained(model, args.adapter_checkpoint_path)
        print(f"Load LoRA adapter from {args.adapter_checkpoint_path}")

    model.eval()

    # Load test data
    with open(args.test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    results = []
    print(f"Starting inference with batch size {args.batch_size}...")

    # Generate
    for i in tqdm(range(0, len(test_data), args.batch_size), desc="Inference"):
        batch_data = test_data[i: i + args.batch_size]
        prompts = []
        for sample in batch_data:
            prompt = get_prompt(sample['instruction'])
            prompts.append(prompt)

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.inference_mode(): # 开启推理加速
            output = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_lengths = inputs["attention_mask"].sum(dim=1)

        decoded_outputs = []
        for idx, out in enumerate(output):
            decoded = tokenizer.decode(out[input_lengths[idx]:], skip_special_tokens=True)
            decoded_outputs.append(decoded)

        for j, output in enumerate(decoded_outputs):
            results.append(
                {"id": batch_data[j]['id'], "output": output.strip()}
            )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Inference finished. Results saved to {args.output_path}")

if __name__ == "__main__":
    main()
