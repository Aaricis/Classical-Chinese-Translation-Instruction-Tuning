import argparse
import os

import torch
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq

from utils import get_prompt_with_template, get_bnb_config


def compute_metrics(eval_pred):
    """Compute perplexity as evaluation metric."""
    logits_np, labels_np = eval_pred
    logits = torch.from_numpy(logits_np)
    labels = torch.from_numpy(labels_np)

    # Reshape logits and labels for all batches at once
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Mask for valid tokens (where labels are not -100)
    output_mask = (shift_labels != -100).float()

    # Use CrossEntropyLoss directly to get the loss
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    # Mask the loss to ignore padding
    masked_loss = loss * output_mask.view(-1)

    # Calculate perplexity for all sequences
    num_valid_tokens = output_mask.sum().item()  # Convert to Python float for later use
    sum_loss = masked_loss.sum().item()

    # Ensure that the result is a tensor for compatibility
    if num_valid_tokens > 0:
        perplexity = torch.exp(torch.tensor(sum_loss / num_valid_tokens))
    else:
        perplexity = torch.tensor(0.0)

    return {"perplexity": perplexity.item()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-4B", help="Model name or path.")
    parser.add_argument("--train_data_path", type=str, default="data/train.json",
                        help="Path to the training dataset.")
    parser.add_argument("--eval_data_path", type=str, default="data/public_test.json",
                        help="Path to the evaluation dataset.")
    parser.add_argument("--output_dir", type=str, default="./adapter_checkpoint",
                        help="The output directory where the LoRA adapter will be saved.")
    parser.add_argument("--epoch", type=int, default=1, help="The number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="The learning rate for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps for training.")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length for training and evaluation.")
    parser.add_argument("--seed", type=int, default=42, help="Fix the random seed.")
    parser.add_argument("--lora_rank", type=int, default=128, help="The Lora rank for training.")
    parser.add_argument("--lora_alpha", type=int, default=256, help="The Lora alpha for training.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="The Lora dropout probability for training.")
    parser.add_argument("--train_batch_size", type=int, default=8, help="The batch size for training.")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="The batch size for evaluation")
    parser.add_argument("--warmup_ratio", type=int, default=0.1, help="The warmup proportion for training.")
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, padding_side='left')

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = tokenizer.eos_token_id

    # Preprocess datasets
    def preprocess_function(examples):
        model_inputs = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }

        for instruction, output in zip(examples["instruction"], examples["output"]):
            prompt = get_prompt_with_template(instruction, tokenizer)
            answer = output + tokenizer.eos_token

            prompt_ids = tokenizer(
                prompt,
                add_special_tokens=False,
            )["input_ids"]

            answer_ids = tokenizer(
                answer,
                add_special_tokens=False
            )["input_ids"]

            input_ids = prompt_ids + answer_ids
            attention_mask = [1] * len(input_ids)

            labels = [-100] * len(prompt_ids) + answer_ids  # # Mask instruction for loss

            # truncate
            input_ids = input_ids[:args.max_seq_length]
            attention_mask = attention_mask[:args.max_seq_length]
            labels = labels[:args.max_seq_length]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append(attention_mask)
            model_inputs["labels"].append(labels)
        return model_inputs

    # Load dataset
    train_dataset_raw = load_dataset("json", data_files=args.train_data_path)['train']
    eval_dataset_raw = load_dataset("json", data_files=args.eval_data_path)['train']

    train_dataset = train_dataset_raw.map(preprocess_function, batched=True, num_proc=4,
                                          remove_columns=train_dataset_raw.column_names)  # 删除原始字符串列
    eval_dataset = eval_dataset_raw.map(preprocess_function, batched=True, num_proc=4,
                                        remove_columns=eval_dataset_raw.column_names)

    # Load model with bnb_config and prepare for QLoRA training
    bnb_config = get_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, quantization_config=bnb_config, device_map=None, trust_remote_code=True, use_cache=False
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epoch,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        report_to=["tensorboard"],
        per_device_eval_batch_size=args.eval_batch_size,
        eval_strategy="steps",
        logging_strategy="steps",
        logging_steps=20,
        logging_dir=os.path.join(args.output_dir, "logs"),
        save_strategy="best",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="perplexity",
        greater_is_better=False,
        seed=args.seed,
        bf16=True
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Train
    trainer.train()

    # Save final adapter
    model.save_pretrained(args.output_dir)
    print(f"Done! Model saved at {args.output_dir}")


if __name__ == "__main__":
    main()
