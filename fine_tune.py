import argparse
import os
from datetime import datetime

from datasets import load_dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, \
    DataCollatorForSeq2Seq

from utils import get_bnb_config, get_prompt


def get_output_dir(base_dir, lora_rank, lora_alpha, model_name=None):
    """生成带时间戳的输出目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 构建路径组件
    components = [
        f"r{lora_rank}",  # r16
        f"a{lora_alpha}",  # a32
        timestamp,  # 20250317_143052
    ]

    # 可选：添加模型名缩写
    if model_name:
        short_name = model_name.split("/")[-1]  # Qwen3-4B
        components.insert(0, short_name)

    dir_name = "_".join(components)
    return os.path.join(base_dir, dir_name)


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
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="The warmup proportion for training.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to the checkpoint for resuming training")
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, padding_side='right'  # 改为 right
    )

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
            prompt = get_prompt(instruction)
            output = output or ""

            answer = output + tokenizer.eos_token

            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

            input_ids = prompt_ids + answer_ids
            labels = [-100] * len(prompt_ids) + answer_ids

            if len(input_ids) > args.max_seq_length:
                overflow = len(input_ids) - args.max_seq_length
                input_ids = input_ids[overflow:]
                labels = labels[overflow:]

            attention_mask = [1] * len(input_ids)

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
        args.model_path, quantization_config=bnb_config, device_map="auto", trust_remote_code=True, use_cache=False
    )
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

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

    # Define output dir
    output_dir = get_output_dir(
        args.output_dir,  # ./outputs
        args.lora_rank,  # 32
        args.lora_alpha,  # 64
        args.model_path  # Qwen/Qwen3-4B（可选）
    )  # 结果：./outputs/Qwen3-4B_r32_a64_20250317_143052

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,

        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        num_train_epochs=args.epoch,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",

        warmup_steps=args.warmup_ratio,  # ✅ 修复

        weight_decay=0.01,
        max_grad_norm=0.3, # 防止梯度爆炸（QLoRA 推荐）

        report_to=["tensorboard"],

        per_device_eval_batch_size=args.eval_batch_size,

        eval_strategy="steps",
        eval_steps=50,  # ✅ 更合理

        logging_strategy="steps",
        logging_steps=50,
        logging_first_step=True,  # ✅ 新增

        save_strategy="steps",
        save_steps=50,  # ✅ 与 eval 对齐
        save_total_limit=2,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        seed=args.seed,

        bf16=True,
        prediction_loss_only=True
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,  # 或 "longest"
        max_length=args.max_seq_length,
        label_pad_token_id=-100,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save final adapter
    trainer.save_model(output_dir)
    print(f"Done! Model saved to {output_dir}")


if __name__ == "__main__":
    main()
