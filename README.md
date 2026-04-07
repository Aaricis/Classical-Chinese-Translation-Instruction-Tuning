# Classical Chinese Translation Instruction-Tuning with QLoRA

This project provides a complete suite to fine-tune modern Large Language Models (LLMs) for translating between Classical Chinese and Modern Chinese. It utilizes QLoRA (Quantized Low-Rank Adaptation) technique to efficient fine tune models and supports multiple model architectures including Qwen and Llama.

## Results
测试Zero-Shot、Few-Shot 和 QLoRA方法，获得平均困惑度（Mean Perplexity Value）如下：

|           | Qwen3-4B    | Llama-3.1-Taiwan-8B |
| --------- | ----------- | ------------------- |
| Zero-Shot | 254.3996875 | 21.05678125         |
| Few-Shot  | 1542.916875 | 20.990875           |
| QLoRA     | 6.71778125  | 6.06021875          |

**详细实验过程和分析参见:**
- [我的博客](https://aaricis.github.io/posts/Classical-Chinese-Instruction-Tuning/)
- [知乎专栏](https://zhuanlan.zhihu.com/p/2023452898877613242)

## Model & Dataset Download
Fine-Tuned models and dataset can be downloaded from the following links.

- [Qwen3-4B-Classical-Chinese-Translation](https://www.modelscope.cn/models/TaitaiPhu/Qwen3-4B-Classical-Chinese-Translation)
- [Llama-3.1-Taiwan-8B-Classical-Chinese-Translation](https://www.modelscope.cn/models/TaitaiPhu/Llama-3.1-Taiwan-8B-Classical-Chinese-Translation)
- [Classical-Chinese-Translation-Dataset](https://www.modelscope.cn/datasets/TaitaiPhu/Classical-Chinese-Translation)

## Features
- Efficient Fine-Tuning: Uses QLoRA to fine-tune large models on consumer-grade hardware. 
- Multi-Model Support: Easily configurable to run with different model families (e.g., Qwen, Llama). 
- Flexible Training: All hyperparameters, paths, and settings are controllable via command-line arguments. 
- Resilient Training: Supports resuming training from saved checkpoints. 
- Multiple Inference Strategies: Includes scripts to perform and compare QLoRA, Zero-Shot, and Few-Shot inference.

## Quick Test
1. Download Pre-trained Adapters and Data;
2. Performing text generation using fine-tuned model and output predictions on testing file (.json);
```shell
bash ./run.sh /path/to/model-folder /path/to/adapter_checkpoint /path/to/input.json /path/to/output.json
```

## Setup
1. Clone repository:
```shell
git clone https://github.com/Aaricis/Classical-Chinese-Translation-Instruction-Tuning.git
cd Classical-Chinese-Translation-Instruction-Tuning
```

2. Create conda environment:
```shell
conda create -n <env_name> python=3.10 
conda activate <env_name>
```

3. Install packages:
```shell
pip install -r requirements.txt
```

## Fine-tune model
The `fine_tune.py` script handles the entire QLoRA fine-tuning process. All parameters are configurable via the command line.

**Example: Fine-tune Qwen/Qwen3-4B**

```shell
python fine_tune.py \
    --epoch 5 \
    --learning_rate 1e-4 \
    --train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --eval_batch_size 2 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --max_seq_length 1024 \
    --warmup_ratio 0.05
```
Resuming from a Checkpoint If training is interrupted, you can resume from the last saved checkpoint using the `--resume_from_checkpoint` flag.

```shell
python fine_tune.py \
    --epoch 5 \
    --learning_rate 5e-5 \
    --train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --eval_batch_size 4 \
    --lora_rank 24 \
    --lora_alpha 48 \
    --lora_dropout 0.05 \
    --max_seq_length 512 \
    --warmup_ratio 0.1 \
    --resume_from_checkpoint "./adapter_checkpoint/Qwen3-4B_r24_a48_20260324_163949/checkpoint-450" # Path to the specific checkpoint
```

## Inference

```shell
python inference.py \
    --model_path "Qwen/Qwen3-4B" \
    --adapter_checkpoint_path "./adapter_checkpoint" \
    --test_data_path "./data/public_test.json" \
    -output_path "output.json" \
    --batch_size 64
    
```

## Evaluation

Calculating mean perplexity.

```shell
python ppl.py \
    --base_model_path "yentinglin/Llama-3.1-Taiwan-8B" \
    --peft_path "./adapter_checkpoint/Llama-3.1-Taiwan-8B_r128_a256_20260401_163101" \
    --test_data_path "data/public_test.json"
```

## Appendix
使用不同的超参数微调模型，测试结果如下：

| 序号 | 超参数                                                       | 模型                           | 平均困惑度  | 措施                                                         |
| ---- | ------------------------------------------------------------ | ------------------------------ | ----------- | ------------------------------------------------------------ |
| 1    | --epoch 1 	<br />--learning_rate 1e-4	<br />--train_batch_size 1	<br />--gradient_accumulation_steps 16 <br />--eval_batch_size 1 	<br />--lora_rank 8 	<br />--lora_alpha 16 	<br />--lora_dropout 0.05 	<br />--max_seq_length 512 | Qwen/Qwen3-4B                  | 18.76153125 |                                                              |
| 2    | --epoch 2 \ <br/>--learning_rate 1e-4 <br />--train_batch_size 1 <br />--gradient_accumulation_steps 16 <br />--eval_batch_size 1 <br />--lora_rank 8 <br />--lora_alpha 16 <br />--lora_dropout 0.05 <br />--max_seq_length 512 | Qwen/Qwen3-4B                  | 18.61840625 | 增加epoch                                                    |
| 3    | --epoch 5 <br />--learning_rate 1e-4 <br />--train_batch_size 1 <br />--gradient_accumulation_steps 16 <br />--eval_batch_size 1 <br />--lora_rank 64 <br />--lora_alpha 16 <br />--lora_dropout 0.05 <br />--max_seq_length 512 | Qwen/Qwen3-4B                  | 25.3800625  | r=64 → 过拟合<br />epoch=5 → 过训练<br />alpha/r = 0.25 → 学习效率低 |
| 4    | --epoch 2 <br />--learning_rate 5e-5 <br />--train_batch_size 1 <br />--gradient_accumulation_steps 16 <br />--eval_batch_size 1 <br />--lora_rank 8 <br />--lora_alpha 16 <br />--lora_dropout 0.05 <br />--max_seq_length 512 | Qwen/Qwen3-4B                  | 19.87884375 | weight_decay=0.01                                            |
| 5    | --epoch 1<br />--learning_rate 2e-5 <br />--train_batch_size 1 <br />--gradient_accumulation_steps 16<br />--eval_batch_size 1<br />--lora_rank 16 <br />--lora_alpha 32 <br />--lora_dropout 0.05 <br />--max_seq_length 1024 | Qwen/Qwen3-4B                  | 20.968625   | 增加微调参数量                                               |
| 6    | --epoch 2<br />--learning_rate 2e-5 <br />--train_batch_size 1 <br />--gradient_accumulation_steps 16<br />--eval_batch_size 1<br />--lora_rank 16 <br />--lora_alpha 32 <br />--lora_dropout 0.05 <br />--max_seq_length 1024 | Qwen/Qwen3-4B                  | 19.89896875 | 增加epoch                                                    |
| 7    | --epoch 3<br />--learning_rate 1e-4 <br />--train_batch_size 2 <br />--gradient_accumulation_steps 8 <br />--eval_batch_size 2<br />--lora_rank 64 <br />--lora_alpha 128 <br />--lora_dropout 0.05 <br />--max_seq_length 1024 <br />--warmup_ratio 0.1 | Qwen/Qwen3-4B                  | 24.64525    | 继续增加微调参数量、epoch                                    |
| 8    | --epoch 1<br />--learning_rate 2e-5<br />--train_batch_size 2<br />--gradient_accumulation_steps 8 <br />--eval_batch_size 2<br />--lora_rank 16<br />--lora_alpha 32 <br />--lora_dropout 0.05<br />--max_seq_length 1024<br />--warmup_ratio 0.1 | Qwen/Qwen3-4B                  | 17.603125   | 调整eval参数                                                 |
| 9    | --epoch 2<br /> --learning_rate 2e-5<br /> --train_batch_size 2 <br />--gradient_accumulation_steps 8 <br />--eval_batch_size 2<br />--lora_rank 16<br />--lora_alpha 32 <br />--lora_dropout 0.05<br />--max_seq_length 256<br />--warmup_ratio 0.05 | Qwen/Qwen3-4B                  | 7.22675     | prompt只输入instruction                                      |
| 10   | --epoch 4 <br />--learning_rate 5e-5 <br />--train_batch_size 2 <br />--gradient_accumulation_steps 16<br />--eval_batch_size 2 <br />--lora_rank 32<br />--lora_alpha 64 <br />--lora_dropout 0.05<br />--max_seq_length 512 <br />--warmup_ratio 0.1 | Qwen/Qwen3-4B                  | 8.1648125   | 增大学习率、epoch、微调参数量、梯度累积                      |
| 11   | --epoch 3<br />--learning_rate 3e-5<br />--train_batch_size 2 <br />--gradient_accumulation_steps 16<br />--eval_batch_size 2<br />--lora_rank 24<br />--lora_alpha 48<br />--lora_dropout 0.05<br />--max_seq_length 512 <br />--warmup_ratio 0.1 | Qwen/Qwen3-4B                  | 8.428       | 减少epoch、学习率、微调参数量、过滤超短样本                  |
| 12   | --epoch 3<br />--learning_rate 5e-5<br />--train_batch_size 4<br />--gradient_accumulation_steps 16<br />--eval_batch_size 4<br />--lora_rank 24<br />--lora_alpha 48<br />--lora_dropout 0.05<br />--max_seq_length 512<br />--warmup_ratio 0.1 | Qwen/Qwen3-4B                  | 7.754625    | 不过滤超短样本，增大batch_size                               |
| 13   | --epoch 5<br />--learning_rate 5e-5<br />--train_batch_size 4<br />--gradient_accumulation_steps 16<br />--eval_batch_size 4<br />--lora_rank 24<br />--lora_alpha 48<br />--lora_dropout 0.05<br />--max_seq_length 512<br />--warmup_ratio 0.1 | Qwen/Qwen3-4B                  | 7.754625    | 增加epoch                                                    |
| 14   | --epoch 3 <br />--learning_rate 1e-4 <br />--train_batch_size 2 <br />--gradient_accumulation_steps 16<br />--eval_batch_size 2<br />--lora_rank 64<br />--lora_alpha 128<br />--lora_dropout 0.05<br />--max_seq_length 1024<br />--warmup_ratio 0.1 | Qwen/Qwen3-4B                  | 7.05903125  | 增加微调参数量，同时调整epoch、学习率，使用early stopping    |
| 15   | --epoch 5 <br />--learning_rate 1e-4 <br />--train_batch_size 2 <br />--gradient_accumulation_steps 16<br />--eval_batch_size 2<br />--lora_rank 64<br />--lora_alpha 128<br />--lora_dropout 0.05<br />--max_seq_length 1024<br />--warmup_ratio 0.05 | Qwen/Qwen3-4B                  | 6.71778125  | 不过滤超短样本，降低warmup_ratio                             |
| 16   | --model_path "yentinglin/Llama-3.1-Taiwan-8B" <br />--epoch 5<br />--learning_rate 7e-5<br />--train_batch_size 2 <br />--gradient_accumulation_steps 16<br />--eval_batch_size 2<br />--lora_rank 128<br />--lora_alpha 256<br />--lora_dropout 0.05<br />--max_seq_length 1024<br />--warmup_ratio 0.05 | yentinglin/Llama-3.1-Taiwan-8B | 9.68309375  |                                                              |
| 17   | --model_path "yentinglin/Llama-3.1-Taiwan-8B" <br />--epoch 5<br />--learning_rate 5e-5<br />--train_batch_size 2 <br />--gradient_accumulation_steps 16<br />--eval_batch_size 2<br />--lora_rank 128<br />--lora_alpha 256<br />--lora_dropout 0.05<br />--max_seq_length 1024<br />--warmup_ratio 0.05 | yentinglin/Llama-3.1-Taiwan-8B | 6.06021875  | 降低学习率                                                   |





## 免责声明 | Disclaimer

本项目仅供学习和研究使用。使用者须遵守当地的法律法规，包括但不限于 DMCA 相关法律。我们不对任何非法使用承担责任。

This project is for research and learning purposes only. Users must comply with local laws and regulations, including but not limited to DMCA-related laws. We do not take any responsibility for illegal usage.