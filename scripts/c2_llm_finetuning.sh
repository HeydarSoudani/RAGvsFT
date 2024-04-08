#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu
#SBATCH --time=20:00:00
#SBATCH --output=script_logging/slurm_%A.out

module load 2022
module load Python/3.10.4-GCCcore-11.3.0

srun $HOME/RAGvsFT/component2_AnsGen/finetuning/llm_finetuning.py \
    --model_name_or_path "HuggingFaceH4/zephyr-7b-beta" \
    --llm_model_name "zephyr" \
    --dataset_name "popQA" \
    --generation_method "prompting" \
    --with_peft True \
    --version 20


# Model name: [
    # llama2: "meta-llama/Llama-2-7b-chat-hf"
    # mistral: "mistralai/Mistral-7B-Instruct-v0.1"
    # zephyr: "HuggingFaceH4/zephyr-7b-beta"
    # tiny_llama: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    # MiniCPM: "openbmb/MiniCPM-2B-sft-fp32"
# ]
# dataset_name: [popQA, witQA, EQ]
