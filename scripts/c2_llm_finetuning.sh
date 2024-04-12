#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --output=script_logging/slurm_%A.out

module load 2022
module load Python/3.10.4-GCCcore-11.3.0

# pip install git+https://github.com/huggingface/transformers 
# pip install transformers==4.38.2
# pip install accelerate==0.27.2

srun $HOME/RAGvsFT/component2_AnsGen/finetuning/llm_finetuning.py \
    --model_name_or_path "meta-llama/Llama-2-7b-chat-hf" \
    --llm_model_name "llama2" \
    --dataset_name "witQA" \
    --generation_method "prompting" \
    --with_peft True \
    --version 43


# Model name: [
    # stable_lm2: "stabilityai/stablelm-2-zephyr-1_6b"
    # tiny_llama: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    # MiniCPM: "openbmb/MiniCPM-2B-sft-fp32"
    
    # llama2: "meta-llama/Llama-2-7b-chat-hf"
    # mistral: "mistralai/Mistral-7B-Instruct-v0.1"
    # zephyr: "HuggingFaceH4/zephyr-7b-beta"
# ]
# dataset_name: [popQA, witQA, EQ]
