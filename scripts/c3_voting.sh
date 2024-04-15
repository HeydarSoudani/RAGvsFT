#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu
#SBATCH --time=15:00:00
#SBATCH --output=script_logging/slurm_%A.out

# Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

# pip install git+https://github.com/huggingface/transformers
srun $HOME/RAGvsFT/component3_voting/main.py \
    --model_name_or_path "HuggingFaceH4/zephyr-7b-beta" \
    --llm_model_name "zephyr"


# output_file_pre_prefix -> 
    # - For Flan: []_bf
# Model name: [
    # flant5: "google/flan-t5-xxl" [small, base, large, xl, xxl]
    
    # tiny_llama: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    # stable_lm2: "stabilityai/stablelm-2-zephyr-1_6b"
    # MiniCPM: "openbmb/MiniCPM-2B-sft-fp32"
    
    # llama2: "meta-llama/Llama-2-7b-chat-hf"
    # mistral: "mistralai/Mistral-7B-Instruct-v0.1"
    # zephyr: "HuggingFaceH4/zephyr-7b-beta"
    
# ]
# dataset_name: [popQA, witQA, EQ]
# retrieval method: ['ideal', 'dpr', 'contriever', 'rerank', 'bm25']

