#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --output=script_logging/slurm_%A.out

# Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

srun $HOME/RAGvsFT/component3_highlighter/sentence_highlighter.py \
    --model_name_or_path "meta-llama/Meta-Llama-3-8B-Instruct" \
    --llm_model_name "llama3" \
    --dataset_name 'popQA' \
    --num_retrieved_passages 3 \
    --passage_concatenation "separate" \
    --retrieval_method "dpr" \
    --seed 42


# Model name: [
    # flant5: "google/flan-t5-xxl" [small, base, large, xl, xxl]
    
    # tiny_llama: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    # stable_lm2: "stabilityai/stablelm-2-zephyr-1_6b"
    # MiniCPM: "openbmb/MiniCPM-2B-sft-fp32"
    
    # llama2: "meta-llama/Llama-2-7b-chat-hf"
    # mistral: "mistralai/Mistral-7B-Instruct-v0.1"
    # zephyr: "HuggingFaceH4/zephyr-7b-beta"
    # llama3: "meta-llama/Meta-Llama-3-8B-Instruct"
    
# ]


