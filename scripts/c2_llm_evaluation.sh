#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu
#SBATCH --time=11:00:00
#SBATCH --output=script_logging/slurm_%A.out

# Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

srun $HOME/RAGvsFT/component2_AnsGen/evaluation/llm_evaluation.py \
    --model_name_or_path "$HOME/RAGvsFT/component2_AnsGen/models/EQ/flant5_lg_EQ_peft_v4" \
    --llm_model_name "flant5" \
    --dataset_name "EQ" \
    --output_file_pre_prefix "lg_af" \
    --with_peft True \
    --with_fs False \
    --with_rag True \
    --retrieval_method "ideal"


# output_file_pre_prefix -> 
    # - For Flan: []_bf
# Model name: [
    # flant5: "google/flan-t5-xxl" [small, base, large, xl, xxl]
    # llama2: "meta-llama/Llama-2-7b-chat-hf"
    # mistral: "mistralai/Mistral-7B-Instruct-v0.1"
    # zephyr: "HuggingFaceH4/zephyr-7b-beta"
    # tiny_llama: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    # MiniCPM: "openbmb/MiniCPM-2B-sft-fp32"
# ]
# dataset_name: [popQA, witQA, EQ]
# retrieval method: ['ideal', 'dpr', 'contriever', 'rerank', 'bm25']

