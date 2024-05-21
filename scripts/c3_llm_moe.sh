#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu
#SBATCH --time=6:00:00
#SBATCH --output=script_logging/slurm_%A.out

# Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

# pip install git+https://github.com/huggingface/transformers
srun $HOME/RAGvsFT/component3_moe/llm_moe.py \
    --model_name_or_path "meta-llama/Meta-Llama-3-8B-Instruct" \
    --dataset_name "popQA" \
    --base_model_name "stable_lm2" \
    --retrieval_method "ideal" \
    --output_file_prefix "10_cot_llm" \
    --chunk_index 10 \
    --seed 42



# dataset_name: [popQA, witQA, EQ]
# base_model_name = [
#     "flant5_sm", "flant5_bs", "flant5_lg", "flant5_xl", "flant5_xxl",
#     "stable_lm2", "tiny_llama", "MiniCPM",
#     "llama2", "mistral", "zephyr"
# ]
# retrieval method: ['ideal', 'dpr', 'contriever', 'rerank', 'bm25']

