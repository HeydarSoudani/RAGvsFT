#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu
#SBATCH --time=0:10:00
#SBATCH --output=script_logging/slurm_%A.out

module load 2022
module load Python/3.10.4-GCCcore-11.3.0

srun $HOME/RAGvsFT/component3_moe/multiple_gpus_test.py



# dataset_name: [popQA, witQA, EQ]
# base_model_name = [
#     "flant5_sm", "flant5_bs", "flant5_lg", "flant5_xl", "flant5_xxl",
#     "stable_lm2", "tiny_llama", "MiniCPM",
#     "llama2", "mistral", "zephyr"
# ]
# retrieval method: ['ideal', 'dpr', 'contriever', 'rerank', 'bm25']

