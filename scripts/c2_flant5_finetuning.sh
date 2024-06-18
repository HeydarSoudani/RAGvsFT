#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu
#SBATCH --time=5:00:00
#SBATCH --output=script_logging/slurm_%A.out

# Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

srun $HOME/RAGvsFT/component2_AnsGen/finetuning/flant5_finetuning.py \
    --model_name_or_path "google/flan-t5-large" \
    --llm_model_name "flant5_lg" \
    --dataset_name "EQ" \
    --generation_method "prompting" \
    --epochs 4 \
    --batch_size 16 \
    --lr 0.0002 \
    --with_peft False \
    --version 110

# Model name: [
    # google/flan-t5-small
    # google/flan-t5-base
    # google/flan-t5-large
    # google/flan-t5-xl
    # google/flan-t5-xxl
# ]
# dataset_name: [popQA, witQA, EQ]
