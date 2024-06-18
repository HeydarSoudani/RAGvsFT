#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --partition=gpu_mig
#SBATCH --time=5:00:00
#SBATCH --output=script_logging/slurm_%A.out

# Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

srun $HOME/RAGvsFT/component2_AnsGen/finetuning/flant5_finetuning.py \
    --model_name_or_path "google/flan-t5-base" \
    --llm_model_name "flant5_bs" \
    --dataset_name "EQ" \
    --generation_method "pipeline" \
    --epochs 5 \
    --batch_size 64 \
    --lr 0.0002 \
    --with_peft True \
    --version 111

# Model name: [
    # google/flan-t5-small
    # google/flan-t5-base
    # google/flan-t5-large
    # google/flan-t5-xl
    # google/flan-t5-xxl
# ]
# dataset_name: [popQA, witQA, EQ]
