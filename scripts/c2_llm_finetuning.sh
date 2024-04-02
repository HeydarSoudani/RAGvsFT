#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu
#SBATCH --time=4:00:00
#SBATCH --output=script_logging/slurm_%A.out

module load 2022
module load Python/3.10.4-GCCcore-11.3.0

srun $HOME/RAGvsFT/component2_AnsGen/finetuning/llm_finetuning.py \
    --model_name_or_path "meta-llama/Llama-2-7b-chat-hf" \
    --llm_model_name "llama2" \
    --generation_method "prompting" \
    --data_dir $HOME/RAGvsFT/component0_preprocessing/generated_data/popQA_EQformat \
    --output_model_dir $HOME/RAGvsFT/component4_CBQA/models \
    --output_result_dir $HOME/RAGvsFT/component0_preprocessing/generated_data/popQA_EQformat \
    --epochs 3 \
    --lr 0.0002 \
    --with_peft True \
    --version 6

