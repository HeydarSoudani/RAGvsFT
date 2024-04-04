#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu
#SBATCH --time=0:10:00
#SBATCH --output=script_logging/slurm_%A.out

# Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

# srun python -uc "import torch; print('GPU available?', torch.cuda.is_available())"

#Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
pip install -r $HOME/RAGvsFT/component1_retrieval/requirements.txt

srun $HOME/RAGvsFT/component1_retrieval/dpr/evaluation.py \
    --model "msmarco-distilbert-base-v3" \
    --dataset_name "popQA" \
    --data_path "$HOME/RAGvsFT/component1_retrieval/data/popqa_religion" \
    --output_results_dir "$HOME/RAGvsFT/component1_retrieval/results/religion" \
    --output_results_filename "dpr_ft_eval.tsv" \
    --results_save_file "dpr_ft-qrels.tsv"

# Zero-shot: msmarco-distilbert-base-v3
# After FT: $HOME/RAGvsFT/component1_retrieval/dpr/models/ft_dpr_10e_v1