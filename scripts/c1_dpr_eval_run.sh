#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu
#SBATCH --time=0:30:00
#SBATCH --output=script_logging/slurm_%A.out

# Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

# srun python -uc "import torch; print('GPU available?', torch.cuda.is_available())"

#Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
pip install -r $HOME/RAGvsFT/component1_retrieval/requirements.txt

srun $HOME/RAGvsFT/component1_retrieval/contriever/evaluation.py \
    --model "msmarco-distilbert-base-v3" \
    --data_path "$HOME/RAGvsFT/component1_retrieval/data/popqa" \
    --output_results_dir "$HOME/RAGvsFT/component1_retrieval/results" \
    --output_results_filename "ft_dpr_eval.tsv" \
    --results_save_file "dpr_ft-qrels.tsv"

# Zero-shot: msmarco-distilbert-base-v3
# After FT: component1_retrieval/dpr/models/ft_dpr_5e