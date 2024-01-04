#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu
#SBATCH --time=03:10:00
#SBATCH --output=script_logging/slurm_%A.out

# Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

# srun python -uc "import torch; print('GPU available?', torch.cuda.is_available())"
# Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
pip install -r $HOME/RAGvsFT/component2_ICL_OBQA/requirements.txt

srun $HOME/RAGvsFT/component2_ICL_OBQA/in_context_learning.py \
    --model_name "facebook/opt-6.7b" \
    --input_file $HOME/RAGvsFT/data/dataset/popQA/popQA.tsv \
    --eval_method BM25 \
    --ret_path $HOME/RAGvsFT/component2_ICL_OBQA/data/popqa/bm25_result.jsonl