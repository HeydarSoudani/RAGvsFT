#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu
#SBATCH --time=6:30:00
#SBATCH --output=script_logging/slurm_%A.out

# Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0
# module load 2023
# module load Python/3.11.3-GCCcore-12.3.0

# srun python -uc "import torch; print('GPU available?', torch.cuda.is_available())"

#Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
# pip install -r $HOME/RAGvsFT/component3_QAGeneration/requirements.txt
# pip install --upgrade pip
# python -m spacy download en_core_web_sm

srun $HOME/RAGvsFT/component3_QAGeneration/zs_prompting_wo_rel.py


# component0_preprocessing/generated_data/popQA_religion/corpus.jsonl
# --corpus_path "$HOME/RAGvsFT/component0_preprocessing/generated_data/popQA_costomized/corpus.jsonl" \
# --corpus_path "$HOME/RAGvsFT/component3_QAGeneration/generated_data/corpus_512tokens.jsonl" \


