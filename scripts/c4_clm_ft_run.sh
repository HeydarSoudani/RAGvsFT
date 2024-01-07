#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu
#SBATCH --time=2:00:00
#SBATCH --output=script_logging/slurm_%A.out

# Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

# srun python -uc "import torch; print('GPU available?', torch.cuda.is_available())"

#Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
pip install -r $HOME/RAGvsFT/component4_CBQA/requirements.txt

srun $HOME/RAGvsFT/component4_CBQA/causalLM_ft.py \
    --model "facebook/opt-1.3b" \
    --corpus_path "$HOME/RAGvsFT/component0_preprocessing/generated_data/popQA_costomized/corpus.jsonl" \
    --model_output_dir "$HOME/RAGvsFT/component4_CBQA/models" \
    --model_output_filename "clm_opt1-3b_1e" \
    --epochs 1