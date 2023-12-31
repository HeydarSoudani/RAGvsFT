#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu
#SBATCH --time=0:30:00
#SBATCH --output=script_logging/slurm_%A.out

# Loading modules
# module load 2022
# module load Python/3.10.4-GCCcore-11.3.0
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

# srun python -uc "import torch; print('GPU available?', torch.cuda.is_available())"

#Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
pip install -r $HOME/RAGvsFT/component4_CBQA/requirements.txt

srun $HOME/RAGvsFT/component4_CBQA/pipline_QAG.py \
    --qg_model 'lmqg/t5-large-squad-qg' \
    --ae_model 'lmqg/t5-large-squad-ae' \
    --corpus_path "$HOME/RAGvsFT/component0_preprocessing/generated_data/popQA_costomized/corpus.jsonl" \
    --results_output_dir "$HOME/RAGvsFT/component4_CBQA/generated_data" \
    --results_output_filename "qag_results.jsonl"
