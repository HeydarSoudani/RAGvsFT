#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu
#SBATCH --time=6:10:00
#SBATCH --output=script_logging/slurm_%A.out

# Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

# srun python -uc "import torch; print('GPU available?', torch.cuda.is_available())"

#Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
pip install -r $HOME/RAGvsFT/component4_CBQA/requirements.txt
pip install -q git+https://github.com/huggingface/transformers.git@main git+https://github.com/huggingface/peft.git

srun $HOME/RAGvsFT/component4_CBQA/clm_lora_ft.py \
    --model_name "facebook/opt-125m" \
    --repo_name "HeydarS/opt-125m-lora-v2" \
    --corpus_path "$HOME/RAGvsFT/component3_preprocessing/generated_data/corpus_splitted.jsonl" \
    --model_output_dir "$HOME/RAGvsFT/component4_CBQA/models" \
    --model_output_filename "opt-350m-lora" \
    --epochs 0.1

# "facebook/opt-1.3b"
# "$HOME/RAGvsFT/component4_CBQA/models/clm_opt1-3b_1e"
