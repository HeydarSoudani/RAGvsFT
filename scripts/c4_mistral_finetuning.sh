#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu
#SBATCH --time=4:00:00
#SBATCH --output=script_logging/slurm_%A.out

# Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

# srun python -uc "import torch; print('GPU available?', torch.cuda.is_available())"

#Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
# pip install -r $HOME/RAGvsFT/component4_CBQA/requirements.txt
# pip install -q -U git+https://github.com/huggingface/accelerate.git
# pip install rouge_score

srun $HOME/RAGvsFT/component4_CBQA/mistral_finetuning.py \
    --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.1" \
    --data_dir $HOME/RAGvsFT/component0_preprocessing/generated_data/popQA_EQformat \
    --generation_method "prompting" \
    --output_model_dir $HOME/RAGvsFT/component4_CBQA/models \
    --output_result_dir $HOME/RAGvsFT/component0_preprocessing/generated_data/popQA_EQformat \
    --epochs 1 \
    --lr 0.0002 \
    --with_peft True \
    --version 3


# For TQA: $HOME/RAGvsFT/data/dataset/TQA
# For EQ: $HOME/RAGvsFT/data/dataset/entity_questions_dataset/dataset
# For popQA: $HOME/RAGvsFT/component0_preprocessing/generated_data/popQA_EQformat
