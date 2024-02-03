#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu
#SBATCH --time=15:10:00
#SBATCH --output=script_logging/slurm_%A.out

# Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

# srun python -uc "import torch; print('GPU available?', torch.cuda.is_available())"

#Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
# pip install -r $HOME/RAGvsFT/component4_CBQA/requirements.txt
# pip install -q -U git+https://github.com/huggingface/accelerate.git

srun $HOME/RAGvsFT/component4_CBQA/opt_finetuning.py \
    --model_name_or_path "facebook/opt-350m" \
    --data_dir $HOME/RAGvsFT/component0_preprocessing/generated_data/popQA_EQformat \
    --output_model_dir $HOME/RAGvsFT/component4_CBQA/models \
    --output_result_dir $HOME/RAGvsFT/component0_preprocessing/generated_data/popQA_EQformat \
    --epochs 10 \
    --lr 0.0002 \
    --version 11


# For TQA: $HOME/RAGvsFT/data/dataset/TQA
# For EQ: $HOME/RAGvsFT/data/dataset/entity_questions_dataset/dataset
# For popQA: $HOME/RAGvsFT/component0_preprocessing/generated_data/popQA_EQformat

# "facebook/opt-125m"
# "facebook/opt-350m"
# "facebook/opt-1.3b"
# "$HOME/RAGvsFT/component4_CBQA/models/clm_opt1-3b_1e"
