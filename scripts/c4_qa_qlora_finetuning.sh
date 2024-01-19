#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu
#SBATCH --time=1:00:00
#SBATCH --output=script_logging/slurm_%A.out

# Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

# srun python -uc "import torch; print('GPU available?', torch.cuda.is_available())"

#Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
pip install -r $HOME/RAGvsFT/component4_CBQA/requirements.txt

srun $HOME/RAGvsFT/component4_CBQA/qa_qlora_finetuning.py \
    --model_name_or_path "facebook/opt-1.3b" \
    --train_data_path $HOME/RAGvsFT/component0_preprocessing/generated_data/popQA_sm/qag_synthetic.jsonl \
    --test_data_path $HOME/RAGvsFT/component0_preprocessing/generated_data/popQA_sm/queries.jsonl \
    --output_dir "$HOME/RAGvsFT/component4_CBQA/models" \
    --epochs 30 \
    --version 3


# "facebook/opt-125m"
# "facebook/opt-350m"
# "facebook/opt-1.3b"
# "$HOME/RAGvsFT/component4_CBQA/models/clm_opt1-3b_1e"
