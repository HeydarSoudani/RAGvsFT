#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu
#SBATCH --time=5:00:00
#SBATCH --output=script_logging/slurm_%A.out

# Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

# srun python -uc "import torch; print('GPU available?', torch.cuda.is_available())"

#Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
# pip install -r $HOME/RAGvsFT/component4_CBQA/requirements.txt

srun $HOME/RAGvsFT/component4_CBQA/llama2_evaluation.py \
    --model_name_or_path $HOME/RAGvsFT/component4_CBQA/models/Llama-2-7b-chat-hf_peft_v1 \
    --data_dir $HOME/RAGvsFT/component0_preprocessing/generated_data/popQA_EQformat \
    --output_result_dir $HOME/RAGvsFT/component0_preprocessing/generated_data/popQA_EQformat \
    --output_file_pre_prefix "af" \
    --with_peft True \
    --with_fs False \
    --with_rag False \
    --retrieval_method ""

# ['ideal', 'dpr', 'contriever', 'rerank', 'bm25']
# Base model: "meta-llama/Llama-2-7b-chat-hf"

