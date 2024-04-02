#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu
#SBATCH --time=6:00:00
#SBATCH --output=script_logging/slurm_%A.out

# Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

# srun python -uc "import torch; print('GPU available?', torch.cuda.is_available())"

#Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
# pip install -r $HOME/RAGvsFT/component4_CBQA/requirements.txt

srun $HOME/RAGvsFT/component2_AnsGen/evaluation/llm_evaluation.py \
    --model_name_or_path "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --llm_model_name "tiny_llama" \
    --data_dir $HOME/RAGvsFT/component0_preprocessing/generated_data/popQA_costomized \
    --output_result_dir $HOME/RAGvsFT/component0_preprocessing/generated_data/popQA_costomized \
    --output_file_pre_prefix "bf" \
    --with_peft False \
    --with_fs False \
    --with_rag True \
    --retrieval_method "ideal"


# Model name: [
    # llama2: "meta-llama/Llama-2-7b-chat-hf"
    # mistral: "mistralai/Mistral-7B-Instruct-v0.1"
    # zephyr: "HuggingFaceH4/zephyr-7b-beta"
    # tiny_llama: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# ]
# retrieval method: ['ideal', 'dpr', 'contriever', 'rerank', 'bm25']

