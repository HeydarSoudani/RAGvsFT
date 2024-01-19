#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu
#SBATCH --time=00:05:00
#SBATCH --output=script_logging/slurm_%A.out

# Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

# srun python -uc "import torch; print('GPU available?', torch.cuda.is_available())"
# Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
pip install -r $HOME/RAGvsFT/component2_ICL_OBQA/requirements.txt
pip install -q git+https://github.com/huggingface/peft.git

srun $HOME/RAGvsFT/component2_ICL_OBQA/icl_arbitrary_input.py \
    --model_name "HeydarS/opt-1.3b_qlora_v2" \
    --knowledge_input_file $HOME/RAGvsFT/data/dataset/popQA/popQA.tsv \
    --test_queries_file $HOME/RAGvsFT/component0_preprocessing/generated_data/popQA_sm/queries.jsonl \
    --eval_method "vanilla"\
    --loading_peft
    # --ret_path $HOME/RAGvsFT/component2_ICL_OBQA/data/popqa/bm25_results.jsonl


# HeydarS/opt-1.3b_qlora_v1
# "facebook/opt-1.3b"
# "$HOME/RAGvsFT/component4_CBQA/models/clm_opt1-3b_1e"