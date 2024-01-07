#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu
#SBATCH --time=03:00:00
#SBATCH --output=script_logging/slurm_%A.out

# Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

# srun python -uc "import torch; print('GPU available?', torch.cuda.is_available())"
# Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
pip install -r $HOME/RAGvsFT/component2_ICL_OBQA/requirements.txt

srun $HOME/RAGvsFT/component2_ICL_OBQA/icl_buckets_input.py \
    --model_name "$HOME/RAGvsFT/component4_CBQA/models/clm_opt1-3b_1e" \
    --input_file $HOME/RAGvsFT/data/dataset/popQA/popQA.tsv \
    --eval_method vanilla \
    --output_resutls_dir $HOME/RAGvsFT/component2_ICL_OBQA/results \
    --output_resutls_filename 'opt1-3_clm_vanilla_bk.tsv'

# "facebook/opt-1.3b"
# "$HOME/RAGvsFT/component4_CBQA/models/clm_opt1-3b_1e"
# --ret_path $HOME/RAGvsFT/component2_ICL_OBQA/data/popqa/bm25_results.jsonl \
