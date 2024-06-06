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

srun $HOME/RAGvsFT/component3_passage_highlighter/sentence_reranker.py \
    --dense_model "msmarco-distilbert-base-v3" \
    --dataset_name 'popQA' \
    --split_type 'word' \
    --word_num 10 \
    --retrieval_method "ideal" \
    --num_retrieved_passages 1 \
    --output_path ''

# popQA, witQA, EQ
# ['ideal', 'dpr', 'contriever', 'rerank', 'bm25']



