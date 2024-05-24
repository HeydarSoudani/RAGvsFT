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

# pip install git+https://github.com/huggingface/transformers
srun $HOME/RAGvsFT/component2_AnsGen/evaluation/llm_evaluation.py \
    --model_name_or_path "openbmb/MiniCPM-2B-sft-fp32" \
    --llm_model_name "MiniCPM" \
    --dataset_name "popQA" \
    --output_file_pre_prefix "2r_2p_bf" \
    --with_peft False \
    --with_rag_corpus True \
    --with_rag_qa_pairs True \
    --num_retrieved_passages 2 \
    --retrieval_method "ideal" \
    --seed 42


# output_file_pre_prefix -> 
    # - For Flan: []_bf
# Model name: [
    # flant5: "google/flan-t5-xxl" [small, base, large, xl, xxl]
    
    # tiny_llama: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    # stable_lm2: "stabilityai/stablelm-2-zephyr-1_6b"
    # MiniCPM: "openbmb/MiniCPM-2B-sft-fp32"
    
    # llama2: "meta-llama/Llama-2-7b-chat-hf"
    # mistral: "mistralai/Mistral-7B-Instruct-v0.1"
    # zephyr: "HuggingFaceH4/zephyr-7b-beta"
    
# ]
# dataset_name: [popQA, witQA, EQ]
# retrieval method: ['ideal', 'dpr', 'contriever', 'rerank', 'bm25']

