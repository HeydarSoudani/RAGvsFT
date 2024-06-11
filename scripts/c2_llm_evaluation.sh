#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --partition=gpu_mig
#SBATCH --time=4:00:00
#SBATCH --output=script_logging/slurm_%A.out

# Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

# pip install git+https://github.com/huggingface/transformers
srun $HOME/RAGvsFT/component2_AnsGen/evaluation/llm_evaluation.py \
    --model_name_or_path "stabilityai/stablelm-2-zephyr-1_6b" \
    --llm_model_name "stable_lm2" \
    --dataset_name "popQA" \
    --output_file_pre_prefix "1rp_3p_bf" \
    --with_peft False \
    --with_fewshot_examples False \
    --with_rag_qa_pairs False \
    --with_rag_sentence_highlight False \
    --with_rag_sentence_rerank False \
    --num_reranked_sentences 2 \
    --with_rag_corpus False \
    --num_retrieved_passages 3 \
    --with_rag_highlighted_passage True \
    --retrieval_method "dpr" \
    --seed 42


# output_file_pre_prefix -> 
    # - For Flan: []_bf
# Model name: [
    # flant5: "google/flan-t5-xxl" [small, base, large, xl, xxl]
    
    # tiny_llama: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    # stable_lm2: "stabilityai/stablelm-2-zephyr-1_6b"
    # MiniCPM: "openbmb/MiniCPM-2B-sft-fp32"
    
    # mistral: "mistralai/Mistral-7B-Instruct-v0.1"
    # zephyr: "HuggingFaceH4/zephyr-7b-beta"
    # llama2: "meta-llama/Llama-2-7b-chat-hf"
    # llama3: "meta-llama/Meta-Llama-3-8B-Instruct"
    
# ]
# dataset_name: [popQA, witQA, EQ]
# retrieval method: ['ideal', 'dpr', 'contriever', 'rerank', 'bm25']

