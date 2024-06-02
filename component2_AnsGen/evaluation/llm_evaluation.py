#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import pipeline
from peft import PeftConfig, PeftModel
import argparse, os, json
import numpy as np
import random
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    handlers=[logging.StreamHandler()]
)
os.environ["WANDB_MODE"] = "offline"

print("Available GPUs:", torch.cuda.device_count())
device = 'cuda:0'
target_relation_ids = 'all'
subset_percentage = 1.0

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

truncate_text_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
def truncate_text(text, max_tokens):
    tokens = truncate_text_tokenizer.tokenize(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    
    truncated_text = truncate_text_tokenizer.convert_tokens_to_string(tokens)
    return truncated_text

def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def load_relations_data(args):
    subfolders = ['test']
        
    relation_files = {}
    for subfolder in subfolders:
        subfolder_path = f"{args.data_dir}/{subfolder}"
        if os.path.exists(subfolder_path):
            for file in os.listdir(subfolder_path):
                relation_id = file.split('.')[0]
                if relation_id not in relation_files:
                    relation_files[relation_id] = []
                relation_files[relation_id].append(os.path.join(subfolder_path, file))    

    # Select relations =================
    if target_relation_ids == "all":
        if args.dataset_name == 'popQA':
            test_relation_ids = ['22', '91', '97', '106', '164', '182', '218', '257', '292', '422', '472', '484', '526', '533', '560', '639']
        elif args.dataset_name == 'witQA':
            test_relation_ids = ['17', '19', '22', '25', '27', '36', '50', '57', '58', '69', '86', '106', '123', '136', '140', '149', '162', '184', '344', '452', '462', '641', '674', '1038', '1050', '1376', '1431', '1433', '2012', '2936', '3301', '4647']
        elif args.dataset_name == 'EQ':
            test_relation_ids = ['17', '19', '20', '26', '30', '36', '40', '50', '69', '106', '112', '127', '131', '136', '159', '170', '175', '176', '264', '276', '407', '413', '495', '740', '800']
    else:
        test_relation_ids = target_relation_ids
    
    test_files = {subfolder: [] for subfolder in subfolders}
    
    for subfolder in subfolders:
        subfolder_path = f"{args.data_dir}/{subfolder}"
        if os.path.exists(subfolder_path):
            for file in os.listdir(subfolder_path):
                file_id = file.split('.')[0]
                if file_id in test_relation_ids:
                    test_files[subfolder].append(os.path.join(subfolder_path, file))

    print("Selected Relation ID:", test_relation_ids)
    logging.info(f"Selected Relation ID: {test_relation_ids}")

    return test_relation_ids, test_files, relation_files

def load_dataset(test_files):
    test_data = []
    for file in test_files['test']:
        relation_id = file.split('/')[-1].split('.')[0]
        data = load_json_file(file)
        for item in data:
            item['relation_id'] = relation_id
        test_data.extend(data) 
        
    test_subset_size = int(subset_percentage * len(test_data))
    subset_test_data = random.sample(test_data, test_subset_size)    
    test_questions = [(item['query_id'], item['question'], item['pageviews'], item['relation_id']) for item in subset_test_data]
    test_answers = [item['answers'] for item in subset_test_data]
    
    logging.info("Test dataset is loaded.")
    # print("Test dataset is loaded.")
    return test_questions, test_answers

def load_model(args):
    if args.with_peft:
        config = PeftConfig.from_pretrained(args.model_name_or_path)
        
        if args.llm_model_name in ["llama3", "llama2", "mistral", "zephyr", "stable_lm2", "tiny_llama", "MiniCPM"]:
            base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,
                low_cpu_mem_usage=True,
                return_dict=True,
                torch_dtype=torch.float16,
                device_map={"":0},
                trust_remote_code=True
            )
        elif args.llm_model_name == "flant5":
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                config.base_model_name_or_path,
                load_in_8bit=True,
                # device_map={"":0}
            )
            
        model = PeftModel.from_pretrained(base_model, args.model_name_or_path)
        model = model.merge_and_unload()
        
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_name_or_path,
            trust_remote_code=True
        )
        
    else:
        if args.llm_model_name in ["llama3", "llama2", "mistral", "zephyr", "stable_lm2", "tiny_llama", "MiniCPM"]:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                low_cpu_mem_usage=True,
                return_dict=True,
                torch_dtype=torch.float16,
                device_map={"":0}, # Load the entire model on the GPU 0
                # device_map='auto',
                trust_remote_code=True
            )
        elif args.llm_model_name == "flant5":
            model = AutoModelForSeq2SeqLM.from_pretrained(
                args.model_name_or_path,
                # load_in_8bit=True,
                # device_map={"": 0}
                # device_map="auto"
            )
            
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True
        )

    if args.llm_model_name in ["llama3", "llama2", "mistral", "zephyr", "stable_lm2", "tiny_llama", "MiniCPM"]:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    # if args.llm_model_name == 'llama2':
    #     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    model.eval()
    logging.info("Model and tokenizer are loaded")
    
    return model, tokenizer

def one_sided_partial_match(pred, possible_answers):
    is_correct = False
    for pa in possible_answers:
            if pa in pred or pa.lower() in pred or pa.capitalize() in pred:
                is_correct = True
    
    return is_correct

def two_sided_partial_match(pred, possible_answers):
    is_correct = False
    for pa in possible_answers:
        if pa in pred or pa.lower() in pred or pa.capitalize() in pred:
            is_correct = True
            break
        
    pred_parts = pred.split()
    for pa in possible_answers:
        for part in pred_parts:
            if part in pa or part.lower() in pa or part.capitalize() in pa:
                is_correct = True
                break  
            
    return is_correct

def main(args):
    
    # == Create data & output dir ===========================
    args.data_dir = f"component0_preprocessing/generated_data/{args.dataset_name}_costomized"
    args.output_result_dir = f"component0_preprocessing/generated_data/{args.dataset_name}_costomized"
    
    # == Create results dir and file ========================
    out_results_dir = f"{args.output_result_dir}/results"
    os.makedirs(out_results_dir, exist_ok=True)
    
    rag_part = "rag" if (args.with_rag_corpus or args.with_rag_sentence_rerank or args.with_rag_qa_pairs) else "norag"
    peft_part = "peft" if args.with_peft else "full"
    if (args.with_rag_corpus or args.with_rag_sentence_rerank or args.with_rag_qa_pairs):
        file_prefix = f"{args.dataset_name}_{args.llm_model_name}_{args.output_file_pre_prefix}_{rag_part}_{args.retrieval_method}_{peft_part}"
    else:
        file_prefix = f"{args.dataset_name}_{args.llm_model_name}_{args.output_file_pre_prefix}_{rag_part}_{peft_part}"
    
    out_results_path = f"{out_results_dir}/{file_prefix}_results.jsonl"
    
    logging.info(f"""
        Model: {args.model_name_or_path}
        Dataset: {args.dataset_name}
        PEFT: {args.with_peft}
        RAG (QA pairs): {args.with_rag_qa_pairs}
        RAG (highlight): {args.with_rag_sentence_highlight}
        RAG (rerank): {args.with_rag_sentence_rerank}
        RAG (corpus): {args.with_rag_corpus}
        Retrieval method: {args.retrieval_method}
        Output file's prefix: {file_prefix}
        Seed: {args.seed}
        """
    )
    print(f"""
        Model: {args.model_name_or_path}
        Dataset: {args.dataset_name}
        PEFT: {args.with_peft}
        RAG (QA pairs): {args.with_rag_qa_pairs}
        RAG (highlight): {args.with_rag_sentence_highlight}
        RAG (rerank): {args.with_rag_sentence_rerank}
        RAG (corpus): {args.with_rag_corpus}
        Retrieval method: {args.retrieval_method}
        Output file's prefix: {file_prefix}
        Seed: {args.seed}
        """
    )
    set_seed(args.seed)
    
    ### === Parameters per model
    if args.llm_model_name == "flant5":
        # V1 -> For small version
        # prompt_template_w_context = """Context: {context} \n Based on the provided context, answer the question: {question}"""
        # prompt_template_wo_context = """Answer the question: {question}"""
        # V2 -> For xl version
        prompt_template_w_context = """Context: {context} \nQuestion: {question}"""
        prompt_template_wo_context = """Question: {question}"""
        
    elif args.llm_model_name in ["llama2", "mistral"]:
        prompt_template_w_context = """<s>[INST] <<SYS>><</SYS>> \nContext: {context}\nQuestion: {question}\n[/INST]"""
        prompt_template_wo_context = """<s>[INST] <<SYS>><</SYS>> \nQuestion: {question}\n[/INST]"""  
        
    elif args.llm_model_name in ["zephyr", "tiny_llama"]:
        prompt_template_w_context = """<|system|> </s>\n <|user|>\nContext: {context}\nQuestion: {question}</s>\n <|assistant|>"""
        prompt_template_wo_context = """<|system|> </s>\n <|user|>\nQuestion: {question}</s>\n <|assistant|>"""
    
    elif args.llm_model_name == "stable_lm2":
        prompt_template_w_context = """<|user|>\nContext: {context}\nQuestion: {question}<|endoftext|>\n<|assistant|>"""
        prompt_template_wo_context = """<|user|>\nQuestion: {question}<|endoftext|>\n<|assistant|>"""
    
    elif args.llm_model_name == "MiniCPM":
        prompt_template_w_context = """<User>\nContext: {context}\nQuestion: {question}\n <AI>"""
        prompt_template_wo_context = """<User>\nQuestion: {question}\n <AI>"""

    elif args.llm_model_name == "llama3":
        prompt_template_w_context = """<|begin_of_text|><|start_header_id|>system<|end_header_id|><|eot_id|><|start_header_id|>user<|end_header_id|>\nContext: {context}\nQuestion: {question}\n<|eot_id|>"""
        prompt_template_wo_context = """<|begin_of_text|><|start_header_id|>system<|end_header_id|><|eot_id|><|start_header_id|>user<|end_header_id|>\nQuestion: {question}\n<|eot_id|>"""


    # == Define maximum number of tokens ====================
    # = Book 50 tokens for QA pairs
    # = Book 20 tokens for input prompt
    if args.llm_model_name in ["flant5", "stable_lm2", "MiniCPM"]:
        max_input_tokens = 512
    elif args.llm_model_name in ["tiny_llama"]:
        max_input_tokens = 1024
    elif args.llm_model_name in ["llama3", "llama2", "mistral", "zephyr"]:
        max_input_tokens = 2048
    
    
    # === Load model and tokenizer ==========================
    logging.info("Inferencing ...")
    model, tokenizer = load_model(args)
    test_relation_ids, test_files, relation_files = load_relations_data(args)
    test_questions, test_answers = load_dataset(test_files)
    
    # == Loading the retrieval results (corpus) ==============
    if args.with_rag_corpus:
        ret_results = {}
        ret_results_dir = f"{args.data_dir}/retrieved_passage/{args.retrieval_method}_3"
        
        for test_relation_id in test_relation_ids:
            ret_results_path = f"{ret_results_dir}/{test_relation_id}.{args.retrieval_method}.ret_results.jsonl"
            with open (ret_results_path, 'r') as file:
                for line in file:
                    data = json.loads(line.strip())
                    ret_results[data['id']] = data
                    
    # == Loading highligted passages =========================
    if args.with_rag_sentence_highlight:
        highlight_results = {}
        highlight_results_file = f'{args.data_dir}/highlighted_sentences/{args.retrieval_method}_3"'
        for test_relation_id in test_relation_ids:
            ret_results_path = f"{ret_results_dir}/{test_relation_id}.{args.retrieval_method}.set_highlighted.jsonl"
            with open (highlight_results_file, 'r') as file:
                for line in file:
                    data = json.loads(line.strip())
                    highlight_results[data['query_id']] = data

    # == Loading the sentence reranking results ==============
    if args.with_rag_sentence_rerank:
        ret_sent_rerank = {}
        ret_results_dir = f"{args.data_dir}/reranked_sentences/{args.retrieval_method}_3"
        for test_relation_id in test_relation_ids:
            ret_results_path = f"{ret_results_dir}/{test_relation_id}.{args.retrieval_method}.set_reranked.jsonl"
            with open (ret_results_path, 'r') as file:
                for line in file:
                    data = json.loads(line.strip())
                    ret_sent_rerank[data['id']] = data

    # == Loading the retrieval results (qa_pairs) ============
    if args.with_rag_qa_pairs:
        ret_qa_results = {}
        ret_results_dir = f"{args.data_dir}/retrieved_qa_pairs/{args.retrieval_method}"
        for test_relation_id in test_relation_ids:
            ret_results_path = f"{ret_results_dir}/{test_relation_id}.retrieved_qa_pairs.jsonl"
            with open (ret_results_path, 'r') as file:
                for line in file:
                    data = json.loads(line.strip())
                    ret_qa_results[data['query_id']] = data
    
    # == Loop over the test questions ========================
    if args.llm_model_name in ["llama3", "llama2", "mistral", "zephyr", "stable_lm2", "tiny_llama", "MiniCPM"]:
        max_output_tokens = 40
        pipe = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens = max_output_tokens
        )
    elif args.llm_model_name == "flant5":
        max_output_tokens = 20
        pipe = pipeline( 
            "text2text-generation", 
            model=model, 
            tokenizer=tokenizer,
            max_new_tokens = max_output_tokens
        ) 
        
    accuracy = []
    with open(out_results_path, 'w') as file:
        # highlight_idx = 0
        for idx, (query_id, query, query_pv, query_relation) in enumerate(tqdm(test_questions)):
            
            # if query_id in qa_list:
            # highlight_idx += 1
            retrieved_text = ""
            has_context = False
            
            # == Apply retrieved QA pairs ====================
            if args.with_rag_qa_pairs:
                qa_pairs_data = ret_qa_results[query_id]['relevant_train_questions']
                qa_pairs_text = ""
                if len(qa_pairs_data) > 0:
                    has_context = True
                    for qa_pair in qa_pairs_data:
                        qa_pairs_text += f"{qa_pair['question']} {qa_pair['answers'][0]}\n"
                
                retrieved_text += f"{qa_pairs_text}\n"
            
            # == Apply highlight text ========================
            if args.with_rag_sentence_highlight:
                
                try:
                    ## == This part is for concatinated highlighted text
                    # highlighted_text = highlight_results[query_id]['highlighted_text']
                    # if 'sentence' in highlighted_text and len(highlighted_text['sentence']) != 0:
                    #     sentences = highlighted_text['sentence']
                    #     retrieved_text += f"{' '.join(sentences)}\n"
                    #     has_context = True
                    # elif 'sentences' in highlighted_text and len(highlighted_text['sentences']) != 0:
                    #     sentences = highlighted_text['sentences']
                    #     retrieved_text += f"{' '.join(sentences)}\n"
                    #     has_context = True
                    # else:
                    #     sentences = []
                    #     logging.info(f"\nNo highlighted text found for query: {query_id}, {query}") 
                    #     print("\nNo highlighted text found for query: {}, {}".format(query_id, query))
                
                    ## == This part is for seperate highlighted text
                    highlighted_text_list = highlight_results[query_id]['highlighted_text']
                    for item in highlighted_text_list:
                        ret_rank = item['ret_rank']
                        highlighted_text = item['highlighted']
                        if 'sentence' in highlighted_text and len(highlighted_text['sentence']) != 0:
                            sentences = highlighted_text['sentence']
                            retrieved_text += f"{' '.join(sentences)}\n"
                            has_context = True
                        elif 'sentences' in highlighted_text and len(highlighted_text['sentences']) != 0:
                            sentences = highlighted_text['sentences']
                            retrieved_text += f"{' '.join(sentences)}\n"
                            has_context = True
                        else:
                            logging.info(f"\nNo highlighted text found for query: {query_id}, {query}, retrieved sentence: {ret_rank}")
                            print(f"\nNo highlighted text found for query: {query_id}, {query}, retrieved sentence: {ret_rank}")
                    
                
                except json.decoder.JSONDecodeError as e:
                    print(f"Error decoding JSON for query_id {query_id}: {e}")
                    print(f"Problematic JSON string: {highlight_results[query_id]['highlighted_text']}")
            
            # == Apply retrieved sentence rerank =============
            if args.with_rag_sentence_rerank:
                rerank_results = ret_sent_rerank[query_id]['sentences']
                reranked_text = "".join(f"{rerank_results[i]['sentence']} \n" for i in range(args.num_reranked_sentences) if i < len(rerank_results))
                retrieved_text += f"{reranked_text}\n"
                has_context = True             
            
            # == Apply retrieved corpus text =================
            if args.with_rag_corpus:
                # if not has_context:
                max_token = max_input_tokens - (70 if args.with_rag_qa_pairs else 20)
                corpus_text = "".join(ret_results[query_id]['ctxs'][i]['text'] for i in range(args.num_retrieved_passages) if i < len(ret_results[query_id]['ctxs']))
                retrieved_text += f"{truncate_text(corpus_text, max_token)}\n"
                has_context = True
            
            if retrieved_text == "":
                logging.info(f"\nNo retrieved text found for query: {query}") 
                print("\nNo retrieved text found for query: {}, {}".format(query_id, query))               
            
            if has_context:
                prompt = prompt_template_w_context.format(context=retrieved_text, question=query)        
            else:
                prompt = prompt_template_wo_context.format(question=query)
                   
            n_max_trial = 5
            for i in range(n_max_trial):
                try:
                    result = pipe(prompt)[0]['generated_text']
                    break
                except Exception as e:
                    print(f"Try #{i+1} for Query: {query_id}")
                    print('Error message:', e)
            
            if args.llm_model_name == 'flant5':
                pred = result
            elif args.llm_model_name in ["llama2", "mistral"]:
                pred = result.split("[/INST]")[1].strip()
            elif args.llm_model_name in ['zephyr', "stable_lm2", "tiny_llama"]:
                pred = result.split("<|assistant|>")[1].strip()
            elif args.llm_model_name == 'MiniCPM':
                pred = result.split("<AI>")[1].strip()
            elif args.llm_model_name == 'llama3':
                pred = result[len(prompt):]
            
            is_correct = one_sided_partial_match(pred, test_answers[idx])         
            # is_correct = two_sided_partial_match(pred, test_answers[idx])
            accuracy.append(is_correct)
            
            # if highlight_idx < 10 or highlight_idx % 200 == 0:
            if idx < 10 or idx % 400 == 0:
                # logging.info('\n')
                # logging.info(f"Prompt: {prompt}")
                # logging.info(f"Query: {query}")
                # logging.info(f"Has context: {has_context}"),
                # logging.info(f"# context: {len(ret_results[query_id]['ctxs'])}"),
                # logging.info(f"Pred: {pred}")
                # logging.info(f"Labels: {test_answers[idx]}")
                # logging.info(f"Final decision: {is_correct}")
                # logging.info('====')
                print('\n')
                print(f"Prompt: {prompt}")
                print(f"Query: {query}")
                print(f"Has context: {has_context}"),
                # print(f"# context: {len(ret_results[query_id]['ctxs'])}"),
                print(f"Pred: {pred}")
                print(f"Labels: {test_answers[idx]}")
                print(f"Final decision: {is_correct}")
                print('====')
            
            item = {
                "query_id": query_id,
                "question": query,
                "possible_answers": test_answers[idx],
                "pred": pred,
                "is_correct": is_correct,
                "has_context": has_context,
                "pageviews": query_pv
            }
            file.write(json.dumps(item) + '\n')
    acc = sum(accuracy) / len(accuracy)
    logging.info(f"Accuracy: {acc * 100:.2f}%")
    print(f"Accuracy: {acc * 100:.2f}%")
    
if __name__ == "__main__":
    
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--llm_model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--output_file_pre_prefix", type=str)
    parser.add_argument("--with_peft", type=str2bool, default=False)
    
    parser.add_argument("--with_rag_qa_pairs", type=str2bool, default=False)
    parser.add_argument("--with_rag_sentence_highlight", type=str2bool, default=False)
    parser.add_argument("--with_rag_sentence_rerank", type=str2bool, default=False)
    parser.add_argument("--num_reranked_sentences", type=int, default=1)
    
    parser.add_argument("--with_rag_corpus", type=str2bool, default=False)
    parser.add_argument("--num_retrieved_passages", type=int, default=1)
    parser.add_argument("--retrieval_method", type=str)
    parser.add_argument("--seed", type=int)
    
    args = parser.parse_args()
    main(args)
    
    