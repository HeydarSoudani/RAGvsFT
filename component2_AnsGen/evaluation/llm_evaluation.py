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
subset_percentage = 0.05

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
    
    # logging.info(f"""
    #     Model: {args.model_name_or_path}
    #     Dataset: {args.dataset_name}
    #     PEFT: {args.with_peft}
    #     RAG (QA pairs): {args.with_rag_qa_pairs}
    #     RAG (highlight): {args.with_rag_sentence_highlight}
    #     RAG (rerank): {args.with_rag_sentence_rerank}
    #     RAG (corpus): {args.with_rag_corpus}
    #     Retrieval method: {args.retrieval_method}
    #     Output file's prefix: {file_prefix}
    #     Seed: {args.seed}
    #     """
    # )
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
        prompt_template_w_context = """<|user|>\n\n{context}\nQuestion: {question}\n<|endoftext|>\n<|assistant|>"""
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
    
    # == Loading test QA results =============================
    if args.with_fewshot_examples:
        qa_test = {}
        qa_test_dir = f"{args.data_dir}/test"
        for test_relation_id in test_relation_ids:
            qa_test_path = f"{qa_test_dir}/{test_relation_id}.test.json"
            with open(qa_test_path, 'r') as file:
                qa_test[test_relation_id] = json.load(file)
    
    # == Loading the retrieval results (corpus) ==============
    # if args.with_rag_corpus:
    ret_results = {}
    ret_results_dir = f"{args.data_dir}/retrieved_passage/{args.retrieval_method}_5"
    
    for test_relation_id in test_relation_ids:
        ret_results_path = f"{ret_results_dir}/{test_relation_id}.{args.retrieval_method}.ret_results.jsonl"
        with open (ret_results_path, 'r') as file:
            for line in file:
                data = json.loads(line.strip())
                ret_results[data['id']] = data
                    
    # == Loading highligted passages =========================
    if args.with_rag_sentence_highlight:
        highlight_results = {}
        highlight_results_dir = f'{args.data_dir}/highlighted_sentences/{args.retrieval_method}_3'
        for test_relation_id in test_relation_ids:
            highlight_results_file = f"{highlight_results_dir}/{test_relation_id}.{args.retrieval_method}.set_highlighted.jsonl"
            with open (highlight_results_file, 'r') as file:
                for line in file:
                    data = json.loads(line.strip())
                    highlight_results[data['query_id']] = data

    # == Loading the sentence reranking results ==============
    # if args.with_rag_sentence_rerank:
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
    
    # == Loading the corpus ==================================
    if args.with_rag_highlighted_passage:
        corpus = {}
        corpus_dir = f"{args.data_dir}/corpus_all"
        for test_relation_id in test_relation_ids:
            corpus_path = f"{corpus_dir}/{test_relation_id}.corpus.json"
            with open (corpus_path, 'r') as file:
                rel_data = json.load(file)
                for line in rel_data:
                    corpus[line['doc_id']] = line
    
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
            pre_context = ""
            main_context = ""
            has_pre_context = False
            has_main_context = False
            
            # == Apply few-shot QA example ====================
            if args.with_fewshot_examples:
                fewshot_examples = ""
                for relation_id, relation_data in qa_test.items():
                    if relation_id != query_relation:
                        # print(len(relation_data))
                        relation_sample = random.choice(relation_data)
                        # print(relation_sample)
                        # print(relation_sample['question'])
                        fewshot_examples += f"{relation_sample['question']} {random.choice(relation_sample['answers'])}\n"
                pre_context += f"Examples: {fewshot_examples}\n"
                has_pre_context = True
                
            # == Apply retrieved QA pairs ====================
            if args.with_rag_qa_pairs:
                qa_pairs_text = ""
                qa_pairs_data = ret_qa_results[query_id]['relevant_train_questions']
                if len(qa_pairs_data) > 0:
                    has_context = True
                    for qa_pair in qa_pairs_data:
                        qa_pairs_text += f"{qa_pair['question']} {qa_pair['answers'][0]}\n"
                
                pre_context += f"Highlight: {qa_pairs_text}\n"
                has_pre_context = True
            
            # == Apply highlight text ========================
            if args.with_rag_sentence_highlight:
                highlight_text = ""
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
                    highlighted_text_list = highlight_results[query_id]['sentences']
                    for item in highlighted_text_list:
                        ret_rank = item['ret_rank']
                        highlighted_text = item['sentence']
                        if 'sentence' in highlighted_text and len(highlighted_text['sentence']) != 0:
                            sentences = highlighted_text['sentence']
                            highlight_text += f"{' '.join(sentences)}\n"
                            has_pre_context = True
                        elif 'sentences' in highlighted_text and len(highlighted_text['sentences']) != 0:
                            sentences = highlighted_text['sentences']
                            for sentence in sentences:
                                if type(sentence) == str:
                                    highlight_text += f"{sentence}\n"
                            has_pre_context = True
                        pre_context += f"Highlight: {highlight_text}\n"
                        # else:
                        #     logging.info(f"\nNo highlighted text found for query: {query_id}, {query}, retrieved sentence: {ret_rank}")
                        #     print(f"\nNo highlighted text found for query: {query_id}, {query}, retrieved sentence: {ret_rank}")
            
                except json.decoder.JSONDecodeError as e:
                    print(f"Error decoding JSON for query_id {query_id}: {e}")
                    print(f"Problematic JSON string: {highlight_results[query_id]['highlighted_text']}")
            
            # == Apply retrieved sentence rerank =============
            if args.with_rag_sentence_rerank:
                rerank_results = ret_sent_rerank[query_id]['sentences']
                reranked_text = "".join(f"{rerank_results[i]['sentence']}, " for i in range(args.num_reranked_sentences) if i < len(rerank_results))
                pre_context += f"Highlight: {reranked_text}\n"
                has_pre_context = True
            
            # == Apply retrieved corpus text =================
            if args.with_rag_corpus:
                max_token = max_input_tokens - (70 if args.with_rag_qa_pairs else 20)
                corpus_text = "".join(ret_results[query_id]['ctxs'][i]['text'] for i in range(args.num_retrieved_passages) if i < len(ret_results[query_id]['ctxs']))
                rag_corpus = truncate_text(corpus_text, max_token) 
                main_context += f"Context: {rag_corpus}\n"
                has_main_context = True   
            
            # == Get highlighted passage =====================
            if args.with_rag_highlighted_passage:
                rerank_results = ret_sent_rerank[query_id]['sentences']
                first_reranked_ref = rerank_results[0]['ref_doc_id']
                highlighted_passage = corpus[first_reranked_ref]['content']
                corpus_text = "".join(ret_results[query_id]['ctxs'][i]['text'] for i in range(args.num_retrieved_passages-1) if i < len(ret_results[query_id]['ctxs']))
                main_context += f"Context: {highlighted_passage}\n{corpus_text}\n"
                has_main_context = True
            
            # # Adaptive text
            # # if query_relation in ['22', '91', '106', '164', '292', '422', '472', '560', '639']:
            # if query_id in ['22_1', '22_6', '22_7', '22_8', '22_16', '22_23', '22_25', '22_42', '22_54', '22_56', '22_69', '22_79', '22_119', '22_124', '22_126', '22_148', '22_194', '22_208', '22_215', '22_235', '22_236', '22_243', '22_246', '22_247', '22_254', '22_264', '22_268', '22_278', '22_299', '22_310', '22_355', '22_372', '22_378', '22_399', '22_401', '22_403', '22_407', '22_429', '22_442', '22_455', '22_494', '22_496', '22_531', '91_2', '91_11', '91_33', '91_48', '91_57', '91_74', '91_86', '91_98', '91_100', '91_103', '91_119', '91_126', '91_132', '91_133', '91_134', '91_141', '91_156', '91_157', '91_159', '91_167', '91_186', '91_193', '91_197', '91_205', '91_208', '91_234', '91_244', '91_249', '91_250', '91_265', '91_268', '91_272', '91_289', '91_304', '91_314', '91_324', '91_334', '91_386', '91_391', '91_397', '91_409', '91_410', '91_411', '91_419', '91_437', '91_447', '91_472', '91_476', '91_497', '91_522', '91_526', '91_528', '91_529', '91_537', '91_556', '91_565', '91_573', '91_576', '91_582', '91_599', '91_633', '91_634', '91_635', '91_647', '91_652', '91_657', '91_670', '91_674', '91_691', '91_698', '91_713', '91_716', '91_740', '91_745', '91_752', '91_760', '91_761', '91_769', '91_771', '91_788', '91_799', '91_814', '91_815', '91_818', '91_825', '91_826', '91_828', '91_832', '91_840', '91_847', '91_853', '91_854', '91_861', '91_864', '91_865', '91_879', '91_880', '91_888', '91_890', '91_895', '91_899', '91_901', '91_921', '91_923', '91_940', '91_951', '91_967', '91_978', '91_1002', '91_1027', '91_1032', '91_1034', '91_1036', '91_1044', '91_1050', '91_1062', '91_1063', '91_1067', '91_1071', '91_1080', '91_1097', '91_1098', '91_1102', '91_1110', '91_1120', '91_1144', '91_1145', '91_1157', '91_1222', '91_1249', '91_1254', '91_1255', '91_1262', '91_1265', '91_1272', '91_1279', '91_1281', '91_1286', '91_1301', '91_1303', '91_1315', '91_1331', '91_1332', '91_1350', '91_1355', '91_1361', '91_1365', '91_1369', '91_1370', '91_1372', '91_1374', '91_1383', '91_1394', '91_1420', '91_1435', '91_1436', '91_1438', '91_1439', '91_1444', '91_1460', '91_1467', '91_1478', '91_1479', '91_1482', '91_1489', '91_1513', '91_1515', '91_1532', '91_1538', '91_1551', '91_1553', '91_1556', '91_1559', '91_1597', '97_2', '97_26', '97_44', '97_53', '97_55', '97_69', '97_74', '97_86', '97_95', '97_100', '97_101', '97_112', '97_133', '97_135', '97_141', '97_143', '97_144', '97_148', '97_155', '97_159', '97_162', '97_175', '97_178', '97_179', '97_183', '97_184', '97_192', '97_200', '97_205', '97_206', '97_207', '97_219', '97_221', '97_230', '97_243', '97_245', '97_247', '97_248', '97_254', '97_267', '97_290', '97_295', '97_297', '97_299', '97_308', '97_309', '97_312', '97_313', '97_318', '97_333', '97_341', '97_362', '106_11', '106_16', '106_28', '106_31', '106_40', '106_54', '106_60', '106_71', '106_72', '106_80', '106_99', '106_100', '106_102', '106_116', '106_126', '106_127', '106_132', '106_135', '106_136', '106_150', '106_153', '106_157', '106_164', '106_173', '106_182', '106_186', '106_201', '106_205', '106_206', '106_218', '106_219', '106_220', '106_225', '106_226', '106_229', '106_235', '106_256', '106_258', '106_267', '106_268', '106_281', '106_298', '106_307', '106_308', '106_309', '106_310', '164_9', '164_22', '164_33', '164_37', '164_46', '164_58', '164_75', '164_82', '164_84', '164_85', '164_107', '164_118', '164_119', '164_127', '164_132', '164_135', '164_136', '164_151', '164_161', '164_192', '164_210', '164_214', '164_220', '164_223', '164_235', '164_236', '164_262', '164_281', '164_282', '164_287', '164_293', '164_312', '164_343', '164_369', '164_375', '164_383', '164_385', '164_386', '164_403', '164_412', '164_413', '164_417', '164_423', '164_425', '164_428', '164_435', '164_447', '164_452', '164_466', '164_475', '164_476', '164_507', '164_510', '164_519', '164_522', '164_542', '164_543', '164_554', '164_556', '164_567', '164_580', '164_593', '164_603', '164_605', '164_611', '164_612', '164_646', '164_656', '164_658', '164_660', '164_661', '164_671', '164_681', '164_686', '164_689', '164_706', '164_708', '164_717', '164_738', '164_748', '164_749', '164_760', '164_761', '164_769', '164_785', '164_788', '164_828', '164_833', '164_837', '164_842', '164_845', '164_846', '164_849', '164_851', '164_862', '164_869', '164_872', '164_873', '164_891', '164_892', '164_935', '164_942', '164_945', '164_957', '164_959', '164_966', '164_970', '164_973', '164_974', '164_975', '164_994', '164_995', '164_1003', '164_1027', '164_1031', '164_1036', '164_1046', '164_1054', '164_1062', '164_1068', '164_1077', '164_1079', '164_1091', '164_1105', '164_1119', '164_1135', '164_1137', '164_1143', '164_1146', '164_1150', '164_1157', '164_1165', '164_1173', '164_1181', '164_1199', '164_1201', '164_1211', '164_1228', '164_1237', '164_1248', '164_1263', '164_1270', '164_1271', '164_1276', '164_1279', '164_1300', '164_1320', '164_1325', '164_1326', '164_1328', '164_1331', '164_1333', '164_1337', '164_1366', '164_1367', '164_1373', '164_1374', '164_1382', '164_1400', '164_1401', '164_1404', '164_1424', '164_1431', '164_1450', '164_1453', '164_1455', '164_1460', '164_1476', '164_1480', '164_1488', '164_1489', '164_1492', '164_1499', '164_1501', '164_1505', '164_1516', '182_7', '182_38', '182_41', '182_53', '182_62', '182_83', '182_96', '182_128', '182_132', '182_185', '182_204', '182_224', '182_226', '182_240', '182_255', '182_259', '182_313', '182_342', '182_379', '182_436', '182_447', '182_488', '182_493', '182_494', '182_529', '182_548', '182_564', '182_612', '182_644', '182_650', '182_657', '182_664', '182_718', '182_719', '182_723', '182_734', '182_803', '182_813', '218_21', '218_23', '218_25', '218_52', '218_58', '218_62', '218_64', '218_71', '218_81', '218_104', '218_116', '218_131', '218_138', '218_164', '218_172', '218_185', '218_202', '218_226', '218_227', '218_231', '218_238', '218_240', '218_243', '218_252', '218_257', '218_272', '218_276', '218_300', '218_322', '218_340', '218_343', '218_346', '218_362', '218_367', '218_368', '218_396', '218_397', '218_426', '218_453', '218_479', '218_486', '218_501', '218_503', '218_505', '218_516', '218_524', '218_525', '218_560', '218_561', '218_573', '257_1', '257_6', '257_11', '257_13', '257_21', '257_31', '257_37', '257_41', '257_49', '257_60', '257_64', '257_82', '257_86', '257_91', '257_126', '257_128', '257_131', '257_139', '257_141', '257_154', '257_155', '257_156', '257_173', '257_192', '257_198', '257_210', '257_212', '257_228', '257_235', '257_237', '257_247', '257_250', '257_256', '257_266', '257_280', '257_287', '257_300', '257_305', '257_309', '257_311', '257_312', '257_319', '257_326', '257_328', '257_331', '257_350', '257_356', '257_371', '257_374', '257_375', '257_389', '257_390', '257_398', '257_423', '257_427', '257_433', '257_448', '257_456', '257_472', '257_492', '257_498', '257_511', '257_515', '257_522', '257_543', '257_559', '257_567', '292_3', '292_4', '292_7', '292_11', '292_12', '292_13', '292_24', '292_33', '292_41', '292_46', '292_52', '292_55', '292_61', '292_62', '292_66', '292_68', '292_70', '292_74', '292_83', '292_85', '292_87', '292_90', '292_98', '292_115', '292_121', '292_128', '292_130', '292_146', '292_155', '292_161', '292_168', '292_177', '292_182', '422_3', '422_12', '422_34', '422_35', '422_37', '422_43', '422_76', '422_83', '422_85', '422_96', '422_123', '422_129', '422_132', '422_138', '422_143', '422_146', '422_148', '422_154', '422_158', '422_159', '422_164', '422_166', '422_177', '422_182', '422_188', '422_193', '422_200', '422_225', '422_256', '422_265', '422_282', '422_283', '422_304', '422_306', '422_310', '422_315', '422_319', '422_323', '422_325', '422_329', '422_337', '422_346', '422_351', '422_382', '422_383', '422_391', '422_395', '422_404', '422_411', '422_417', '422_423', '422_424', '422_425', '422_427', '422_431', '422_448', '422_456', '422_460', '422_481', '422_497', '422_498', '422_503', '422_518', '422_524', '422_525', '422_535', '422_537', '422_562', '422_565', '422_604', '422_607', '422_612', '422_613', '422_615', '422_623', '422_625', '422_627', '422_630', '472_14', '472_15', '472_16', '472_17', '472_22', '472_26', '472_33', '484_9', '484_22', '484_36', '484_38', '484_43', '484_51', '484_54', '484_57', '484_123', '484_124', '484_141', '484_143', '484_152', '484_172', '484_208', '484_226', '484_254', '484_267', '484_268', '484_270', '484_271', '484_287', '484_306', '484_314', '484_337', '484_343', '484_347', '484_375', '484_386', '484_389', '484_403', '484_416', '484_476', '484_480', '484_507', '484_515', '484_519', '484_522', '484_523', '484_525', '484_534', '484_539', '484_543', '484_578', '484_586', '484_597', '484_601', '484_612', '484_629', '484_681', '484_684', '484_690', '484_706', '484_748', '484_763', '484_785', '484_788', '484_790', '484_792', '484_796', '484_830', '484_838', '484_862', '484_865', '484_913', '484_930', '484_962', '484_964', '484_978', '484_985', '484_1041', '484_1043', '484_1058', '484_1088', '484_1090', '484_1091', '484_1093', '484_1094', '484_1117', '484_1120', '484_1137', '484_1178', '484_1206', '484_1219', '484_1249', '484_1258', '484_1262', '484_1267', '484_1275', '484_1334', '484_1339', '484_1355', '484_1365', '484_1379', '484_1432', '484_1438', '484_1445', '484_1449', '484_1488', '484_1497', '484_1504', '526_12', '526_37', '526_67', '526_78', '526_84', '526_97', '526_98', '526_121', '526_123', '526_125', '526_132', '526_136', '526_148', '526_160', '526_165', '526_170', '526_175', '526_194', '526_200', '526_207', '526_242', '526_257', '526_267', '526_318', '526_339', '526_362', '526_382', '526_395', '526_398', '526_400', '526_408', '526_422', '526_442', '526_443', '526_460', '526_482', '526_503', '526_521', '526_550', '526_577', '526_591', '526_600', '526_603', '526_613', '526_616', '526_627', '526_635', '526_647', '526_681', '526_704', '526_708', '526_712', '526_716', '526_717', '526_729', '526_747', '526_759', '526_767', '526_782', '526_788', '526_803', '526_804', '526_854', '526_865', '526_867', '526_869', '526_900', '526_924', '526_928', '526_970', '526_979', '526_982', '526_986', '526_1009', '526_1029', '526_1032', '526_1034', '526_1051', '526_1058', '526_1060', '526_1062', '526_1072', '526_1081', '526_1087', '526_1105', '526_1116', '526_1125', '526_1128', '526_1148', '526_1174', '526_1181', '526_1193', '526_1209', '526_1226', '526_1238', '526_1246', '526_1250', '526_1259', '526_1303', '526_1311', '526_1326', '526_1362', '526_1372', '526_1381', '526_1449', '526_1478', '526_1485', '526_1515', '526_1528', '526_1534', '526_1536', '526_1539', '526_1547', '526_1556', '526_1559', '526_1564', '526_1566', '526_1579', '526_1590', '526_1629', '526_1643', '526_1678', '526_1692', '526_1737', '526_1742', '526_1743', '526_1745', '526_1760', '526_1763', '526_1764', '526_1774', '526_1785', '526_1794', '526_1802', '526_1803', '526_1833', '526_1855', '526_1857', '526_1862', '526_1866', '526_1868', '526_1877', '526_1885', '526_1911', '526_1914', '526_1925', '526_1934', '526_1958', '526_1968', '526_1970', '533_17', '533_44', '533_78', '533_82', '533_114', '533_139', '533_151', '533_155', '533_167', '533_171', '533_172', '533_175', '533_182', '533_188', '533_190', '533_224', '533_235', '533_251', '533_252', '533_259', '533_261', '533_272', '533_314', '533_346', '533_348', '533_361', '533_372', '533_381', '533_383', '533_425', '533_436', '533_442', '533_448', '533_449', '533_451', '533_467', '533_480', '533_483', '533_490', '533_493', '533_494', '533_498', '533_538', '533_543', '533_580', '533_584', '533_598', '533_603', '533_614', '533_626', '533_647', '533_648', '533_680', '533_698', '533_708', '533_713', '533_717', '533_735', '533_742', '533_754', '533_758', '533_772', '533_777', '533_830', '533_864', '533_866', '533_870', '533_880', '533_903', '533_905', '533_922', '533_924', '533_926', '533_943', '533_967', '533_972', '533_979', '533_985', '533_991', '533_1024', '533_1037', '533_1045', '533_1048', '533_1058', '533_1066', '533_1113', '533_1136', '533_1139', '533_1141', '533_1149', '533_1166', '533_1179', '533_1205', '533_1215', '533_1222', '533_1231', '533_1248', '533_1253', '533_1290', '533_1301', '533_1304', '533_1307', '533_1312', '533_1318', '533_1345', '533_1350', '533_1365', '533_1389', '533_1402', '533_1415', '533_1418', '533_1422', '533_1432', '533_1433', '533_1439', '533_1441', '533_1452', '533_1483', '533_1485', '533_1554', '533_1564', '533_1566', '533_1570', '533_1573', '533_1583', '533_1595', '533_1598', '533_1623', '533_1644', '533_1645', '533_1647', '533_1679', '533_1696', '533_1701', '533_1703', '533_1732', '533_1742', '533_1747', '533_1768', '533_1773', '533_1788', '533_1815', '533_1823', '533_1842', '533_1845', '533_1855', '533_1859', '533_1867', '533_1873', '533_1877', '533_1902', '533_1913', '533_1918', '533_1928', '533_1942', '533_1950', '533_1956', '533_1963', '533_1979', '560_4', '560_47', '560_53', '560_65', '560_68', '560_77', '560_111', '560_130', '560_149', '560_158', '560_159', '560_169', '560_175', '560_185', '560_203', '560_238', '560_246', '560_249', '560_257', '560_305', '560_325', '560_326', '560_379', '560_415', '560_418', '560_426', '560_446', '560_465', '560_468', '560_527', '560_531', '560_535', '560_544', '560_545', '639_0', '639_3', '639_5', '639_11', '639_15', '639_17', '639_18', '639_26', '639_27', '639_35', '639_39', '639_56', '639_57', '639_80', '639_109', '639_112', '639_119', '639_124', '639_129', '639_134', '639_144', '639_146', '639_159', '639_180', '639_191', '639_222', '639_228', '639_230', '639_240', '639_249', '639_250', '639_259', '639_260', '639_267', '639_271', '639_273', '639_281', '639_328', '639_330', '639_332', '639_354', '639_359', '639_361', '639_411', '639_413', '639_416', '639_425', '639_456', '639_465', '639_475', '639_483', '639_495', '639_504', '639_509', '639_511', '639_534', '639_536', '639_537', '639_542', '639_555', '639_564', '639_576', '639_579', '639_593', '639_598', '639_605', '639_608', '639_611', '639_615', '639_625', '639_628', '639_629', '639_632', '639_644', '639_650', '639_658', '639_671', '639_705', '639_706', '639_707', '639_717', '639_723', '639_757', '639_758', '639_791', '639_800', '639_808', '639_814', '639_820', '639_826', '639_828', '639_833', '639_846', '639_849', '639_859', '639_875', '639_881', '639_886', '639_903', '639_920', '639_934', '639_939', '639_943', '639_959']:
            #     main_context += f"Context: {highlighted_passage}\n"
            #     has_main_context = True
            # else:
            #     main_context += f"Context: {rag_corpus}\n"
            #     has_main_context = True
            
            if has_main_context and has_pre_context:
                context = f"{pre_context}{main_context}"
                prompt = prompt_template_w_context.format(context=context, question=query)  
            elif has_main_context and not has_pre_context:
                prompt = prompt_template_w_context.format(context=main_context, question=query)
            elif not has_main_context and has_pre_context:
                prompt = prompt_template_w_context.format(context=pre_context, question=query)
            elif not has_main_context and not has_pre_context:
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
            if idx < 10 or idx % 100 == 0:
                # logging.info('\n')
                # logging.info(f"Prompt: {prompt}")
                # logging.info(f"Query: {query}")
                # logging.info(f"Has context: {has_main_context}"),
                # logging.info(f"Pred: {pred}")
                # logging.info(f"Labels: {test_answers[idx]}")
                # logging.info(f"Final decision: {is_correct}")
                # logging.info('====')
                print('\n')
                print(f"Prompt: {prompt}")
                print(f"Query: {query}")
                print(f"Has context: {has_main_context}"),
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
                "has_context": has_main_context,
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
    
    parser.add_argument("--with_fewshot_examples", type=str2bool, default=False)
    parser.add_argument("--with_rag_qa_pairs", type=str2bool, default=False)
    parser.add_argument("--with_rag_sentence_highlight", type=str2bool, default=False)
    parser.add_argument("--with_rag_sentence_rerank", type=str2bool, default=False)
    parser.add_argument("--with_rag_highlighted_passage", type=str2bool, default=False)
    parser.add_argument("--num_reranked_sentences", type=int, default=1)
    
    parser.add_argument("--with_rag_corpus", type=str2bool, default=False)
    parser.add_argument("--num_retrieved_passages", type=int, default=1)
    parser.add_argument("--retrieval_method", type=str)
    parser.add_argument("--seed", type=int)
    
    args = parser.parse_args()
    main(args)
    
    