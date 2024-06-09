
import torch
import random
import logging
import numpy as np
import argparse, os, json
from transformers import pipeline
from peft import PeftConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_model(args):

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


def main(args):
    
    print(f"""
        Model: {args.model_name_or_path}
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
        prompt_template = """{prompt}"""
        
    elif args.llm_model_name in ["llama2", "mistral"]:
        prompt_template = """<s>[INST] <<SYS>><</SYS>> \n{prompt}\n[/INST]"""  
        
    elif args.llm_model_name in ["zephyr", "tiny_llama"]:
        prompt_template = """<|system|> </s>\n <|user|>\n{prompt}\n</s>\n <|assistant|>"""
    
    elif args.llm_model_name == "stable_lm2":
        prompt_template = """<|user|>\n{prompt}\n<|endoftext|>\n<|assistant|>"""
    
    elif args.llm_model_name == "MiniCPM":
        prompt_template = """<User>\n{prompt}\n <AI>"""

    elif args.llm_model_name == "llama3":
        prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|><|eot_id|><|start_header_id|>user<|end_header_id|>\n{prompt}\n<|eot_id|>"""

    print("Inferencing ...")
    model, tokenizer = load_model(args)
    
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
    
    prompt = prompt_template.format(prompt=args.prompt)
    result = pipe(prompt)[0]['generated_text']
    
    print(f"\nPrompt: {prompt}\n")
    
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
    
    print(f'Prediction: {pred}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--llm_model_name", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--seed", type=int)
    
    args = parser.parse_args()
    main(args)
    