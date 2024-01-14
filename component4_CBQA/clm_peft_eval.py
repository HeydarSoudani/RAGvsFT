import argparse, json
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import numpy as np
from tqdm import tqdm

q_templates = {
    22: "What is {}'s occupation?",
    218: "In what city was {} born?",
    91: "What genre is {}?",
    257: "Who is the father of {}?",
    182: "In what country is {}?",
    164: "Who was the producer of {}?",
    526: "Who was the director of {}?",
    97: "What is {} the capital of?",
    533: "Who was the screenwriter for {}?",
    639: "Who was the composer of {}?",
    472: "What color is {}?",
    106: "What is the religion of {}?",
    560: "What sport does {} play?",
    484: "Who is the author of {}?",
    292: "Who is the mother of {}?",
    422: "What is the capital of {}?"
}
completion_template = "Q: {} A:"  # "{}" # "Query: {}\nResult:" # "Q: {} A:" # "{} The answer is"

def call_model(prompt, model, tokenizer, device, max_new_tokens=15, model_max_length=None):
    max_inpt_tokens = tokenizer.model_max_length if model_max_length is None else model_max_length
    inpts = tokenizer(prompt, return_tensors="pt").to(device)
    gen = model.generate(input_ids=inpts.input_ids[:, -(max_inpt_tokens - max_new_tokens):], attention_mask=inpts.attention_mask[:, -(max_inpt_tokens - max_new_tokens):], pad_token_id=tokenizer.eos_token_id, max_new_tokens=max_new_tokens, num_beams=1, do_sample=False)
    text = tokenizer.decode(gen[0])
    actual_prompt = tokenizer.decode(inpts.input_ids[0, -(max_inpt_tokens - max_new_tokens):])
    pred = text[len(actual_prompt):]
    if pred.startswith("\n\n"):
        pred = pred[2:]
    pred = pred.split("\n")[0]
    return pred, text

def clip_paragraph(text, eval_method):
    if eval_method in ["BM25", "genread"]:
        return text
    split = text.split(". ")
    return ". ".join(split[:-1]) + "."

def get_few_shot_text_with_retrieval(row, retrieval_dict, eval_method):
    if eval_method == "vanilla":
        return completion_template.format(row.question) + " " + row.obj
      # retrieval_dict[row.id]["ctxs"][0]
    if row.question.replace("?", "").lower() not in retrieval_dict:
        print("missing retrieval")
        return completion_template.format(row.question) + " " + row.obj
    else:
        retrieval = retrieval_dict[row.question.replace("?", "").lower()]["ctxs"][0]
        retrieved_text = clip_paragraph(retrieval["text"], eval_method)
        return retrieved_text + "\n\n" + completion_template.format(row.question) + " " + row.obj

def get_few_shot_text(row, eval_method):
    return completion_template.format(row.question) + " " + row.obj


def main(args):
    # === model intro ====
    repo_name = "HeydarS/opt-350m-lora"
    device = args.device
    
    config = PeftConfig.from_pretrained(repo_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path, return_dict=True, load_in_8bit=True, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    # = Add peft model ===
    model = PeftModel.from_pretrained(model, repo_name)
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    generate = lambda prompt, max_new_tokens: call_model(prompt, model=model, tokenizer=tokenizer, device=device, max_new_tokens=max_new_tokens, model_max_length=2048)
    
    # === Load data ========
    input_path = args.input_file
    knowledge = pd.read_csv(input_path, sep="\t")
    
    # 
    if args.continue_from is not None:
        results = pd.read_csv(args.continue_from, sep="\t")
        knowledge = knowledge[~knowledge.id.isin(results.id)]
    n = len(knowledge) if args.sample == 0 else args.sample
    sample = knowledge.sample(n=n, replace=False)
    if args.parallel is not None:
        worker_num, n_workers = map(int, args.parallel.split("."))
        sample = sample.iloc[worker_num::n_workers]

    n_examples = args.n_examples
    is_templatedQA = True
    examples_per_template = n_examples // (len(q_templates) - 1)

    preds = []
    prompts =[]
    accuracy = []
    responses = []
    retrieval_dict = {}
    
    # main loop
    for row in tqdm(sample.iloc, total=n):

        # get few shot examples text
        if n_examples == 0:
            few_shot_examples_text = ""
        else:
            few_shot_examples = []
            if is_templatedQA:
                other_pids = list(q_templates.keys())
                other_pids.remove(row.prop_id)
                for pid in other_pids:
                    for row2 in knowledge[knowledge.prop_id == pid].sample(n=examples_per_template).iloc:
                        few_shot_examples.append(get_few_shot_text_with_retrieval(row2, retrieval_dict, args.eval_method) if args.eval_method in ["BM25", "contriever"] else get_few_shot_text(row2, args.eval_method))
            else:
                for row2 in knowledge[knowledge.question != row.question].sample(n=n_examples).iloc:
                    few_shot_examples.append(get_few_shot_text_with_retrieval(row2, retrieval_dict, args.eval_method) if args.eval_method in ["BM25", "contriever"] else get_few_shot_text(row2, args.eval_method))
                
            np.random.shuffle(few_shot_examples)
            few_shot_examples_text = "\n\n".join(few_shot_examples) + "\n\n"
        
        # get prompt
        prompt = few_shot_examples_text + completion_template.format(row.question)
        
        # generate response
        pred, response = generate(prompt, max_new_tokens=args.max_new_tokens)
        prompts.append(prompt)
        preds.append(pred)
        responses.append(response)

        # compute accuracy
        possible_answers = json.loads(row.possible_answers)        
        is_correct = False
        for pa in possible_answers:
            if pa in pred or pa.lower() in pred or pa.capitalize() in pred:
                is_correct = True
        accuracy.append(is_correct)
        
        # save results intermittently
        if len(preds) % 100 == 0:
            temp_sample = sample.iloc[:len(preds)].copy()
            temp_sample["pred"] = preds
            temp_sample["prompt"] = prompts
            temp_sample["generation"] = responses
            temp_sample["is_correct"] = accuracy
    
    sample["is_correct"] = accuracy
    sample["prompt"] = prompts
    sample["pred"] = preds
    sample["generation"] = responses
    print(sample.is_correct.mean())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("-m", "--model_name", required=True)
    # parser.add_argument("--repo_name", type=str)
    # parser.add_argument("--corpus_path", type=str)
    # parser.add_argument("--model_output_dir", type=str)
    # parser.add_argument("--model_output_filename", type=str)
    # parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--n_examples', type=int, default=15)
    parser.add_argument('--eval_method', type=str, default="vanilla", choices=["vanilla", "BM25", "contriever", "genread"])
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--max_new_tokens', type=int, default=15)
    parser.add_argument('--sample', type=int, default=0, help="if 0, use all examples")
    parser.add_argument('--continue_from', type=str, help="path to previous results file")
    parser.add_argument('--int8bit', action="store_true")
    parser.add_argument('--parallel', type=str, help="string of format 'i.n_workers' where i is the index of the worker")

    
    args = parser.parse_args()
    main(args)