from src.contriever import Contriever
from transformers import AutoTokenizer
import torch

import src.slurm
from src.beir_utils import evaluate_model
import src.utils
import src.dist_utils

if torch.cuda.is_available():
    device = torch.device("cuda:0") 
    print("Running on the GPU")
elif torch.backends.mps.is_available():
    device = torch.device("mps:0")
    print("Running on the mps")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

# Parameters
dataset = 'popqa'
per_gpu_batch_size = 128
norm_query = True
norm_doc = True
score_function = 'dot'
dataset_dir = "component1_retrieval/popqa_data"
save_results_path = "results"
lower_case = True
normalize_text = True

if __name__ == "__main__":
    model = Contriever.from_pretrained("facebook/contriever") 
    tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
    
    model.to(device)
    model.eval()
    query_encoder = model
    doc_encoder = model
    
    metrics = evaluate_model(
        query_encoder=query_encoder,
        doc_encoder=doc_encoder,
        tokenizer=tokenizer,
        dataset=dataset,
        batch_size=per_gpu_batch_size,
        norm_query=norm_query,
        norm_doc=norm_doc,
        is_main=src.dist_utils.is_main(),
        split="dev" if dataset == "msmarco" else "test",
        score_function=score_function,
        data_path=dataset_dir,
        save_results_path=save_results_path,
        lower_case=lower_case,
        normalize_text=normalize_text,
        device=device
    )
    
    if src.dist_utils.is_main():
        for key, value in metrics.items():
            print(f"{dataset} : {key}: {value:.1f}")