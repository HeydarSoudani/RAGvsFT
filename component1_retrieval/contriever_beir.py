from contriever.src.contriever import Contriever
from transformers import AutoTokenizer

import contriever.src.slurm
import contriever.src.contriever
from contriever.src.beir_utils import evaluate_model
import contriever.src.utils
import contriever.src.dist_utils
import contriever.src.contriever

# Parameters
dataset = 'popqa'
per_gpu_batch_size = 128
norm_query = True
norm_doc = True
score_function = 'dot'
dataset_dir = "popqa_data"
save_results_path = "results"
lower_case = True
normalize_text = True

if __name__ == "__main__":
    model = Contriever.from_pretrained("facebook/contriever") 
    tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
    
    # model.cuda()
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
        is_main=contriever.src.dist_utils.is_main(),
        split="dev" if dataset == "msmarco" else "test",
        score_function=score_function,
        data_path=dataset_dir,
        save_results_path=save_results_path,
        lower_case=lower_case,
        normalize_text=normalize_text,
    )
    
    if contriever.src.dist_utils.is_main():
        for key, value in metrics.items():
            print(f"{dataset} : {key}: {value:.1f}")