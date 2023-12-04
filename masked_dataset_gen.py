from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling
from transformers import default_data_collator, get_scheduler, TrainingArguments, Trainer
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from datasets import Dataset, DatasetDict
from huggingface_hub import get_full_repo_name
from huggingface_hub import Repository
from transformers import pipeline

from utils import load_json_file, read_tsv_column
import json
from tqdm.auto import tqdm
import math

# 1) Get wikipedia text
# 2) Split by sentences
# 3) Use AE module to extract appropriate entities
# 4) Create samples in dataset style
#   - 
#   - 

def main():
    
    model_checkpoint = "distilbert-base-uncased"
    # model_checkpoint = "google/t5-v1_1-base"
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    chunk_size = 128
    batch_size = 64
    wwm_probability = 0.2
    train_size = 12222
    test_size = int(0.1 * train_size)
    num_train_epochs = 10
    
    # === Defining functions =============
    def tokenize_function(examples):
        result = tokenizer(examples['text'])
        if tokenizer.is_fast:
            result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        return result
    
    def group_texts(examples):
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        # Compute length of concatenated texts
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the last chunk if it's smaller than chunk_size
        total_length = (total_length // chunk_size) * chunk_size
        # Split by chunks of max_len
        result = {
            k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
            for k, t in concatenated_examples.items()
        }
        # Create a new labels column
        result["labels"] = result["input_ids"].copy()
        return result

    def insert_random_mask(batch):
        features = [dict(zip(batch, t)) for t in zip(*batch.values())]
        masked_inputs = data_collator(features)
        # Create a new "masked" column for each column in the dataset
        return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}
    # ====================================

    wiki_context = load_json_file("data/generated/wikiapi_results.json")
    #TODO: add loop one list of context
    context_dataset = DatasetDict({
        'train': Dataset.from_dict({
            "entity": [entity for entity, context in wiki_context.items()], 
            "text": [context[0] for entity, context in wiki_context.items()]
        })
    })
    print(context_dataset)
    tokenized_datasets = context_dataset.map(
        tokenize_function, batched=True, remove_columns=["text", "entity"]
    )

    lm_datasets = tokenized_datasets.map(group_texts, batched=True)
    downsampled_dataset = lm_datasets["train"].train_test_split(
        train_size=train_size, test_size=test_size, seed=42
    )
    downsampled_dataset = downsampled_dataset.remove_columns(["word_ids"])
    
    eval_dataset = downsampled_dataset["test"].map(
        insert_random_mask,
        batched=True,
        remove_columns=downsampled_dataset["test"].column_names,
    )
    eval_dataset = eval_dataset.rename_columns(
        {
            "masked_input_ids": "input_ids",
            "masked_attention_mask": "attention_mask",
            "masked_labels": "labels",
        }
    )

    # # ========= other version of training ========
    # logging_steps = len(downsampled_dataset["train"]) // batch_size
    # model_name = model_checkpoint.split("/")[-1]

    # training_args = TrainingArguments(
    #     output_dir="./data/{}-finetuned-wiki".format(model_name),
    #     overwrite_output_dir=True,
    #     evaluation_strategy="epoch",
    #     learning_rate=2e-5,
    #     weight_decay=0.01,
    #     per_device_train_batch_size=batch_size,
    #     per_device_eval_batch_size=batch_size,
    #     push_to_hub=True,
    #     fp16=True,
    #     logging_steps=logging_steps,
    # )
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=downsampled_dataset["train"],
    #     eval_dataset=downsampled_dataset["test"],
    #     data_collator=data_collator,
    #     tokenizer=tokenizer,
    # )

    # eval_results = trainer.evaluate()
    # print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    # trainer.train()

    # eval_results = trainer.evaluate()
    # print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")    
    # trainer.push_to_hub()



    train_dataloader = DataLoader(
        downsampled_dataset["train"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=batch_size, collate_fn=default_data_collator
    )

    optimizer = AdamW(model.parameters(), lr=5e-5)
    accelerator = Accelerator()
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    progress_bar = tqdm(range(num_training_steps))

    model_name = "distilbert-base-uncased-finetuned-wiki-accelerate"
    output_dir = "./data/saved-model"
    # repo_name = get_full_repo_name(model_name)
    # output_dir = model_name
    # repo = Repository(output_dir, clone_from=repo_name)

    ### === Train loop ========== 
    for epoch in range(num_train_epochs):
        # Training
        model.train()
        for batch in train_dataloader:

            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        # Evaluation
        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(accelerator.gather(loss.repeat(batch_size)))

        losses = torch.cat(losses)
        losses = losses[: len(eval_dataset)]
        try:
            perplexity = math.exp(torch.mean(losses))
        except OverflowError:
            perplexity = float("inf")

        print(f">>> Epoch {epoch}: Perplexity: {perplexity}")

        # Save and upload
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)
            # repo.push_to_hub(
            #     commit_message=f"Training in progress epoch {epoch}", blocking=False
            # )


def popqa_inference():
    ## === Test on PopQA ========== 
    # compute accuracy: from popQA paper
    preds = []
    prompts =[]
    accuracy = []
    responses = []

    print("test on PopQA ....")
    dataset_test_path = "./data/dataset/popQA.tsv"
    questions = read_tsv_column(dataset_test_path, 'question')
    possible_answers = read_tsv_column(dataset_test_path, 'possible_answers')
    completion_template = "Q: {} A: [MASK]"
    

    # ===== (1)
    # inputs = tokenizer(text, return_tensors="pt")
    # token_logits = model(**inputs).logits
    # # Find the location of [MASK] and extract its logits
    # mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    # mask_token_logits = token_logits[0, mask_token_index, :]
    # # Pick the [MASK] candidates with the highest logits
    # top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()


    # ===== (2)
    tokenizer = AutoTokenizer.from_pretrained("./data/saved-model")
    model = AutoModelForMaskedLM.from_pretrained("./data/saved-model") 
    
    progress_bar = tqdm(range(len(questions)))
    for idx, items in enumerate(possible_answers):
        is_correct = False
        inputs = tokenizer(completion_template.format(questions[idx]), return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
        predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
        output = tokenizer.decode(predicted_token_id)
        # print(output)

        for pa in items:
            if pa in output or pa.lower() in output or pa.capitalize() in output:
                is_correct = True
        accuracy.append(is_correct)
        progress_bar.update(1)
    
    correct_predictions = sum(pred_val for pred_val in accuracy)
    acc = correct_predictions / len(accuracy)
    print("acc: {}".format(acc*100))



    # inputs = tokenizer("The internet [MASK] amazing.", return_tensors="pt")
    # with torch.no_grad():
    #     logits = model(**inputs).logits
    # # retrieve index of [MASK]
    # mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    # predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
    # output = tokenizer.decode(predicted_token_id)
    # print(output) 


    # mask_filler = pipeline(
    #     "fill-mask", model='./data/saved-model', tokenizer="./data/saved-model"
    # )

    # for idx, items in enumerate(possible_answers):
    #     is_correct = False
    #     pred = mask_filler(completion_template.format(questions[idx]))

    #     for pa in items:
    #         if pa in pred or pa.lower() in pred or pa.capitalize() in pred:
    #             is_correct = True
    #     accuracy.append(is_correct)
    
    # correct_predictions = sum(pred_val for pred_val in accuracy)
    # acc = correct_predictions / len(accuracy)
    # print("acc: {}".format(acc))





if __name__ == "__main__":
    main()
    popqa_inference()