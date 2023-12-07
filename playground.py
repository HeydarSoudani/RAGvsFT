from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset, DatasetDict
from utils import load_json_file, read_tsv_column
import nltk
import string
import numpy as np
# from rouge_score import rouge_scorer, scoring
import evaluate
import torch

nltk.download('punkt')

def main():
    model_checkpoint = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    
    # == loading dataset =========
    dataset_test_path = "./data/dataset/popQA.tsv"
    questions = read_tsv_column(dataset_test_path, 'question')
    possible_answers = read_tsv_column(dataset_test_path, 'possible_answers')
    
    popqa_dataset = DatasetDict({
        'train': Dataset.from_dict({
            "question": questions, 
            "answer": [pa[0] for pa in possible_answers]
        })
    })
    
    datasets_train_test = popqa_dataset["train"].train_test_split(test_size=1500)
    datasets_train_validation = datasets_train_test["train"].train_test_split(test_size=1500)
    popqa_dataset["train"] = datasets_train_validation["train"]
    popqa_dataset["validation"] = datasets_train_validation["test"]
    popqa_dataset["test"] = datasets_train_test["test"]
    
    n_samples_train = len(popqa_dataset["train"])
    n_samples_validation = len(popqa_dataset["validation"])
    n_samples_test = len(popqa_dataset["test"])
    n_samples_total = n_samples_train + n_samples_validation + n_samples_test

    print(f"- Training set: {n_samples_train*100/n_samples_total:.2f}%")
    print(f"- Validation set: {n_samples_validation*100/n_samples_total:.2f}%")
    print(f"- Test set: {n_samples_test*100/n_samples_total:.2f}%")
    
    # keep only a subsample of the datasets
    popqa_dataset["train"] = popqa_dataset["train"].shuffle().select(range(1000))
    popqa_dataset["validation"] = popqa_dataset["validation"].shuffle().select(range(100))
    popqa_dataset["test"] = popqa_dataset["test"].shuffle().select(range(100))
    
    print(popqa_dataset)

    # === Data preprocessing ==============
    prefix = "qa: "
    max_input_length = 512
    max_target_length = 64
    
    def clean_text(text):
        sentences = nltk.sent_tokenize(text.strip())
        sentences_cleaned = [s for sent in sentences for s in sent.split("\n")]
        sentences_cleaned_no_titles = [sent for sent in sentences_cleaned
                                        if len(sent) > 0 and
                                        sent[-1] in string.punctuation]
        text_cleaned = "\n".join(sentences_cleaned_no_titles)
        return text_cleaned

    def preprocess_data(examples):
        texts_cleaned = [clean_text(text) for text in examples["question"]]
        inputs = [prefix + text for text in texts_cleaned]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["answer"], max_length=max_target_length, 
                            truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # popqa_dataset_cleaned = popqa_dataset.filter(lambda example: (len(example['question']) >= 500) and (len(example['answer']) >= 20))
    tokenized_datasets = popqa_dataset.map(preprocess_data, batched=True)
    print(tokenized_datasets)
    
    
    # === Fine-tune T5 ====================
    batch_size = 8
    metric = evaluate.load("rouge")
    # metric = evaluate.load("exact_match", module_type="comparison")
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        print(len(predictions))
        print(predictions[0])
        print(len(labels))
        print(labels[0])
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip()))
                        for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) 
                        for label in decoded_labels]
        
        # Compute ROUGE scores
        result = metric.compute(predictions=decoded_preds, references=decoded_labels,
                                use_stemmer=True)

        # Extract ROUGE f1 scores
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        
        # Add mean generated length to metrics
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id)
                        for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
        
        return {k: round(v, 4) for k, v in result.items()}
    
    args = Seq2SeqTrainingArguments(
        output_dir="my_fine_tuned_t5_small_model",
        evaluation_strategy="steps",
        eval_steps=100,
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="steps",
        save_steps=200,
        learning_rate=4e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=1,
        predict_with_generate=True,
        # fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="rouge1",
        report_to="tensorboard"
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer)
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    trainer.save_model()
    
    # === Test =========================
    path = "my_fine_tuned_t5_small_model"
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(path)
    
    # get test split
    test_tokenized_dataset = tokenized_datasets["test"]

    # pad texts to the same length
    def preprocess_test(examples):
        inputs = [prefix + text for text in examples["question"]]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True,
                                padding="max_length")
        return model_inputs

    test_tokenized_dataset = test_tokenized_dataset.map(preprocess_test, batched=True)

    # prepare dataloader
    test_tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    dataloader = torch.utils.data.DataLoader(test_tokenized_dataset, batch_size=32)

    # generate text for each batch
    all_predictions = []
    print(dataloader)
    for i,batch in enumerate(dataloader):
        print(batch)
        predictions = model.generate(**batch)
        all_predictions.append(predictions)

    # flatten predictions
    all_predictions_flattened = [pred for preds in all_predictions for pred in preds]

    # tokenize and pad titles
    all_titles = tokenizer(test_tokenized_dataset["answer"], max_length=max_target_length,
                        truncation=True, padding="max_length")["input_ids"]

    # compute metrics
    predictions_labels = [all_predictions_flattened, all_titles]
    compute_metrics(predictions_labels)

if __name__ == "__main__":
    main()
    