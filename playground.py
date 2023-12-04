from transformers import AutoModelForMaskedLM, AutoTokenizer

def MLMloss():
    model_checkpoint = "distilbert-base-uncased"
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    text = "Who was Jim Paterson ? Jim Paterson is a doctor".lower()
    inputs  =  tokenizer([text],  return_tensors="pt")
    
    print(inputs)
    
    input_ids = inputs["input_ids"]
    input_ids[0][7] = tokenizer.mask_token_id

    print(inputs)

    labels = inputs["input_ids"].clone()
    labels[labels != tokenizer.mask_token_id] = -100 # only calculate loss on masked tokens

    print(labels)

    loss, logits = model(
        input_ids=input_ids,
        labels=labels,
        attention_mask=inputs["attention_mask"],
        # token_type_ids=inputs["token_type_ids"]
    )
    print(loss)
    print(logits)


if __name__ == "__main__":
    MLMloss()