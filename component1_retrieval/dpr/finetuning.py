from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever
from sentence_transformers import SentenceTransformer, losses, models

import os

prefix = "gen"
data_path = "component1_retrieval/popqa_data"
corpus, gen_queries, gen_qrels = GenericDataLoader(data_path, prefix=prefix).load(split="train")

model = SentenceTransformer("msmarco-distilbert-base-v3")
retriever = TrainRetriever(model=model, batch_size=64)

train_samples = retriever.load_train(corpus, gen_queries, gen_qrels)
train_dataloader = retriever.prepare_train(train_samples, shuffle=True)
train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model)

ir_evaluator = retriever.load_dummy_evaluator()

model_path = "msmarco-distilbert-base-v3"
model_save_path = os.path.join("component1_retrieval/results", "{}-GenQ-popqa".format(model_path))
os.makedirs(model_save_path, exist_ok=True)
num_epochs = 1
evaluation_steps = 5000
warmup_steps = int(len(train_samples) * num_epochs / retriever.batch_size * 0.1)

retriever.fit(train_objectives=[(train_dataloader, train_loss)], 
                evaluator=ir_evaluator, 
                epochs=num_epochs,
                output_path=model_save_path,
                warmup_steps=warmup_steps,
                evaluation_steps=evaluation_steps,
                use_amp=True)
