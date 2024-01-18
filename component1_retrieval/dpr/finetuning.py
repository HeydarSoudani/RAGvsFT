#!/usr/bin/env python3

import os
import logging
import argparse
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever
from sentence_transformers import SentenceTransformer, losses, models

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


def main(args):
    prefix = "gen"
    
    data_path = "component1_retrieval/data/popqa_religion"
    corpus, gen_queries, gen_qrels = GenericDataLoader(data_path, prefix=prefix).load(split="train")

    model = SentenceTransformer(args.model)
    retriever = TrainRetriever(model=model, batch_size=64)

    train_samples = retriever.load_train(corpus, gen_queries, gen_qrels)
    train_dataloader = retriever.prepare_train(train_samples, shuffle=True)
    train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model)

    ir_evaluator = retriever.load_dummy_evaluator()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    model_save_path = os.path.join(args.output_dir, args.output_filename)
    os.makedirs(model_save_path, exist_ok=True)
    
    # warmup_steps = int(len(train_samples) * args.epochs / retriever.batch_size * 0.1)

    retriever.fit(
        train_objectives=[(train_dataloader, train_loss)], 
        evaluator=ir_evaluator, 
        epochs=args.epochs,
        output_path=model_save_path,
        warmup_steps=0,
        use_amp=True
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--output_filename", type=str)
    parser.add_argument("--epochs", default=1, type=int)
    # parser.add_argument("--evaluation_steps", default=5000, type=int)
    args = parser.parse_args()
    
    main(args)
