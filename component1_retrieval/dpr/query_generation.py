#!/usr/bin/env python3

import logging, os
import argparse
import torch

from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.generation import QueryGenerator as QGen
from beir.generation.models import QGenModel


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda:0") 
        print("Running on the GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps:0")
        print("Running on the mps")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    data_path = "component1_retrieval/popqa_data"
    corpus = GenericDataLoader(data_path).load_corpus()
    
    generator = QGen(model=QGenModel(args.model).to(device))

    prefix = "gen"
    ques_per_passage = 3
    batch_size = 64

    generator.generate(
        corpus,
        output_dir=data_path,
        ques_per_passage=ques_per_passage,
        prefix=prefix,
        batch_size=batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("--output_dir", type=str, )
    args = parser.parse_args()
    main(args)