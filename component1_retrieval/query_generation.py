from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.generation import QueryGenerator as QGen
from beir.generation.models import QGenModel

import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

data_path = "component1_retrieval/popqa_data"
corpus = GenericDataLoader(data_path).load_corpus()

model_path = "BeIR/query-gen-msmarco-t5-base-v1"
generator = QGen(model=QGenModel(model_path))

prefix = "gen-3"
ques_per_passage = 3
batch_size = 64

generator.generate(
    corpus,
    output_dir=data_path,
    ques_per_passage=ques_per_passage,
    prefix=prefix,
    batch_size=batch_size)
