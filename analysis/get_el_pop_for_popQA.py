import numpy as np


if __name__ == "__main__":
    queries_file = "data/generated/popQA_costomized/queries.jsonl"
    npz_path = 'analysis/some_data/pretraining_entities/wikipedia_entity_map.npz'
    # el_pop_path = "data/generated/popQA_costomized/el_popularity.json"
    
    with np.load(npz_path) as data:
        pass
    
    