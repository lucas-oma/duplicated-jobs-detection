from src.vector_search import VectorSearch
from src.evaluate import threshold_analysis, metrics_evaluation
from src.find_duplicates import find_all_duplicates
import numpy as np
import pandas as pd

def test():
    vs = VectorSearch(dim=384)
    vs.delete_all()

    # Sample data
    embeddings = [np.random.rand(384).tolist() for _ in range(5)]
    lids = [f"vec_{i}" for i in range(5)]

    vs.add(lids, embeddings)

    # Pick one of them and search
    query = embeddings[0]
    results = vs.search(query, top_k=3)


    # Add
    print("Adding vectors...")
    vs.add(lids, embeddings)

    # Search
    print("Searching for similar vectors...")
    results = vs.search(embeddings[0], top_k=3)
    for i, (found_lid, score) in enumerate(results):
        print(f"Result {i + 1}: lid={found_lid}, score={score:.4f}")

    # Remove
    print("Removing vectors...")
    vs.remove(lids)

    # Deleting all
    vs.delete_all()


def start(input_csv, output_npy, output_indexes):
    print(f"Reading {input_csv}")

    embeddings = np.load(output_npy)
    print(f"Loaded embeddings, shape: {embeddings.shape}")

    df_ids = pd.read_csv(output_indexes)
    lid_list = df_ids["lid"].tolist()
    print(f"Loaded {len(lid_list)} job IDs")

    assert len(lid_list) == len(embeddings), "Mismatch between number of IDs and embeddings!"

    vs = VectorSearch(dim=embeddings.shape[1])
    
    vs.add(lid_list, embeddings)

    print(f"Successfully inserted {len(lid_list)} embeddings into Milvus!")
    


    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--test", action="store_true")
    parser.add_argument("--insert", action="store_true")
    parser.add_argument("--t_analysis", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--find_all", action="store_true")
    parser.add_argument("--reset_db", action="store_true")

    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.92)

    parser.add_argument("--input", default="data/raw/jobs.csv")
    parser.add_argument("--embeddings", default="data/processed/embeddings.npy")
    parser.add_argument("--ids", default="data/processed/job_ids.csv")

    args = parser.parse_args()

    if args.reset_db:
        vs = VectorSearch()
        vs.delete_all()

    if args.test:
        test()
    
    elif args.insert:
        start(args.input, args.embeddings, args.ids)

    elif args.t_analysis:
        threshold_analysis(args.embeddings, args.ids)

    elif args.evaluate:
        metrics_evaluation(args.ids, args.input, args.embeddings, args.top_k, args.threshold)

    elif args.find_all:
        find_all_duplicates(args.embeddings, args.ids, args.threshold)
    

    
