import pandas as pd
import numpy as np
from tqdm import tqdm
from src.vector_search import VectorSearch

def find_all_duplicates(embeddings_npy, id_path, threshold, top_k=5):    
    print(f"Finding duplicates with threshold={threshold}")

    df = pd.read_csv(id_path)
    ids = df["lid"].tolist()
    embeddings = np.load(embeddings_npy)

    vs = VectorSearch(dim=embeddings.shape[1])

    evaluated = set()
    pairs = []

    for i, (lid, emb) in tqdm(enumerate(zip(ids, embeddings)), total=len(ids)):
        results = vs.search(emb.tolist(), top_k=top_k)
        for hit_id, score in results:
            if lid != hit_id and score > threshold:
                pair = tuple(sorted([lid, hit_id]))
                if pair not in evaluated:
                    evaluated.add(pair)
                    pairs.append((*pair, score))

    df_out = pd.DataFrame(pairs, columns=["JOB ID 1", "JOB ID 2", "Similarity Score"])
    df_out.to_csv("data/eval/all_duplicates.csv", index=False, float_format="%.5f")
    print(f"Saved final output to data/eval/all_duplicates.csv (total pairs: {len(df_out)})")
    

    unique_ids = set(df_out["JOB ID 1"]).union(set(df_out["JOB ID 2"]))
    print(f"Total unique job IDs involved in (posible) duplicates: {len(unique_ids)}")

