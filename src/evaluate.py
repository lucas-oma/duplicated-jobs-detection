import numpy as np
import pandas as pd
from tqdm import tqdm
from src.vector_search import VectorSearch
import matplotlib.pyplot as plt
from collections import defaultdict


def threshold_analysis(embedding_path, id_path, k=5):
    print("Threshold analysis in progress...")
    embeddings = np.load(embedding_path)
    lids = pd.read_csv(id_path)["lid"].tolist()

    vs = VectorSearch(dim=embeddings.shape[1])

    scores = []

    for i, (lid, emb) in tqdm(enumerate(zip(lids, embeddings)), total=len(lids)):
        results = vs.search(emb.tolist(), top_k=k)
        scores.extend([hit_score for hit_id, hit_score in results if hit_id != lid])

    plt.hist(scores, bins=100)
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.title("Similarity Score Distribution")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("data/eval/threshold_distribution.png", dpi=300)
    print("Saved plot to data/eval/threshold_distribution.png")


def precision_at_k(vs, embeddings, lids, threshold=0.92, k=5, sample_size=100):
    print("Running Precision@K via manual inspection sample")
    pairs = []

    for i, (query_id, query_emb) in tqdm(enumerate(zip(lids, embeddings)), total=len(lids)):
        results = vs.search(query_emb.tolist(), top_k=k)
        for hit_id, hit_score in results:
            if query_id != hit_id and hit_score > threshold:
                pairs.append((query_id, hit_id, hit_score))

    sample = pd.DataFrame(pairs, columns=["job_id_1", "job_id_2", "similarity"]).head(sample_size)
    sample.to_csv("data/eval/manual_sample.csv", index=False, float_format="%.5f")
    print(f"Saved manual inspection sample to data/eval/manual_sample.csv (sample of {sample_size})")
    return sample

def threshold_curve(vs, embeddings, ids, thresholds=np.arange(0.80, 0.99, 0.01), k=5):
    print("Running threshold vs. count analysis")
    counts = defaultdict(int)

    for i, (id, emb) in tqdm(enumerate(zip(ids, embeddings)), total=len(ids)):
        results = vs.search(emb.tolist(), top_k=k)
        for hit_id, score in results:
            for t in thresholds:
                if id != hit_id and score > t:
                    counts[round(t, 2)] += 1

    xs = sorted(counts.keys())
    ys = [counts[t] for t in xs]

    plt.plot(xs, ys)
    plt.xlabel("Similarity Threshold")
    plt.ylabel("Num. Duplicate Pairs Detected")
    plt.title("Threshold vs. Duplicate Pair Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("data/eval/threshold_curve.png", dpi=300)
    print("Saved threshold curve to data/eval/threshold_curve.png")

    return xs, ys


def heuristic_check(df, embeddings, vs, threshold=0.92, k=5):
    print("Running heuristic match check")
    matches = []
    lid_index = {lid: i for i, lid in enumerate(df['lid'])}

    for i, row in df.iterrows():
        title = row['jobTitle']
        city = row['finalCity']
        company = row['companyName']

        same = df[(df['jobTitle'] == title) & (df['finalCity'] == city) & (df['companyName'] == company)]
        for _, r in same.iterrows():
            if row['lid'] != r['lid']:
                i1, i2 = lid_index[row['lid']], lid_index[r['lid']]
                score = float(np.dot(embeddings[i1], embeddings[i2]) / \
                              (np.linalg.norm(embeddings[i1]) * np.linalg.norm(embeddings[i2])))
                if score > threshold:
                    matches.append((row['lid'], r['lid'], score))

    df_out = pd.DataFrame(matches, columns=["job_id_1", "job_id_2", "similarity"])
    df_out.to_csv("data/eval/heuristic_matches.csv", index=False, float_format="%.5f")
    print(f"Saved heuristic match results to data/eval/heuristic_matches.csv (total: {len(matches)})")
    return df_out

def metrics_evaluation(id_path, raw_input_path, embeddings_npy, top_k, threshold):
    df = pd.read_csv(id_path)
    ids = df["lid"].tolist()
    full_df = pd.read_csv(raw_input_path)
    embeddings = np.load(embeddings_npy)

    vs = VectorSearch(dim=embeddings.shape[1])

    precision_at_k(vs, embeddings, ids, k=top_k, threshold=threshold)
    threshold_curve(vs, embeddings, ids, k=top_k)
    heuristic_check(full_df, embeddings, vs, k=top_k, threshold=threshold)
