from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from src.vector_search import VectorSearch
from src.embeddings import generate_embeddings
from typing import List


app = FastAPI()

EMBEDDINGS_PATH = "data/processed/embeddings.npy"
IDS_PATH = "data/processed/job_ids.csv"
DIM = 384

embeddings = np.load(EMBEDDINGS_PATH)
lids = pd.read_csv(IDS_PATH)["lid"].tolist()
vs = VectorSearch(dim=DIM)


class JobQuery(BaseModel):
    jobTitle: str = ""
    companyName: str = ""
    companyBranchName: str = ""
    nlpEmployment: str = ""
    nlpSeniority: str = ""
    finalCity: str = ""
    finalState: str = ""
    nlpDegreeLevel: List[str] = []
    jobDescRaw: str = ""
    top_k: int = 5
    threshold: float = 0.95

@app.post("/search")
def search_duplicate(query: JobQuery):
    try:
        # Convert input to one-row DataFrame
        df = pd.DataFrame([query.dict()])
        top_k = df.pop("top_k").iloc[0]
        threshold = df.pop("threshold").iloc[0]

        query_emb = generate_embeddings(df)[0]

        results = vs.search(query_emb.tolist(), top_k=top_k)
        duplicated = [{"job_id": r[0], "score": round(r[1], 5)} for r in results if r[1] > threshold]

        return {"results": duplicated}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

