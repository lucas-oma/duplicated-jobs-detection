import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import re
import ast
from tqdm import tqdm

# Show progress bar with pandas: df.progress_apply() instead of using .apply()
tqdm.pandas()

model = SentenceTransformer("all-MiniLM-L6-v2")


def clean_text(text):
    if pd.isnull(text):
        return ""

    # Cleaning whitespaces and converting to lowercase
    text = re.sub(r"\s+", " ", text).strip()
    text = text.lower()
    return text


def clean_job_description(raw_html):
    if raw_html == "":
        return ""

    soup = BeautifulSoup(raw_html, "html.parser")
    text = soup.get_text()
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()
    

def preprocess_text(job):

    # Cleaning string columns
    title = clean_text(job.get("jobTitle", ""))
    title = f"POSITION: {title}" if title else ""

    company = clean_text(job.get("companyName", ""))
    company = f"COMPANY: {company}" if company else ""

    branch = clean_text(job.get("companyBranchName", ""))
    branch = f"COMPANY BRANCH: {branch}" if branch else ""

    employment = clean_text(job.get("nlpEmployment", ""))
    employment = f"EMPLOYMENT TYPE: {employment}" if employment else ""

    seniority = clean_text(job.get("nlpSeniority", ""))
    seniority = f"SENIORITY: {seniority}" if seniority else ""


    # From the EDA I've found some States were ending with a comma, so lets take care of similar cases
    city = clean_text(job.get("finalCity", ""))
    city = re.sub(r'[^a-zA-Z0-9 ]', '', city)

    state = clean_text(job.get("finalState", ""))
    state = re.sub(r'[^a-zA-Z0-9 ]', '', state)

    location = ",".join([city, state])
    location = f"LOCATION: {location}" if len(location)>1 else ""

    # Cleaning array-like columns
    raw_degrees = job.get("nlpDegreeLevel", "[]")
    try:
        degree_arr = ast.literal_eval(raw_degrees)
    except:
        degree_arr = []
    degree = f"DEGREE LEVEL: {' or '.join([clean_text(d) for d in degree_arr])}" if degree_arr else ""


    # Cleaning description
    description_raw = job.get("jobDescRaw", "")
    description = clean_job_description(description_raw)
    description = f"JOB DESCRIPTION: {description}" if description else ""

    # Final text for embedding
    return "\n".join([x for x in [title, company, branch, employment, seniority, location, degree, description] if x])
    

def generate_embeddings(df):
    texts = df.progress_apply(preprocess_text, axis=1).tolist()
    embeddings = model.encode(texts, show_progress_bar=True)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms

    return normalized_embeddings


def main(input_csv, output_npy, output_indexes):
    print(f"Reading csv file: {input_csv}")
    df = pd.read_csv(input_csv, sep=",")

    print("Creating embeddings")
    embeddings = generate_embeddings(df)
    
    print("Embeddings created")
    
    print(f"Saving embeddings: {output_npy}")
    np.save(output_npy, embeddings)

    print(f"Saving indexes: {output_indexes}")
    df[['lid']].to_csv(output_indexes, index=False)

    print("Done!")





if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/jobs.csv")
    parser.add_argument("--output_emb", default="data/processed/embeddings.npy")
    parser.add_argument("--output_ids", default="data/processed/job_ids.csv")

    args = parser.parse_args()

    main(args.input, args.output_emb, args.output_ids)
