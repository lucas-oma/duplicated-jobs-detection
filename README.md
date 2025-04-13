# Job Duplication Detection via Vector Similarity Search

## Overview
This project performs job duplication detection by comparing vector embeddings of job postings using approximate nearest neighbor search. It uses Sentence Transformers (MiniLM-L6-v2) to generate embeddings and Milvus as the vector database. The system is containerized using Docker and can be easily run and evaluated using `docker compose`.

### Core technologies:
- **Python 3.10.16** (Jupyter notebooks)
- **Milvus v2.5.9** (standalone mode)
- **Docker** with `docker-compose`
- **SentenceTransformers (MiniLM-L6-v2)** for embeddings
- **Pandas, NumPy, Matplotlib, tqdm** for analysis and visualization

### Why SentenceTransformers (MiniLM-L6-v2) and Milvus?

#### SentenceTransformers (MiniLM-L6-v2)
I selected `all-MiniLM-L6-v2` from the [SentenceTransformers](https://www.sbert.net/) library due to its popular use for text embeddings (good balance between **semantic quality and speed**).

#### Milvus (v2.5.9)
Milvus was chosen as the vector database for its:

- **Scalability and performance** on large-scale similarity search.
- **Support for real-time vector insertion**. New embeddings can be added without the need to re-index or reload the entire database.
- **Persistence by default**, ensuring data survives container restarts.

Also used the **HNSW index** with the **COSINE** similarity metric.

- HNSW (Hierarchical Navigable Small World) offers **high recall with low latency** which is perfect for fast and accurate nearest neighbor retrieval in production settings.
- COSINE similarity, combined with our normalized sentence embeddings, makes distance scores directly interpretable as semantic similarity. It is also widely adopted for text-based similarity tasks, making it a natural choice for this application.

This setup supports efficient, on-demand execution without sacrificing performance.

---

## Environment Setup
Ensure you have the following installed:
- Python 3.10+
- Docker
- Docker Compose

To install Python dependencies locally (for notebook use):
```bash
pip install -r eda_notebooks/requirements.txt
```

To use the full system in Docker:
```bash
sudo docker compose build
```

---

## Data Analysis
The project includes exploratory data analysis (EDA) in the form of a Jupyter notebook, located at:
```bash
eda_notebooks/eda.ipynb
```

The purpose of EDA was to:
- Understand field distributions (e.g., degree level, employment type)
- Detect formatting inconsistencies (e.g., trailing commas in city/state)
- Understand text format of job descriptions (e.g., html format)

> **Note**: Data is not modified during EDA to preserve "plug-and-play" compatibility. Any new CSV following the same structure can be used directly. This behavior is intended.

---

## Data Preparation & Storage

To generate and save the job embeddings, run:

```bash
python src/embeddings.py
```
or inside the container:

```bash
docker compose run --rm app python src/embeddings.py
```

> It is recommended to enable GPU support for docker since this process might take a while

This script supports the following optional arguments:
```bash

    --input         Path to the input CSV file (default: data/raw/jobs.csv)

    --output_emb    Path to save the generated embeddings in .npy format (default: data/processed/embeddings.npy)

    --output_ids    Path to save the job ID mapping in .csv format (default: data/processed/job_ids.csv)
```


The job embeddings are generated through the following steps:

1. **Field Selection**: We select key fields to construct the text input for embedding. These are:  
    > job title, company name, company branch, employment type, seniority, location, degree level, and job description

2. **Text Cleaning**: All selected fields are cleaned by removing redundant whitespaces, trimming trailing spaces, and converting text to lowercase. The `jobDescRaw` field (HTML format), is first parsed using BeautifulSoup to extract clean text content, and then the regular cleaning is applied.

3. **Text Assembly**: The cleaned fields are concatenated into a single string to be used as input for the embedding model.

4. **Embedding Generation**: 384-dimensional vector embeddings are generated using the `all-MiniLM-L6-v2` model. Since we use cosine similarity, these embeddings are also normalized as usually recommended.

6. **(Temp) Storage**: The output files are saved for downstream use:
   - `data/processed/embeddings.npy`: stores the normalized embedding vectors.
   - `data/processed/job_ids.csv`: stores the corresponding job IDs (`lid` values).

Once the embeddings are generated, they are later loaded into Milvus for indexing and similarity search.

---

## Running the Application Inside Docker

Once the Docker environment is built, you can run different parts of the pipeline as follows:

### Inserting and Indexing Embeddings
```bash
sudo docker compose run --rm app python main.py --insert
```

This process:

- Loads and cleans the input CSV

- Generates sentence embeddings using MiniLM-L6-v2

- Stores the vectors and their associated IDs (for later use) in:

    - `data/processed/embeddings.npy`
    - `data/processed/job_ids.csv`

- Inserts them into the Milvus vector database for similarity search


> Flags are available to specify input/output files. Check the Flags section for more information.

### (optional) Health Check: Dummy Test
```bash
sudo docker compose run --rm app python main.py --test
```

This process is meant as a test to ensure Milvus is working as expected:

- Creates dummy embeddings and dummy job IDs

- Verifies that all VectorSearch methods work as expected (add, search, delete, etc.)

- Useful for validating system behavior before inserting real data

> ⚠️ **Warning:** This will drop your current Milvus collection and recreate it.


### Finding all duplicates
```bash
sudo docker compose run --rm app python main.py --find_all --threshold 0.93
```

Searches the vector database for job pairs with high similarity based on a threshold value.

The output will be saved to: `data/eval/all_duplicates.csv`. Each row contains:
- `JOB ID 1`: The first job in the similar pair
- `JOB ID 2`: The second/matched job
- `Similarity Score`: The similarity score between the two (>threshold)


---

## Evaluation

### Threshold Analysis (for setting similarity cutoff)
```bash
sudo docker compose run --rm app python main.py --t_analysis
```

This process:

- Runs similarity searches across the entire dataset

- Collects similarity scores for nearest neighbors

- Plots a histogram of those scores to help visualize how "tight" or "loose" the similarity range is

Outputs a histogram to help visualize similarity score distribution: `data/eval/threshold_distribution.png`

### Evaluation Metrics (heuristic checks and inspection samples)
```bash
sudo docker compose run --rm app python main.py --evaluate
```

Runs multiple evaluation routines:

    - Threshold vs. Count Curve: how many pairs would be matched at different thresholds

    - Manual Inspection Sample: samples N high-scoring pairs for inspection

    - Heuristic Matching: e.g., identical titles + cities + degree levels for score reference

Helps validate threshold

Outputs:
    - `data/eval/threshold_curve.png`
    - `data/eval/manual_sample.csv`
    - `data/eval/heuristic_matches.csv`

---

## Final Output
The final result consists of job pairs likely to be duplicates:
```
Job ID 1, Job ID 2, Similarity Score
```
This is stored in:
```
data/eval/all_duplicates.csv
```

---

## Flags in `main.py`

```bash
--input <string>         Input CSV file containing job postings
                         (default: data/raw/jobs.csv)

--embeddings <string>    Path to the output or existing .npy file containing embeddings
                         (default: data/processed/embeddings.npy)

--ids <string>           Path to the output or existing .csv file containing job IDs
                         (default: data/processed/job_ids.csv)

--top_k <int>            Number of nearest neighbors to retrieve during search and analysis
                         (default: 5)

--threshold <float>      Similarity threshold used in evaluation and pair detection
                         (default: 0.92)

--insert                 Runs the insertion of embeddings using the VectorSearch system

--test                   Runs a dummy test of the VectorSearch system using synthetic data
                         (⚠️ This resets the database and is intended only as a health check)

--t_analysis             Runs a threshold analysis to help select a good similarity cutoff
                         (outputs a histogram of similarity score distribution: data/eval/threshold_distribution.png)

--evaluate               Runs evaluation methods including manual inspection sampling,
                         threshold sweep, and heuristic matching checks. Outputs can be found at data/eval/manual_sample.csv,
                         data/eval/threshold_curve.png, and data/eval/heuristic_matches.csv respectively

--reset_db               Drops and reinitializes the Milvus collection before proceeding
                         (useful for rebuilding from scratch)
```
---

---

## EXTRA: FastAPI
This project includes a FastAPI server to perform real-time duplicate job detection via API calls.

### Running the API server
The Docker container will run the FastAPI server by default:

```bash
sudo docker compose up
```

The API will be available at: `http://localhost:8000`


### Endpoint: `/search`

**Method:** `POST`  

**URL:** `http://localhost:8000/search`  

**Request Body (example):**
```json
{
  "jobTitle": "Software Engineer",
  "companyName": "OpenAI",
  "companyBranchName": "San Francisco",
  "nlpEmployment": "Full-time",
  "nlpSeniority": "Entry level",
  "finalCity": "San Francisco",
  "finalState": "CA",
  "nlpDegreeLevel": ["Bachelors"],
  "jobDescRaw": "Work on cutting-edge AI research and deployment.",
  "top_k": 5,
  "threshold": 0.95
}
```

**Response (example):**
```json
Response:

{
  "results": [
    {"job_id": "abc123", "score": 0.96},
    {"job_id": "def456", "score": 0.98}
  ]
}
```
> ⚠️ Results only include matches with similarity score greater than the specified threshold (default 0.95).


> Test it with CURL, you can find an example in `curl_example.txt`

---

## Follow-ups

For duplicate detection, I would also recommend to incorporate an LLM model for score matching to the results that are above the threshold. BERT should be good (if GPU allows it) or simply a cheap OpenAI model such as gpt-o4-mini should do the job. ALthough this might increase cost greatly

---

## Notes
- `.env` file is optional and not required for this project
- Vector dimensions can be customized, but 384 is recommended for `all-MiniLM-L6-v2`
- Milvus uses cosine distance; embeddings are normalized prior to insertion (this is done internaly by the `VectorSearch` class)

---

## Author
Lucas Martinez, 2025

