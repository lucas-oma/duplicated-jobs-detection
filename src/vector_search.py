import numpy as np
from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection, utility
)
import time

class VectorSearch:
    def __init__(self, collection_name="job_embeddings", dim=384):
        self.collection_name = collection_name
        self.dim = dim
        self._connect_milvus()
        self._init_collection()

    def _connect_milvus(self, host="standalone", port="19530", max_tries=5):
        for i in range(max_tries):
            try:
                connections.connect("default", host=host, port=port)
                print("Connected to Milvus!")
                return
            except Exception as e:
                print(f"Waiting for Milvus... try {i + 1}/{max_tries}")
                time.sleep(2)
        raise RuntimeError("Milvus is not responding after multiple attempts.")


    def _init_collection(self):
        if not utility.has_collection(self.collection_name):
            fields = [
                FieldSchema(name="lid", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
            ]
            schema = CollectionSchema(fields, description="Job embeddings collection")
            collection = Collection(name=self.collection_name, schema=schema)
            index_params = {
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 64, "efConstruction": 200}
            }
            collection.create_index("embedding", index_params)
            print(f"NEW collection created: {self.collection_name}")

        self.collection = Collection(name=self.collection_name)
        self.collection.load()

    def add(self, lid_strings, embeddings, batch_size=1000):
        assert len(lid_strings) == len(embeddings)
        
        for i in range(0, len(lid_strings), batch_size):
            batch_lids = lid_strings[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            data = [batch_lids, batch_embeddings]
            
            print(f"Inserting batch {i // batch_size + 1} with {len(batch_lids)} items...")
            self.collection.insert(data)
        
        self.collection.flush()
        print(f"Inserted all {len(lid_strings)} items into Milvus.")

    def remove(self, lid_list, batch_size=1000):
        if not lid_list:
            return

        total = len(lid_list)
        for i in range(0, total, batch_size):
            batch = lid_list[i:i + batch_size]
            expr = f"lid in [{', '.join(repr(lid) for lid in batch)}]"
            print(f"Deleting batch {i // batch_size + 1} ({len(batch)} items)...")
            self.collection.delete(expr=expr)

        self.collection.flush()
        print(f"Deleted {total} items from Milvus.")


    def search(self, query_embedding: list, top_k=5):
        self.collection.load()
        search_params = {"metric_type": "COSINE", "params": {"ef": max(top_k, 64)}}

        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k
        )

        return [(hit.id, hit.distance) for hit in results[0]]

    def delete_all(self):
        self.collection.drop()
        self._init_collection()

    def save_index(self):
        self.collection.flush()
        print("Collection flushed to disk.")

    def load_index(self):
        self.collection.load()
        print("Collection loaded into memory.")
