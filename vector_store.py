import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import os

class VectorStore:
    def __init__(self, model_name='all-MiniLM-L6-v2', db_path="data/faiss_index.pkl"):
        self.model = SentenceTransformer(model_name)
        self.db_path = db_path
        self.index = None
        self.metadata = [] # List of {text, page, confidence}

    def add_texts(self, texts, page_numbers, confidences):
        """Embeds and adds texts to the FAISS index."""
        embeddings = self.model.encode(texts)
        dimension = embeddings.shape[1]
        
        if self.index is None:
            self.index = faiss.IndexFlatL2(dimension)
        
        self.index.add(np.array(embeddings).astype('float32'))
        
        for i in range(len(texts)):
            self.metadata.append({
                "text": texts[i],
                "page": page_numbers[i],
                "confidence": confidences[i]
            })
            
    def search(self, query, top_k=3):
        """Searches for the most relevant chunks."""
        if self.index is None:
            return []
            
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), top_k)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx < len(self.metadata):
                res = self.metadata[idx]
                res["distance"] = distances[0][i]
                results.append(res)
        
        return results

    def save(self):
        with open(self.db_path, "wb") as f:
            pickle.dump({"index": self.index, "metadata": self.metadata}, f)

    def load(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "rb") as f:
                data = pickle.load(f)
                self.index = data["index"]
                self.metadata = data["metadata"]
            return True
        return False
