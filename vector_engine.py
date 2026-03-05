import numpy as np
from collections import Counter
import math
import re

class CustomTFIDF:
    """
    Implementation of TF-IDF from scratch.
    This model is 'trained' (fitted) on the extracted notes.
    """
    def __init__(self):
        self.vocabulary = {}
        self.idf = {}
        self.documents = []
        self.tfidf_matrix = None

    def _tokenize(self, text):
        return re.findall(r'\w+', text.lower())

    def fit_transform(self, docs):
        self.documents = docs
        doc_count = len(docs)
        
        # Build Vocabulary and calculate DF
        df = Counter()
        tokenized_docs = []
        for doc in docs:
            tokens = set(self._tokenize(doc))
            df.update(tokens)
            tokenized_docs.append(self._tokenize(doc))
        
        self.vocabulary = {word: i for i, word in enumerate(sorted(df.keys()))}
        
        # Calculate IDF
        for word, count in df.items():
            self.idf[word] = math.log(doc_count / (count + 1))
            
        # Create TF-IDF Matrix
        self.tfidf_matrix = np.zeros((doc_count, len(self.vocabulary)))
        for i, tokens in enumerate(tokenized_docs):
            tf = Counter(tokens)
            for word, count in tf.items():
                if word in self.vocabulary:
                    self.tfidf_matrix[i, self.vocabulary[word]] = count * self.idf[word]
        
        # Normalize matrix (L2)
        norms = np.linalg.norm(self.tfidf_matrix, axis=1, keepdims=True)
        self.tfidf_matrix = self.tfidf_matrix / (norms + 1e-9)
        
        return self.tfidf_matrix

    def transform(self, query):
        """Vectorizes a user question."""
        tokens = self._tokenize(query)
        vec = np.zeros(len(self.vocabulary))
        tf = Counter(tokens)
        for word, count in tf.items():
            if word in self.vocabulary:
                vec[self.vocabulary[word]] = count * self.idf[word]
        
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

def cosine_similarity(vec_a, matrix_b):
    """Manual implementation of cosine similarity for search."""
    # (A . B) / (||A|| * ||B||) -> Since rows are normalized, it's just dot product
    return np.dot(matrix_b, vec_a)

class VectorEngineScratch:
    def __init__(self):
        self.tfidf = CustomTFIDF()
        self.metadata = []

    def index_notes(self, chunks, page_nums):
        self.tfidf.fit_transform(chunks)
        for i in range(len(chunks)):
            self.metadata.append({
                "text": chunks[i],
                "page": page_nums[i]
            })

    def search(self, query, top_k=3):
        query_vec = self.tfidf.transform(query)
        scores = cosine_similarity(query_vec, self.tfidf.tfidf_matrix)
        
        best_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in best_indices:
            res = self.metadata[idx].copy()
            res["score"] = float(scores[idx])
            results.append(res)
        return results
