import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util

class SimilarityMatcher:
    def __init__(self, dataset_path="data/bfsi_dataset.json", model_name="all-MiniLM-L6-v2", threshold=0.85):
        """
        Initializes the Tier 1 Similarity Matcher.
        Loads the dataset and computes embeddings for all 'input' fields.
        """
        self.threshold = threshold
        
        # Load Dataset
        with open(dataset_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)
            
        self.corpus = [item["input"] for item in self.dataset]
        self.responses = [item["output"] for item in self.dataset]
        
        print(f"Loading SentenceTransformer model '{model_name}'...")
        # Use CPU by default for the matcher since it's very lightweight and quick
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        
        print(f"Encoding {len(self.corpus)} queries from the dataset...")
        self.corpus_embeddings = self.model.encode(self.corpus, convert_to_tensor=True)
        print("Dataset embeddings ready.")

    def match(self, query):
        """
        Matches a user query against the dataset corpus.
        Returns the best matched output if similarity > threshold.
        Otherwise returns None.
        """
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        # Compute cosine similarities
        cosine_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]
        
        best_score_idx = torch.argmax(cosine_scores).item()
        best_score = cosine_scores[best_score_idx].item()
        
        if best_score >= self.threshold:
            return {
                "match_found": True,
                "score": best_score,
                "matched_query": self.corpus[best_score_idx],
                "response": self.responses[best_score_idx]
            }
        
        return {
            "match_found": False,
            "score": best_score,
            "matched_query": self.corpus[best_score_idx] if best_score > 0 else None,
            "response": None
        }

if __name__ == "__main__":
    # Smoke test
    matcher = SimilarityMatcher()
    
    test_queries = [
        "What's the status for my car loan?",
        "How do I change my EMI debit date?", # Exact match
        "Can I get a loan to buy a private jet?", # Out of domain / low score
    ]
    
    for q in test_queries:
        print(f"\nUser Query: {q}")
        result = matcher.match(q)
        print(f"Score: {result['score']:.4f}")
        if result['match_found']:
            print(f"✅ Exact Match Triggered! \nResponse: {result['response']}")
        else:
            print(f"❌ No Tier 1 Match. Pass to Tier 2 (SLM).")
