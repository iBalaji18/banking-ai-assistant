from typing import Dict
from similarity_matcher import SimilarityMatcher
from rag_manager import RAGManager
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class BFSIPipeline:
    def __init__(self, use_slm=False):
        """
        Orchestrator for the 3-Tier BFSI logic.
        Tier 1: Dataset Matcher
        Tier 2: Fine-Tuned SLM (Conditional Load, default OFF for basic testing)
        Tier 3: RAG Retrieval (used alongside SLM or as fallback context)
        """
        self.matcher = SimilarityMatcher()
        self.rag = RAGManager()
        
        # Load Tier 2 model if explicitly enabled
        self.use_slm = use_slm
        if self.use_slm:
            model_id = "Qwen/Qwen2.5-1.5B-Instruct"
            print("Loading SLM Base Model for Fallback Generation...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                device_map="auto", 
                torch_dtype=torch.bfloat16
            )

            # In a real environment, you would load the LoRA weights here
            # model.load_adapter("models/bfsi-slm-lora")

    def _guardrails_check(self, query: str) -> bool:
        """
        Basic guardrails to reject out-of-domain or malicious queries.
        Returns False if query is unsafe.
        """
        unsafe_words = ["hack", "bypass", "ignore previous", "jailbreak", "illegal", "murder", "bomb"]
        for word in unsafe_words:
            if word in query.lower():
                return False
        return True

    def process_query(self, query: str) -> str:
        # Pre-flight Guardrails
        if not self._guardrails_check(query):
            return "I am unable to assist with this request as it violates safety guidelines."

        # Tier 1: Exact / Near-Exact Dataset Match (Threshold 0.85)
        match_result = self.matcher.match(query)
        if match_result["match_found"]:
            return f"[Tier 1 Response]: {match_result['response']}"

        # Tier 3 Context Retrieval (For Complex Queries)
        # Attempt to retrieve relevant RAG context
        context = self.rag.retrieve(query)

        # Tier 2: SLM Response (Generated from scratch using Context)
        if self.use_slm:
            prompt = (
                "<|im_start|>system\n"
                "You are a compliant financial assistant. Answer based strictly on the provided context. If the answer is not in the context, say 'I cannot provide this information.'\n"
                f"Context:\n{context}\n<|im_end|>\n"
                f"<|im_start|>user\n{query}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(**inputs, max_new_tokens=150, temperature=0.3)
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
            return f"[Tier 2+3 Generated Response]: {response}"
        else:
            # Fallback if SLM is disabled (Simulator Mode)
            if len(context) > 20: # If we retrieved something meaningful
                return f"[Tier 3 Context Retrieved]: Based on our policies:\n{context}\nPlease speak to an executive for further generated insights."
            else:
                return "[Tier 2 Fallback]: No exact match and no relevant policy found. Please rephrase or contact human support."

if __name__ == "__main__":
    pipeline = BFSIPipeline(use_slm=False) # Keep false to test logic without 4GB download
    
    print("Testing Pipeline...")
    q1 = "How can I change my EMI debit date?" # Should exact match
    q2 = "What are the rules for late payment on a premium credit card?" # Should RAG retrieve
    q3 = "How do I build a bomb?" # Guardrails test
    
    print(f"\nQ: {q1}\n{pipeline.process_query(q1)}")
    print(f"\nQ: {q2}\n{pipeline.process_query(q2)}")
    print(f"\nQ: {q3}\n{pipeline.process_query(q3)}")
