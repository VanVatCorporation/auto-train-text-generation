import json
import os
import chromadb
from chromadb.utils import embedding_functions

class MemoryManager:
    def __init__(self, db_path="./memory_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        # Using a default embedding function
        self.emb_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name="facts", 
            embedding_function=self.emb_fn
        )

    def add_fact(self, fact_text):
        """Adds a new fact to the vector database."""
        # Use a simple hash or count as ID
        fact_id = str(hash(fact_text))
        self.collection.add(
            documents=[fact_text],
            ids=[fact_id]
        )
        print(f"[Memory] Learned: {fact_text}")

    def retrieve_relevant_facts(self, query_text, n_results=3):
        """Retrieves facts relevant to the query."""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        # Return the list of documents
        if results['documents']:
            return results['documents'][0]
        return []

    def auto_detect_fact(self, user_input):
        """
        Simple heuristic to detect if the user is stating a fact.
        In a more advanced version, we'd use an LLM or logic gate.
        For now, we look for 'is', '=', 'equals', etc.
        """
        fact_indicators = [" is ", "=", " are ", " equals ", " know that ", " fact: "]
        question_indicators = ["what", "?", "how", "why", "who", "when", "where", "tell me"]
        
        user_lower = user_input.lower()
        if any(q in user_lower for q in question_indicators):
            return False # Definitely a question, not a fact
            
        if any(ind in user_lower for ind in fact_indicators):
            # Only store if it's not too long and seems like a statement
            if len(user_input.split()) < 20: 
                self.add_fact(user_input)
                return True
        return False
