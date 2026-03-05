import ollama
import json

class LLMManager:
    def __init__(self, model="mistral"):
        self.model = model

    def generate_answer(self, query, context_chunks):
        """Generates a grounded answer using the provided context."""
        
        context_text = "\n\n".join([f"Source (Page {c['page']}): {c['text']}" for c in context_chunks])
        
        prompt = f"""
You are NOTEVAULT AI, a personal study assistant. 
Your goal is to answer questions based ONLY on the provided notes.

### CONSTRAINTS:
1. If the answer is NOT in the notes, say: "I don't have enough information in the notes to answer this."
2. Do NOT use outside knowledge.
3. Use the context provided below to formulate your answer.
4. Mention the page number in your explanation.

### CONTEXT FROM NOTES:
{context_text}

### QUESTION:
{query}

### ANSWER:
"""

        try:
            response = ollama.generate(model=self.model, prompt=prompt)
            return response['response']
        except Exception as e:
            return f"Error connecting to Ollama: {str(e)}. Make sure Ollama is running."

    def get_summary(self, full_text):
        """Generates a summary of the entire document."""
        prompt = f"Summarize the following notes in a concise study guide format:\n\n{full_text[:4000]}" # Truncate for safety
        try:
            response = ollama.generate(model=self.model, prompt=prompt)
            return response['response']
        except Exception as e:
            return f"Error: {str(e)}"
