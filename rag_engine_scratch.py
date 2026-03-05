from preprocessing import pdf_to_images
from ocr_pipeline import OCRManager
from vector_engine import VectorEngineScratch
from llm_scratch import LocalLLMScratch
import os
import sqlite3
import json

class NoteVaultScratchEngine:
    def __init__(self):
        self.ocr_manager = None
        self.vector_engine = VectorEngineScratch()
        self.llm = None
        self.processed_pages = []
        self._llm_model_id = 'mistral'  # default; overridden by sidebar
        self._init_db()

    def _init_db(self):
        """Initializes a local SQLite database for persistence."""
        conn = sqlite3.connect('data/notevault.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS notes 
                     (id INTEGER PRIMARY KEY, filename TEXT, page_num INTEGER, content TEXT)''')
        conn.commit()
        conn.close()

    def _save_to_db(self, filename, page_num, content):
        conn = sqlite3.connect('data/notevault.db')
        c = conn.cursor()
        c.execute("INSERT INTO notes (filename, page_num, content) VALUES (?, ?, ?)", 
                  (filename, page_num, content))
        conn.commit()
        conn.close()

    def load_ocr(self):
        if self.ocr_manager is None:
            self.ocr_manager = OCRManager()

    def load_llm(self, model_id=None):
        if model_id:
            self._llm_model_id = model_id
        if self.llm is None:
            self.llm = LocalLLMScratch(self._llm_model_id)
        elif model_id:
            self.llm.set_model(model_id)

    def process_pdf(self, pdf_path):
        """Full pipeline: PDF -> OCR -> TF-IDF fitting (training) -> Indexing."""
        self.load_ocr()
        image_paths = pdf_to_images(pdf_path)
        all_chunks = []
        all_page_nums = []

        for i, img_path in enumerate(image_paths):
            text, _ = self.ocr_manager.extract_text(img_path)
            
            # Semantic chunking (Simple period-based)
            chunks = [c.strip() for c in text.split(".") if len(c.strip()) > 10]
            
            all_chunks.extend(chunks)
            all_page_nums.extend([i + 1] * len(chunks))
            self.processed_pages.append({"page": i + 1, "text": text})
            self._save_to_db(os.path.basename(pdf_path), i + 1, text)

        # "TRAIN" the TF-IDF model on these specific notes
        self.vector_engine.index_notes(all_chunks, all_page_nums)
        return " ".join(all_chunks)

    def ask(self, question):
        """Retrieval + generation with confidence scoring."""
        relevant_chunks = self.vector_engine.search(question, top_k=3)
        
        # 1. Out-of-context detection
        # If max similarity is very low, we are confident we don't know.
        if not relevant_chunks or relevant_chunks[0]['score'] < 0.12:
             return "I don't have enough information in the notes to answer this.", [], 0, ""

        self.load_llm()
        context_text = "\n".join([f"[Page {c['page']}]: {c['text']}" for c in relevant_chunks])
        
        prompt = f"""<|system|>
You are NoteVault AI. Answer based ONLY on the notes provided. 
If not in notes, say "I don't have enough information in the notes to answer this."
Mention page numbers where the info was found.</s>
<|user|>
Context:
{context_text}

Question: {question}</s>
<|assistant|>
Answer:"""

        answer = self.llm.generate(prompt)
        
        # 2. Strict check on LLM response
        if "don't know" in answer.lower() or "enough information" in answer.lower():
             return "I don't have enough information in the notes to answer this.", [], 0, ""

        # 3. Calculate Confidence Score
        confidence = self._calculate_confidence(question, answer, relevant_chunks)
        
        # 4. Referenced Text (the best matching chunk)
        referenced_text = relevant_chunks[0]['text']

        return answer, relevant_chunks, confidence, referenced_text

    def _calculate_confidence(self, question, answer, chunks):
        """
        Calculates a confidence percentage based on:
        - Vector similarity (60%)
        - Ranking gap (20%) - how much better is #1 than #2
        - Semantic reinforcement (20%) - do multiple chunks agree
        """
        if not chunks: return 0
        
        sim = chunks[0]['score']
        # Normalize similarity (cosine typical 0.1-0.8 for notes)
        sim_score = min(sim * 100 / 0.7, 100)
        
        # Gap between top 2
        gap = (chunks[0]['score'] - chunks[1]['score']) if len(chunks) > 1 else 0.1
        gap_score = min(gap * 100 / 0.3, 100)
        
        # Total
        confidence = (0.7 * sim_score) + (0.3 * gap_score)
        return int(max(min(confidence, 99), 10))

    def generate_quiz(self):
        """Generates 5 MCQs based on the notes with a strict parseable format."""
        if not self.processed_pages: return ""
        
        full_text = " ".join([p['text'] for p in self.processed_pages])[:4500]
        self.load_llm()
        
        prompt = f"""<|system|>
You are a professor. Create a 5-question multiple choice quiz based ONLY on the provided notes.
Each question MUST have exactly 4 options (A, B, C, D) and one correct letter.
STRICT FORMAT:
Q: [Question]
A) [Option 1]
B) [Option 2]
C) [Option 3]
D) [Option 4]
Correct: [A/B/C/D]
---</s>
<|user|>
Notes: {full_text}</s>
<|assistant|>
"""
        return self.llm.generate(prompt, max_new_tokens=1200)

    def extract_flashcards(self):
        """Extracts key terms and definitions for flashcards."""
        if not self.processed_pages: return []
        
        full_text = " ".join([p['text'] for p in self.processed_pages])[:3000]
        self.load_llm()
        
        prompt = f"""<|system|>
Extract 8 key terms and their definitions from the notes for flashcards.
Format: Term | Definition
One per line.</s>
<|user|>
Notes: {full_text}</s>
<|assistant|>
Flashcards:"""
        raw = self.llm.generate(prompt, max_new_tokens=800)
        
        cards = []
        for line in raw.split("\n"):
            if "|" in line:
                term, b = line.split("|", 1)
                cards.append({"front": term.strip(), "back": b.strip()})
        return cards

    def get_summary(self, full_text):
        self.load_llm()
        return self.llm.get_summary(full_text)
