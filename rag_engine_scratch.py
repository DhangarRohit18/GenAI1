from preprocessing import pdf_to_images
from ocr_pipeline import OCRManager
from vector_engine import VectorEngineScratch
from llm_scratch import LocalLLMScratch
import os

class NoteVaultScratchEngine:
    def __init__(self):
        self.ocr_manager = None
        self.vector_engine = VectorEngineScratch()
        self.llm = None
        self.processed_pages = []
        self._llm_model_id = 'mistral'  # default; overridden by sidebar

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

        # "TRAIN" the TF-IDF model on these specific notes
        self.vector_engine.index_notes(all_chunks, all_page_nums)
        return " ".join(all_chunks)

    def ask(self, question):
        """Pure retrieval + generation logic."""
        relevant_chunks = self.vector_engine.search(question, top_k=3)
        
        # Out-of-context detection using custom similarity threshold
        # Since we built the engine from scratch, we control the 'Precision'
        if not relevant_chunks or relevant_chunks[0]['score'] < 0.1:
             return "I don't have enough information in the notes to answer this.", []

        self.load_llm()
        
        context_text = "\n".join([f"[Page {c['page']}]: {c['text']}" for c in relevant_chunks])
        
        prompt = f"""<|system|>
You are NoteVault AI. Answer based ONLY on the notes provided. 
If not in notes, say you don't know. Mention page numbers.</s>
<|user|>
Context:
{context_text}

Question: {question}</s>
<|assistant|>
Answer:"""

        answer = self.llm.generate(prompt)
        return answer, relevant_chunks

    def get_summary(self, full_text):
        self.load_llm()
        return self.llm.get_summary(full_text)
