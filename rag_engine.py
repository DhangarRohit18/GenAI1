from preprocessing import pdf_to_images, preprocess_image
from ocr_pipeline import OCRManager
from vector_store import VectorStore
from llm_interface import LLMManager
import os

class NoteVaultEngine:
    def __init__(self, llm_model="mistral"):
        self.ocr_manager = OCRManager()
        self.vector_store = VectorStore()
        self.llm_manager = LLMManager(model=llm_model)
        self.processed_pages = []

    def process_pdf(self, pdf_path):
        """Full pipeline: PDF -> Images -> OCR -> FAISS."""
        image_paths = pdf_to_images(pdf_path)
        all_text = []

        for i, img_path in enumerate(image_paths):
            # Preprocess (Optional enhancement)
            # prep_path = preprocess_image(img_path) 
            
            # OCR
            text, details = self.ocr_manager.extract_text(img_path)
            
            # We can chunk within a page if needed, but for notes, page-level or paragraph-level is good.
            # Let's chunk by sentences or simple blocks for better retrieval.
            chunks = text.split(". ") # Crude semantic chunking
            page_num = i + 1
            
            self.vector_store.add_texts(
                chunks, 
                [page_num] * len(chunks), 
                [1.0] * len(chunks) # Default confidence
            )
            all_text.append(text)
            self.processed_pages.append({"page": page_num, "text": text})

        self.vector_store.save()
        return " ".join(all_text)

    def ask(self, question):
        """Retrieves context and generates an answer."""
        relevant_chunks = self.vector_store.search(question, top_k=4)
        
        # Check confidence/threshold
        if not relevant_chunks or relevant_chunks[0]['distance'] > 1.5: # FAISS L2 distance threshold
             return "I don't have enough information in the notes to answer this.", []

        answer = self.llm_manager.generate_answer(question, relevant_chunks)
        return answer, relevant_chunks
