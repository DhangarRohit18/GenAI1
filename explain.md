# NoteVault AI: System Technical Overview

## 1. Introduction
NoteVault AI is a Document Intelligence system designed to act as a personal study assistant for students. It processes handwritten PDF notes and allows users to query their content using natural language. The system's primary focus is **precision and honesty**, ensuring that all answers are grounded in the provided notes without using external AI knowledge.

## 2. Core Pipeline
The system operates through a multi-stage RAG (Retrieval-Augmented Generation) pipeline:

### Stage 1: Document Ingestion & Preprocessing
When a PDF is uploaded, it is converted into high-resolution images using `PyMuPDF`. Each image undergoes OpenCV-based enhancement (contrast adjustment) to prepare for OCR.

### Stage 2: Handwritten OCR
We utilize **PaddleOCR**, which employs a Deep Learning-based approach (using a CRNN-based recognition model) that is highly resilient to varying handwriting styles, tilted text, and messy annotations. This stage outputs text blocks with specific confidence scores and coordinates.

### Stage 3: Mathematical Retrieval (From Scratch)
Unlike typical RAG systems that use pre-trained black-box embeddings, NoteVault AI implements a **Custom TF-IDF & Cosine Similarity Engine** using only NumPy. 
- **TF-IDF**: The system calculates the 'importance' of words within your specific notes. This "trains" the search engine to understand which terms are unique to your study material.
- **Cosine Similarity**: We manually implement the dot-product math to compare your question with the indexed note sections.

### Stage 4: Low-Level Local Inference
We bypass high-level wrappers like Ollama or LangChain. The system talks directly to the **TinyLlama** model using the `transformers` library. This allows us to manually tune the **generation parameters** (Temperature, Top-P, Sampling) to ensure the AI remains strictly grounded in the provided notes.

### Stage 5: Local Inference
The retrieved context chunks and the user's question are packaged into a "Strict Grounding Prompt." This prompt is sent to a **local Mistral 7B LLM** running via Ollama. The LLM is instructed to generate an answer *only* using the provided context and to cite specific pages.

## 3. Key Design Decisions
- **Fully Offline**: By using Ollama and local embedding models, the system honors the "No Internet" rule, ensuring privacy and reliability in a competition environment.
- **Explainability**: Every answer is accompanied by the exact text snippets and page numbers used to generate it, allowing the user to verify the AI's "thought process."
- **Robustness**: The use of `all-MiniLM` provides a great balance between performance on consumer hardware and semantic accuracy.

## 4. Future Enhancements
- **Multi-modal Retrieval**: Using CLIP or similar models to search for diagrams and drawings directly.
- **Table Parsing**: Enhanced structural analysis for complex handwritten tables.
- **Incremental Indexing**: Allowing the user to add new notes to an existing session seamlessly.

---
**NoteVault AI** - *Document Intelligence made local, private, and precise.*
